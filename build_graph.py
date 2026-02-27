"""
build_graph.py
──────────────
Loads ontology.ttl into an RDFLib ConjunctiveGraph, validates it,
and persists it to graph.pkl for fast reuse by the rest of the app.
"""

import pickle
import logging
from pathlib import Path
from rdflib import ConjunctiveGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
TTL_PATH   = BASE_DIR / "ontology.ttl"
GRAPH_PATH = BASE_DIR / "graph.pkl"


def build_graph(ttl_path: Path = TTL_PATH) -> ConjunctiveGraph:
    log.info("Building graph from %s", ttl_path)
    g = ConjunctiveGraph()
    g.parse(str(ttl_path), format="turtle")
    log.info("Graph built — %d triples loaded", len(g))
    return g


def save_graph(g: ConjunctiveGraph, path: Path = GRAPH_PATH) -> None:
    with open(path, "wb") as fh:
        pickle.dump(g, fh)
    log.info("Graph saved to %s", path)


def load_graph(path: Path = GRAPH_PATH) -> ConjunctiveGraph:
    """Load pickled graph, rebuilding from TTL if pickle is missing."""
    if not path.exists():
        log.warning("Pickle not found — rebuilding graph …")
        g = build_graph()
        save_graph(g, path)
        return g
    with open(path, "rb") as fh:
        g = pickle.load(fh)
    log.info("Graph loaded from pickle — %d triples", len(g))
    return g


if __name__ == "__main__":
    g = build_graph()
    save_graph(g)
    print(f"✅  PV Solar graph built and saved ({len(g)} triples).")
