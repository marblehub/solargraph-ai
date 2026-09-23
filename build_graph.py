"""
build_graph.py
──────────────
Loads the PV Solar TBox and ABox, validates them with SHACL, and persists the
combined RDFLib graph to graph.pkl for fast reuse by the application.
"""

import logging
import pickle
from collections.abc import Iterable
from pathlib import Path

from pyshacl import validate
from rdflib import ConjunctiveGraph, Graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CORE_ONTOLOGY_PATHS = (
    BASE_DIR / "ontology.ttl",
    BASE_DIR / "data.ttl",
)
INGESTED_DATA_PATH = BASE_DIR / "ingested_data.ttl"
SHAPES_PATH = BASE_DIR / "shapes.ttl"
GRAPH_PATH = BASE_DIR / "graph.pkl"


def graph_source_paths() -> tuple[Path, ...]:
    """Return core sources plus the optional persisted literature ABox."""
    paths = list(CORE_ONTOLOGY_PATHS)
    if INGESTED_DATA_PATH.exists():
        paths.append(INGESTED_DATA_PATH)
    return tuple(paths)


def validate_graph(graph: Graph, shapes_path: Path = SHAPES_PATH) -> None:
    """Validate the combined knowledge graph and raise on any SHACL violation."""
    shapes = Graph().parse(str(shapes_path), format="turtle")
    conforms, _, report = validate(
        graph,
        shacl_graph=shapes,
        inference="rdfs",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
    )
    if not conforms:
        raise ValueError(f"SHACL validation failed:\n{report}")
    log.info("SHACL validation passed")


def build_graph(
    ontology_paths: Iterable[Path] | None = None,
    *,
    run_validation: bool = True,
) -> ConjunctiveGraph:
    graph = ConjunctiveGraph()
    paths = tuple(ontology_paths) if ontology_paths is not None else graph_source_paths()
    for path in paths:
        log.info("Loading graph data from %s", path)
        graph.parse(str(path), format="turtle")
    if run_validation:
        validate_graph(graph)
    log.info("Graph built — %d triples loaded", len(graph))
    return graph


def save_graph(graph: ConjunctiveGraph, path: Path = GRAPH_PATH) -> None:
    with open(path, "wb") as file_handle:
        pickle.dump(graph, file_handle)
    log.info("Graph saved to %s", path)


def load_graph(path: Path = GRAPH_PATH) -> ConjunctiveGraph:
    """Load the pickled graph, rebuilding when any source file is newer."""
    source_paths = (*graph_source_paths(), SHAPES_PATH)
    is_stale = path.exists() and any(
        source_path.stat().st_mtime > path.stat().st_mtime
        for source_path in source_paths
    )
    if not path.exists() or is_stale:
        reason = "missing" if not path.exists() else "stale"
        log.warning("Pickle is %s — rebuilding graph …", reason)
        graph = build_graph()
        save_graph(graph, path)
        return graph
    with open(path, "rb") as file_handle:
        graph = pickle.load(file_handle)
    log.info("Graph loaded from pickle — %d triples", len(graph))
    return graph


if __name__ == "__main__":
    knowledge_graph = build_graph()
    save_graph(knowledge_graph)
    print(f"✅  PV Solar graph built, validated, and saved ({len(knowledge_graph)} triples).")
