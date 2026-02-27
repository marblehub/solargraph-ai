"""
ingest_literature.py
────────────────────
Heterogeneous data ingestion pipeline: OpenAlex API → LLM entity extraction → RDF graph.

Pipeline:
  1. Query OpenAlex (free, no API key needed) for recent PV papers
  2. Send title + abstract to Groq LLM
  3. LLM extracts structured entities (materials, processes, metrics, institutions)
  4. New entities are inserted as RDF triples into the knowledge graph
  5. Updated graph is re-pickled and graph.html is regenerated

This addresses the job requirement:
  "develop data and agent architectures for integrating heterogeneous sources
   and enabling knowledge extraction"

Usage:
  python ingest_literature.py                  # default: 10 papers on perovskite
  python ingest_literature.py --query "CIGS thin film" --limit 5
  python ingest_literature.py --dry-run        # extract only, don't write to graph
  python ingest_literature.py --query "silicon heterojunction" --limit 3 --dry-run
"""

import os
import re
import json
import time
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("ingest")

# ── Namespaces ────────────────────────────────────────────────────────────────
PV  = Namespace("http://example.org/pvsolar#")
LIT = Namespace("http://example.org/pvsolar/literature#")   # for paper provenance nodes

# ── Paths ─────────────────────────────────────────────────────────────────────
GRAPH_PKL  = Path(__file__).parent / "graph.pkl"
ONTOLOGY   = Path(__file__).parent / "ontology.ttl"
INGESTED   = Path(__file__).parent / "ingested_papers.json"   # audit log

# ── OpenAlex ──────────────────────────────────────────────────────────────────
OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_HEADERS = {"User-Agent": "SolarGraphAI/1.0 (mailto:your@email.com)"}


def fetch_papers(query: str = "perovskite solar cell efficiency",
                 limit: int = 10) -> list[dict]:
    """
    Fetch recent open-access PV papers from OpenAlex.
    Returns a list of {id, title, abstract, doi, year, authors, institution_names}.
    No API key required.
    """
    params = {
        "search":   query,
        "per-page": min(limit, 25),
        "filter":   "has_abstract:true",
        "sort":     "cited_by_count:desc",
        "select":   "id,title,abstract_inverted_index,doi,publication_year,authorships",
    }
    log.info("Querying OpenAlex: '%s' (limit=%d)", query, limit)
    try:
        resp = requests.get(OPENALEX_BASE, params=params,
                            headers=OPENALEX_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("OpenAlex request failed: %s", e)
        return []

    works = resp.json().get("results", [])
    papers = []
    for w in works:
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index") or {})
        if not abstract or len(abstract) < 80:
            continue
        institutions = []
        for auth in (w.get("authorships") or []):
            for inst in (auth.get("institutions") or []):
                name = inst.get("display_name")
                if name:
                    institutions.append(name)
        papers.append({
            "id":           w.get("id", ""),
            "doi":          w.get("doi", ""),
            "title":        w.get("title", "").strip(),
            "abstract":     abstract[:1500],     # cap for LLM context
            "year":         w.get("publication_year"),
            "institutions": list(set(institutions)),
        })
    log.info("Fetched %d papers with abstracts", len(papers))
    return papers


def _reconstruct_abstract(inv_index: dict) -> str:
    """OpenAlex stores abstracts as inverted index {word: [positions]}. Reconstruct."""
    if not inv_index:
        return ""
    word_pos = []
    for word, positions in inv_index.items():
        for pos in positions:
            word_pos.append((pos, word))
    word_pos.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_pos)


# ── LLM Entity Extraction ─────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a materials science knowledge extraction system.
Given the title and abstract of a PV solar energy paper, extract structured entities.

Return ONLY a JSON object with these keys (omit keys with no findings):
{
  "absorbers": [
    {"name": "...", "bandgap_eV": 1.55, "crystal": "...", "description": "..."}
  ],
  "architectures": [
    {"name": "...", "efficiency_pct": 26.1, "description": "..."}
  ],
  "fabrication_processes": [
    {"name": "...", "temperature_C": 150, "description": "..."}
  ],
  "characterisation_techniques": [
    {"name": "...", "description": "..."}
  ],
  "defects": [
    {"name": "...", "description": "...", "affects_metrics": ["PCE", "Voc"]}
  ],
  "institutions": [
    {"name": "...", "country": "..."}
  ],
  "researchers": [
    {"name": "...", "institution": "..."}
  ],
  "performance_metrics": [
    {"name": "...", "value": "...", "unit": "..."}
  ]
}

Rules:
- Only extract entities clearly stated in the text. No inference.
- Use standard names (e.g. "MAPbI3" not "methylammonium lead triiodide")
- Numeric values must be numbers, not strings
- Return valid JSON only, no markdown fences
"""


def extract_entities(paper: dict, groq_client: Groq,
                     model: str = "llama-3.1-8b-instant") -> dict:
    """Send paper to Groq LLM and return extracted entity dict."""
    text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content or "{}"
        # Strip possible markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("JSON parse error for paper '%s': %s", paper["title"][:50], e)
        return {}
    except Exception as e:
        log.error("LLM extraction error: %s", e)
        return {}


# ── RDF Triple Construction ───────────────────────────────────────────────────
def _safe_uri(name: str) -> str:
    """Convert a name to a safe URI fragment."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name.strip())


def _add_paper_provenance(g: Graph, paper: dict) -> URIRef:
    """Add a LiteratureSource node for provenance tracking."""
    pid  = _safe_uri(paper["id"].split("/")[-1] if paper["id"] else paper["title"][:30])
    node = LIT[pid]
    g.add((node, RDF.type,       LIT.LiteratureSource))
    g.add((node, PV.name,        Literal(paper["title"])))
    g.add((node, LIT.doi,        Literal(paper.get("doi", ""))))
    g.add((node, LIT.year,       Literal(str(paper.get("year", "")))))
    g.add((node, LIT.ingestedAt, Literal(datetime.now(timezone.utc).isoformat())))
    return node


def entities_to_triples(g: Graph, entities: dict,
                         paper_node: URIRef) -> int:
    """
    Convert extracted entity dict → RDF triples, added to graph g.
    Returns count of new triples added.
    """
    added = 0

    # ── Absorbers ─────────────────────────────────────────────────────────────
    for item in entities.get("absorbers", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.Absorber) not in g:
            g.add((node, RDF.type,    PV.Absorber))
            g.add((node, PV.name,     Literal(name)))
            added += 1
        if item.get("bandgap_eV"):
            g.add((node, PV.bandgap_eV, Literal(float(item["bandgap_eV"]), datatype=XSD.decimal)))
            added += 1
        if item.get("crystal"):
            g.add((node, PV.crystalStructure, Literal(str(item["crystal"]))))
            added += 1
        if item.get("description"):
            g.add((node, PV.description, Literal(str(item["description"])[:300])))
            added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    # ── Cell Architectures ────────────────────────────────────────────────────
    for item in entities.get("architectures", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.CellArchitecture) not in g:
            g.add((node, RDF.type, PV.CellArchitecture))
            g.add((node, PV.name,  Literal(name)))
            added += 1
        if item.get("efficiency_pct"):
            g.add((node, PV.recordEfficiency_pct,
                   Literal(float(item["efficiency_pct"]), datatype=XSD.decimal)))
            added += 1
        if item.get("description"):
            g.add((node, PV.description, Literal(str(item["description"])[:300])))
            added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    # ── Fabrication Processes ─────────────────────────────────────────────────
    for item in entities.get("fabrication_processes", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.FabricationProcess) not in g:
            g.add((node, RDF.type, PV.FabricationProcess))
            g.add((node, PV.name,  Literal(name)))
            added += 1
        if item.get("temperature_C"):
            g.add((node, PV.depositionTemp_C,
                   Literal(float(item["temperature_C"]), datatype=XSD.decimal)))
            added += 1
        if item.get("description"):
            g.add((node, PV.description, Literal(str(item["description"])[:300])))
            added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    # ── Characterisation Techniques ───────────────────────────────────────────
    for item in entities.get("characterisation_techniques", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.CharacterisationTechnique) not in g:
            g.add((node, RDF.type, PV.CharacterisationTechnique))
            g.add((node, PV.name,  Literal(name)))
            added += 1
        if item.get("description"):
            g.add((node, PV.description, Literal(str(item["description"])[:300])))
            added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    # ── Defects ───────────────────────────────────────────────────────────────
    for item in entities.get("defects", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.Defect) not in g:
            g.add((node, RDF.type, PV.Defect))
            g.add((node, PV.name,  Literal(name)))
            added += 1
        if item.get("description"):
            g.add((node, PV.description, Literal(str(item["description"])[:300])))
            added += 1
        for metric in item.get("affects_metrics", []):
            metric_node = PV[_safe_uri(metric)]
            g.add((node, PV.affectsMetric, metric_node)); added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    # ── Institutions ──────────────────────────────────────────────────────────
    for item in entities.get("institutions", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.Institution) not in g:
            g.add((node, RDF.type,    PV.Institution))
            g.add((node, PV.name,     Literal(name)))
            added += 1
        if item.get("country"):
            g.add((node, PV.country, Literal(str(item["country"]))))
            added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    # ── Researchers ───────────────────────────────────────────────────────────
    for item in entities.get("researchers", []):
        name = item.get("name", "").strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        if (node, RDF.type, PV.Researcher) not in g:
            g.add((node, RDF.type, PV.Researcher))
            g.add((node, PV.name,  Literal(name)))
            added += 1
        if item.get("institution"):
            inst_node = PV[_safe_uri(item["institution"])]
            g.add((node, PV.studiedAt, inst_node)); added += 1
        g.add((node, LIT.mentionedIn, paper_node)); added += 1

    return added


# ── Audit log ─────────────────────────────────────────────────────────────────
def _load_ingested() -> dict:
    if INGESTED.exists():
        try:
            return json.loads(INGESTED.read_text())
        except Exception:
            return {}
    return {}


def _save_ingested(log_dict: dict):
    INGESTED.write_text(json.dumps(log_dict, indent=2))


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_ingestion(query: str = "perovskite solar cell efficiency",
                  limit: int = 10,
                  dry_run: bool = False,
                  model: str = "llama-3.1-8b-instant") -> dict:
    """
    Full pipeline: fetch → extract → insert → save.
    Returns summary dict.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")
    groq_client = Groq(api_key=api_key)

    # Load existing graph
    from build_graph import load_graph, save_graph
    g = load_graph()
    triples_before = len(g)
    log.info("Graph loaded: %d triples", triples_before)

    # Load audit log
    ingested_log = _load_ingested()

    # Fetch papers
    papers = fetch_papers(query=query, limit=limit)
    if not papers:
        return {"status": "no_papers", "papers_processed": 0, "triples_added": 0}

    stats = {
        "papers_fetched":    len(papers),
        "papers_skipped":    0,
        "papers_processed":  0,
        "triples_added":     0,
        "entities_extracted": {},
        "dry_run":           dry_run,
    }

    for i, paper in enumerate(papers, 1):
        paper_id = paper["id"] or hashlib.md5(paper["title"].encode()).hexdigest()

        # Skip already-ingested papers
        if paper_id in ingested_log:
            log.info("[%d/%d] Already ingested — skipping: %s",
                     i, len(papers), paper["title"][:60])
            stats["papers_skipped"] += 1
            continue

        log.info("[%d/%d] Processing: %s", i, len(papers), paper["title"][:70])

        # LLM extraction
        entities = extract_entities(paper, groq_client, model=model)
        if not entities:
            log.warning("  No entities extracted, skipping.")
            continue

        # Count entity types found
        for etype, items in entities.items():
            if items:
                stats["entities_extracted"][etype] = \
                    stats["entities_extracted"].get(etype, 0) + len(items)

        if dry_run:
            log.info("  [DRY RUN] Would add entities: %s",
                     {k: len(v) for k, v in entities.items() if v})
        else:
            paper_node = _add_paper_provenance(g, paper)
            new_triples = entities_to_triples(g, entities, paper_node)
            stats["triples_added"]   += new_triples
            stats["papers_processed"] += 1
            ingested_log[paper_id] = {
                "title":      paper["title"],
                "doi":        paper.get("doi", ""),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "triples_added": new_triples,
            }
            log.info("  ✓ Added %d triples", new_triples)

        # Polite delay for OpenAlex API
        time.sleep(0.5)

    if not dry_run and stats["papers_processed"] > 0:
        triples_after = len(g)
        log.info("Saving updated graph: %d → %d triples (+%d)",
                 triples_before, triples_after,
                 triples_after - triples_before)
        save_graph(g)
        _save_ingested(ingested_log)

        # Regenerate graph visualisation
        try:
            from query_engine import QueryEngine
            from visualize    import generate_graph_html
            qe = QueryEngine(g)
            generate_graph_html(qe)
            log.info("Graph visualisation regenerated.")
        except Exception as e:
            log.warning("Could not regenerate graph HTML: %s", e)

    log.info("Ingestion complete: %s", stats)
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenAlex → LLM entity extraction → RDF knowledge graph ingestion"
    )
    parser.add_argument("--query",   default="perovskite solar cell efficiency",
                        help="OpenAlex search query (default: 'perovskite solar cell efficiency')")
    parser.add_argument("--limit",   type=int, default=10,
                        help="Number of papers to fetch (default: 10)")
    parser.add_argument("--model",   default="llama-3.1-8b-instant",
                        help="Groq model to use for extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract entities but don't write to graph")
    args = parser.parse_args()

    result = run_ingestion(
        query   = args.query,
        limit   = args.limit,
        dry_run = args.dry_run,
        model   = args.model,
    )

    print("\n── Ingestion Summary ──────────────────────────────")
    print(f"  Papers fetched:      {result.get('papers_fetched', 0)}")
    print(f"  Papers skipped:      {result.get('papers_skipped', 0)}")
    print(f"  Papers processed:    {result.get('papers_processed', 0)}")
    print(f"  New triples added:   {result.get('triples_added', 0)}")
    print(f"  Dry run:             {result.get('dry_run', False)}")
    if result.get("entities_extracted"):
        print("  Entity types found:")
        for etype, count in result["entities_extracted"].items():
            print(f"    {etype}: {count}")
    print("───────────────────────────────────────────────────")
