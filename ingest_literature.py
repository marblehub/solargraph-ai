"""
ingest_literature.py
────────────────────
Heterogeneous data ingestion pipeline: OpenAlex API → LLM entity extraction → RDF graph.

Pipeline:
  1. Query OpenAlex (free, no API key needed) for recent PV papers
  2. Send title + abstract to Groq LLM
  3. LLM extracts structured entities (materials, processes, metrics, institutions)
  4. New entities are inserted as RDF triples into the knowledge graph
  5. Validated additions are persisted to ingested_data.ttl and graph.pkl

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
from decimal import Decimal, InvalidOperation

import requests
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, XSD
from rdflib.namespace import DCTERMS, SKOS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("ingest")

# ── Namespaces ────────────────────────────────────────────────────────────────
PV = Namespace("https://w3id.org/pvsolar#")
LIT = Namespace("https://w3id.org/pvsolar/literature#")
PROV = Namespace("http://www.w3.org/ns/prov#")
QUDT = Namespace("http://qudt.org/schema/qudt/")
UNIT = Namespace("http://qudt.org/vocab/unit/")

# ── Paths ─────────────────────────────────────────────────────────────────────
INGESTED_DATA = Path(__file__).parent / "ingested_data.ttl"
INGESTED = Path(__file__).parent / "ingested_papers.json"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# ── OpenAlex ──────────────────────────────────────────────────────────────────
OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_HEADERS = {"User-Agent": "SolarGraphAI/1.0 (https://w3id.org/pvsolar)"}

ENTITY_KEYS = (
    "absorbers",
    "architectures",
    "fabrication_processes",
    "characterisation_techniques",
    "defects",
    "institutions",
    "researchers",
    "performance_metrics",
)

METRIC_IRIS = {
    "pce": PV.PCE,
    "powerconversionefficiency": PV.PCE,
    "voc": PV.Voc,
    "opencircuitvoltage": PV.Voc,
    "jsc": PV.Jsc,
    "shortcircuitcurrentdensity": PV.Jsc,
    "ff": PV.FF,
    "fillfactor": PV.FF,
    "carrierlifetime": PV.CarrierLifetime,
    "minoritycarrierlifetime": PV.CarrierLifetime,
    "hysteresis": PV.Hysteresis,
    "hysteresisindex": PV.Hysteresis,
    "jvhysteresisindex": PV.Hysteresis,
}

UNIT_MAPPINGS = {
    "%": (UNIT.PERCENT, Decimal("1")),
    "percent": (UNIT.PERCENT, Decimal("1")),
    "percentage": (UNIT.PERCENT, Decimal("1")),
    "v": (UNIT.V, Decimal("1")),
    "volt": (UNIT.V, Decimal("1")),
    "volts": (UNIT.V, Decimal("1")),
    "a/m2": (UNIT["A-PER-M2"], Decimal("1")),
    "a/m^2": (UNIT["A-PER-M2"], Decimal("1")),
    "ma/cm2": (UNIT["A-PER-M2"], Decimal("10")),
    "ma/cm^2": (UNIT["A-PER-M2"], Decimal("10")),
    "s": (UNIT.SEC, Decimal("1")),
    "sec": (UNIT.SEC, Decimal("1")),
    "second": (UNIT.SEC, Decimal("1")),
    "seconds": (UNIT.SEC, Decimal("1")),
    "ms": (UNIT.MilliSEC, Decimal("1")),
    "millisecond": (UNIT.MilliSEC, Decimal("1")),
    "milliseconds": (UNIT.MilliSEC, Decimal("1")),
    "us": (UNIT.MicroSEC, Decimal("1")),
    "microsecond": (UNIT.MicroSEC, Decimal("1")),
    "microseconds": (UNIT.MicroSEC, Decimal("1")),
    "fraction": (UNIT.FRACTION, Decimal("1")),
    "dimensionless": (UNIT.UNITLESS, Decimal("1")),
    "unitless": (UNIT.UNITLESS, Decimal("1")),
    "1": (UNIT.UNITLESS, Decimal("1")),
}


def fetch_papers(query: str = "perovskite solar cell efficiency",
                 limit: int = 10) -> list[dict]:
    """Fetch PV papers with abstracts from OpenAlex; an API key is optional."""
    if not query.strip():
        raise ValueError("OpenAlex query must not be empty.")
    if not 1 <= limit <= 25:
        raise ValueError("OpenAlex limit must be between 1 and 25.")

    params = {
        "search": query.strip(),
        "per-page": limit,
        "filter": "has_abstract:true",
        "sort": "cited_by_count:desc",
        "select": "id,title,abstract_inverted_index,doi,publication_year,authorships",
    }
    openalex_api_key = os.getenv("OPENALEX_API_KEY")
    if openalex_api_key:
        params["api_key"] = openalex_api_key
    log.info("Querying OpenAlex: '%s' (limit=%d)", query, limit)
    response = None
    for attempt in range(3):
        try:
            response = requests.get(
                OPENALEX_BASE,
                params=params,
                headers=OPENALEX_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            break
        except requests.RequestException as error:
            status_code = getattr(error.response, "status_code", None)
            if status_code == 429 and attempt < 2:
                retry_after = getattr(error.response, "headers", {}).get("Retry-After")
                delay = float(retry_after) if retry_after else 2 ** attempt
                log.warning("OpenAlex rate limited; retrying in %.1f seconds", delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"OpenAlex request failed: {error}") from error

    if response is None:
        raise RuntimeError("OpenAlex request failed without a response.")
    try:
        works = response.json().get("results", [])
    except requests.JSONDecodeError as error:
        raise RuntimeError("OpenAlex returned invalid JSON.") from error
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
            "institutions": sorted(set(institutions)),
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
    {"name": "...", "bandgap": 1.55, "crystal": "...", "description": "..."}
  ],
  "architectures": [
    {"name": "...", "efficiency": 26.1, "description": "..."}
  ],
  "fabrication_processes": [
    {"name": "...", "deposition_temperature": 150, "description": "..."}
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
    {"name": "PCE", "value": 25.2, "unit": "%"}
  ]
}

Rules:
- Only extract entities clearly stated in the text. No inference.
- Use standard names (e.g. "MAPbI3" not "methylammonium lead triiodide")
- Numeric values must be numbers, not strings
- Bandgap values are in eV, efficiency values are percentages, and deposition temperatures are in degrees Celsius
- Performance metric units must use one of: %, V, A/m2, mA/cm2, s, ms, us, fraction, dimensionless
- Return valid JSON only, no markdown fences
"""


def _normalise_entities(payload: object) -> dict[str, list[dict]]:
    """Keep only supported extraction collections containing JSON objects."""
    if not isinstance(payload, dict):
        raise ValueError("LLM extraction response must be a JSON object.")

    normalised = {}
    for key in ENTITY_KEYS:
        items = payload.get(key, [])
        if items is None:
            continue
        if not isinstance(items, list):
            log.warning("Ignoring non-list extraction field: %s", key)
            continue
        valid_items = [item for item in items if isinstance(item, dict)]
        if len(valid_items) != len(items):
            log.warning("Ignoring malformed entries in extraction field: %s", key)
        if valid_items:
            normalised[key] = valid_items
    return normalised


def extract_entities(
    paper: dict,
    groq_client: Groq,
    model: str = DEFAULT_MODEL,
) -> dict[str, list[dict]]:
    """Extract schema-shaped entities from one paper using Groq JSON mode."""
    paper_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": paper_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1200,
        )
        raw = response.choices[0].message.content or "{}"
        return _normalise_entities(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"Invalid extraction response for '{paper['title'][:50]}': {error}"
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"Groq extraction failed for '{paper['title'][:50]}': {error}"
        ) from error


# ── RDF Triple Construction ───────────────────────────────────────────────────
def _safe_uri(name: str) -> str:
    """Convert a name to a stable URI fragment."""
    fragment = re.sub(r"[^A-Za-z0-9_\-]", "_", name.strip()).strip("_")
    if fragment:
        return fragment
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"entity_{digest}"


def _normalise_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _normalise_unit(value: str) -> str:
    return (
        value.strip()
        .casefold()
        .replace("μ", "u")
        .replace("µ", "u")
        .replace("²", "2")
        .replace(" ", "")
    )


def _decimal_value(value: object, field_name: str) -> Decimal | None:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        log.warning("Ignoring invalid %s value: %r", field_name, value)
        return None
    if not decimal_value.is_finite():
        log.warning("Ignoring non-finite %s value: %r", field_name, value)
        return None
    return decimal_value


def _add_paper_provenance(graph: Graph, paper: dict) -> URIRef:
    """Add a literature source node used by statement-level provenance."""
    paper_id = _safe_uri(
        paper["id"].split("/")[-1] if paper["id"] else paper["title"][:30]
    )
    node = LIT[paper_id]
    graph.add((node, RDF.type, PROV.Entity))
    graph.add((node, RDF.type, LIT.LiteratureSource))
    graph.add((node, DCTERMS.title, Literal(paper["title"])))
    if paper.get("doi"):
        graph.add((node, DCTERMS.identifier, Literal(paper["doi"])))
    if paper.get("year"):
        graph.add((node, DCTERMS.issued, Literal(str(paper["year"]), datatype=XSD.gYear)))
    graph.add(
        (
            node,
            PROV.generatedAtTime,
            Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime),
        )
    )
    return node


def _add_literal_with_provenance(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    value: Literal,
    source: URIRef,
) -> int:
    """Add one literal assertion and a deterministic PROV-O RDF statement."""
    triple = (subject, predicate, value)
    is_new = triple not in graph
    graph.add(triple)

    digest_input = "\u241f".join(
        term.n3() for term in (subject, predicate, value, source)
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:20]
    statement = PV[f"assertion_{digest}"]
    graph.add((statement, RDF.type, RDF.Statement))
    graph.add((statement, RDF.type, PROV.Entity))
    graph.add((statement, RDF.subject, subject))
    graph.add((statement, RDF.predicate, predicate))
    graph.add((statement, RDF.object, value))
    graph.add((statement, PROV.wasDerivedFrom, source))
    graph.add((statement, PROV.wasAttributedTo, PV.Marblehub))
    graph.add(
        (
            statement,
            PROV.generatedAtTime,
            Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime),
        )
    )
    return int(is_new)


def _ensure_entity(
    graph: Graph,
    node: URIRef,
    entity_class: URIRef,
    name: str,
    description: str,
    source: URIRef,
) -> int:
    """Ensure an entity has its type and provenance-backed preferred labels."""
    added = 0
    type_triple = (node, RDF.type, entity_class)
    if type_triple not in graph:
        graph.add(type_triple)
        added += 1

    label = Literal(name, lang="en")
    if not any(graph.objects(node, RDFS.label)):
        added += _add_literal_with_provenance(graph, node, RDFS.label, label, source)
    if not any(graph.objects(node, SKOS.prefLabel)):
        added += _add_literal_with_provenance(graph, node, SKOS.prefLabel, label, source)
    if description and not any(graph.objects(node, DCTERMS.description)):
        added += _add_literal_with_provenance(
            graph,
            node,
            DCTERMS.description,
            Literal(description[:300]),
            source,
        )
    return added


def _add_metric_observation(
    graph: Graph,
    item: dict,
    paper_node: URIRef,
) -> None:
    name = str(item.get("name", "")).strip()
    numeric_value = _decimal_value(item.get("value"), "performance metric")
    unit_text = str(item.get("unit", "")).strip()
    unit_mapping = UNIT_MAPPINGS.get(_normalise_unit(unit_text))
    if not name or numeric_value is None:
        return
    if unit_mapping is None:
        log.warning("Ignoring performance metric with unsupported unit: %r", unit_text)
        return

    unit_iri, conversion_factor = unit_mapping
    metric_node = METRIC_IRIS.get(_normalise_token(name), PV[_safe_uri(name)])
    _ensure_entity(
        graph,
        metric_node,
        PV.PerformanceMetric,
        name,
        str(item.get("description", "")),
        paper_node,
    )
    if not any(graph.objects(metric_node, QUDT.hasUnit)):
        graph.add((metric_node, QUDT.hasUnit, unit_iri))

    converted_value = numeric_value * conversion_factor
    digest_input = "\u241f".join(
        (metric_node.n3(), converted_value.to_eng_string(), unit_iri.n3(), paper_node.n3())
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:20]
    observation = LIT[f"observation_{digest}"]
    graph.add((observation, RDF.type, QUDT.QuantityValue))
    graph.add((observation, QUDT.hasUnit, unit_iri))
    graph.add((observation, PROV.wasDerivedFrom, paper_node))
    graph.add((observation, PROV.wasAttributedTo, PV.Marblehub))
    graph.add(
        (
            observation,
            PROV.generatedAtTime,
            Literal(datetime.now(timezone.utc).isoformat(), datatype=XSD.dateTime),
        )
    )
    _add_literal_with_provenance(
        graph,
        observation,
        QUDT.numericValue,
        Literal(converted_value, datatype=XSD.decimal),
        paper_node,
    )
    graph.add((metric_node, PV.hasReportedValue, observation))
    graph.add((metric_node, LIT.mentionedIn, paper_node))


def entities_to_triples(graph: Graph, entities: dict, paper_node: URIRef) -> int:
    """Convert extracted entities into schema-aligned RDF with provenance."""
    triples_before = len(graph)

    for item in entities.get("absorbers", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        _ensure_entity(
            graph, node, PV.Absorber, name, str(item.get("description", "")), paper_node
        )
        if item.get("bandgap") is not None:
            bandgap = _decimal_value(item["bandgap"], "bandgap")
            if bandgap is not None:
                _add_literal_with_provenance(
                    graph, node, PV.bandgap,
                    Literal(bandgap, datatype=XSD.decimal), paper_node
                )
        if item.get("crystal"):
            _add_literal_with_provenance(
                graph, node, PV.crystalStructure,
                Literal(str(item["crystal"])), paper_node
            )
        graph.add((node, LIT.mentionedIn, paper_node))

    for item in entities.get("architectures", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        _ensure_entity(
            graph, node, PV.CellArchitecture, name,
            str(item.get("description", "")), paper_node
        )
        if item.get("efficiency") is not None:
            efficiency = _decimal_value(item["efficiency"], "efficiency")
            if efficiency is not None:
                _add_literal_with_provenance(
                    graph, node, PV.recordEfficiency,
                    Literal(efficiency, datatype=XSD.decimal), paper_node
                )
        graph.add((node, LIT.mentionedIn, paper_node))

    for item in entities.get("fabrication_processes", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        _ensure_entity(
            graph, node, PV.FabricationProcess, name,
            str(item.get("description", "")), paper_node
        )
        if item.get("deposition_temperature") is not None:
            temperature = _decimal_value(
                item["deposition_temperature"], "deposition temperature"
            )
            if temperature is not None:
                _add_literal_with_provenance(
                    graph, node, PV.depositionTemperature,
                    Literal(temperature, datatype=XSD.decimal), paper_node
                )
        graph.add((node, LIT.mentionedIn, paper_node))

    simple_groups = (
        ("characterisation_techniques", PV.CharacterisationTechnique),
        ("institutions", PV.Institution),
        ("researchers", PV.Researcher),
    )
    for group_name, entity_class in simple_groups:
        for item in entities.get(group_name, []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            node = PV[_safe_uri(name)]
            _ensure_entity(
                graph, node, entity_class, name,
                str(item.get("description", "")), paper_node
            )
            if group_name == "institutions" and item.get("country"):
                _add_literal_with_provenance(
                    graph, node, PV.country, Literal(str(item["country"])), paper_node
                )
            if group_name == "researchers" and item.get("institution"):
                institution_name = str(item["institution"]).strip()
                institution = PV[_safe_uri(institution_name)]
                _ensure_entity(
                    graph, institution, PV.Institution, institution_name, "", paper_node
                )
                graph.add((node, PV.studiedAt, institution))
            graph.add((node, LIT.mentionedIn, paper_node))

    for item in entities.get("defects", []):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        node = PV[_safe_uri(name)]
        _ensure_entity(
            graph, node, PV.Defect, name, str(item.get("description", "")), paper_node
        )
        for metric_name in item.get("affects_metrics", []):
            metric_node = METRIC_IRIS.get(_normalise_token(str(metric_name)))
            if metric_node is None:
                log.warning("Ignoring unknown affected metric: %r", metric_name)
                continue
            graph.add((node, PV.affectsMetric, metric_node))
        graph.add((node, LIT.mentionedIn, paper_node))

    for item in entities.get("performance_metrics", []):
        _add_metric_observation(graph, item, paper_node)

    return len(graph) - triples_before


# ── Audit log ─────────────────────────────────────────────────────────────────
def _load_ingested() -> dict:
    if INGESTED.exists():
        try:
            return json.loads(INGESTED.read_text())
        except Exception:
            return {}
    return {}


def _save_ingested(log_dict: dict) -> None:
    temporary_path = INGESTED.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(log_dict, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(INGESTED)


def _persist_ingested_triples(graph: Graph, baseline: set[tuple]) -> int:
    """Persist graph additions as a separate ABox so rebuilds retain them."""
    additions = set(graph) - baseline
    if not additions:
        return 0

    persisted = Graph()
    if INGESTED_DATA.exists():
        persisted.parse(str(INGESTED_DATA), format="turtle")
    persisted.bind("pv", PV)
    persisted.bind("lit", LIT)
    persisted.bind("prov", PROV)
    persisted.bind("qudt", QUDT)
    persisted.bind("unit", UNIT)
    persisted.bind("dcterms", DCTERMS)
    persisted.bind("skos", SKOS)
    for triple in additions:
        persisted.add(triple)

    temporary_path = INGESTED_DATA.with_suffix(".ttl.tmp")
    persisted.serialize(destination=str(temporary_path), format="turtle")
    temporary_path.replace(INGESTED_DATA)
    return len(additions)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_ingestion(
    query: str = "perovskite solar cell efficiency",
    limit: int = 10,
    dry_run: bool = False,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Fetch, extract, validate, and atomically persist literature assertions."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")
    groq_client = Groq(api_key=api_key)

    from build_graph import load_graph, save_graph, validate_graph

    graph = load_graph()
    baseline = set(graph)
    triples_before = len(graph)
    log.info("Graph loaded: %d triples", triples_before)
    ingested_log = _load_ingested()

    papers = fetch_papers(query=query, limit=limit)
    if not papers:
        return {
            "status": "no_papers",
            "papers_fetched": 0,
            "papers_processed": 0,
            "papers_failed": 0,
            "triples_added": 0,
            "dry_run": dry_run,
        }

    stats = {
        "status": "ok",
        "model": model,
        "papers_fetched": len(papers),
        "papers_skipped": 0,
        "papers_processed": 0,
        "papers_failed": 0,
        "papers_without_entities": 0,
        "triples_added": 0,
        "triples_prepared": 0,
        "entities_extracted": {},
        "errors": [],
        "dry_run": dry_run,
    }

    for index, paper in enumerate(papers, 1):
        paper_id = paper["id"] or hashlib.sha256(
            paper["title"].encode("utf-8")
        ).hexdigest()
        if paper_id in ingested_log:
            log.info(
                "[%d/%d] Already ingested — skipping: %s",
                index, len(papers), paper["title"][:60],
            )
            stats["papers_skipped"] += 1
            continue

        log.info(
            "[%d/%d] Processing: %s",
            index, len(papers), paper["title"][:70],
        )
        try:
            entities = extract_entities(paper, groq_client, model=model)
        except RuntimeError as error:
            log.error("%s", error)
            stats["papers_failed"] += 1
            stats["errors"].append(str(error))
            continue

        stats["papers_processed"] += 1
        if not entities:
            stats["papers_without_entities"] += 1
            log.info("  No supported entities extracted.")
            continue

        for entity_type, items in entities.items():
            stats["entities_extracted"][entity_type] = (
                stats["entities_extracted"].get(entity_type, 0) + len(items)
            )

        paper_triples_before = len(graph)
        paper_node = _add_paper_provenance(graph, paper)
        entities_to_triples(graph, entities, paper_node)
        new_triples = len(graph) - paper_triples_before
        stats["triples_prepared"] += new_triples

        if dry_run:
            log.info(
                "  [DRY RUN] Prepared %d triples from entities: %s",
                new_triples,
                {key: len(value) for key, value in entities.items()},
            )
            continue

        stats["triples_added"] += new_triples
        ingested_log[paper_id] = {
            "title": paper["title"],
            "doi": paper.get("doi", ""),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "triples_added": new_triples,
            "model": model,
        }
        log.info("  Added %d RDF triples", new_triples)
        time.sleep(0.5)

    if stats["papers_failed"]:
        stats["status"] = (
            "failed" if stats["papers_processed"] == 0 else "partial_failure"
        )

    if dry_run and stats["triples_prepared"] > 0:
        validate_graph(graph)
        log.info("Dry-run graph passed SHACL validation.")

    if not dry_run and stats["triples_added"] > 0:
        validate_graph(graph)
        persisted_count = _persist_ingested_triples(graph, baseline)
        if persisted_count != stats["triples_added"]:
            raise RuntimeError(
                "Persisted triple count does not match the validated graph delta."
            )
        save_graph(graph)
        _save_ingested(ingested_log)
        log.info(
            "Saved updated graph: %d → %d triples (+%d)",
            triples_before, len(graph), persisted_count,
        )

        try:
            from query_engine import QueryEngine
            from visualize import generate_graph_html

            generate_graph_html(QueryEngine(graph))
            log.info("Graph visualisation regenerated.")
        except Exception as error:
            log.warning("Could not regenerate graph HTML: %s", error)

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
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help="Groq model to use for extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract entities but don't write to graph")
    args = parser.parse_args()

    try:
        result = run_ingestion(
            query=args.query,
            limit=args.limit,
            dry_run=args.dry_run,
            model=args.model,
        )
    except Exception as error:
        parser.exit(1, f"Ingestion failed: {error}\n")

    print("\n── Ingestion Summary ──────────────────────────────")
    print(f"  Status:              {result.get('status', 'unknown')}")
    print(f"  Model:               {result.get('model', args.model)}")
    print(f"  Papers fetched:      {result.get('papers_fetched', 0)}")
    print(f"  Papers skipped:      {result.get('papers_skipped', 0)}")
    print(f"  Papers processed:    {result.get('papers_processed', 0)}")
    print(f"  Papers failed:       {result.get('papers_failed', 0)}")
    print(f"  Triples prepared:    {result.get('triples_prepared', 0)}")
    print(f"  New triples added:   {result.get('triples_added', 0)}")
    print(f"  Dry run:             {result.get('dry_run', False)}")
    if result.get("entities_extracted"):
        print("  Entity types found:")
        for etype, count in result["entities_extracted"].items():
            print(f"    {etype}: {count}")
    print("───────────────────────────────────────────────────")
