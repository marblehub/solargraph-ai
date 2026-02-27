"""
provenance.py  —  Triple-level answer provenance and traceability.

For every answer the agent produces, this module:
  1. Detects entity names mentioned in the answer
  2. Retrieves the supporting KG triples for those entities
  3. Returns a structured, human-readable ProvenanceRecord

Addresses the job requirement:
  "implement the necessary logic for data access, provenance,
   reproducibility, and traceability"
"""

import re, json, hashlib, logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from query_engine import QueryEngine

log = logging.getLogger(__name__)


@dataclass
class ProvenanceRecord:
    query_id:            str
    query_text:          str
    timestamp_utc:       str
    cited_entities:      list
    supporting_triples:  list   # [{subject, predicate, object}]
    sparql_queries_used: list
    agent_iterations:    int
    cached:              bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "### Provenance Record",
            f"- **Query ID:** `{self.query_id[:16]}…`",
            f"- **Timestamp:** {self.timestamp_utc}",
            f"- **Agent iterations:** {self.agent_iterations}",
            f"- **Served from cache:** {'Yes ⚡' if self.cached else 'No 🔍'}",
        ]
        if self.cited_entities:
            lines += ["", "**Cited Entities:**"]
            for e in self.cited_entities:
                lines.append(f"- `{e}`")
        if self.supporting_triples:
            lines += ["", "**Supporting Knowledge Graph Triples:**",
                      "| Subject | Predicate | Object |", "|---|---|---|"]
            for t in self.supporting_triples[:25]:
                lines.append(f"| {t['subject']} | *{t['predicate']}* | {t['object']} |")
        if self.sparql_queries_used:
            lines += ["", "**SPARQL Queries Executed:**"]
            for q in self.sparql_queries_used[:5]:
                lines.append(f"```sparql\n{q[:300]}\n```")
        return "\n".join(lines)


def build_provenance(query: str, answer: str, qe: QueryEngine,
                     agent_steps: list = None, agent_iterations: int = 1,
                     cached: bool = False) -> ProvenanceRecord:
    agent_steps = agent_steps or []

    # 1. Find entity names in the answer
    all_entities = qe.get_all_entities()
    entity_names = [e.get("name", "") for e in all_entities if e.get("name")]
    answer_lower = answer.lower()
    cited = []
    for name in entity_names:
        abbrev = _abbrev(name)
        if name.lower() in answer_lower or (abbrev and abbrev in answer_lower):
            cited.append(name)
    cited = list(dict.fromkeys(cited))[:15]

    # 2. Fetch supporting triples
    triples, seen = [], set()
    for ename in cited[:8]:
        for d in qe.get_entity_details(ename):
            pred = d.get("predLabel", "")
            obj  = d.get("objectName") or d.get("dataValue", "")
            if pred and obj and pred not in ("name", "description"):
                key = (ename, pred, str(obj))
                if key not in seen:
                    seen.add(key)
                    triples.append({"subject": ename, "predicate": pred, "object": str(obj)})

    # 3. Collect SPARQL queries from agent steps
    sparql_used = [s["args"].get("query", "") for s in agent_steps
                   if s.get("tool") == "sparql_query" and s.get("args", {}).get("query")]

    return ProvenanceRecord(
        query_id            = hashlib.sha256(query.encode()).hexdigest(),
        query_text          = query,
        timestamp_utc       = datetime.now(timezone.utc).isoformat(),
        cited_entities      = cited,
        supporting_triples  = triples[:25],
        sparql_queries_used = sparql_used,
        agent_iterations    = agent_iterations,
        cached              = cached,
    )


def _abbrev(name: str) -> str:
    m = re.search(r'\(([^)]{2,20})\)', name)
    return m.group(1).lower() if m else ""
