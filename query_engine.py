"""
query_engine.py
───────────────
All SPARQL queries over the PV Solar Materials Science knowledge graph.
Provides a clean Python API and a context-builder for the LLM agent.
"""

import logging
from typing import Any
from rdflib import ConjunctiveGraph

log = logging.getLogger(__name__)

PREFIXES = """
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
    PREFIX pv:   <http://example.org/pvsolar#>
"""

# Class labels to IRI suffix mapping (for domain lookups)
CLASS_MAP = {
    "absorber":         "Absorber",
    "absorbers":        "Absorber",
    "material":         "Material",
    "materials":        "Material",
    "semiconductor":    "Semiconductor",
    "semiconductors":   "Semiconductor",
    "transport":        "TransportLayer",
    "transport layer":  "TransportLayer",
    "electrode":        "Electrode",
    "electrodes":       "Electrode",
    "encapsulant":      "Encapsulant",
    "architecture":     "CellArchitecture",
    "architectures":    "CellArchitecture",
    "cell":             "CellArchitecture",
    "cells":            "CellArchitecture",
    "process":          "FabricationProcess",
    "processes":        "FabricationProcess",
    "fabrication":      "FabricationProcess",
    "characterisation": "CharacterisationTechnique",
    "characterization": "CharacterisationTechnique",
    "technique":        "CharacterisationTechnique",
    "defect":           "Defect",
    "defects":          "Defect",
    "metric":           "PerformanceMetric",
    "metrics":          "PerformanceMetric",
    "performance":      "PerformanceMetric",
    "degradation":      "DegradationMechanism",
    "institution":      "Institution",
    "institutions":     "Institution",
    "researcher":       "Researcher",
    "researchers":      "Researcher",
    "standard":         "StandardTest",
    "test":             "StandardTest",
}


class QueryEngine:
    def __init__(self, graph: ConjunctiveGraph):
        self.g = graph

    def _sparql(self, query: str) -> list[dict[str, Any]]:
        full = PREFIXES + query
        log.debug("SPARQL:\n%s", full)
        rows = []
        for row in self.g.query(full):
            rows.append({str(k): str(v) for k, v in row.asdict().items()})
        return rows

    # ── Entity queries ────────────────────────────────────────────────────────

    def get_all_entities(self) -> list[dict]:
        return self._sparql("""
            SELECT ?entity ?type ?name ?description WHERE {
                ?entity a ?type ;
                        pv:name ?name .
                OPTIONAL { ?entity pv:description ?description }
                FILTER(?type != owl:NamedIndividual)
                FILTER(STRSTARTS(STR(?type), "http://example.org/pvsolar#"))
            }
            ORDER BY ?type ?name
        """)

    def get_entities_by_type(self, class_suffix: str) -> list[dict]:
        return self._sparql(f"""
            SELECT ?entity ?name ?description WHERE {{
                ?entity a pv:{class_suffix} ;
                        pv:name ?name .
                OPTIONAL {{ ?entity pv:description ?description }}
            }}
            ORDER BY ?name
        """)

    def get_relationships(self) -> list[dict]:
        return self._sparql("""
            SELECT ?subjectName ?predLabel ?objectName WHERE {
                ?s ?p ?o .
                ?s pv:name ?subjectName .
                ?o pv:name ?objectName .
                ?p rdfs:label ?predLabel .
            }
            ORDER BY ?subjectName ?predLabel
        """)

    def get_entity_details(self, entity_name: str) -> list[dict]:
        return self._sparql(f"""
            SELECT ?predLabel ?objectName ?dataValue WHERE {{
                ?s pv:name "{entity_name}" .
                ?s ?p ?o .
                OPTIONAL {{ ?p rdfs:label ?predLabel }}
                OPTIONAL {{ ?o pv:name ?objectName }}
                OPTIONAL {{
                    FILTER(isLiteral(?o))
                    BIND(?o as ?dataValue)
                }}
            }}
        """)

    def search_by_keyword(self, keyword: str) -> list[dict]:
        kw = keyword.lower()
        return self._sparql(f"""
            SELECT ?entity ?name ?type ?description WHERE {{
                ?entity pv:name ?name ;
                        a ?type .
                OPTIONAL {{ ?entity pv:description ?description }}
                FILTER(
                    CONTAINS(LCASE(STR(?name)), "{kw}") ||
                    CONTAINS(LCASE(STR(?description)), "{kw}")
                )
                FILTER(STRSTARTS(STR(?type), "http://example.org/pvsolar#"))
            }}
            ORDER BY ?name
        """)

    # ── Domain-specific queries ───────────────────────────────────────────────

    def get_absorbers(self) -> list[dict]:
        return self._sparql("""
            SELECT ?name ?description ?bandgap ?crystal WHERE {
                ?e a pv:Absorber ;
                   pv:name ?name .
                OPTIONAL { ?e pv:description ?description }
                OPTIONAL { ?e pv:bandgap_eV ?bandgap }
                OPTIONAL { ?e pv:crystalStructure ?crystal }
            }
            ORDER BY ?bandgap
        """)

    def get_cell_architectures(self) -> list[dict]:
        return self._sparql("""
            SELECT ?name ?description ?efficiency WHERE {
                ?e a pv:CellArchitecture ;
                   pv:name ?name .
                OPTIONAL { ?e pv:description ?description }
                OPTIONAL { ?e pv:recordEfficiency_pct ?efficiency }
            }
            ORDER BY DESC(?efficiency)
        """)

    def get_defects_and_impacts(self) -> list[dict]:
        return self._sparql("""
            SELECT ?defectName ?defectDesc ?metricName WHERE {
                ?d a pv:Defect ;
                   pv:name ?defectName .
                OPTIONAL { ?d pv:description ?defectDesc }
                OPTIONAL {
                    ?d pv:affectsMetric ?m .
                    ?m pv:name ?metricName .
                }
            }
            ORDER BY ?defectName
        """)

    def get_materials_for_architecture(self, arch_name: str) -> list[dict]:
        return self._sparql(f"""
            SELECT ?materialName ?materialType ?materialDesc WHERE {{
                ?m pv:usedIn ?a ;
                   pv:name ?materialName ;
                   a ?materialType .
                ?a pv:name "{arch_name}" .
                OPTIONAL {{ ?m pv:description ?materialDesc }}
            }}
            ORDER BY ?materialType ?materialName
        """)

    def get_fabrication_for_material(self, material_name: str) -> list[dict]:
        return self._sparql(f"""
            SELECT ?processName ?processDesc ?temp WHERE {{
                ?m pv:name "{material_name}" ;
                   pv:fabricatedBy ?p .
                ?p pv:name ?processName .
                OPTIONAL {{ ?p pv:description ?processDesc }}
                OPTIONAL {{ ?p pv:deposition_temp_C ?temp }}
            }}
        """)

    def get_characterisation_for_material(self, material_name: str) -> list[dict]:
        return self._sparql(f"""
            SELECT ?techName ?techDesc WHERE {{
                ?m pv:name "{material_name}" ;
                   pv:characterisedBy ?t .
                ?t pv:name ?techName .
                OPTIONAL {{ ?t pv:description ?techDesc }}
            }}
        """)

    def get_researchers_and_institutions(self) -> list[dict]:
        return self._sparql("""
            SELECT ?resName ?resDesc ?instName ?instCountry WHERE {
                ?r a pv:Researcher ;
                   pv:name ?resName .
                OPTIONAL { ?r pv:description ?resDesc }
                OPTIONAL {
                    ?r pv:studiedAt ?i .
                    ?i pv:name ?instName ;
                       pv:country ?instCountry .
                }
            }
            ORDER BY ?resName
        """)

    def get_institutions(self) -> list[dict]:
        return self._sparql("""
            SELECT ?name ?description ?country ?founded WHERE {
                ?e a pv:Institution ;
                   pv:name ?name .
                OPTIONAL { ?e pv:description ?description }
                OPTIONAL { ?e pv:country ?country }
                OPTIONAL { ?e pv:founded ?founded }
            }
            ORDER BY ?country ?name
        """)

    def get_performance_metrics(self) -> list[dict]:
        return self._sparql("""
            SELECT ?name ?description ?unit ?range WHERE {
                ?e a pv:PerformanceMetric ;
                   pv:name ?name .
                OPTIONAL { ?e pv:description ?description }
                OPTIONAL { ?e pv:unit ?unit }
                OPTIONAL { ?e pv:typicalRange ?range }
            }
            ORDER BY ?name
        """)

    def get_degradation_mechanisms(self) -> list[dict]:
        return self._sparql("""
            SELECT ?mechName ?mechDesc ?defectName WHERE {
                ?m a pv:DegradationMechanism ;
                   pv:name ?mechName .
                OPTIONAL { ?m pv:description ?mechDesc }
                OPTIONAL {
                    ?m pv:causedBy ?d .
                    ?d pv:name ?defectName .
                }
            }
            ORDER BY ?mechName
        """)

    def get_compatible_materials(self, material_name: str) -> list[dict]:
        return self._sparql(f"""
            SELECT ?compName ?compDesc WHERE {{
                ?m pv:name "{material_name}" ;
                   pv:compatibleWith ?c .
                ?c pv:name ?compName .
                OPTIONAL {{ ?c pv:description ?compDesc }}
            }}
        """)

    def get_graph_summary(self) -> dict:
        def count(cls):
            r = self._sparql(f"SELECT (COUNT(?x) as ?c) WHERE {{ ?x a pv:{cls} }}")
            return int(r[0]["c"]) if r else 0
        return {
            "total_triples":          len(self.g),
            "absorbers":              count("Absorber"),
            "cell_architectures":     count("CellArchitecture"),
            "fabrication_processes":  count("FabricationProcess"),
            "characterisation_techniques": count("CharacterisationTechnique"),
            "defects":                count("Defect"),
            "performance_metrics":    count("PerformanceMetric"),
            "degradation_mechanisms": count("DegradationMechanism"),
            "institutions":           count("Institution"),
            "researchers":            count("Researcher"),
        }

    def get_graph_data_for_viz(self) -> dict:
        entities = self.get_all_entities()
        rels     = self.get_relationships()

        type_colors = {
            "Absorber":               "#f97316",   # orange
            "TransportLayer":         "#34d399",   # green
            "Electrode":              "#94a3b8",   # slate
            "Encapsulant":            "#a78bfa",   # violet
            "CellArchitecture":       "#60a5fa",   # blue
            "FabricationProcess":     "#fbbf24",   # amber
            "CharacterisationTechnique": "#2dd4bf", # teal
            "Defect":                 "#fb7185",   # rose
            "PerformanceMetric":      "#c084fc",   # purple
            "DegradationMechanism":   "#f43f5e",   # red
            "Institution":            "#38bdf8",   # sky
            "Researcher":             "#86efac",   # light-green
            "StandardTest":           "#e2e8f0",   # white-ish
        }

        nodes = {}
        for e in entities:
            name  = e.get("name", "")
            etype = e.get("type", "").split("#")[-1]
            if name and name not in nodes:
                nodes[name] = {
                    "id":    name,
                    "label": name,
                    "color": type_colors.get(etype, "#64748b"),
                    "type":  etype,
                    "title": e.get("description", name),
                }

        edges = []
        seen  = set()
        for r in rels:
            s, p, o = r.get("subjectName"), r.get("predLabel"), r.get("objectName")
            if s and p and o:
                key = (s, p, o)
                if key not in seen:
                    seen.add(key)
                    edges.append({"from": s, "to": o, "label": p})

        return {"nodes": list(nodes.values()), "edges": edges}

    # ── Context builder for LLM ───────────────────────────────────────────────

    def build_context_for_query(self, user_query: str) -> str:
        q = user_query.lower()
        lines = []

        # ── Detect class-level intent ─────────────────────────────────────────
        matched_classes = set()
        for keyword, cls in CLASS_MAP.items():
            if keyword in q:
                matched_classes.add(cls)

        # Special domain keywords
        if any(w in q for w in ["perovskite", "mapbi3", "fapbi3", "silicon", "cigs", "cdte", "gaas", "organic"]):
            matched_classes.add("Absorber")
        if any(w in q for w in ["efficiency", "record", "pce", "voc", "jsc", "fill factor"]):
            matched_classes.add("PerformanceMetric")
            matched_classes.add("CellArchitecture")
        if any(w in q for w in ["degrade", "stability", "moisture", "thermal", "uv"]):
            matched_classes.add("DegradationMechanism")
            matched_classes.add("Defect")
        if any(w in q for w in ["ito", "spiro", "ptaa", "tio2", "sno2", "c60", "fullerene"]):
            matched_classes.add("TransportLayer")
            matched_classes.add("Electrode")
        if any(w in q for w in ["spin coat", "sputtering", "evaporation", "czochralski", "pecvd", "anneal"]):
            matched_classes.add("FabricationProcess")
        if any(w in q for w in ["xrd", "sem", "tem", "pl ", "trpl", "eqe", "dlts", "ellipsom"]):
            matched_classes.add("CharacterisationTechnique")
        if any(w in q for w in ["nrel", "fraunhofer", "epfl", "kaust", "hzb", "oxford", "longi", "first solar"]):
            matched_classes.add("Institution")

        # ── Dump entities for matched classes ─────────────────────────────────
        for cls in matched_classes:
            rows = self.get_entities_by_type(cls)
            if rows:
                lines.append(f"\n## {cls.replace('_',' ')}s")
                for r in rows:
                    lines.append(f"- **{r['name']}**: {r.get('description', 'N/A')}")

        # ── Extra detail for absorbers ─────────────────────────────────────────
        if "Absorber" in matched_classes or any(w in q for w in ["bandgap", "absorber", "semiconductor"]):
            absorbers = self.get_absorbers()
            if absorbers:
                lines.append("\n## Absorber Materials — Key Properties")
                for a in absorbers:
                    bg = a.get("bandgap", "N/A")
                    cs = a.get("crystal", "N/A")
                    lines.append(f"- **{a['name']}** | Bandgap: {bg} eV | Crystal: {cs}")

        # ── Extra detail for architectures ────────────────────────────────────
        if "CellArchitecture" in matched_classes or any(w in q for w in ["tandem", "perc", "topcon", "shj", "pin", "nip"]):
            archs = self.get_cell_architectures()
            if archs:
                lines.append("\n## Cell Architectures — Record Efficiencies")
                for a in archs:
                    eff = a.get("efficiency", "N/A")
                    lines.append(f"- **{a['name']}**: {eff}% record PCE — {a.get('description', '')[:120]}")

        # ── Defects + impacts ─────────────────────────────────────────────────
        if "Defect" in matched_classes:
            defects = self.get_defects_and_impacts()
            if defects:
                lines.append("\n## Defects and Performance Impacts")
                current = None
                for d in defects:
                    if d.get("defectName") != current:
                        current = d.get("defectName")
                        lines.append(f"- **{current}**: {d.get('defectDesc','')[:100]}")
                    if d.get("metricName"):
                        lines.append(f"  → affects: {d['metricName']}")

        # ── Relationship dump ─────────────────────────────────────────────────
        if any(w in q for w in ["relationship", "connect", "related", "compatible", "link", "used in"]):
            rels = self.get_relationships()
            if rels:
                lines.append("\n## All Relationships in the Graph")
                for r in rels:
                    lines.append(f"- {r['subjectName']} → [{r['predLabel']}] → {r['objectName']}")

        # ── Researchers ───────────────────────────────────────────────────────
        if "Researcher" in matched_classes or any(w in q for w in ["gratzel", "snaith", "miyasaka", "sargent"]):
            researchers = self.get_researchers_and_institutions()
            if researchers:
                lines.append("\n## Researchers")
                for r in researchers:
                    inst = f" at {r['instName']} ({r.get('instCountry','')})" if r.get("instName") else ""
                    lines.append(f"- **{r['resName']}**{inst}: {r.get('resDesc','')[:120]}")

        # ── Performance metrics ───────────────────────────────────────────────
        if "PerformanceMetric" in matched_classes:
            metrics = self.get_performance_metrics()
            if metrics:
                lines.append("\n## Performance Metrics")
                for m in metrics:
                    rng = f" (typical: {m['range']})" if m.get("range") else ""
                    unit = f" [{m['unit']}]" if m.get("unit") else ""
                    lines.append(f"- **{m['name']}**{unit}{rng}: {m.get('description','')[:100]}")

        # ── Degradation mechanisms ────────────────────────────────────────────
        if "DegradationMechanism" in matched_classes:
            meches = self.get_degradation_mechanisms()
            if meches:
                lines.append("\n## Degradation Mechanisms")
                for m in meches:
                    cause = f" (caused by defect: {m['defectName']})" if m.get("defectName") else ""
                    lines.append(f"- **{m['mechName']}**{cause}: {m.get('mechDesc','')[:120]}")

        # ── Institutions ──────────────────────────────────────────────────────
        if "Institution" in matched_classes:
            insts = self.get_institutions()
            if insts:
                lines.append("\n## Research Institutions & Companies")
                for i in insts:
                    lines.append(f"- **{i['name']}** ({i.get('country','?')}): {i.get('description','')[:120]}")

        # ── Keyword entity search fallback ────────────────────────────────────
        if not lines:
            all_entities = self.get_all_entities()
            matched = [e for e in all_entities if e.get("name","").lower() in q]
            for ent in matched:
                details = self.get_entity_details(ent["name"])
                lines.append(f"\n## {ent['name']} ({ent.get('type','').split('#')[-1]})")
                lines.append(f"**Description**: {ent.get('description','N/A')}")
                for d in details:
                    pred = d.get("predLabel","")
                    obj  = d.get("objectName","")
                    val  = d.get("dataValue","")
                    if pred and pred not in ("name","description"):
                        if obj:
                            lines.append(f"- {pred}: {obj}")
                        elif val:
                            lines.append(f"- {pred}: {val}")

        # ── Ultimate fallback → full graph overview ────────────────────────────
        if not lines:
            summary = self.get_graph_summary()
            lines.append("## PV Solar Knowledge Graph — Overview")
            for k, v in summary.items():
                lines.append(f"- {k.replace('_',' ').title()}: {v}")
            archs = self.get_cell_architectures()
            lines.append("\n## Cell Architectures by Record Efficiency")
            for a in archs:
                lines.append(f"- **{a['name']}**: {a.get('efficiency','?')}% — {a.get('description','')[:80]}")
            absorbers = self.get_absorbers()
            lines.append("\n## Absorber Materials")
            for ab in absorbers:
                lines.append(f"- **{ab['name']}** (bandgap {ab.get('bandgap','?')} eV)")

        return "\n".join(lines)
