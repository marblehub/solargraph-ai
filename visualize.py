"""
visualize.py
────────────
Generates a fully self-contained interactive graph HTML for the PV Solar KG.
Uses vis.js loaded from CDN — no local library files required, no 404 errors.
Writes to templates/graph.html (served by Flask via /graph).
"""

import json
import logging
from pathlib import Path
from query_engine import QueryEngine

log = logging.getLogger(__name__)
OUTPUT_PATH = Path(__file__).parent / "templates" / "graph.html"

TYPE_COLORS = {
    "Absorber":                  "#f97316",
    "TransportLayer":            "#34d399",
    "Electrode":                 "#94a3b8",
    "Encapsulant":               "#a78bfa",
    "CellArchitecture":          "#60a5fa",
    "FabricationProcess":        "#fbbf24",
    "CharacterisationTechnique": "#2dd4bf",
    "Defect":                    "#fb7185",
    "PerformanceMetric":         "#c084fc",
    "DegradationMechanism":      "#f43f5e",
    "Institution":               "#38bdf8",
    "Researcher":                "#86efac",
    "StandardTest":              "#e2e8f0",
}

NODE_SIZES = {
    "CellArchitecture":          44,
    "Absorber":                  40,
    "Institution":               36,
    "Researcher":                30,
    "FabricationProcess":        28,
    "CharacterisationTechnique": 26,
    "Defect":                    26,
    "TransportLayer":            24,
    "PerformanceMetric":         24,
    "DegradationMechanism":      24,
    "Electrode":                 22,
    "Encapsulant":               20,
    "StandardTest":              20,
}

LEGEND = [
    ("Absorber",             "#f97316"),
    ("Transport Layer",      "#34d399"),
    ("Electrode",            "#94a3b8"),
    ("Encapsulant",          "#a78bfa"),
    ("Cell Architecture",    "#60a5fa"),
    ("Fabrication Process",  "#fbbf24"),
    ("Characterisation",     "#2dd4bf"),
    ("Defect",               "#fb7185"),
    ("Performance Metric",   "#c084fc"),
    ("Degradation",          "#f43f5e"),
    ("Institution",          "#38bdf8"),
    ("Researcher",           "#86efac"),
    ("Standard Test",        "#e2e8f0"),
]


def generate_graph_html(qe: QueryEngine, output_path: Path = OUTPUT_PATH) -> Path:
    data  = qe.get_graph_data_for_viz()
    nodes = data["nodes"]
    edges = data["edges"]

    # ── Build vis.js node/edge datasets ──────────────────────────────────────
    vis_nodes = []
    for n in nodes:
        etype = n.get("type", "")
        color = TYPE_COLORS.get(etype, "#64748b")
        size  = NODE_SIZES.get(etype, 24)
        # Escape for safe HTML tooltip
        desc  = n.get("title", "")[:220].replace("'", "&#39;").replace("<", "&lt;")
        vis_nodes.append({
            "id":    n["id"],
            "label": n["label"],
            "title": (
                f"<div style='font-family:sans-serif;max-width:260px;padding:4px'>"
                f"<b style='color:{color}'>{n['label']}</b>"
                f"<br><i style='color:#94a3b8;font-size:11px'>{etype}</i>"
                f"<br><br><span style='color:#cbd5e1;font-size:12px'>{desc}</span>"
                f"</div>"
            ),
            "color": {
                "background": color,
                "border":     "#0f172a",
                "highlight":  {"background": color, "border": "#ffffff"},
                "hover":      {"background": color, "border": "#e2e8f0"},
            },
            "size":        size,
            "shape":       "dot",
            "font":        {"color": "#f1f5f9", "size": 13},
            "borderWidth": 2,
            "shadow":      {"enabled": True, "size": 10, "color": "rgba(0,0,0,0.55)"},
        })

    vis_edges = []
    for i, e in enumerate(edges):
        vis_edges.append({
            "id":     i,
            "from":   e["from"],
            "to":     e["to"],
            "label":  e["label"],
            "title":  e["label"],
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.65}},
            "color":  {"color": "#1e3a5f", "highlight": "#3b82f6", "hover": "#60a5fa"},
            "font":   {"size": 10, "color": "#4a6885", "align": "middle",
                       "background": "#0f1e33", "strokeWidth": 0},
            "smooth": {"type": "curvedCW", "roundness": 0.15},
            "width":  1.2,
        })

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False)
    edges_json = json.dumps(vis_edges, ensure_ascii=False)

    # ── Legend HTML ───────────────────────────────────────────────────────────
    legend_html = "\n".join(
        f'<div class="leg-item">'
        f'<span class="leg-dot" style="background:{c}"></span>{lbl}'
        f'</div>'
        for lbl, c in LEGEND
    )

    # ── Full self-contained HTML ──────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>PV Solar Knowledge Graph</title>

  <!-- vis.js from CDN — no local files needed -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
  <link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet"/>
  <link  href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet"/>

  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      background: #06101f;
      font-family: 'DM Sans', sans-serif;
      overflow: hidden;
      height: 100vh; width: 100vw;
    }}

    #graph-container {{
      width: 100vw;
      height: 100vh;
      background: #06101f;
    }}

    /* ── Back button ── */
    #back-btn {{
      position: fixed; top: 16px; left: 16px; z-index: 9999;
      background: #f97316; color: #fff;
      border: none; border-radius: 8px;
      padding: 10px 20px; font-size: 13px; font-weight: 700;
      cursor: pointer; text-decoration: none;
      display: inline-flex; align-items: center; gap: 8px;
      box-shadow: 0 4px 16px rgba(249,115,22,0.45);
      transition: background .2s;
      font-family: 'DM Sans', sans-serif;
    }}
    #back-btn:hover {{ background: #ea6c07; }}

    /* ── Title bar ── */
    #title-bar {{
      position: fixed; top: 16px; left: 50%; transform: translateX(-50%);
      z-index: 9999;
      background: rgba(8,15,26,0.88); border: 1px solid #1e3a5f;
      border-radius: 100px; padding: 8px 24px;
      font-size: 13px; color: #7e9cbf;
      backdrop-filter: blur(12px);
      white-space: nowrap;
      pointer-events: none;
    }}

    /* ── Stats pill ── */
    #stats-pill {{
      position: fixed; top: 16px; right: 16px; z-index: 9999;
      background: rgba(8,15,26,0.88); border: 1px solid #1e3a5f;
      border-radius: 100px; padding: 8px 18px;
      font-size: 12px; color: #94a3b8;
      backdrop-filter: blur(12px);
    }}
    #stats-pill span {{ color: #f97316; font-weight: 700; }}

    /* ── Legend ── */
    #legend {{
      position: fixed; bottom: 20px; right: 20px; z-index: 9999;
      background: rgba(8,15,26,0.92); border: 1px solid #1e3a5f;
      border-radius: 12px; padding: 14px 18px;
      font-size: 11px; color: #94a3b8;
      backdrop-filter: blur(12px);
      max-height: 70vh; overflow-y: auto;
    }}
    #legend h4 {{
      color: #e2e8f0; margin: 0 0 10px;
      font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
    }}
    .leg-item {{
      display: flex; align-items: center; gap: 8px; margin: 5px 0;
    }}
    .leg-dot {{
      width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    }}

    /* ── Search box ── */
    #search-wrap {{
      position: fixed; bottom: 20px; left: 16px; z-index: 9999;
      display: flex; gap: 8px;
    }}
    #search-input {{
      background: rgba(8,15,26,0.92); border: 1px solid #1e3a5f;
      border-radius: 8px; padding: 8px 14px;
      font-size: 13px; color: #e2e8f0;
      font-family: 'DM Sans', sans-serif;
      outline: none; width: 220px;
      backdrop-filter: blur(12px);
      transition: border-color .2s;
    }}
    #search-input::placeholder {{ color: #3d607f; }}
    #search-input:focus {{ border-color: #f97316; }}
    #search-btn {{
      background: #1e3a5f; border: 1px solid #243d62;
      border-radius: 8px; padding: 8px 14px;
      color: #93c5fd; font-size: 12px; font-weight: 600;
      cursor: pointer; font-family: 'DM Sans', sans-serif;
      transition: background .2s;
    }}
    #search-btn:hover {{ background: #243d62; }}

    /* ── Loading overlay ── */
    #loading {{
      position: fixed; inset: 0; z-index: 99999;
      background: #06101f;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 20px;
    }}
    .spinner-ring {{
      width: 52px; height: 52px;
      border: 4px solid #1e3a5f;
      border-top-color: #f97316;
      border-radius: 50%;
      animation: spin .8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    #loading p {{
      color: #7e9cbf; font-size: 14px;
    }}

    /* vis.js tooltip override */
    .vis-tooltip {{
      background: rgba(8,15,26,0.95) !important;
      border: 1px solid #1e3a5f !important;
      border-radius: 10px !important;
      padding: 10px 14px !important;
      font-family: 'DM Sans', sans-serif !important;
      font-size: 12px !important;
      color: #e2e8f0 !important;
      max-width: 280px !important;
      box-shadow: 0 8px 32px rgba(0,0,0,.5) !important;
    }}
  </style>
</head>
<body>

<!-- Loading screen -->
<div id="loading">
  <div class="spinner-ring"></div>
  <p>Building knowledge graph…</p>
</div>

<!-- Back button -->
<a id="back-btn" href="/">← Home</a>

<!-- Title -->
<div id="title-bar">☀ PV Solar &amp; Materials Science — Knowledge Graph</div>

<!-- Stats -->
<div id="stats-pill">
  <span>{len(vis_nodes)}</span> nodes &nbsp;·&nbsp; <span>{len(vis_edges)}</span> edges
</div>

<!-- Graph canvas -->
<div id="graph-container"></div>

<!-- Legend -->
<div id="legend">
  <h4>Node Types</h4>
  {legend_html}
</div>

<!-- Search -->
<div id="search-wrap">
  <input id="search-input" type="text" placeholder="Search nodes…"/>
  <button id="search-btn">Find</button>
</div>

<script>
// ── Data ─────────────────────────────────────────────────────────────────────
const RAW_NODES = {nodes_json};
const RAW_EDGES = {edges_json};

// ── Network ──────────────────────────────────────────────────────────────────
const nodes    = new vis.DataSet(RAW_NODES);
const edges    = new vis.DataSet(RAW_EDGES);
const container = document.getElementById("graph-container");

const options = {{
  physics: {{
    solver: "forceAtlas2Based",
    forceAtlas2Based: {{
      gravitationalConstant: -80,
      centralGravity: 0.004,
      springLength: 180,
      springConstant: 0.07,
      damping: 0.5,
    }},
    stabilization: {{ iterations: 300, fit: true }},
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 120,
    navigationButtons: true,
    keyboard: {{ enabled: true }},
    multiselect: true,
    zoomView: true,
  }},
  nodes: {{
    borderWidth: 2,
    borderWidthSelected: 4,
  }},
  edges: {{
    width: 1.2,
  }},
}};

const network = new vis.Network(container, {{ nodes, edges }}, options);

// Hide loading overlay once stabilised
network.once("stabilizationIterationsDone", () => {{
  document.getElementById("loading").style.display = "none";
}});
// Fallback: hide after 6s regardless
setTimeout(() => {{
  const el = document.getElementById("loading");
  if (el) el.style.display = "none";
}}, 6000);

// ── Search ────────────────────────────────────────────────────────────────────
function doSearch() {{
  const q = document.getElementById("search-input").value.trim().toLowerCase();
  if (!q) {{ network.unselectAll(); return; }}
  const matches = RAW_NODES
    .filter(n => n.label.toLowerCase().includes(q))
    .map(n => n.id);
  if (matches.length > 0) {{
    network.selectNodes(matches);
    network.fit({{ nodes: matches, animation: {{ duration: 800, easingFunction: "easeInOutQuad" }} }});
  }} else {{
    network.unselectAll();
    alert("No nodes found for: " + q);
  }}
}}

document.getElementById("search-btn").addEventListener("click", doSearch);
document.getElementById("search-input").addEventListener("keydown", e => {{
  if (e.key === "Enter") doSearch();
}});
</script>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    log.info("Graph HTML written to %s — %d nodes, %d edges",
             output_path, len(vis_nodes), len(vis_edges))
    return output_path
