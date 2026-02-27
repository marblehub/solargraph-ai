"""
app.py  —  Flask application for PV Solar Materials Science Knowledge Graph.

Routes:
  GET  /                → home page
  POST /ask             → fast single-shot LLM (cached)
  POST /ask/react       → multi-step ReAct agent + provenance
  GET  /graph           → interactive vis.js graph
  GET  /api/stats       → KG statistics
  GET  /api/entities    → entity list (?type= filter)
  GET  /api/absorbers   → absorber materials
  GET  /api/architectures → cell architectures by efficiency
  GET  /api/relationships → all KG triples
  GET  /api/search      → keyword search (?q=)
  GET  /api/cache/stats → cache statistics
  POST /api/cache/clear → clear all caches
"""
import logging, os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET", "pvsolar-dev-secret")

from build_graph  import load_graph
from query_engine import QueryEngine
from llm_agent    import LLMAgent
from react_agent  import ReActAgent
from provenance   import build_provenance
from visualize    import generate_graph_html

log.info("Loading PV Solar knowledge graph …")
graph = load_graph()
qe    = QueryEngine(graph)

log.info("Initialising agents …")
agent       = LLMAgent(qe)
react_agent = ReActAgent(qe)

log.info("Generating graph visualisation …")
generate_graph_html(qe)
log.info("✅  PV Solar KG App ready")


@app.route("/")
def home():
    stats = qe.get_graph_summary()
    archs = qe.get_cell_architectures()[:5]
    cache = agent.cache_stats()
    return render_template("home.html", stats=stats, archs=archs, cache=cache)

@app.route("/ask", methods=["POST"])
def ask():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    log.info("Fast query: %s", query)
    answer = agent.answer(query)
    return jsonify({"answer": answer, "cache": agent.cache_stats(), "mode": "fast"})

@app.route("/ask/react", methods=["POST"])
def ask_react():
    data  = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    log.info("ReAct query: %s", query)
    result = react_agent.answer(query)
    prov   = build_provenance(
        query=query, answer=result["answer"], qe=qe,
        agent_steps=result.get("steps", []),
        agent_iterations=result.get("iterations", 1),
        cached=result.get("cached", False),
    )
    return jsonify({
        "answer":        result["answer"],
        "steps":         result.get("steps", []),
        "iterations":    result.get("iterations", 1),
        "cached":        result.get("cached", False),
        "provenance":    prov.to_dict(),
        "provenance_md": prov.to_markdown(),
        "mode":          "react",
    })

@app.route("/graph")
def graph_view():
    return render_template("graph.html")

@app.route("/api/stats")
def api_stats():
    return jsonify(qe.get_graph_summary())

@app.route("/api/entities")
def api_entities():
    t = request.args.get("type", "").strip()
    return jsonify(qe.get_entities_by_type(t) if t else qe.get_all_entities())

@app.route("/api/absorbers")
def api_absorbers():
    return jsonify(qe.get_absorbers())

@app.route("/api/architectures")
def api_architectures():
    return jsonify(qe.get_cell_architectures())

@app.route("/api/relationships")
def api_relationships():
    return jsonify(qe.get_relationships())

@app.route("/api/search")
def api_search():
    kw = request.args.get("q", "").strip()
    return jsonify(qe.search_by_keyword(kw) if kw else [])

@app.route("/api/cache/stats")
def api_cache_stats():
    return jsonify({"fast_agent": agent.cache_stats(), "react_agent": react_agent.cache_stats()})

@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear():
    n1, n2 = agent.clear_cache(), react_agent.clear_cache()
    return jsonify({"message": f"Cleared {n1 + n2} total cache entries."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)

@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Trigger OpenAlex ingestion. Body: {query, limit, dry_run}"""
    data    = request.get_json(force=True)
    query   = (data.get("query") or "perovskite solar cell efficiency").strip()
    limit   = int(data.get("limit", 5))
    dry_run = bool(data.get("dry_run", False))
    log.info("Ingestion: query='%s' limit=%d dry_run=%s", query, limit, dry_run)
    try:
        from ingest_literature import run_ingestion
        result = run_ingestion(query=query, limit=limit, dry_run=dry_run)
        if not dry_run and result.get("triples_added", 0) > 0:
            global graph, qe, agent, react_agent
            from build_graph import load_graph
            graph = load_graph()
            qe    = QueryEngine(graph)
            agent       = LLMAgent(qe)
            react_agent = ReActAgent(qe)
            log.info("Graph reloaded after ingestion.")
        return jsonify(result)
    except Exception as e:
        log.error("Ingestion error: %s", e)
        return jsonify({"error": str(e)}), 500
