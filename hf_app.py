"""
hf_app.py  —  HuggingFace Spaces entry point (Gradio SDK).

Deploy steps:
  1. Create a new Space at huggingface.co/new-space
     - SDK: Gradio
     - Visibility: Public
  2. Upload all project files to the Space repo
  3. Add your GROQ_API_KEY in Space Settings → Secrets
  4. HF will auto-run this file as the app entry point
"""
import os, gradio as gr
from dotenv import load_dotenv

load_dotenv()

from build_graph  import load_graph
from query_engine import QueryEngine
from llm_agent    import LLMAgent
from react_agent  import ReActAgent
from provenance   import build_provenance

print("🔄 Loading knowledge graph…")
graph = load_graph()
qe    = QueryEngine(graph)
agent       = LLMAgent(qe)
react_agent = ReActAgent(qe)
stats = qe.get_graph_summary()
print("✅ Ready")

EXAMPLES = [
    "List all absorber materials with their bandgap energies.",
    "What are the record efficiencies of PERC, TOPCon, SHJ, and Perovskite/Silicon tandem?",
    "What defects affect perovskite solar cells and which metrics do they impact?",
    "What fabrication processes are used to deposit MAPbI3?",
    "Which characterisation techniques are used for CIGS?",
    "What are the degradation mechanisms in perovskite PV?",
    "Tell me about NREL, Fraunhofer ISE, and KAUST.",
    "Which transport layers are compatible with FAPbI3?",
]

def chat_fast(message, history):
    if not message.strip():
        return history
    answer = agent.answer(message)
    history.append((message, answer))
    return history

def chat_react(message, history):
    if not message.strip():
        return history, ""
    result = react_agent.answer(message)
    prov   = build_provenance(
        query=message, answer=result["answer"], qe=qe,
        agent_steps=result.get("steps", []),
        agent_iterations=result.get("iterations", 1),
        cached=result.get("cached", False),
    )
    answer_with_meta = (
        result["answer"]
        + f"\n\n---\n*Agent steps: {result.get('iterations',1)} | "
        + ("⚡ cached" if result.get("cached") else "🔍 live query") + "*"
    )
    history.append((message, answer_with_meta))
    return history, prov.to_markdown()

def browse_entities(entity_type):
    rows = qe.get_entities_by_type(entity_type)
    if not rows:
        return f"No entities found for type: {entity_type}"
    lines = [f"### {entity_type}s ({len(rows)} found)\n",
             "| Name | Description |", "|---|---|"]
    for r in rows:
        name = r.get("name", "")
        desc = (r.get("description") or "")[:100]
        lines.append(f"| **{name}** | {desc} |")
    return "\n".join(lines)

STATS_MD = "\n".join([
    "### Knowledge Graph Statistics", "| Class | Count |", "|---|---|",
    f"| Total RDF Triples | **{stats['total_triples']}** |",
    f"| Absorber Materials | **{stats['absorbers']}** |",
    f"| Cell Architectures | **{stats['cell_architectures']}** |",
    f"| Fabrication Processes | **{stats['fabrication_processes']}** |",
    f"| Characterisation Techniques | **{stats['characterisation_techniques']}** |",
    f"| Defects | **{stats['defects']}** |",
    f"| Performance Metrics | **{stats['performance_metrics']}** |",
    f"| Degradation Mechanisms | **{stats['degradation_mechanisms']}** |",
    f"| Institutions | **{stats['institutions']}** |",
    f"| Researchers | **{stats['researchers']}** |",
])

with gr.Blocks(title="SolarGraph AI",
               theme=gr.themes.Base(primary_hue="orange", neutral_hue="slate")) as demo:

    gr.Markdown("# ☀ SolarGraph AI\n### LLM Agent & Knowledge Graph for PV Materials Science")
    gr.Markdown(
        "> Answers grounded strictly in a formal **RDF/OWL knowledge graph** — "
        "no hallucinations, full provenance tracing via SPARQL."
    )

    with gr.Tabs():
        with gr.Tab("⚡ Fast Agent"):
            gr.Markdown("Single-shot LLM with SPARQL context injection and dual-layer caching.")
            fast_chat = gr.Chatbot(height=420, label="SolarGraph AI")
            with gr.Row():
                fast_input = gr.Textbox(placeholder="Ask about PV materials…", scale=5, lines=2)
                gr.Button("Ask", variant="primary", scale=1).click(
                    chat_fast, [fast_input, fast_chat], [fast_chat]
                ).then(lambda: "", outputs=fast_input)
            fast_input.submit(chat_fast, [fast_input, fast_chat], [fast_chat])
            gr.Examples(EXAMPLES[:4], fast_input)

        with gr.Tab("🔍 ReAct Agent"):
            gr.Markdown("Multi-step tool-use agent: iterates Thought→Tool→Observation until ready to answer.")
            react_chat = gr.Chatbot(height=380, label="ReAct Agent")
            prov_out   = gr.Markdown(label="Provenance Record")
            with gr.Row():
                react_input = gr.Textbox(placeholder="Ask a complex question…", scale=5, lines=2)
                gr.Button("ReAct Ask", variant="primary", scale=1).click(
                    chat_react, [react_input, react_chat], [react_chat, prov_out]
                ).then(lambda: "", outputs=react_input)
            react_input.submit(chat_react, [react_input, react_chat], [react_chat, prov_out])
            gr.Examples(EXAMPLES[4:], react_input)

        with gr.Tab("📊 KG Stats"):
            gr.Markdown(STATS_MD)

        with gr.Tab("🔎 Browse Entities"):
            dd = gr.Dropdown(
                choices=["Absorber","TransportLayer","Electrode","Encapsulant",
                         "CellArchitecture","FabricationProcess","CharacterisationTechnique",
                         "Defect","PerformanceMetric","DegradationMechanism",
                         "Institution","Researcher","StandardTest"],
                value="Absorber", label="Entity Type"
            )
            out = gr.Markdown()
            dd.change(browse_entities, dd, out)
            gr.Button("Load", variant="primary").click(browse_entities, dd, out)

        with gr.Tab("ℹ About"):
            gr.Markdown("""
## Architecture

```
User Question
    ↓
ReAct Agent (Groq LLM)
    ↓  tool calls in loop
SPARQL Query Engine (RDFLib)
    ↓
PV Solar OWL Ontology (Turtle RDF)
    ↓
Grounded Answer + Provenance Record
```

## Key Design Decisions
- **Grounding**: System prompt prohibits answers beyond KG context — eliminates hallucination
- **ReAct Loop**: LLM decides which SPARQL tools to call and when to stop
- **Provenance**: Every answer records cited entities, supporting triples, and SPARQL queries used
- **Dual Cache**: LRU (in-process) + JSON file cache — minimises API calls, survives restarts

## Ontology Coverage
13 OWL classes · 13 object properties · 8 data properties · 70+ individuals

Relevant to: NOMAD, Materials Project, OPTIMADE, BattINFO, EMMO

**GitHub**: https://github.com/YOUR_USERNAME/solargraph-ai
""")

    gr.Markdown("---\n*SolarGraph AI · PV Solar Materials Science · RDFLib · SPARQL · Groq · ReAct*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
