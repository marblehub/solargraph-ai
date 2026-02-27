"""
react_agent.py  —  Multi-step ReAct (Reason + Act) agent with SPARQL tools.

Instead of one LLM call, the agent loops:
  Thought → Action (tool call) → Observation → Thought → … → Final Answer

This demonstrates:
  - Tool use beyond prompt engineering
  - Agent control flow and state management
  - Multi-step reasoning over a formal knowledge graph
  - Full provenance: every answer records which tools were invoked
"""

import os, json, time, hashlib, logging
from pathlib import Path
from groq import Groq
from query_engine import QueryEngine

log = logging.getLogger(__name__)

CACHE_PATH        = Path(__file__).parent / "react_cache.json"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 86400))
MAX_ITERATIONS    = 6

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sparql_query",
            "description": (
                "Execute a SPARQL SELECT query against the PV Solar knowledge graph. "
                "Always use: PREFIX pv: <http://example.org/pvsolar#>. "
                "Returns a JSON list of result rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A valid SPARQL SELECT query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_details",
            "description": "Get all properties and relationships for a named entity in the knowledge graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The pv:name value, e.g. 'MAPbI3' or 'PERC (Passivated Emitter and Rear Cell)'"}
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Full-text search across all entity names and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Search term e.g. 'perovskite', 'defect', 'silicon'"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_absorbers",
            "description": "Get all absorber materials with bandgap energies and crystal structures.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_architectures",
            "description": "Get all cell architectures with record power conversion efficiencies, sorted best first.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_defects",
            "description": "Get all defects and which performance metrics (PCE, Voc, FF…) they impact.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_relationships",
            "description": "Get all subject-predicate-object triples / relationships in the knowledge graph.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

SYSTEM_PROMPT = """You are SolarGraph AI — a scientific assistant for Photovoltaic (PV)
Solar Energy and Materials Science Engineering.

You have tools to query a formal RDF/OWL knowledge graph. ALWAYS use at least one
tool before giving a final answer. Never rely on training knowledge alone.

Rules:
1. Use tools to retrieve facts, then reason over the results.
2. If a first query is insufficient, run a follow-up query.
3. Cite specific entity names and relationships from your tool results.
4. Include units for all numeric data (%, eV, mA/cm², V …).
5. Structure your final answer with clear headings and bullet points.
6. End your answer with a brief "## Sources" section listing the tools you used.
"""


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    except Exception:
        return {}

def _save_cache(cache: dict):
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception as e:
        log.warning("Cache write error: %s", e)

def _cache_key(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode()).hexdigest()


def _execute_tool(name: str, args: dict, qe: QueryEngine) -> str:
    try:
        if name == "sparql_query":
            return json.dumps(qe._sparql(args["query"])[:40])
        elif name == "get_entity_details":
            return json.dumps(qe.get_entity_details(args["entity_name"]))
        elif name == "keyword_search":
            return json.dumps(qe.search_by_keyword(args["keyword"])[:20])
        elif name == "get_absorbers":
            return json.dumps(qe.get_absorbers())
        elif name == "get_architectures":
            return json.dumps(qe.get_cell_architectures())
        elif name == "get_defects":
            return json.dumps(qe.get_defects_and_impacts())
        elif name == "get_relationships":
            return json.dumps(qe.get_relationships()[:60])
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        log.error("Tool error [%s]: %s", name, e)
        return json.dumps({"error": str(e)})


class ReActAgent:
    def __init__(self, query_engine: QueryEngine, model: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=api_key)
        self.model  = model
        self.qe     = query_engine
        self._cache = _load_cache()
        log.info("ReActAgent ready | model=%s | cache=%d entries", model, len(self._cache))

    def answer(self, user_query: str) -> dict:
        key   = _cache_key(user_query)
        entry = self._cache.get(key)
        if entry and time.time() - entry["ts"] < CACHE_TTL_SECONDS:
            log.info("ReAct cache HIT: %.70s", user_query)
            return {**entry["data"], "cached": True}

        log.info("ReAct agent loop starting: %.70s", user_query)
        messages   = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": user_query}]
        steps      = []
        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1
            log.info("ReAct iteration %d", iterations)
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=TOOLS, tool_choice="auto",
                temperature=0.1, max_tokens=2000,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                result = {
                    "answer": msg.content or "No answer generated.",
                    "steps":  steps, "iterations": iterations, "cached": False,
                }
                self._cache[key] = {"data": result, "ts": time.time()}
                _save_cache(self._cache)
                return result

            # Append assistant message with tool_calls
            messages.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                log.info("Tool call: %s(%s)", tool_name, str(tool_args)[:80])
                result_str = _execute_tool(tool_name, tool_args, self.qe)
                steps.append({
                    "iteration": iterations, "tool": tool_name,
                    "args": tool_args, "result_preview": result_str[:300],
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result_str,
                })

        return {"answer": "Reached maximum reasoning steps.", "steps": steps,
                "iterations": iterations, "cached": False}

    def clear_cache(self) -> int:
        n = len(self._cache)
        self._cache.clear(); _save_cache(self._cache)
        return n

    def cache_stats(self) -> dict:
        return {"react_cache_entries": len(self._cache),
                "cache_ttl_seconds": CACHE_TTL_SECONDS, "max_iterations": MAX_ITERATIONS}
