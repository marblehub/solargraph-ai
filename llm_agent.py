"""
llm_agent.py
────────────
Groq-powered LLM agent grounded strictly in the PV Solar knowledge graph.

Caching strategy (two layers):
  1. In-process LRU cache  (functools.lru_cache) — instant hits within a session.
  2. Persistent file cache (JSON on disk, cache.json) — survives restarts.
     Entries expire after CACHE_TTL_SECONDS (default 24 h).
"""

import os
import json
import time
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from groq import Groq
from query_engine import QueryEngine

log = logging.getLogger(__name__)

# ── Cache settings ─────────────────────────────────────────────────────────────
CACHE_PATH        = Path(__file__).parent / "cache.json"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 86400))   # 24 hours
LRU_MAXSIZE       = int(os.getenv("LRU_MAXSIZE", 256))

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are SolarGraph AI, a specialist Knowledge Graph Assistant for
Photovoltaic (PV) Solar Energy and Materials Science Engineering.

Your ONLY source of truth is the context block provided inside
<KNOWLEDGE_GRAPH_CONTEXT> … </KNOWLEDGE_GRAPH_CONTEXT> tags.

Rules you must always follow:
1. Answer STRICTLY and ONLY using facts present in the knowledge graph context.
2. If the information is not in the context, respond exactly:
   "I could not find that information in the PV Solar knowledge graph."
3. Format answers in clear, plain English — use bullet points and section headers
   where helpful. Never output raw URIs, SPARQL syntax, or RDF notation.
4. For numeric values (efficiencies, bandgaps, temperatures), always include units.
5. When listing materials, group them by type (absorber, transport layer, electrode, …).
6. Be precise and scientifically accurate. Do not speculate beyond the graph data.
7. Keep answers concise but complete — avoid padding or repetition.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistent file-based cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception as exc:
            log.warning("Cache read error: %s", exc)
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2))
    except Exception as exc:
        log.warning("Cache write error: %s", exc)


def _make_key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _get_cached(key: str, cache: dict) -> str | None:
    entry = cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        log.info("Cache entry expired for key %s", key[:12])
        del cache[key]
        return None
    return entry["answer"]


def _set_cached(key: str, answer: str, cache: dict) -> None:
    cache[key] = {"answer": answer, "ts": time.time()}
    _save_cache(cache)


# ═══════════════════════════════════════════════════════════════════════════════
#  LRU-cached context builder (in-process layer)
# ═══════════════════════════════════════════════════════════════════════════════

def make_cached_context_builder(qe: QueryEngine):
    """
    Wrap QueryEngine.build_context_for_query with an LRU cache.
    Returns a cached function bound to the given QueryEngine instance.
    """
    @lru_cache(maxsize=LRU_MAXSIZE)
    def _build_context(query: str) -> str:
        return qe.build_context_for_query(query)
    return _build_context


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Agent
# ═══════════════════════════════════════════════════════════════════════════════

class LLMAgent:
    def __init__(self, query_engine: QueryEngine, model: str = "llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it before starting the app."
            )
        self.client        = Groq(api_key=api_key)
        self.model         = model
        self.qe            = query_engine
        self._build_ctx    = make_cached_context_builder(query_engine)
        self._file_cache   = _load_cache()
        log.info("LLMAgent ready | model=%s | file-cache entries=%d",
                 model, len(self._file_cache))

    # ── Public ────────────────────────────────────────────────────────────────

    def answer(self, user_query: str) -> str:
        """
        Return a grounded LLM answer for user_query.

        Cache lookup order:
          1. File cache (persistent across restarts, TTL-gated)
          2. In-process LRU cache for KG context building
          3. Groq API call (only when both caches miss)
        """
        key = _make_key(user_query)

        # ── Layer 1: file cache hit ─────────────────────────────────────────
        cached_answer = _get_cached(key, self._file_cache)
        if cached_answer:
            log.info("File-cache HIT for query: %.80s", user_query)
            return cached_answer

        log.info("Cache MISS — querying KG + Groq | query: %.80s", user_query)

        # ── Layer 2: KG context (LRU-cached) ───────────────────────────────
        context = self._build_ctx(user_query)
        log.debug("KG context length: %d chars", len(context))

        if not context.strip():
            return "I could not find relevant information in the PV Solar knowledge graph for that query."

        # ── Layer 3: Groq API ───────────────────────────────────────────────
        user_message = (
            f"<KNOWLEDGE_GRAPH_CONTEXT>\n{context}\n</KNOWLEDGE_GRAPH_CONTEXT>\n\n"
            f"Question: {user_query}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content.strip()
            log.info("Groq response (%d chars) for query: %.60s", len(answer), user_query)

            # Store in file cache
            _set_cached(key, answer, self._file_cache)
            return answer

        except Exception as exc:
            log.error("Groq API error: %s", exc, exc_info=True)
            return f"Error communicating with the LLM: {exc}"

    def clear_cache(self) -> int:
        """Clear both LRU and file cache. Returns number of entries removed."""
        n = len(self._file_cache)
        self._file_cache.clear()
        _save_cache(self._file_cache)
        self._build_ctx.cache_clear()
        log.info("Cache cleared (%d entries removed)", n)
        return n

    def cache_stats(self) -> dict:
        lru_info = self._build_ctx.cache_info()
        return {
            "file_cache_entries":  len(self._file_cache),
            "lru_hits":            lru_info.hits,
            "lru_misses":          lru_info.misses,
            "lru_maxsize":         lru_info.maxsize,
            "lru_currsize":        lru_info.currsize,
            "cache_ttl_seconds":   CACHE_TTL_SECONDS,
        }
