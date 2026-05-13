"""
rag/generator.py

RAG response generation via Azure OpenAI Responses API.
Non-streaming variants are used by RAGAS evaluation.
Streaming variants are consumed directly by FastAPI StreamingResponse.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

from openai import AsyncAzureOpenAI

from config.settings import settings
from rag.query_parser import NumericCondition, apply_post_filters
from retrieval.retriever import _pool_k, retrieve

_RESPONSES_API_VERSION = "2025-04-01-preview"

_client: Optional[AsyncAzureOpenAI] = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            api_version=_RESPONSES_API_VERSION,
        )
    return _client


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an enterprise-grade multilingual RAG assistant.

Strict rules:
1. Ground every statement exclusively in the numbered context blocks below.
2. Cite every factual claim inline using [N] where N is the context block number.
3. Reproduce numerical values, percentages, dates, and KPIs exactly as they appear.
4. Deliver a thorough, well-structured response of at least 6-8 sentences.
5. FORMAT YOUR ENTIRE RESPONSE IN MARKDOWN:
   - Use ## or ### headings for major sections
   - Use **bold** for key terms and important figures
   - Use bullet lists (-) or numbered lists for items/metrics/steps
   - Use tables for comparisons when context supports it
6. Synthesize across multiple context blocks into a coherent narrative.
7. State clearly what is and isn't covered by the context.
8. Respond in the same language as the question (English or Spanish).
9. Never invent figures, names, or conclusions not in the context.
"""

CHAT_SYSTEM_PROMPT = """You are an enterprise-grade multilingual RAG assistant in a multi-turn conversation.

Strict rules:
1. Ground every statement exclusively in the numbered context blocks below.
2. Cite every factual claim inline using [N].
3. Use conversation history for context, but draw facts ONLY from current context blocks.
4. Deliver a thorough, well-structured response of at least 6-8 sentences.
5. FORMAT YOUR ENTIRE RESPONSE IN MARKDOWN:
   - Use ## or ### headings for major sections
   - Use **bold** for key terms and important figures  
   - Use bullet lists or numbered lists for items/metrics
   - Use tables for comparisons
6. Maintain a professional yet conversational tone.
7. Respond in the same language as the question (English or Spanish).
8. Never invent figures, names, or conclusions not in the context.
"""

# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(chunks: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    citations: List[Dict[str, Any]] = []
    parts: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        meta   = chunk["metadata"]
        source = meta.get("source", "unknown")

        if "page" in meta:
            location = f"page {meta['page']}"
        elif "row_start" in meta:
            location = f"rows {meta['row_start']}-{meta['row_end']}"
        elif meta.get("doc_type") == "summary":
            location = "summary"
        else:
            location = f"chunk {meta.get('chunk_index', i)}"

        parts.append(f"[{i}] Source: {source} ({location})\n{chunk['text']}")
        citations.append({
            "index":       i,
            "source":      source,
            "location":    location,
            "score":       chunk["score"],
            "preview":     chunk["text"][:300],
            "source_type": meta.get("source_type", "unknown"),
        })

    return "\n\n---\n\n".join(parts), citations


# ── helpers ───────────────────────────────────────────────────────────────────

def _format_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for msg in history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


async def _fetch_chunks(
    query: str,
    embedding_model: str,
    filters: Optional[Dict[str, Any]],
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], float, bool]:
    """
    Shared retrieve → post-filter → trim helper used by all four generators.

    Pool sizing (fix for DeepEval path):
      Previously callers passed top_k directly here — routes sent 500 (correct
      but wasteful), deepeval sent 5 (too small, missed candidates).
      Now pool_k is always computed from final_k via _pool_k(), so every code
      path (streaming, non-streaming, DeepEval) gets proper candidate pooling:
        final_k=5  → pool_k=50
        final_k=10 → pool_k=100

    Fallback chain:
      1. Pinecone with metadata filters (score_threshold disabled — see retriever.py)
      2. Pinecone semantic-only if filtered search returns 0 results (in retriever.py)
      3. If numeric post_filters wipe everything → use unfiltered semantic results
         so the LLM always has context.

    Returns: (chunks, retrieval_latency_ms, filter_fallback_used)
    """
    effective_final_k = final_k if final_k is not None else settings.top_k
    pool_k = _pool_k(effective_final_k)

    retrieval = await retrieve(query, embedding_model, pool_k, filters)
    chunks    = retrieval["results"]

    if post_filters:
        filtered = apply_post_filters(chunks, post_filters)
        # Fallback: post_filters wiped all results → keep semantic results so
        # the LLM can still answer (it will caveat appropriately).
        chunks = filtered if filtered else chunks

    # Trim to what the LLM should actually see
    chunks = chunks[:effective_final_k]

    return chunks, retrieval["latency_ms"], retrieval["filter_fallback"]


# ── Non-streaming (used by RAGAS evaluation) ──────────────────────────────────

async def generate_response(
    query: str,
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    # Legacy positional arg (top_k) kept for backwards compat — ignored,
    # pool is now computed internally from final_k via _pool_k().
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    chunks, retrieval_ms, filter_fallback = await _fetch_chunks(
        query, embedding_model, filters, post_filters, final_k
    )

    if not chunks:
        return {
            "answer":                "No relevant documents found in the knowledge base.",
            "citations":             [],
            "chunks":                [],
            "retrieval_latency_ms":  retrieval_ms,
            "generation_latency_ms": 0,
            "embedding_model":       embedding_model,
            "filter_fallback":       filter_fallback,
            "tokens_used":           None,
        }

    context, citations = _build_context(chunks)

    t0 = time.perf_counter()
    response = await _get_client().responses.create(
        model=settings.openai_chat_model,
        instructions=SYSTEM_PROMPT,
        input=f"Context:\n{context}\n\nQuestion: {query}",
        max_output_tokens=1800,
    )
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = response.output_text
    usage  = response.usage
    prompt_tokens     = getattr(usage, "input_tokens",  0)
    completion_tokens = getattr(usage, "output_tokens", 0)
    total_tokens      = getattr(usage, "total_tokens",  prompt_tokens + completion_tokens)

    return {
        "answer":                answer,
        "citations":             citations,
        "chunks":                chunks,
        "retrieval_latency_ms":  retrieval_ms,
        "generation_latency_ms": gen_ms,
        "total_latency_ms":      retrieval_ms + gen_ms,
        "embedding_model":       embedding_model,
        "filter_fallback":       filter_fallback,
        "tokens_used": {
            "prompt":     prompt_tokens,
            "completion": completion_tokens,
            "total":      total_tokens,
        },
    }


async def generate_chat_response(
    query: str,
    history: List[Dict[str, str]],
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,  # legacy compat, ignored
) -> Dict[str, Any]:
    chunks, retrieval_ms, filter_fallback = await _fetch_chunks(
        query, embedding_model, filters, post_filters, final_k
    )

    if not chunks:
        return {
            "answer":                "No relevant documents found in the knowledge base.",
            "citations":             [],
            "chunks":                [],
            "retrieval_latency_ms":  retrieval_ms,
            "generation_latency_ms": 0,
            "embedding_model":       embedding_model,
            "filter_fallback":       filter_fallback,
            "tokens_used":           None,
        }

    context, citations = _build_context(chunks)
    history_text = _format_history(history)

    input_text = (
        f"Context:\n{context}\n\n"
        f"Conversation so far:\n{history_text}\n\nUser: {query}"
        if history_text
        else f"Context:\n{context}\n\nQuestion: {query}"
    )

    t0 = time.perf_counter()
    response = await _get_client().responses.create(
        model=settings.openai_chat_model,
        instructions=CHAT_SYSTEM_PROMPT,
        input=input_text,
        max_output_tokens=1800,
    )
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = response.output_text
    usage  = response.usage
    prompt_tokens     = getattr(usage, "input_tokens",  0)
    completion_tokens = getattr(usage, "output_tokens", 0)
    total_tokens      = getattr(usage, "total_tokens",  prompt_tokens + completion_tokens)

    return {
        "answer":                answer,
        "citations":             citations,
        "chunks":                chunks,
        "retrieval_latency_ms":  retrieval_ms,
        "generation_latency_ms": gen_ms,
        "total_latency_ms":      retrieval_ms + gen_ms,
        "embedding_model":       embedding_model,
        "filter_fallback":       filter_fallback,
        "tokens_used": {
            "prompt":     prompt_tokens,
            "completion": completion_tokens,
            "total":      total_tokens,
        },
    }


# ── Streaming versions ────────────────────────────────────────────────────────

async def stream_generate_response(
    query: str,
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,  # legacy compat, ignored
) -> AsyncGenerator[str, None]:
    """
    SSE event sequence:
      {"type":"citations", "citations":[...], "retrieval_latency_ms": N, "filter_fallback": bool}
      {"type":"token",     "content": "..."}  x many
      {"type":"done",      "generation_latency_ms": N, "tokens_used": {...}}
      {"type":"error",     "message": "..."}
    """
    chunks, retrieval_ms, filter_fallback = await _fetch_chunks(
        query, embedding_model, filters, post_filters, final_k
    )

    if not chunks:
        yield _sse({"type": "error", "message": "No relevant documents found in the knowledge base."})
        return

    context, citations = _build_context(chunks)

    yield _sse({
        "type":                 "citations",
        "citations":            citations,
        "retrieval_latency_ms": retrieval_ms,
        "embedding_model":      embedding_model,
        # Surface fallback so the UI can show a caveat banner when filters
        # found nothing and we fell back to semantic-only results.
        "filter_fallback":      filter_fallback,
    })

    t0 = time.perf_counter()
    prompt_tokens = completion_tokens = total_tokens = 0

    try:
        stream = await _get_client().responses.create(
            model=settings.openai_chat_model,
            instructions=SYSTEM_PROMPT,
            input=f"Context:\n{context}\n\nQuestion: {query}",
            max_output_tokens=1800,
            stream=True,
        )
        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield _sse({"type": "token", "content": delta})
            elif etype == "response.completed":
                resp = getattr(event, "response", None)
                if resp:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        prompt_tokens     = getattr(usage, "input_tokens",  0)
                        completion_tokens = getattr(usage, "output_tokens", 0)
                        total_tokens      = getattr(usage, "total_tokens",  prompt_tokens + completion_tokens)
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    gen_ms = round((time.perf_counter() - t0) * 1000, 2)
    yield _sse({
        "type":                  "done",
        "generation_latency_ms": gen_ms,
        "total_latency_ms":      retrieval_ms + gen_ms,
        "tokens_used": {
            "prompt":     prompt_tokens,
            "completion": completion_tokens,
            "total":      total_tokens,
        },
    })


async def stream_generate_chat_response(
    query: str,
    history: List[Dict[str, str]],
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,  # legacy compat, ignored
) -> AsyncGenerator[str, None]:
    chunks, retrieval_ms, filter_fallback = await _fetch_chunks(
        query, embedding_model, filters, post_filters, final_k
    )

    if not chunks:
        yield _sse({"type": "error", "message": "No relevant documents found in the knowledge base."})
        return

    context, citations = _build_context(chunks)

    yield _sse({
        "type":                 "citations",
        "citations":            citations,
        "retrieval_latency_ms": retrieval_ms,
        "embedding_model":      embedding_model,
        "filter_fallback":      filter_fallback,
    })

    history_text = _format_history(history)
    input_text = (
        f"Context:\n{context}\n\n"
        f"Conversation so far:\n{history_text}\n\nUser: {query}"
        if history_text
        else f"Context:\n{context}\n\nQuestion: {query}"
    )

    t0 = time.perf_counter()
    prompt_tokens = completion_tokens = total_tokens = 0

    try:
        stream = await _get_client().responses.create(
            model=settings.openai_chat_model,
            instructions=CHAT_SYSTEM_PROMPT,
            input=input_text,
            max_output_tokens=1800,
            stream=True,
        )
        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield _sse({"type": "token", "content": delta})
            elif etype == "response.completed":
                resp = getattr(event, "response", None)
                if resp:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        prompt_tokens     = getattr(usage, "input_tokens",  0)
                        completion_tokens = getattr(usage, "output_tokens", 0)
                        total_tokens      = getattr(usage, "total_tokens",  prompt_tokens + completion_tokens)
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    gen_ms = round((time.perf_counter() - t0) * 1000, 2)
    yield _sse({
        "type":                  "done",
        "generation_latency_ms": gen_ms,
        "total_latency_ms":      retrieval_ms + gen_ms,
        "tokens_used": {
            "prompt":     prompt_tokens,
            "completion": completion_tokens,
            "total":      total_tokens,
        },
    })