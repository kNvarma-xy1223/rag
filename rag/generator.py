import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

from openai import AsyncAzureOpenAI

from config.settings import settings
from retrieval.retriever import retrieve

# gpt-5.4-pro requires Responses API with api-version >= 2025-03-01-preview
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
    """Format a dict as a Server-Sent Event string."""
    return f"data: {json.dumps(data)}\n\n"


SYSTEM_PROMPT = """You are a precise multilingual RAG assistant. Your rules:
1. Answer strictly from the provided numbered context blocks.
2. Cite every factual claim using [N] where N is the context block number.
3. Preserve exact numerical values — never round or paraphrase numbers.
4. If context is insufficient, state "Insufficient context to answer fully."
5. Respond in the same language as the question (English or Spanish).
6. Never fabricate information not present in the context."""

CHAT_SYSTEM_PROMPT = """You are a precise multilingual RAG assistant in a conversation. Your rules:
1. Answer strictly from the provided numbered context blocks.
2. Cite every factual claim using [N] where N is the context block number.
3. Preserve exact numerical values — never round or paraphrase numbers.
4. Use conversation history for follow-up context but only answer from the current context blocks.
5. If context is insufficient, state "Insufficient context to answer fully."
6. Respond in the same language as the question (English or Spanish).
7. Never fabricate information not present in the context.
8. Keep answers concise and conversational."""


def _build_context(chunks: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    citations: List[Dict[str, Any]] = []
    parts: List[str] = []

    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
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
        citations.append(
            {
                "index": i,
                "source": source,
                "location": location,
                "score": chunk["score"],
                "preview": chunk["text"][:300],
                "source_type": meta.get("source_type", "unknown"),
            }
        )

    return "\n\n---\n\n".join(parts), citations


# ── Non-streaming (kept for RAGAS evaluation / benchmark) ────────────────────

async def generate_response(
    query: str,
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    retrieval = await retrieve(query, embedding_model, top_k, filters)
    chunks = retrieval["results"]

    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "citations": [],
            "chunks": [],
            "retrieval_latency_ms": retrieval["latency_ms"],
            "generation_latency_ms": 0,
            "embedding_model": embedding_model,
            "tokens_used": None,
        }

    context, citations = _build_context(chunks)

    t0 = time.perf_counter()
    response = await _get_client().responses.create(
        model=settings.openai_chat_model,
        instructions=SYSTEM_PROMPT,
        input=f"Context:\n{context}\n\nQuestion: {query}",
        max_output_tokens=1500,
    )
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = response.output_text
    usage = response.usage
    prompt_tokens     = getattr(usage, "input_tokens", 0)
    completion_tokens = getattr(usage, "output_tokens", 0)
    total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks,
        "retrieval_latency_ms": retrieval["latency_ms"],
        "generation_latency_ms": gen_ms,
        "total_latency_ms": retrieval["latency_ms"] + gen_ms,
        "embedding_model": embedding_model,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    }


async def generate_chat_response(
    query: str,
    history: List[Dict[str, str]],
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """RAG response with conversation history (last 6 turns = 3 exchanges)."""
    retrieval = await retrieve(query, embedding_model, top_k, filters)
    chunks = retrieval["results"]

    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "citations": [],
            "chunks": [],
            "retrieval_latency_ms": retrieval["latency_ms"],
            "generation_latency_ms": 0,
            "embedding_model": embedding_model,
            "tokens_used": None,
        }

    context, citations = _build_context(chunks)

    history_lines = []
    for msg in history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {msg.get('content', '')}")
    history_text = "\n".join(history_lines)

    if history_text:
        input_text = (
            f"Context:\n{context}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"User: {query}"
        )
    else:
        input_text = f"Context:\n{context}\n\nQuestion: {query}"

    t0 = time.perf_counter()
    response = await _get_client().responses.create(
        model=settings.openai_chat_model,
        instructions=CHAT_SYSTEM_PROMPT,
        input=input_text,
        max_output_tokens=1500,
    )
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    answer = response.output_text
    usage = response.usage
    prompt_tokens     = getattr(usage, "input_tokens", 0)
    completion_tokens = getattr(usage, "output_tokens", 0)
    total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks,
        "retrieval_latency_ms": retrieval["latency_ms"],
        "generation_latency_ms": gen_ms,
        "total_latency_ms": retrieval["latency_ms"] + gen_ms,
        "embedding_model": embedding_model,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    }


# ── Streaming versions ────────────────────────────────────────────────────────
# IMPORTANT: These must be defined as `async def` functions that `yield` —
# FastAPI StreamingResponse consumes the generator directly by calling the
# function. Do NOT wrap with `await` in the route; pass the call expression.

async def stream_generate_response(
    query: str,
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted strings.

    Event sequence:
      1. {"type":"citations", "citations":[...], "retrieval_latency_ms": N}
      2. {"type":"token", "content": "..."} x many
      3. {"type":"done", "generation_latency_ms": N, "tokens_used": {...}}
      OR {"type":"error", "message":"..."}
    """
    retrieval = await retrieve(query, embedding_model, top_k, filters)
    chunks = retrieval["results"]

    if not chunks:
        yield _sse({"type": "error", "message": "No relevant documents found in the knowledge base."})
        return

    context, citations = _build_context(chunks)

    # Send citations immediately so UI can render source cards while tokens stream
    yield _sse({
        "type": "citations",
        "citations": citations,
        "retrieval_latency_ms": retrieval["latency_ms"],
        "embedding_model": embedding_model,
    })

    t0 = time.perf_counter()
    prompt_tokens = completion_tokens = total_tokens = 0

    try:
        stream = await _get_client().responses.create(
            model=settings.openai_chat_model,
            instructions=SYSTEM_PROMPT,
            input=f"Context:\n{context}\n\nQuestion: {query}",
            max_output_tokens=1500,
            stream=True,
        )

        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield _sse({"type": "token", "content": delta})
            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        prompt_tokens     = getattr(usage, "input_tokens", 0)
                        completion_tokens = getattr(usage, "output_tokens", 0)
                        total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    gen_ms = round((time.perf_counter() - t0) * 1000, 2)
    yield _sse({
        "type": "done",
        "generation_latency_ms": gen_ms,
        "total_latency_ms": retrieval["latency_ms"] + gen_ms,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    })


async def stream_generate_chat_response(
    query: str,
    history: List[Dict[str, str]],
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming RAG chat response. Same SSE event sequence as stream_generate_response.
    """
    retrieval = await retrieve(query, embedding_model, top_k, filters)
    chunks = retrieval["results"]

    if not chunks:
        yield _sse({"type": "error", "message": "No relevant documents found in the knowledge base."})
        return

    context, citations = _build_context(chunks)

    yield _sse({
        "type": "citations",
        "citations": citations,
        "retrieval_latency_ms": retrieval["latency_ms"],
        "embedding_model": embedding_model,
    })

    history_lines = []
    for msg in history[-6:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {msg.get('content', '')}")
    history_text = "\n".join(history_lines)

    if history_text:
        input_text = (
            f"Context:\n{context}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"User: {query}"
        )
    else:
        input_text = f"Context:\n{context}\n\nQuestion: {query}"

    t0 = time.perf_counter()
    prompt_tokens = completion_tokens = total_tokens = 0

    try:
        stream = await _get_client().responses.create(
            model=settings.openai_chat_model,
            instructions=CHAT_SYSTEM_PROMPT,
            input=input_text,
            max_output_tokens=1500,
            stream=True,
        )

        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield _sse({"type": "token", "content": delta})
            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        prompt_tokens     = getattr(usage, "input_tokens", 0)
                        completion_tokens = getattr(usage, "output_tokens", 0)
                        total_tokens      = getattr(usage, "total_tokens", prompt_tokens + completion_tokens)

    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    gen_ms = round((time.perf_counter() - t0) * 1000, 2)
    yield _sse({
        "type": "done",
        "generation_latency_ms": gen_ms,
        "total_latency_ms": retrieval["latency_ms"] + gen_ms,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    })