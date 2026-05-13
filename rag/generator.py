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


# ─────────────────────────────────────────────
# Azure OpenAI Client
# ─────────────────────────────────────────────

def _get_client() -> AsyncAzureOpenAI:
    global _client

    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            api_version=_RESPONSES_API_VERSION,
        )

    return _client


# ─────────────────────────────────────────────
# SSE Helper
# ─────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────
# Lightweight Fast System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an enterprise multilingual RAG assistant.

Rules:
- Use ONLY retrieved context.
- Never hallucinate or infer missing facts.
- Cite factual claims using [N].
- Preserve exact numbers and KPIs.
- Respond in user's language.
- Keep responses concise and professional.
- Avoid repetition.
- Use markdown only when useful.
- If information is missing, say:
  "Insufficient information found in the retrieved context."
"""


# ─────────────────────────────────────────────
# Deduplicate Chunks
# ─────────────────────────────────────────────

def _deduplicate_chunks(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    seen = set()
    unique_chunks = []

    for chunk in chunks:

        normalized = (
            " ".join(chunk["text"].split())
            .strip()
            .lower()
        )

        if normalized not in seen:
            seen.add(normalized)
            unique_chunks.append(chunk)

    return unique_chunks


# ─────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────

def _build_context(
    chunks: List[Dict[str, Any]]
) -> tuple[str, List[Dict[str, Any]]]:

    citations = []
    parts = []

    chunks = _deduplicate_chunks(chunks)

    for i, chunk in enumerate(chunks, 1):

        meta = chunk["metadata"]

        source = meta.get("source", "unknown")

        if "page" in meta:
            location = f"page {meta['page']}"

        elif meta.get("doc_type") == "row":
            location = f"row {meta.get('row_number', '?')}"

        else:
            location = f"chunk {meta.get('chunk_index', i)}"

        # aggressive context compression
        text = chunk["text"].strip()[:1200]

        parts.append(
            f"[{i}] Source: {source} ({location})\n{text}"
        )

        citations.append({
            "index": i,
            "source": source,
            "location": location,
        })

    return "\n\n---\n\n".join(parts), citations


# ─────────────────────────────────────────────
# History Formatter
# ─────────────────────────────────────────────

def _format_history(
    history: List[Dict[str, str]]
) -> str:

    lines = []

    for msg in history[-4:]:

        role = (
            "User"
            if msg.get("role") == "user"
            else "Assistant"
        )

        content = msg.get("content", "")

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Query Complexity Router
# ─────────────────────────────────────────────

def _determine_final_k(query: str) -> int:

    query = query.lower()

    analytical_keywords = [
        "compare",
        "analysis",
        "trend",
        "summary",
        "performance",
        "regional",
        "ranking",
        "insights",
    ]

    if any(word in query for word in analytical_keywords):
        return 6

    return 4


# ─────────────────────────────────────────────
# Retrieval Pipeline
# ─────────────────────────────────────────────

async def _fetch_chunks(
    query: str,
    embedding_model: str,
    filters: Optional[Dict[str, Any]],
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
):

    effective_final_k = (
        final_k
        if final_k is not None
        else _determine_final_k(query)
    )

    effective_final_k = min(effective_final_k, 6)

    pool_k = _pool_k(effective_final_k)

    retrieval = await retrieve(
        query=query,
        embedding_model=embedding_model,
        top_k=pool_k,
        filters=filters,
    )

    chunks = retrieval["results"]

    # numeric filtering
    if post_filters:

        filtered = apply_post_filters(
            chunks,
            post_filters,
        )

        chunks = filtered if filtered else chunks

    chunks = chunks[:effective_final_k]

    return (
        chunks,
        retrieval["latency_ms"],
        retrieval["filter_fallback"],
    )


# ─────────────────────────────────────────────
# Shared Completion Call
# ─────────────────────────────────────────────

async def _generate_completion(
    input_text: str,
):

    client = _get_client()

    return await client.responses.create(
        model=settings.openai_chat_model,
        instructions=SYSTEM_PROMPT,
        input=input_text,
        temperature=0,
        max_output_tokens=500,
    )


# ─────────────────────────────────────────────
# Non Streaming Response
# ─────────────────────────────────────────────

async def generate_response(
    query: str,
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:

    (
        chunks,
        retrieval_ms,
        filter_fallback,
    ) = await _fetch_chunks(
        query,
        embedding_model,
        filters,
        post_filters,
        final_k,
    )

    if not chunks:

        return {
            "answer": (
                "Insufficient information found "
                "in the retrieved context."
            ),
            "citations": [],
            "chunks": [],
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": 0,
            "embedding_model": embedding_model,
            "filter_fallback": filter_fallback,
            "tokens_used": None,
        }

    context, citations = _build_context(chunks)

    input_text = f"""
Context:
{context}

Question:
{query}
"""

    t0 = time.perf_counter()

    response = await _generate_completion(
        input_text
    )

    generation_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    answer = response.output_text.strip()

    usage = response.usage

    prompt_tokens = getattr(
        usage,
        "input_tokens",
        0,
    )

    completion_tokens = getattr(
        usage,
        "output_tokens",
        0,
    )

    total_tokens = getattr(
        usage,
        "total_tokens",
        prompt_tokens + completion_tokens,
    )

    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks,
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": generation_ms,
        "total_latency_ms": (
            retrieval_ms + generation_ms
        ),
        "embedding_model": embedding_model,
        "filter_fallback": filter_fallback,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    }


# ─────────────────────────────────────────────
# Chat Response
# ─────────────────────────────────────────────

async def generate_chat_response(
    query: str,
    history: List[Dict[str, str]],
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,
):

    (
        chunks,
        retrieval_ms,
        filter_fallback,
    ) = await _fetch_chunks(
        query,
        embedding_model,
        filters,
        post_filters,
        final_k,
    )

    if not chunks:

        return {
            "answer": (
                "Insufficient information found "
                "in the retrieved context."
            ),
            "citations": [],
            "chunks": [],
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": 0,
            "embedding_model": embedding_model,
            "filter_fallback": filter_fallback,
            "tokens_used": None,
        }

    context, citations = _build_context(chunks)

    history_text = _format_history(history)

    input_text = f"""
Context:
{context}

Conversation:
{history_text}

Question:
{query}
"""

    t0 = time.perf_counter()

    response = await _generate_completion(
        input_text
    )

    generation_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    answer = response.output_text.strip()

    usage = response.usage

    prompt_tokens = getattr(
        usage,
        "input_tokens",
        0,
    )

    completion_tokens = getattr(
        usage,
        "output_tokens",
        0,
    )

    total_tokens = getattr(
        usage,
        "total_tokens",
        prompt_tokens + completion_tokens,
    )

    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks,
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": generation_ms,
        "total_latency_ms": (
            retrieval_ms + generation_ms
        ),
        "embedding_model": embedding_model,
        "filter_fallback": filter_fallback,
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    }


# ─────────────────────────────────────────────
# Streaming Response
# ─────────────────────────────────────────────

async def stream_generate_response(
    query: str,
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,
) -> AsyncGenerator[str, None]:

    (
        chunks,
        retrieval_ms,
        filter_fallback,
    ) = await _fetch_chunks(
        query,
        embedding_model,
        filters,
        post_filters,
        final_k,
    )

    if not chunks:

        yield _sse({
            "type": "error",
            "message": (
                "Insufficient information found "
                "in the retrieved context."
            ),
        })

        return

    context, citations = _build_context(chunks)

    yield _sse({
        "type": "citations",
        "citations": citations,
        "retrieval_latency_ms": retrieval_ms,
        "filter_fallback": filter_fallback,
    })

    input_text = f"""
Context:
{context}

Question:
{query}
"""

    client = _get_client()

    t0 = time.perf_counter()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:

        stream = await client.responses.create(
            model=settings.openai_chat_model,
            instructions=SYSTEM_PROMPT,
            input=input_text,
            temperature=0,
            max_output_tokens=500,
            stream=True,
        )

        async for event in stream:

            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":

                delta = getattr(event, "delta", "")

                if delta:

                    yield _sse({
                        "type": "token",
                        "content": delta,
                    })

            elif event_type == "response.completed":

                response = getattr(event, "response", None)

                if response:

                    usage = getattr(response, "usage", None)

                    if usage:

                        prompt_tokens = getattr(
                            usage,
                            "input_tokens",
                            0,
                        )

                        completion_tokens = getattr(
                            usage,
                            "output_tokens",
                            0,
                        )

                        total_tokens = getattr(
                            usage,
                            "total_tokens",
                            prompt_tokens + completion_tokens,
                        )

    except Exception as exc:

        yield _sse({
            "type": "error",
            "message": str(exc),
        })

        return

    generation_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    yield _sse({
        "type": "done",
        "generation_latency_ms": generation_ms,
        "total_latency_ms": (
            retrieval_ms + generation_ms
        ),
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    })


# ─────────────────────────────────────────────
# Streaming Chat Response
# ─────────────────────────────────────────────

async def stream_generate_chat_response(
    query: str,
    history: List[Dict[str, str]],
    embedding_model: str = "openai",
    filters: Optional[Dict[str, Any]] = None,
    post_filters: Optional[List[NumericCondition]] = None,
    final_k: Optional[int] = None,
    top_k: Optional[int] = None,
) -> AsyncGenerator[str, None]:

    (
        chunks,
        retrieval_ms,
        filter_fallback,
    ) = await _fetch_chunks(
        query,
        embedding_model,
        filters,
        post_filters,
        final_k,
    )

    if not chunks:

        yield _sse({
            "type": "error",
            "message": (
                "Insufficient information found "
                "in the retrieved context."
            ),
        })

        return

    context, citations = _build_context(chunks)

    yield _sse({
        "type": "citations",
        "citations": citations,
        "retrieval_latency_ms": retrieval_ms,
        "filter_fallback": filter_fallback,
    })

    history_text = _format_history(history)

    input_text = f"""
Context:
{context}

Conversation:
{history_text}

Question:
{query}
"""

    client = _get_client()

    t0 = time.perf_counter()

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    try:

        stream = await client.responses.create(
            model=settings.openai_chat_model,
            instructions=SYSTEM_PROMPT,
            input=input_text,
            temperature=0,
            max_output_tokens=500,
            stream=True,
        )

        async for event in stream:

            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":

                delta = getattr(event, "delta", "")

                if delta:

                    yield _sse({
                        "type": "token",
                        "content": delta,
                    })

            elif event_type == "response.completed":

                response = getattr(event, "response", None)

                if response:

                    usage = getattr(response, "usage", None)

                    if usage:

                        prompt_tokens = getattr(
                            usage,
                            "input_tokens",
                            0,
                        )

                        completion_tokens = getattr(
                            usage,
                            "output_tokens",
                            0,
                        )

                        total_tokens = getattr(
                            usage,
                            "total_tokens",
                            prompt_tokens + completion_tokens,
                        )

    except Exception as exc:

        yield _sse({
            "type": "error",
            "message": str(exc),
        })

        return

    generation_ms = round(
        (time.perf_counter() - t0) * 1000,
        2,
    )

    yield _sse({
        "type": "done",
        "generation_latency_ms": generation_ms,
        "total_latency_ms": (
            retrieval_ms + generation_ms
        ),
        "tokens_used": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
    })