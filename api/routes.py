"""
api/routes.py

Every query endpoint now runs through the full pipeline:
  NL query → query_parser → Pinecone (metadata filters) → LLM stream
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    CompareRequest,
    QueryRequest,
    RagasBenchmarkRequest,
    RagasEvaluateRequest,
)
from chunking.semantic_chunker import chunk_documents
from config.settings import settings
from evaluation.deepeval_evaluator import (
    evaluate_with_ragas,
    run_ragas_benchmark,
    run_ragas_benchmark_compare,
)
from ingestion.csv_ingestor import ingest_csv
from ingestion.pdf_ingestor import ingest_pdf
from rag.generator import (
    _sse,
    generate_response,
    generate_chat_response,
    stream_generate_chat_response,
    stream_generate_response,
)
from rag.query_parser import apply_post_filters, parse_query
from retrieval.retriever import _pool_k, compare_retrievals, retrieve
from vectordb.pinecone_manager import pinecone_manager

router = APIRouter()

_SUPPORTED = {".pdf", ".csv"}

_SSE_HEADERS = {
    "Cache-Control":    "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":       "keep-alive",
}

_RAGAS_BENCHMARK_SAMPLE = [
    {
        "query":        "What is the total revenue for Q2 2024?",
        "ground_truth": "The total revenue for Q2 2024 is reported in the corporate financial report.",
        "language":     "en",
        "category":     "financial",
    },
    {
        "query":        "What is the gross margin percentage?",
        "ground_truth": "The gross margin percentage is derived from revenue minus cost of goods sold.",
        "language":     "en",
        "category":     "financial",
    },
    {
        "query":        "How many active customers did the company have at quarter-end?",
        "ground_truth": "The number of active customers at quarter-end is stated in the metrics section.",
        "language":     "en",
        "category":     "metrics",
    },
    {
        "query":        "What is the Annual Recurring Revenue (ARR)?",
        "ground_truth": "ARR is the annualised value of subscription revenue reported in the SaaS metrics section.",
        "language":     "en",
        "category":     "saas_metrics",
    },
    {
        "query":        "What is the revenue guidance for Q3 2024?",
        "ground_truth": "Revenue guidance for Q3 2024 is provided in the forward-looking statements section.",
        "language":     "en",
        "category":     "guidance",
    },
]


def _get_embedder(model: str):
    if model == "openai":
        from embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    from embeddings.cohere_embedder import CohereEmbedder
    return CohereEmbedder()


def _attach_k(result: dict, k: int) -> dict:
    result["k"] = k
    return result


def _merge_filters(
    explicit: Optional[Dict[str, Any]],
    parsed: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Merge caller-supplied filters with parser-extracted filters.
    Caller-supplied filters always win on conflict.
    If both exist, wrap in $and.

    Note: language is NOT added here.  parse_query(language=req.language)
    already injects language into parsed.metadata_filters via
    _build_pinecone_filter().  Calling req.effective_filters() and also
    passing language to parse_query would double-inject the language clause,
    corrupting the $and structure.  Always pass req.filters (raw, no language)
    as the `explicit` argument.
    """
    if explicit and parsed:
        return {"$and": [explicit, parsed]}
    return explicit or parsed


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/api/ragas-benchmark-sample")
async def get_ragas_benchmark_sample():
    return _RAGAS_BENCHMARK_SAMPLE


@router.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    embedding_model: str = Form("both"),
):
    suffix     = Path(file.filename).suffix.lower()
    filename   = file.filename
    file_bytes = await file.read()

    async def progress_stream():
        if suffix not in _SUPPORTED:
            yield _sse({"type": "error", "message": f"Unsupported file type '{suffix}'. Use PDF or CSV."})
            return

        yield _sse({"type": "progress", "percent": 5, "message": "File received, saving to disk..."})

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            yield _sse({"type": "progress", "percent": 15, "message": "Parsing document..."})
            raw_docs = (await ingest_pdf(tmp_path)) if suffix == ".pdf" else (await ingest_csv(tmp_path))
            for doc in raw_docs:
                doc["metadata"]["source"] = filename

            detected_lang = raw_docs[0]["metadata"].get("language", "en") if raw_docs else "en"
            yield _sse({
                "type": "progress", "percent": 30,
                "message": (
                    f"Parsed {len(raw_docs)} page(s) "
                    f"[language detected: {detected_lang}]. "
                    "Splitting into chunks..."
                ),
            })

            chunks = await chunk_documents(raw_docs)
            texts  = [c["text"] for c in chunks]
            results: dict = {}

            yield _sse({"type": "progress", "percent": 45,
                        "message": f"Created {len(chunks)} chunks. Starting embedding..."})

            if embedding_model in ("openai", "both"):
                yield _sse({"type": "progress", "percent": 55, "message": "Generating OpenAI embeddings..."})
                try:
                    embedder   = _get_embedder("openai")
                    embeddings = await embedder.embed(texts)
                except Exception as emb_exc:
                    yield _sse({
                        "type":    "error",
                        "stage":   "openai_embedding",
                        "message": (
                            f"OpenAI embedding failed: {emb_exc}. "
                            "Check OPENAI_API_KEY and OPENAI_ENDPOINT in .env"
                        ),
                    })
                    return
                yield _sse({"type": "progress", "percent": 68, "message": "Indexing OpenAI vectors in Pinecone..."})
                try:
                    results["openai"] = await pinecone_manager.upsert_chunks(
                        settings.pinecone_openai_index, chunks, embeddings
                    )
                except Exception as ups_exc:
                    yield _sse({
                        "type":    "error",
                        "stage":   "pinecone_openai_upsert",
                        "message": f"Pinecone upsert (OpenAI index) failed: {ups_exc}",
                    })
                    return

            if embedding_model in ("cohere", "both"):
                yield _sse({"type": "progress", "percent": 78, "message": "Generating Cohere embeddings..."})
                try:
                    embedder   = _get_embedder("cohere")
                    embeddings = await embedder.embed(texts)
                except Exception as emb_exc:
                    yield _sse({
                        "type":    "error",
                        "stage":   "cohere_embedding",
                        "message": (
                            f"Cohere embedding failed: {emb_exc}. "
                            "Check COHERE_API_KEY and COHERE_ENDPOINT in .env"
                        ),
                    })
                    return
                yield _sse({"type": "progress", "percent": 92, "message": "Indexing Cohere vectors in Pinecone..."})
                try:
                    results["cohere"] = await pinecone_manager.upsert_chunks(
                        settings.pinecone_cohere_index, chunks, embeddings
                    )
                except Exception as ups_exc:
                    yield _sse({
                        "type":    "error",
                        "stage":   "pinecone_cohere_upsert",
                        "message": f"Pinecone upsert (Cohere index) failed: {ups_exc}",
                    })
                    return

            yield _sse({
                "type":        "done",
                "status":      "success",
                "filename":    filename,
                "source_type": suffix[1:],
                "raw_documents": len(raw_docs),
                "chunks":      len(chunks),
                "indexed":     results,
                "language":    detected_lang,
                "percent":     100,
            })
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    return StreamingResponse(progress_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/api/query")
async def query_rag(req: QueryRequest):
    """
    Full pipeline:
      NL query → query_parser → Pinecone (metadata filters) → LLM stream

    Pool sizing: _pool_k(req.top_k) is computed inside _fetch_chunks in
    generator.py, so routes no longer need to manage retrieval_k at all.
    Pass final_k=req.top_k; generator handles the rest.

    Language: passed to parse_query() which injects it into
    metadata_filters.  Do NOT call req.effective_filters() here —
    that would add language a second time, corrupting the $and structure.
    """
    async def _stream():
        # 1. Parse natural language → structured filters
        parsed = await parse_query(req.query, language=req.language)

        yield _sse({
            "type":              "parsing",
            "semantic_query":    parsed.semantic_query,
            "filters_extracted": parsed.raw_filters_extracted,
            "pinecone_filter":   parsed.metadata_filters,
        })

        # 2. Merge caller-supplied raw filters with parser-extracted filters.
        #    Use req.filters (not req.effective_filters()) — language is
        #    already in parsed.metadata_filters.
        merged_filters = _merge_filters(req.filters, parsed.metadata_filters)

        # 3. Stream — generator computes pool_k internally from final_k
        async for chunk in stream_generate_response(
            parsed.semantic_query,
            req.embedding_model,
            merged_filters,
            post_filters=parsed.numeric_post_filters,
            final_k=req.top_k,
        ):
            yield chunk

    return StreamingResponse(_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/api/retrieve")
async def retrieve_chunks(req: QueryRequest):
    """
    Retrieve with full query parsing.
    Returns parsed filter info alongside results for debugging.

    pool_k is exposed in the response so you can verify how many candidates
    Pinecone fetched (e.g. pool_k=50 when top_k=5).
    """
    parsed         = await parse_query(req.query, language=req.language)
    # Use req.filters (not req.effective_filters()) — language already in
    # parsed.metadata_filters via parse_query(language=req.language).
    merged_filters = _merge_filters(req.filters, parsed.metadata_filters)

    # Compute pool_k the same way _fetch_chunks does so the debug output
    # reflects the actual number of candidates sent to Pinecone.
    pool_k = _pool_k(req.top_k)
    result = await retrieve(
        parsed.semantic_query,
        req.embedding_model,
        pool_k,
        merged_filters,
        req.score_threshold,
    )

    # Post-filter safety net then trim to requested top_k
    result["results"]  = apply_post_filters(result["results"], parsed.numeric_post_filters)
    result["results"]  = result["results"][:req.top_k]
    result["returned"] = len(result["results"])
    result["pool_k"]   = pool_k  # expose for debugging

    result["query_parsing"] = {
        "original_query":    req.query,
        "semantic_query":    parsed.semantic_query,
        "filters_extracted": parsed.raw_filters_extracted,
        "pinecone_filter":   parsed.metadata_filters,
    }
    return result


@router.post("/api/compare")
async def compare_models(req: CompareRequest):
    """Compare OpenAI vs Cohere retrieval, both with full query parsing."""
    parsed         = await parse_query(req.query, language=req.language)
    merged_filters = _merge_filters(req.filters, parsed.metadata_filters)

    # Pool from final_k so both models get the same candidate budget
    pool_k = _pool_k(req.top_k)
    result = await compare_retrievals(parsed.semantic_query, pool_k, merged_filters)

    for model_key in ("openai", "cohere"):
        if model_key in result and "results" in result[model_key]:
            result[model_key]["results"]  = apply_post_filters(
                result[model_key]["results"], parsed.numeric_post_filters
            )[:req.top_k]
            result[model_key]["returned"] = len(result[model_key]["results"])

    result["pool_k"] = pool_k
    result["query_parsing"] = {
        "original_query":    req.query,
        "semantic_query":    parsed.semantic_query,
        "filters_extracted": parsed.raw_filters_extracted,
        "pinecone_filter":   parsed.metadata_filters,
    }
    return result


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """Multi-turn chat with full query parsing on each turn."""
    parsed         = await parse_query(req.query, language=req.language)
    merged_filters = _merge_filters(req.filters, parsed.metadata_filters)

    return StreamingResponse(
        stream_generate_chat_response(
            parsed.semantic_query,
            req.history,
            req.embedding_model,
            merged_filters,
            post_filters=parsed.numeric_post_filters,
            final_k=req.top_k,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _run_single_evaluate(req: "RagasEvaluateRequest") -> dict:
    """
    Execute the full DeepEval pipeline for a single embedding model.

    Separation from the route handler lets us call this twice in parallel
    when embedding_model == "both" without recursion quirks.

    Double post-filter fix
    ----------------------
    The old code ran apply_post_filters() TWICE when req.answer was None:
      1. Inside generate_response → _fetch_chunks (with a semantic fallback
         when no chunk passed the numeric conditions).
      2. Again here on rag_result["chunks"] — which wipes the semantic
         fallback chunks, leaving contexts=[] → 422.

    Fix: in the else-branch, consume rag_result["chunks"] directly.
    _fetch_chunks already applied post_filters and fell back gracefully;
    we must NOT re-apply them here.

    The if-branch (pre-supplied answer) correctly calls apply_post_filters
    once on the raw retrieve() results (which have NOT been post-filtered).
    That path is unchanged.
    """
    parsed         = await parse_query(req.query, language=req.language)
    merged_filters = _merge_filters(None, parsed.metadata_filters)
    pool_k         = _pool_k(req.top_k)

    if req.answer:
        # Pre-supplied answer: retrieve context separately (no generate_response).
        # apply_post_filters once here — retrieve() does NOT filter numerically.
        answer        = req.answer
        retrieval     = await retrieve(parsed.semantic_query, req.embedding_model, pool_k, merged_filters)
        raw_results   = apply_post_filters(retrieval["results"], parsed.numeric_post_filters)[:req.top_k]
        contexts      = [r["text"] for r in raw_results]
        retrieval_ms  = retrieval["latency_ms"]
        generation_ms = 0
        tokens_used   = None
    else:
        # Full RAG pipeline: _fetch_chunks inside generate_response applies
        # post_filters once and falls back to semantic results when needed.
        # Do NOT call apply_post_filters again here — that would wipe the
        # semantic fallback results and cause contexts=[] → 422.
        rag_result    = await generate_response(
            parsed.semantic_query,
            req.embedding_model,
            merged_filters,
            post_filters=parsed.numeric_post_filters,
            final_k=req.top_k,
        )
        answer        = rag_result["answer"]
        # Slice to top_k only — _fetch_chunks already trims but be explicit.
        contexts      = [c["text"] for c in rag_result.get("chunks", [])[:req.top_k]]
        retrieval_ms  = rag_result["retrieval_latency_ms"]
        generation_ms = rag_result["generation_latency_ms"]
        tokens_used   = rag_result.get("tokens_used")

    if not contexts:
        raise HTTPException(status_code=422, detail="No contexts retrieved — cannot evaluate.")

    scores = await evaluate_with_ragas(
        query=req.query,
        answer=answer,
        contexts=contexts,
        ground_truth=req.ground_truth,
    )

    return {
        **scores,
        "answer_preview":        answer[:300],
        "num_contexts":          len(contexts),
        "retrieval_latency_ms":  retrieval_ms,
        "generation_latency_ms": generation_ms,
        "tokens_used":           tokens_used,
        "embedding_model":       req.embedding_model,
        "query_parsing": {
            "semantic_query":    parsed.semantic_query,
            "filters_extracted": parsed.raw_filters_extracted,
        },
    }


@router.post("/api/ragas-evaluate")
async def ragas_evaluate(req: RagasEvaluateRequest):
    """
    DeepEval single-query evaluation with query parsing.

    embedding_model handling
    ------------------------
    • "openai" / "cohere" — runs the pipeline for that model, returns a
      single result dict.
    • "both" — runs OpenAI and Cohere pipelines in parallel (same as the
      benchmark endpoint) and returns {"openai": {...}, "cohere": {...}}.
      This mirrors what the UI expects when "Both (Side-by-Side)" is selected.

    Previously "both" silently fell through to retrieve(), which used the
    else-branch and picked Cohere — giving inconsistent results and, when
    combined with the double-post-filter bug, always raising 422.
    """
    import asyncio

    if req.embedding_model == "both":
        # Build two single-model request objects and evaluate in parallel.
        # Pydantic v2: model_copy(update=...) is the canonical way to clone
        # with field overrides.  We also support the v1 dict() path as a
        # fallback so the code works across pydantic versions.
        try:
            oa_req = req.model_copy(update={"embedding_model": "openai"})
            co_req = req.model_copy(update={"embedding_model": "cohere"})
        except AttributeError:
            # Pydantic v1 fallback
            data   = req.dict()
            oa_req = RagasEvaluateRequest(**{**data, "embedding_model": "openai"})
            co_req = RagasEvaluateRequest(**{**data, "embedding_model": "cohere"})

        oa_result, co_result = await asyncio.gather(
            _run_single_evaluate(oa_req),
            _run_single_evaluate(co_req),
            return_exceptions=True,
        )

        # Surface any exceptions as error dicts rather than crashing the whole request
        def _safe(r):
            if isinstance(r, Exception):
                return {"error": str(r)}
            return r

        return {"openai": _safe(oa_result), "cohere": _safe(co_result)}

    # Single-model path
    return await _run_single_evaluate(req)


@router.post("/api/ragas-benchmark")
async def ragas_benchmark(req: RagasBenchmarkRequest):
    """DeepEval batch benchmark (API shape unchanged)."""
    if req.embedding_model == "both":
        result = await run_ragas_benchmark_compare(req.benchmark_queries, req.k)
        _attach_k(result["openai"], req.k)
        _attach_k(result["cohere"], req.k)
        return result

    result = await run_ragas_benchmark(req.benchmark_queries, req.embedding_model, req.k)
    return _attach_k(result, req.k)


@router.get("/api/collections")
async def get_collections():
    return {
        "openai": await pinecone_manager.get_collection_info(settings.pinecone_openai_index),
        "cohere": await pinecone_manager.get_collection_info(settings.pinecone_cohere_index),
    }


@router.delete("/api/collections/{model}")
async def clear_collection(model: str):
    if model not in ("openai", "cohere", "both"):
        raise HTTPException(400, "model must be 'openai', 'cohere', or 'both'")
    if model in ("openai", "both"):
        await pinecone_manager.reset_collection(settings.pinecone_openai_index, settings.openai_embedding_dim)
    if model in ("cohere", "both"):
        await pinecone_manager.reset_collection(settings.pinecone_cohere_index, settings.cohere_embedding_dim)
    return {"status": "cleared", "model": model}


@router.get("/api/health")
async def health_check():
    """
    Diagnose each API key / endpoint independently.
    Hit this first when ingestion returns 401 / 403 / connection errors.
    Returns per-service status so you know exactly which .env key is wrong.
    """
    report: Dict[str, Any] = {}

    # ── Pinecone ──────────────────────────────────────────────────────────────
    try:
        info = await pinecone_manager.get_collection_info(settings.pinecone_openai_index)
        report["pinecone"] = {"status": "ok", "openai_index_vectors": info["vectors_count"]}
    except Exception as exc:
        report["pinecone"] = {"status": "error", "detail": str(exc)}

    # ── OpenAI Embedding ──────────────────────────────────────────────────────
    try:
        from embeddings.openai_embedder import OpenAIEmbedder
        emb  = OpenAIEmbedder()
        vecs = await emb.embed(["health check"])
        report["openai_embedding"] = {
            "status": "ok",
            "model":  emb.model_name,
            "dim":    len(vecs[0]),
        }
    except Exception as exc:
        report["openai_embedding"] = {
            "status": "error",
            "detail": str(exc),
            "fix":    "Check OPENAI_API_KEY + OPENAI_ENDPOINT + OPENAI_EMBEDDING_API_VERSION in .env",
        }

    # ── Cohere Embedding ──────────────────────────────────────────────────────
    try:
        from embeddings.cohere_embedder import CohereEmbedder
        emb  = CohereEmbedder()
        vecs = await emb.embed(["health check"])
        report["cohere_embedding"] = {
            "status": "ok",
            "model":  emb.model_name,
            "dim":    len(vecs[0]),
        }
    except Exception as exc:
        report["cohere_embedding"] = {
            "status": "error",
            "detail": str(exc),
            "fix":    "Check COHERE_API_KEY + COHERE_ENDPOINT in .env",
        }

    # ── OpenAI Chat / Responses API ───────────────────────────────────────────
    try:
        from openai import AsyncAzureOpenAI
        client = AsyncAzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            api_version=settings.openai_api_version,
        )
        resp = await client.responses.create(
            model=settings.openai_chat_model,
            instructions="Reply with one word.",
            input="Say OK",
            max_output_tokens=5,
        )
        report["openai_chat"] = {"status": "ok", "model": settings.openai_chat_model, "reply": resp.output_text}
    except Exception as exc:
        report["openai_chat"] = {
            "status": "error",
            "detail": str(exc),
            "fix":    "Check OPENAI_API_KEY + OPENAI_ENDPOINT + OPENAI_API_VERSION in .env",
        }

    overall = "ok" if all(v.get("status") == "ok" for v in report.values()) else "degraded"
    return {"overall": overall, "services": report}