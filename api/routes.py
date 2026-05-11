import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.schemas import BenchmarkRequest, ChatRequest, CompareRequest, EvaluateRequest, QueryRequest
from chunking.semantic_chunker import chunk_documents
from config.settings import settings
from evaluation.metrics import evaluate_query, run_benchmark
from ingestion.csv_ingestor import ingest_csv
from ingestion.pdf_ingestor import ingest_pdf
from rag.generator import (
    _sse,
    generate_chat_response,
    generate_response,
    stream_generate_chat_response,
    stream_generate_response,
)
from retrieval.retriever import compare_retrievals, retrieve
from vectordb.pinecone_manager import pinecone_manager

router = APIRouter()

_SUPPORTED = {".pdf", ".csv"}

# ── SSE headers (prevents nginx/proxy buffering) ──────────────────────────────
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# ── Benchmark sample ──────────────────────────────────────────────────────────
_BENCHMARK_SAMPLE = [
    {"query": "What is the total revenue for Q2 2024?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page1_0",
         "corporate_report_en.pdf_page1_1",
         "corporate_report_en.pdf_page2_0",
     ],
     "language": "en", "category": "financial"},
    {"query": "What is the gross margin percentage?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page2_0",
         "corporate_report_en.pdf_page2_1",
         "sales_data.csv_summary_0",
     ],
     "language": "en", "category": "financial"},
    {"query": "How many active customers did the company have at quarter-end?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page2_0",
         "corporate_report_en.pdf_page2_1",
     ],
     "language": "en", "category": "metrics"},
    {"query": "What is the Annual Recurring Revenue (ARR)?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page2_0",
         "corporate_report_en.pdf_page2_1",
     ],
     "language": "en", "category": "saas_metrics"},
    {"query": "What is the revenue guidance for Q3 2024?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page2_0",
         "corporate_report_en.pdf_page2_1",
     ],
     "language": "en", "category": "guidance"},
    {"query": "Cuales son los ingresos totales del Q2 2024?",
     "relevant_doc_ids": [
         "informe_corporativo_es.pdf_page1_0",
         "informe_corporativo_es.pdf_page1_1",
         "informe_corporativo_es.pdf_page2_0",
     ],
     "language": "es", "category": "financial"},
    {"query": "Cual es el margen EBITDA?",
     "relevant_doc_ids": [
         "informe_corporativo_es.pdf_page2_0",
         "informe_corporativo_es.pdf_page2_1",
     ],
     "language": "es", "category": "financial"},
    {"query": "Cuantos clientes activos tiene la empresa?",
     "relevant_doc_ids": [
         "informe_corporativo_es.pdf_page2_0",
         "informe_corporativo_es.pdf_page2_1",
     ],
     "language": "es", "category": "metrics"},
    {"query": "What are the research and development expenses?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page2_0",
         "corporate_report_en.pdf_page2_1",
     ],
     "language": "en", "category": "costs"},
    {"query": "What is the operating cash flow?",
     "relevant_doc_ids": [
         "corporate_report_en.pdf_page2_0",
         "corporate_report_en.pdf_page2_1",
     ],
     "language": "en", "category": "cash_flow"},
    {"query": "What is the average revenue per transaction?",
     "relevant_doc_ids": [
         "sales_data.csv_summary_0",
         "sales_data.csv_rows1_rows_0",
     ],
     "language": "en", "category": "numerical"},
    {"query": "What is the highest gross profit recorded in the sales data?",
     "relevant_doc_ids": [
         "sales_data.csv_summary_0",
         "sales_data.csv_rows1_rows_0",
     ],
     "language": "en", "category": "numerical"},
]


def _get_embedder(model: str):
    if model == "openai":
        from embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder()
    from embeddings.cohere_embedder import CohereEmbedder
    return CohereEmbedder()


# ── Benchmark sample ──────────────────────────────────────────────────────────

@router.get("/api/benchmark-sample")
async def get_benchmark_sample():
    return _BENCHMARK_SAMPLE


# ── Ingestion (SSE progress stream) ──────────────────────────────────────────

@router.post("/api/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    embedding_model: str = Form("both"),
):
    """
    Returns a Server-Sent Events stream reporting ingestion progress.

    Event types:
      {"type":"progress", "percent": 0-99, "message": "..."}
      {"type":"done", "status":"success", "filename":"...", "chunks":N, "indexed":{...}, "percent":100}
      {"type":"error", "message":"..."}
    """
    # Read the file into memory before starting the generator so FastAPI
    # doesn't close the upload stream mid-generator.
    suffix = Path(file.filename).suffix.lower()
    filename = file.filename
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

            raw_docs = (
                ingest_pdf(tmp_path) if suffix == ".pdf" else ingest_csv(tmp_path)
            )
            for doc in raw_docs:
                doc["metadata"]["source"] = filename

            yield _sse({
                "type": "progress",
                "percent": 30,
                "message": f"Parsed {len(raw_docs)} page(s). Splitting into chunks...",
            })

            chunks = chunk_documents(raw_docs)
            texts = [c["text"] for c in chunks]
            results: dict = {}

            yield _sse({
                "type": "progress",
                "percent": 45,
                "message": f"Created {len(chunks)} chunks. Starting embedding...",
            })

            if embedding_model in ("openai", "both"):
                yield _sse({"type": "progress", "percent": 55, "message": "Generating OpenAI embeddings..."})
                embedder = _get_embedder("openai")
                embeddings = await embedder.embed(texts)
                yield _sse({"type": "progress", "percent": 68, "message": "Indexing OpenAI vectors in Pinecone..."})
                count = pinecone_manager.upsert_chunks(settings.pinecone_openai_index, chunks, embeddings)
                results["openai"] = count

            if embedding_model in ("cohere", "both"):
                yield _sse({"type": "progress", "percent": 78, "message": "Generating Cohere embeddings..."})
                embedder = _get_embedder("cohere")
                embeddings = await embedder.embed(texts)
                yield _sse({"type": "progress", "percent": 92, "message": "Indexing Cohere vectors in Pinecone..."})
                count = pinecone_manager.upsert_chunks(settings.pinecone_cohere_index, chunks, embeddings)
                results["cohere"] = count

            yield _sse({
                "type": "done",
                "status": "success",
                "filename": filename,
                "source_type": suffix[1:],
                "raw_documents": len(raw_docs),
                "chunks": len(chunks),
                "indexed": results,
                "percent": 100,
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


# ── Query / RAG (streaming) ───────────────────────────────────────────────────

@router.post("/api/query")
async def query_rag(req: QueryRequest):
    """
    Streams the RAG answer as SSE.

    Event sequence:
      {"type":"citations", "citations":[...], "retrieval_latency_ms":N}
      {"type":"token", "content":"..."}  × many
      {"type":"done", "generation_latency_ms":N, "tokens_used":{...}}
    """
    return StreamingResponse(
        stream_generate_response(req.query, req.embedding_model, req.top_k, req.filters),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/api/retrieve")
async def retrieve_chunks(req: QueryRequest):
    return await retrieve(
        req.query, req.embedding_model, req.top_k, req.filters, req.score_threshold
    )


@router.post("/api/compare")
async def compare_models(req: CompareRequest):
    return await compare_retrievals(req.query, req.top_k, req.filters)


# ── Chat (streaming) ──────────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Streams the chat RAG answer as SSE. Same event types as /api/query.
    """
    return StreamingResponse(
        stream_generate_chat_response(
            req.query, req.history, req.embedding_model, req.top_k, req.filters
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

@router.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    return await evaluate_query(
        req.query, req.relevant_doc_ids, req.embedding_model, req.k
    )


@router.post("/api/benchmark")
async def benchmark(req: BenchmarkRequest):
    return await run_benchmark(req.benchmark_queries, req.embedding_model, req.k)


@router.post("/api/benchmark-compare")
async def benchmark_compare(req: BenchmarkRequest):
    """Run the same benchmark on both OpenAI and Cohere and return side-by-side."""
    openai_result = await run_benchmark(req.benchmark_queries, "openai", req.k)
    cohere_result = await run_benchmark(req.benchmark_queries, "cohere", req.k)
    return {"openai": openai_result, "cohere": cohere_result}


# ── Collections ───────────────────────────────────────────────────────────────

@router.get("/api/collections")
async def get_collections():
    return {
        "openai": pinecone_manager.get_collection_info(settings.pinecone_openai_index),
        "cohere": pinecone_manager.get_collection_info(settings.pinecone_cohere_index),
    }


@router.delete("/api/collections/{model}")
async def clear_collection(model: str):
    if model not in ("openai", "cohere", "both"):
        raise HTTPException(400, "model must be 'openai', 'cohere', or 'both'")

    if model in ("openai", "both"):
        pinecone_manager.reset_collection(
            settings.pinecone_openai_index, settings.openai_embedding_dim
        )
    if model in ("cohere", "both"):
        pinecone_manager.reset_collection(
            settings.pinecone_cohere_index, settings.cohere_embedding_dim
        )

    return {"status": "cleared", "model": model}
