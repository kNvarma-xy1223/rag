from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Core schemas ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    score_threshold: float = Field(0.0, ge=0.0, le=1.0)


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = []      # [{"role":"user","content":"..."}]
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None


# ── RAGAS schemas ─────────────────────────────────────────────────────────────

class RagasEvaluateRequest(BaseModel):
    """
    Single-query RAGAS evaluation.

    Provide `answer` if you already have one; otherwise the system will
    run the full RAG pipeline to generate it automatically.

    `ground_truth` is an optional reference answer (plain text, not doc IDs).
    When supplied, context_recall and context_precision are also computed.

    Set embedding_model="both" to run evaluation with both OpenAI and Cohere
    embeddings side-by-side and compare results.
    """
    query: str = Field(..., min_length=1)
    embedding_model: str = Field("openai", pattern="^(openai|cohere|both)$")
    top_k: int = Field(5, ge=1, le=20)
    answer: Optional[str] = Field(
        None,
        description="Pre-generated answer. If omitted the RAG pipeline generates one.",
    )
    ground_truth: Optional[str] = Field(
        None,
        description="Reference answer text. Enables context_recall + context_precision.",
    )


class RagasBenchmarkRequest(BaseModel):
    """
    Batch RAGAS benchmark.

    Each entry in benchmark_queries should contain:
      - query        : str   (required)
      - ground_truth : str   (optional)
      - language     : str   (optional, for grouping)
      - category     : str   (optional, for grouping)

    The system runs the full RAG pipeline for every query then scores with RAGAS.
    Set embedding_model="both" to run OpenAI and Cohere side-by-side.
    """
    benchmark_queries: List[Dict[str, Any]]
    embedding_model: str = Field("openai", pattern="^(openai|cohere|both)$")
    k: int = Field(5, ge=1, le=20)