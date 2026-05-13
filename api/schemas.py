from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Core schemas ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    top_k: int = Field(
        5, ge=1, le=100,
        description=(
            "Max chunks sent to the LLM. The retrieval pool is always "
            "settings.pinecone_retrieval_k (default 500) so increasing "
            "top_k never misses candidates — it only widens what the LLM sees."
        ),
    )
    filters: Optional[Dict[str, Any]] = None
    score_threshold: float = Field(0.35, ge=0.0, le=1.0)
    language: Optional[str] = Field(
        None,
        pattern="^(en|es)$",
        description=(
            "Optional language filter. Pass 'en' or 'es' to restrict retrieval "
            "to documents ingested from that language. When omitted, all languages "
            "are searched."
        ),
    )

    def effective_filters(self) -> Optional[Dict[str, Any]]:
        base = dict(self.filters) if self.filters else {}
        if self.language:
            base["language"] = self.language
        return base or None


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    language: Optional[str] = Field(
        None,
        pattern="^(en|es)$",
        description="Optional language filter ('en' or 'es'). Merged into filters.",
    )

    def effective_filters(self) -> Optional[Dict[str, Any]]:
        base = dict(self.filters) if self.filters else {}
        if self.language:
            base["language"] = self.language
        return base or None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = []
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    top_k: int = Field(5, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    language: Optional[str] = Field(
        None,
        pattern="^(en|es)$",
        description="Optional language filter ('en' or 'es'). Merged into filters.",
    )

    def effective_filters(self) -> Optional[Dict[str, Any]]:
        base = dict(self.filters) if self.filters else {}
        if self.language:
            base["language"] = self.language
        return base or None


# ── Evaluation schemas ────────────────────────────────────────────────────────

class RagasEvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    embedding_model: str = Field("openai", pattern="^(openai|cohere|both)$")
    top_k: int = Field(5, ge=1, le=100)
    language: Optional[str] = Field(None, pattern="^(en|es)$")
    answer: Optional[str] = Field(
        None,
        description="Pre-generated answer. If omitted the RAG pipeline generates one.",
    )
    ground_truth: Optional[str] = Field(
        None,
        description="Reference answer text. Enables contextrecall scoring.",
    )


class RagasBenchmarkRequest(BaseModel):
    benchmark_queries: List[Dict[str, Any]]
    embedding_model: str = Field("openai", pattern="^(openai|cohere|both)$")
    k: int = Field(5, ge=1, le=100)
    language: Optional[str] = Field(
        None,
        pattern="^(en|es)$",
        description=(
            "Global language filter applied to every query. "
            "Per-query language entries in benchmark_queries take precedence."
        ),
    )