from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


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


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    relevant_doc_ids: List[str]
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    k: int = Field(5, ge=1, le=20)


class BenchmarkRequest(BaseModel):
    benchmark_queries: List[Dict[str, Any]]
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    k: int = Field(5, ge=1, le=20)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = []          # [{"role":"user","content":"..."}]
    embedding_model: str = Field("openai", pattern="^(openai|cohere)$")
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
