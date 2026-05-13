"""
retrieval/retriever.py

Core retrieval with:
  - Proper Pinecone candidate pool sizing
  - Score threshold disabled when metadata filters are active
    (CSV row chunks have low cosine scores 0.18-0.28; threshold would drop them)
  - Fallback to semantic-only search if filtered search returns nothing
"""

import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config.settings import settings
from vectordb.pinecone_manager import pinecone_manager


@lru_cache(maxsize=1)
def _get_openai_embedder():
    from embeddings.openai_embedder import OpenAIEmbedder
    return OpenAIEmbedder()


@lru_cache(maxsize=1)
def _get_cohere_embedder():
    from embeddings.cohere_embedder import CohereEmbedder
    return CohereEmbedder()


def _pool_k(final_k: int) -> int:
    """
    Compute the Pinecone candidate pool size from the desired final_k.

    For CSV datasets with metadata filters we need a large enough pool so
    that all matching rows are candidates before post-filtering trims to final_k.

    Scale: final_k × 100, clamped to [50, 10000].
      K=5   → pool=500    (covers datasets up to ~500 matching rows)
      K=10  → pool=1000
      K=100 → pool=10000  (Pinecone hard limit)

    This is intentionally generous: Pinecone's metadata filter is the real
    precision gate, and the pool is fetched in one round-trip so there's no
    latency cost from a larger pool vs a smaller one.
    """
    return max(50, min(final_k * 100, 10_000))


async def retrieve(
    query: str,
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Core retrieval.

    score_threshold behaviour:
      When metadata filters are supplied, score_threshold is forced to 0.0.
      CSV row chunks have low cosine scores (0.18-0.28) that fall below the
      default threshold even though they are exactly the records we want.
      Pinecone's metadata filter is the precision gate — the score threshold
      would silently drop matching records and trigger the fallback.

    fallback behaviour:
      If the filtered search returns 0 results we retry without filters.
      filter_fallback=True is set so callers can surface a caveat to the user.
    """
    top_k = top_k or settings.top_k

    if embedding_model not in ("openai", "cohere"):
        raise ValueError(
            f"retrieve() received embedding_model={embedding_model!r}. "
            "Valid values are 'openai' or 'cohere'. "
            "For dual-model retrieval call compare_retrievals() instead."
        )

    t0 = time.perf_counter()

    if embedding_model == "openai":
        embedder   = _get_openai_embedder()
        index_name = settings.pinecone_openai_index
        vectors    = await embedder.embed([query])
    else:
        embedder   = _get_cohere_embedder()
        index_name = settings.pinecone_cohere_index
        vectors    = await embedder.embed([query], input_type="search_query")

    query_vector = vectors[0]

    # When filters are active: disable score threshold entirely.
    # When no filters: apply caller-supplied threshold (default from settings).
    primary_threshold = 0.0 if filters else (
        score_threshold if score_threshold != 0.0 else settings.similarity_threshold
    )

    fallback_used = False
    results = pinecone_manager.search(
        index_name=index_name,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
        score_threshold=primary_threshold,
    )

    # Fallback: filtered search returned nothing → retry semantic-only
    if not results and filters:
        results = pinecone_manager.search(
            index_name=index_name,
            query_vector=query_vector,
            top_k=top_k,
            filters=None,
            score_threshold=0.0,
        )
        fallback_used = True

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "query":           query,
        "embedding_model": embedding_model,
        "results":         results,
        "latency_ms":      latency_ms,
        "top_k":           top_k,
        "total_found":     len(results),
        "filter_fallback": fallback_used,
    }


async def compare_retrievals(
    query: str,
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    openai_r = await retrieve(query, "openai", top_k, filters)
    cohere_r = await retrieve(query, "cohere", top_k, filters)

    texts_oa = {r["text"][:150] for r in openai_r["results"]}
    texts_co = {r["text"][:150] for r in cohere_r["results"]}
    union    = texts_oa | texts_co
    overlap  = round(len(texts_oa & texts_co) / len(union), 4) if union else 0.0

    return {
        "query":  query,
        "openai": openai_r,
        "cohere": cohere_r,
        "comparison": {
            "openai_latency_ms": openai_r["latency_ms"],
            "cohere_latency_ms": cohere_r["latency_ms"],
            "openai_top_score":  openai_r["results"][0]["score"] if openai_r["results"] else 0,
            "cohere_top_score":  cohere_r["results"][0]["score"] if cohere_r["results"] else 0,
            "result_overlap":    overlap,
            "openai_total":      openai_r["total_found"],
            "cohere_total":      cohere_r["total_found"],
        },
    }