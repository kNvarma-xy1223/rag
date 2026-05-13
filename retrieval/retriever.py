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

    Scale: final_k × 10, clamped to [20, 500].
      K=5  → pool=50    (enough headroom for metadata + score filtering)
      K=10 → pool=100
      K=50 → pool=500   (Pinecone soft limit; hard limit is 10 000)

    Floor of 20 ensures single-result queries still get real candidates.
    """
    return max(20, min(final_k * 10, 500))


async def retrieve(
    query: str,
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Core retrieval function.

    score_threshold behaviour (critical fix):
      • When metadata filters are supplied, score_threshold is forced to 0.0
        for the filtered search.  CSV row chunks have low semantic similarity
        scores (0.18–0.28) that fall below the default 0.3 threshold even
        though they are the EXACT records the filter is looking for.
        Pinecone's metadata filter is already the precision gate — adding a
        score threshold on top silently drops matching records and triggers
        the fallback, returning completely unrelated chunks to the LLM.
      • When no filters are supplied (pure semantic search), the caller-
        supplied score_threshold (or settings.similarity_threshold) applies
        normally.

    fallback behaviour:
      If the filtered search returns 0 results we retry without filters so
      the LLM always has context to work with.  filter_fallback=True is set
      so callers (routes, generator) can surface this information.
    """
    top_k = top_k or settings.top_k

    # Guard: "both" is only valid for compare_retrievals().
    # If it reaches here the caller has a bug — raise immediately so the stack
    # trace points at the real site rather than silently falling through to
    # Cohere and returning confusing results.
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

    # ── Primary search ────────────────────────────────────────────────────────
    # When filters are active, disable the score threshold: Pinecone's metadata
    # filter is the precision gate. Applying score_threshold here would silently
    # drop CSV row records whose cosine scores (0.18–0.28) fall below the
    # default 0.3 threshold, causing the fallback to fire and returning wrong
    # chunks to the LLM.
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

    # ── Fallback: filtered search returned nothing ────────────────────────────
    # Retry semantic-only so the LLM always has context.
    # score_threshold=0.0 intentionally: don't let the threshold swallow the
    # fallback results; the LLM will caveat based on filter_fallback=True.
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
        "query":          query,
        "embedding_model": embedding_model,
        "results":        results,
        "latency_ms":     latency_ms,
        "top_k":          top_k,
        "total_found":    len(results),
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