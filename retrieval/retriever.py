import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from config.settings import settings
from vectordb.pinecone_manager import pinecone_manager   # ← was qdrant_manager


@lru_cache(maxsize=1)
def _get_openai_embedder():
    from embeddings.openai_embedder import OpenAIEmbedder
    return OpenAIEmbedder()


@lru_cache(maxsize=1)
def _get_cohere_embedder():
    from embeddings.cohere_embedder import CohereEmbedder
    return CohereEmbedder()


async def retrieve(
    query: str,
    embedding_model: str = "openai",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    top_k = top_k or settings.top_k
    t0 = time.perf_counter()

    if embedding_model == "openai":
        embedder = _get_openai_embedder()
        index_name = settings.pinecone_openai_index     # ← was qdrant_openai_collection
        vectors = await embedder.embed([query])
    else:
        embedder = _get_cohere_embedder()
        index_name = settings.pinecone_cohere_index     # ← was qdrant_cohere_collection
        vectors = await embedder.embed([query], input_type="search_query")

    query_vector = vectors[0]

    results = pinecone_manager.search(
        index_name=index_name,
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
        score_threshold=score_threshold,
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "query": query,
        "embedding_model": embedding_model,
        "results": results,
        "latency_ms": latency_ms,
        "top_k": top_k,
        "total_found": len(results),
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
    union = texts_oa | texts_co
    overlap = round(len(texts_oa & texts_co) / len(union), 4) if union else 0.0

    return {
        "query": query,
        "openai": openai_r,
        "cohere": cohere_r,
        "comparison": {
            "openai_latency_ms": openai_r["latency_ms"],
            "cohere_latency_ms": cohere_r["latency_ms"],
            "openai_top_score": openai_r["results"][0]["score"] if openai_r["results"] else 0,
            "cohere_top_score": cohere_r["results"][0]["score"] if cohere_r["results"] else 0,
            "result_overlap": overlap,
            "openai_total": openai_r["total_found"],
            "cohere_total": cohere_r["total_found"],
        },
    }
