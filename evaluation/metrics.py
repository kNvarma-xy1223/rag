import math
import time
from typing import Any, Dict, List
from retrieval.retriever import retrieve


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return min(hits / len(relevant), 1.0)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    rel_set = set(relevant)
    seen = set()
    for i, r in enumerate(retrieved):
        if r in rel_set and r not in seen:
            return 1.0 / (i + 1)
        seen.add(r)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    rel_set = set(relevant)

    def dcg(items: List[str], top_k: int) -> float:
        seen = set()
        score = 0.0
        rank = 1
        for item in items[:top_k]:
            if item in rel_set and item not in seen:
                score += 1.0 / math.log2(rank + 1)
                seen.add(item)
            rank += 1
        return score

    ideal = dcg(list(rel_set), k)
    if ideal == 0:
        return 0.0
    return min(dcg(retrieved, k) / ideal, 1.0)


def _doc_id(result: Dict[str, Any]) -> str:
    meta = result.get("metadata", {})
    source = meta.get("source", "unknown")
    # Build a stable, unique ID using all available location fields
    parts = [source]
    if "page" in meta:
        parts.append(f"page{meta['page']}")
    if "row_start" in meta:
        parts.append(f"rows{meta['row_start']}")
    if "doc_type" in meta:
        parts.append(meta["doc_type"])
    parts.append(str(meta.get("chunk_index", 0)))
    return "_".join(parts)


async def evaluate_query(
    query: str,
    relevant_doc_ids: List[str],
    embedding_model: str = "openai",
    k: int = 5,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = await retrieve(query, embedding_model, k)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    retrieved_ids = [_doc_id(r) for r in result["results"]]

    return {
        "query": query,
        "embedding_model": embedding_model,
        "k": k,
        "precision_at_k": round(precision_at_k(retrieved_ids, relevant_doc_ids, k), 4),
        "recall_at_k": round(recall_at_k(retrieved_ids, relevant_doc_ids, k), 4),
        "mrr": round(mrr(retrieved_ids, relevant_doc_ids), 4),
        "ndcg_at_k": round(ndcg_at_k(retrieved_ids, relevant_doc_ids, k), 4),
        "latency_ms": latency_ms,
        "retrieved_ids": retrieved_ids,
    }


async def run_benchmark(
    benchmark_queries: List[Dict[str, Any]],
    embedding_model: str = "openai",
    k: int = 5,
) -> Dict[str, Any]:
    if not benchmark_queries:
        return {"error": "No benchmark queries provided"}

    per_query = []
    for bq in benchmark_queries:
        r = await evaluate_query(
            bq["query"],
            bq.get("relevant_doc_ids", []),
            embedding_model,
            k,
        )
        per_query.append(r)

    n = len(per_query)
    sorted_latencies = sorted(r["latency_ms"] for r in per_query)
    p95_idx = max(0, int(n * 0.95) - 1)

    agg = {
        "avg_precision_at_k": round(sum(r["precision_at_k"] for r in per_query) / n, 4),
        "avg_recall_at_k": round(sum(r["recall_at_k"] for r in per_query) / n, 4),
        "avg_mrr": round(sum(r["mrr"] for r in per_query) / n, 4),
        "avg_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in per_query) / n, 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in per_query) / n, 2),
        "p95_latency_ms": round(sorted_latencies[p95_idx], 2),
    }

    return {
        "embedding_model": embedding_model,
        "k": k,
        "num_queries": n,
        "aggregate": agg,
        "per_query": per_query,
    }
