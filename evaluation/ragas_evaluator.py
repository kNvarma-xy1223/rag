"""
evaluation/ragas_evaluator.py
─────────────────────────────
End-to-end RAGAS evaluation for the multilingual RAG system.

Metrics computed
────────────────
Always (no ground-truth required):
  • faithfulness          – does the answer stay faithful to the retrieved contexts?
  • answer_relevancy      – is the answer relevant to the question asked?

When `ground_truth` answer text is supplied:
  • context_recall        – do the retrieved contexts cover the ground-truth answer?
  • context_precision     – are retrieved contexts precise relative to the ground truth?

RAGAS requires an LLM (judge) and an embedding model.
Both are wired to the same Azure OpenAI deployment used by the rest of the system
via LangChain wrappers, so no extra API keys are needed.

Dependencies (add to requirements.txt):
  ragas>=0.2.0
  langchain-openai>=0.1.0
  langchain>=0.2.0
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from config.settings import settings


# ── LLM / Embeddings wrappers ─────────────────────────────────────────────────

def _build_ragas_llm():
    """Wrap Azure OpenAI chat model for RAGAS judge calls."""
    from langchain_openai import AzureChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    lc_llm = AzureChatOpenAI(
        azure_endpoint=settings.openai_endpoint,
        api_key=settings.openai_api_key,
        api_version=settings.ragas_judge_api_version,
        azure_deployment=settings.resolved_ragas_judge_model,
        temperature=0,
        max_tokens=settings.ragas_max_tokens,
    )
    return LangchainLLMWrapper(lc_llm)


def _build_ragas_embeddings():
    """Wrap Azure OpenAI embeddings for RAGAS AnswerRelevancy metric."""
    from langchain_openai import AzureOpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    lc_emb = AzureOpenAIEmbeddings(
        azure_endpoint=settings.openai_endpoint,
        api_key=settings.openai_api_key,
        openai_api_version=settings.openai_embedding_api_version,
        azure_deployment=settings.openai_embedding_model,
    )
    return LangchainEmbeddingsWrapper(lc_emb)


# ── Core single-sample evaluation ─────────────────────────────────────────────

async def evaluate_with_ragas(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run RAGAS on a single (query, answer, contexts[, ground_truth]) sample.

    Returns a flat dict of metric scores + eval_latency_ms.
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerRelevancy, Faithfulness

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    # Base metrics — no ground truth needed
    metrics: list = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ]

    # Reference-based metrics — only when a ground-truth answer is provided
    if ground_truth:
        from ragas.metrics import (
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )
        metrics += [
            LLMContextRecall(llm=ragas_llm),
            LLMContextPrecisionWithReference(llm=ragas_llm),
        ]

    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth or "",
    )

    dataset = EvaluationDataset(samples=[sample])

    t0 = time.perf_counter()
    # RAGAS evaluate() is synchronous; run in thread pool to keep FastAPI async
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: evaluate(dataset=dataset, metrics=metrics),
    )
    eval_ms = round((time.perf_counter() - t0) * 1000, 2)

    scores: Dict[str, Any] = result.to_pandas().iloc[0].to_dict()

    output: Dict[str, Any] = {
        "query": query,
        "faithfulness": _safe_float(scores.get("faithfulness")),
        "answer_relevancy": _safe_float(scores.get("answer_relevancy")),
        "eval_latency_ms": eval_ms,
    }
    if ground_truth:
        output["context_recall"] = _safe_float(scores.get("context_recall"))
        output["context_precision"] = _safe_float(scores.get("context_precision"))

    return output


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a potentially-None or NaN RAGAS score to a safe float."""
    try:
        f = float(val)
        return round(f if f == f else default, 4)   # NaN guard
    except (TypeError, ValueError):
        return default


# ── Benchmark runner ──────────────────────────────────────────────────────────

async def run_ragas_benchmark(
    benchmark_queries: List[Dict[str, Any]],
    embedding_model: str = "openai",
    k: int = 5,
) -> Dict[str, Any]:
    """
    Full RAG pipeline → RAGAS evaluation for every benchmark query.

    Each entry in benchmark_queries should have:
      - query        : str   (required)
      - ground_truth : str   (optional — enables context_recall / context_precision)

    Note: relevant_doc_ids from the existing benchmark format are ignored here;
    RAGAS works with answer text, not document IDs.
    """
    from rag.generator import generate_response  # local import avoids circular refs

    if not benchmark_queries:
        return {"error": "No benchmark queries provided"}

    per_query: List[Dict[str, Any]] = []

    for bq in benchmark_queries:
        query: str = bq.get("query", "")
        ground_truth: Optional[str] = bq.get("ground_truth")  # may be absent

        try:
            # ── Step 1: full RAG generation ───────────────────────────────────
            rag_result = await generate_response(query, embedding_model, k)
            answer: str = rag_result["answer"]
            contexts: List[str] = [c["text"] for c in rag_result.get("chunks", [])]

            if not contexts:
                raise ValueError("No contexts retrieved — cannot run RAGAS evaluation.")

            # ── Step 2: RAGAS evaluation ──────────────────────────────────────
            ragas_scores = await evaluate_with_ragas(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

            per_query.append(
                {
                    **ragas_scores,
                    "answer_preview": answer[:200],
                    "num_contexts": len(contexts),
                    "retrieval_latency_ms": rag_result["retrieval_latency_ms"],
                    "generation_latency_ms": rag_result["generation_latency_ms"],
                    "embedding_model": embedding_model,
                    "language": bq.get("language", "en"),
                    "category": bq.get("category", ""),
                }
            )

        except Exception as exc:
            per_query.append(
                {
                    "query": query,
                    "error": str(exc),
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_recall": 0.0 if ground_truth else None,
                    "context_precision": 0.0 if ground_truth else None,
                    "embedding_model": embedding_model,
                }
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    valid = [r for r in per_query if "error" not in r]
    n = len(valid) or 1

    agg: Dict[str, Any] = {
        "avg_faithfulness": round(sum(r["faithfulness"] for r in valid) / n, 4),
        "avg_answer_relevancy": round(sum(r["answer_relevancy"] for r in valid) / n, 4),
    }

    # Context metrics only when at least one query had ground_truth
    cr_vals = [r["context_recall"] for r in valid if r.get("context_recall") is not None]
    cp_vals = [r["context_precision"] for r in valid if r.get("context_precision") is not None]
    if cr_vals:
        agg["avg_context_recall"] = round(sum(cr_vals) / len(cr_vals), 4)
        agg["avg_context_precision"] = round(sum(cp_vals) / len(cp_vals), 4)

    return {
        "framework": "ragas",
        "embedding_model": embedding_model,
        "k": k,
        "num_queries": len(benchmark_queries),
        "valid_evaluations": len(valid),
        "failed_evaluations": len(benchmark_queries) - len(valid),
        "aggregate": agg,
        "per_query": per_query,
    }


# ── Dual-model comparison ─────────────────────────────────────────────────────

async def run_ragas_benchmark_compare(
    benchmark_queries: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """Run run_ragas_benchmark on both OpenAI and Cohere and return side-by-side."""
    openai_result, cohere_result = await asyncio.gather(
        run_ragas_benchmark(benchmark_queries, "openai", k),
        run_ragas_benchmark(benchmark_queries, "cohere", k),
    )
    return {"openai": openai_result, "cohere": cohere_result}
