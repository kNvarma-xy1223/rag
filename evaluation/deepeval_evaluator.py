from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from config.settings import settings

# ── DeepEval imports (validated against deepeval 4.0.0) ───────────────────────
from deepeval.models import AzureOpenAIModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase


# ── Azure judge model factory ─────────────────────────────────────────────────
# Fresh instance per call so metrics never share state across concurrent evals.
def _azure_judge() -> AzureOpenAIModel:
    """Return a DeepEval AzureOpenAIModel pointed at the judge deployment."""
    return AzureOpenAIModel(
        model=settings.deepeval_judge_model,
        deployment_name=settings.deepeval_judge_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_endpoint,
        api_version=settings.deepeval_judge_api_version,
    )


# ── Metric factories (fresh instance per call avoids state leakage) ───────────
def _faith() -> FaithfulnessMetric:
    return FaithfulnessMetric(model=_azure_judge(), verbose_mode=False)


def _arelevancy() -> AnswerRelevancyMetric:
    return AnswerRelevancyMetric(model=_azure_judge(), verbose_mode=False)


def _cprecision() -> ContextualPrecisionMetric:
    return ContextualPrecisionMetric(model=_azure_judge(), verbose_mode=False)


def _crecall() -> ContextualRecallMetric:
    return ContextualRecallMetric(model=_azure_judge(), verbose_mode=False)


def _halluc() -> HallucinationMetric:
    return HallucinationMetric(model=_azure_judge(), verbose_mode=False)


# ── Async executor wrapper ────────────────────────────────────────────────────
async def _run_metric(metric, test_case: LLMTestCase) -> float:
    """Run a synchronous DeepEval metric in a thread, return normalised score."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, metric.measure, test_case)
    score = getattr(metric, "score", 0.0) or 0.0
    return round(float(score), 4)


# ── Main evaluation entry point ───────────────────────────────────────────────
async def evaluate_with_deepeval(
    query: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run DeepEval metrics in parallel.

    Always computed      : faithfulness, answerrelevancy, hallucination
    Requires ground_truth: contextprecision, contextrecall
    """
    t0 = time.time()

    # DeepEval requires non-empty actual_output — raise early with a clear message.
    if not answer or not answer.strip():
        raise ValueError(
            "actual_output is empty. The RAG pipeline returned no answer; "
            "check that documents are ingested and the generation model is reachable."
        )

    # LLMTestCase shared by all metrics
    tc_base = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=contexts[:5],
        context=contexts[:5],           # HallucinationMetric uses `context`
        expected_output=ground_truth or "",
    )

    # Core 3 always run in parallel
    faithfulness, answerrelevancy, hallucination = await asyncio.gather(
        _run_metric(_faith(),      tc_base),
        _run_metric(_arelevancy(), tc_base),
        _run_metric(_halluc(),     tc_base),
    )

    contextprecision: Optional[float] = None
    contextrecall:    Optional[float] = None

    if ground_truth:
        contextprecision, contextrecall = await asyncio.gather(
            _run_metric(_cprecision(), tc_base),
            _run_metric(_crecall(),    tc_base),
        )

    return {
        "faithfulness":     faithfulness,
        "answerrelevancy":  answerrelevancy,
        "contextprecision": contextprecision,
        "contextrecall":    contextrecall,
        "hallucination":    hallucination,
        "eval_latency_ms":  round((time.time() - t0) * 1000),
        "judge_model":      settings.deepeval_judge_model,
        "framework":        "deepeval",
    }


# ── Benchmark ─────────────────────────────────────────────────────────────────
async def run_deepeval_benchmark(
    queries: List[Dict[str, Any]],
    embedding_model: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    from rag.generator import generate_response

    results: List[Dict[str, Any]] = []

    for item in queries:
        q  = item.get("query", "")
        gt = item.get("ground_truth")
        try:
            # FIX: use keyword arguments to match the new generate_response
            # signature. The old call `generate_response(q, embedding_model, top_k)`
            # passed top_k as the third positional arg, which in the new signature
            # is `filters` — Pinecone would receive an integer as a filter and crash.
            rag      = await generate_response(
                q,
                embedding_model,
                final_k=top_k,
            )
            answer   = rag["answer"]
            contexts = [c["text"] for c in rag.get("chunks", [])]
            sc       = await evaluate_with_deepeval(q, answer, contexts, gt)
            results.append({
                "query":          q,
                "ground_truth":   gt,
                "answer_preview": answer[:200],
                **sc,
            })
        except Exception as exc:
            results.append({"query": q, "error": str(exc)})

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "results": results,
        "summary": {
            "total":                 len(results),
            "faithfulness_avg":      _avg("faithfulness"),
            "answer_relevancy_avg":  _avg("answerrelevancy"),
            "context_precision_avg": _avg("contextprecision"),
            "context_recall_avg":    _avg("contextrecall"),
            "hallucination_avg":     _avg("hallucination"),
        },
        "embedding_model": embedding_model,
        "framework":       "deepeval",
    }


async def run_deepeval_benchmark_compare(
    queries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    oa, co = await asyncio.gather(
        run_deepeval_benchmark(queries, "openai", top_k),
        run_deepeval_benchmark(queries, "cohere", top_k),
    )
    return {"openai": oa, "cohere": co}


# ── Legacy aliases — routes.py imports these ──────────────────────────────────
evaluate_with_ragas         = evaluate_with_deepeval
run_ragas_benchmark         = run_deepeval_benchmark
run_ragas_benchmark_compare = run_deepeval_benchmark_compare