"""
rag/query_parser.py  —  Natural-language → structured query

Pipeline:
  raw query  →  Azure OpenAI (gpt-5.4-pro)  →  ParsedQuery
                                                    ├── semantic_query        (clean text for embedding)
                                                    ├── metadata_filters      (Pinecone $eq / $gte / $lte)
                                                    └── numeric_post_filters  (safety-net after retrieval)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NumericCondition:
    """A single numeric comparison: field OP value  (e.g. performance_score >= 85)."""
    field: str   # safe_meta_key matching Pinecone metadata, e.g. "performance_score"
    op: str      # "gt" | "gte" | "lt" | "lte" | "eq"
    value: float


@dataclass
class ParsedQuery:
    semantic_query: str
    metadata_filters: Optional[Dict[str, Any]] = None
    numeric_post_filters: List[NumericCondition] = field(default_factory=list)
    raw_filters_extracted: Dict[str, Any] = field(default_factory=dict)


# ── Pinecone filter builder ───────────────────────────────────────────────────

_OP_MAP = {"gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte", "eq": "$eq"}


def _build_pinecone_filter(
    eq_filters: Dict[str, str],
    numeric_conditions: List[NumericCondition],
    language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Combine categorical equality + numeric range + optional language into one
    Pinecone filter dict.

    Example:
      {"$and": [
          {"role":              {"$eq": "Sales Manager"}},
          {"region":            {"$eq": "North"}},
          {"performance_score": {"$gte": 85.0}},
      ]}
    """
    clauses: List[Dict[str, Any]] = []

    for meta_key, val in eq_filters.items():
        clauses.append({meta_key: {"$eq": val}})

    for cond in numeric_conditions:
        clauses.append({cond.field: {_OP_MAP[cond.op]: cond.value}})

    if language:
        clauses.append({"language": {"$eq": language}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ── LLM system prompt ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a query-parsing assistant for an enterprise analytics RAG system.

Given a natural-language question, extract structured filter information.

Return ONLY valid JSON (no markdown fences, no explanation):
{
  "semantic_query": "<clean question stripped of all filter constraints, for embedding search>",
  "eq_filters": { "<key>": "<value>" },
  "numeric_conditions": [
    {"field": "<safe_field_name>", "op": "<gt|gte|lt|lte|eq>", "value": <number>}
  ],
  "top_k_hint": <integer or null>
}

RULES:

1. semantic_query — remove all filter constraints, keep the core topic.
   "Sales Managers in North region with performance score above 85"
   → "employees with high performance scores"

2. eq_filters — use EXACTLY these key names (they match Pinecone metadata):
   role, region, department, status, category, language, doc_type, source_type
   Normalise values to Title Case.
   Examples:
     "sales managers"   → {"role": "Sales Manager"}
     "north region"     → {"region": "North"}
     "field managers"   → {"role": "Field Manager"}
     "HR department"    → {"department": "Hr"}

3. numeric_conditions — field names use safe_meta_key format (snake_case, lowercase).
   Common fields: performance_score, salary, revenue_usd, field_visits,
                  customer_satisfaction, target_achievement_pct, operational_cost_usd,
                  score, rating, quantity, price, amount, percentage, count
   For any numeric field mentioned, infer the most likely safe key name.
   Operator mapping:
     "above N" / "greater than N" / "more than N" / "over N"  → gte N
     "below N" / "less than N" / "under N"                    → lt  N
     "at least N" / "minimum N" / "≥ N"                       → gte N
     "at most N" / "maximum N" / "≤ N"                        → lte N
     "exactly N" / "equal to N" / "= N"                       → eq  N
     "between N and M" → TWO conditions: gte N AND lte M

4. top_k_hint — integer if user specified a count ("top 3", "first 5", "show 10"), else null.

EXAMPLES:
  Input:  "Show me Sales Managers in the North region with performance score above 85"
  Output: {"semantic_query": "employees with high performance scores", "eq_filters": {"role": "Sales Manager", "region": "North"}, "numeric_conditions": [{"field": "performance_score", "op": "gte", "value": 85.0}], "top_k_hint": null}

  Input:  "Which Field Managers have customer satisfaction below 3?"
  Output: {"semantic_query": "employees with low customer satisfaction", "eq_filters": {"role": "Field Manager"}, "numeric_conditions": [{"field": "customer_satisfaction", "op": "lt", "value": 3.0}], "top_k_hint": null}

  Input:  "Top 5 employees in South region by revenue"
  Output: {"semantic_query": "employees with high revenue", "eq_filters": {"region": "South"}, "numeric_conditions": [], "top_k_hint": 5}

  Input:  "What was Q2 revenue growth?"
  Output: {"semantic_query": "Q2 revenue growth", "eq_filters": {}, "numeric_conditions": [], "top_k_hint": null}
"""


# ── Parser ────────────────────────────────────────────────────────────────────

async def parse_query(
    raw_query: str,
    language: Optional[str] = None,
) -> ParsedQuery:
    """
    Call Azure OpenAI (gpt-5.4-pro) to extract structured filters.

    Falls back to passthrough ParsedQuery on any failure so the pipeline
    never crashes due to parsing errors.
    """
    try:
        from config.settings import settings
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            api_version=getattr(settings, "openai_api_version", "2025-04-01-preview"),
        )
        model = getattr(settings, "openai_chat_model", "gpt-5.4-pro")
    except Exception as exc:
        logger.warning("query_parser: could not initialise client (%s) — passthrough", exc)
        return ParsedQuery(semantic_query=raw_query)

    try:
        response = await client.responses.create(
            model=model,
            instructions=_SYSTEM_PROMPT,
            input=raw_query,
            max_output_tokens=512,
        )
        raw_text = response.output_text or ""
    except Exception as exc:
        logger.warning("query_parser: LLM call failed (%s) — passthrough", exc)
        return ParsedQuery(semantic_query=raw_query)

    try:
        clean = re.sub(r"^```[a-z]*\n?", "", raw_text.strip())
        clean = re.sub(r"\n?```$", "", clean)
        parsed_json = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("query_parser: JSON parse failed (%s) raw=%r — passthrough", exc, raw_text[:200])
        return ParsedQuery(semantic_query=raw_query)

    semantic_query: str = parsed_json.get("semantic_query") or raw_query
    eq_filters: Dict[str, str] = parsed_json.get("eq_filters") or {}
    raw_numeric: List[Dict] = parsed_json.get("numeric_conditions") or []

    numeric_conditions: List[NumericCondition] = []
    for nc in raw_numeric:
        try:
            numeric_conditions.append(NumericCondition(
                field=str(nc["field"]),
                op=str(nc["op"]),
                value=float(nc["value"]),
            ))
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("query_parser: skipping bad numeric condition %r (%s)", nc, exc)

    pinecone_filter = _build_pinecone_filter(eq_filters, numeric_conditions, language)

    logger.info(
        "query_parser: '%s' → semantic='%s' eq=%s numeric=%s filter=%s",
        raw_query, semantic_query, eq_filters,
        [(c.field, c.op, c.value) for c in numeric_conditions],
        pinecone_filter,
    )

    return ParsedQuery(
        semantic_query=semantic_query,
        metadata_filters=pinecone_filter,
        numeric_post_filters=numeric_conditions,
        raw_filters_extracted={
            "eq_filters":         eq_filters,
            "numeric_conditions": [vars(c) for c in numeric_conditions],
            "top_k_hint":         parsed_json.get("top_k_hint"),
        },
    )


# ── Post-retrieval numeric filter (safety net) ────────────────────────────────

def apply_post_filters(
    chunks: List[Dict[str, Any]],
    conditions: List[NumericCondition],
) -> List[Dict[str, Any]]:
    """
    Applied AFTER Pinecone retrieval as a safety net.

    Handles cases where the Pinecone metadata filter was not precise enough
    or the field value was stored in an unexpected format. Silently skips
    chunks that are missing the filtered field (non-row documents like summaries).
    """
    if not conditions:
        return chunks

    def _passes(chunk: Dict[str, Any]) -> bool:
        meta = chunk.get("metadata", {})
        for cond in conditions:
            raw = meta.get(cond.field)
            if raw is None:
                # Field absent (e.g. summary doc) — let it through; LLM will handle
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if cond.op == "gt"  and not (val >  cond.value): return False
            if cond.op == "gte" and not (val >= cond.value): return False
            if cond.op == "lt"  and not (val <  cond.value): return False
            if cond.op == "lte" and not (val <= cond.value): return False
            if cond.op == "eq"  and not (val == cond.value): return False
        return True

    return [c for c in chunks if _passes(c)]