"""
rag/query_parser.py  —  Natural-language → structured query

Pipeline:
  raw query  →  Azure OpenAI (gpt-5.4-pro)  →  ParsedQuery
                                                    ├── semantic_query        (clean text for embedding)
                                                    ├── metadata_filters      (Pinecone $eq / $gte / $lte)
                                                    └── numeric_post_filters  (safety-net after retrieval)

Place this file at:
  your_project/
  └── rag/
      └── query_parser.py      ← HERE

It uses the same settings keys already in your project:
  settings.openai_api_key
  settings.openai_endpoint
  settings.openai_chat_model          (should be your gpt-5.4-pro deployment name)
  settings.openai_api_version         (should be "2025-04-01-preview" for gpt-5.4-pro)
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
    field: str      # safe_meta_key name stored in Pinecone, e.g. "performance_score"
    op: str         # "gt" | "gte" | "lt" | "lte" | "eq"
    value: float


@dataclass
class ParsedQuery:
    semantic_query: str                                    # clean text → embedding
    metadata_filters: Optional[Dict[str, Any]] = None     # Pinecone filter dict
    numeric_post_filters: List[NumericCondition] = field(default_factory=list)
    raw_filters_extracted: Dict[str, Any] = field(default_factory=dict)  # for debug/SSE


# ── Pinecone filter builder ───────────────────────────────────────────────────

_OP_MAP = {"gt": "$gt", "gte": "$gte", "lt": "$lt", "lte": "$lte", "eq": "$eq"}


def _build_pinecone_filter(
    eq_filters: Dict[str, str],
    numeric_conditions: List[NumericCondition],
    language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Combine categorical equality + numeric range + optional language into one
    Pinecone metadata filter dict.

    Example output:
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


# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a query-parsing assistant for an HR/Sales analytics RAG system.

Given a natural-language question, extract structured filter information.

Return ONLY a JSON object (no markdown, no explanation):
{
  "semantic_query": "<clean question with all filter constraints removed, for embedding search>",
  "eq_filters": {
    "<key>": "<value>"
  },
  "numeric_conditions": [
    {"field": "<safe_field_name>", "op": "<gt|gte|lt|lte|eq>", "value": <number>}
  ],
  "top_k_hint": <integer or null>
}

Rules:
1. semantic_query: strip all constraints, keep the core topic.
   "Sales Managers in the North region with performance score above 85"
   → "Sales Managers with high performance scores"

2. eq_filters keys (use EXACTLY these key names — they match Pinecone metadata):
   role, region, department, language, doc_type, source_type
   Normalise values to Title Case.
   Examples:
     "sales managers"      → role: "Sales Manager"
     "north region"        → region: "North"
     "HR department"       → department: "HR"

3. numeric_conditions field names (safe_meta_key format, matches Pinecone metadata):
   performance_score, salary, revenue_usd, field_visits,
   customer_satisfaction, target_achievement, operational_cost
   Operator mapping:
     "above 85" / "greater than 85" / "more than 85"  → gte 85.0
     "below 90" / "less than 90" / "under 90"          → lt  90.0
     "at least 80" / "minimum 80"                      → gte 80.0
     "exactly 90"                                       → eq  90.0
     "between 80 and 90" → TWO conditions: gte 80.0 AND lte 90.0

4. top_k_hint: integer if user asked for a specific count ("top 3", "first 5"), else null.

Examples:
  Input:  "Show me Sales Managers in the North region with performance score above 85"
  Output: {"semantic_query": "Sales Managers with high performance scores", "eq_filters": {"role": "Sales Manager", "region": "North"}, "numeric_conditions": [{"field": "performance_score", "op": "gte", "value": 85.0}], "top_k_hint": null}

  Input:  "Which employees in HR have salary below 60000?"
  Output: {"semantic_query": "employees with low salary", "eq_filters": {"department": "HR"}, "numeric_conditions": [{"field": "salary", "op": "lt", "value": 60000.0}], "top_k_hint": null}

  Input:  "What was Q2 revenue?"
  Output: {"semantic_query": "Q2 revenue", "eq_filters": {}, "numeric_conditions": [], "top_k_hint": null}
"""


# ── Parser ────────────────────────────────────────────────────────────────────

async def parse_query(
    raw_query: str,
    language: Optional[str] = None,
) -> ParsedQuery:
    """
    Call Azure OpenAI (gpt-5.4-pro) to extract structured filters from a
    natural-language query.

    Falls back gracefully to a passthrough ParsedQuery if:
      - settings are missing
      - the LLM call fails
      - the response is not valid JSON
    So the pipeline NEVER crashes due to parsing failure.
    """
    # Lazy import to avoid circular dependencies at module load time
    try:
        from config.settings import settings
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            # gpt-5.4-pro requires the Responses API version
            api_version=getattr(settings, "openai_api_version", "2025-04-01-preview"),
        )
        model = getattr(settings, "openai_chat_model", "gpt-5.4-pro")

    except Exception as exc:
        logger.warning("query_parser: could not initialise Azure client (%s) — passthrough", exc)
        return ParsedQuery(semantic_query=raw_query)

    # ── Call the LLM via Responses API (gpt-5.4-pro requires this) ──────────
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

    # ── Parse JSON response ───────────────────────────────────────────────────
    try:
        # Strip accidental markdown fences the model might add
        clean = re.sub(r"^```[a-z]*\n?", "", raw_text.strip())
        clean = re.sub(r"\n?```$", "", clean)
        parsed_json = json.loads(clean)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("query_parser: JSON parse failed (%s) raw=%r — passthrough", exc, raw_text[:200])
        return ParsedQuery(semantic_query=raw_query)

    # ── Build structured result ───────────────────────────────────────────────
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
            continue

    pinecone_filter = _build_pinecone_filter(eq_filters, numeric_conditions, language)

    logger.info(
        "query_parser: '%s' → semantic='%s' filters=%s numeric=%s",
        raw_query, semantic_query, eq_filters,
        [(c.field, c.op, c.value) for c in numeric_conditions],
    )

    return ParsedQuery(
        semantic_query=semantic_query,
        metadata_filters=pinecone_filter,
        numeric_post_filters=numeric_conditions,  # also used as post-filter safety net
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

    Handles cases where:
      - The field wasn't indexed as a numeric scalar in Pinecone
      - The value was stored as a string instead of float
      - Pinecone's filter missed an edge case

    chunks: list of result dicts, each with a "metadata" sub-dict.
    """
    if not conditions:
        return chunks

    def _passes(chunk: Dict[str, Any]) -> bool:
        meta = chunk.get("metadata", {})
        for cond in conditions:
            raw = meta.get(cond.field)
            if raw is None:
                # Field absent from this chunk — don't filter it out,
                # let the LLM decide based on context
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