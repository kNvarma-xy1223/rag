"""
ingestion/csv_ingestor.py

Design principles:
  - Each CSV row → exactly ONE document (chunk_index always 0).
    Row documents must never be merged by the semantic chunker — their
    per-row numeric metadata is what makes Pinecone metadata filters work.
    The semantic chunker is for free-text (PDFs, notes); CSV rows are
    already atomic units.
  - ALL numeric columns stored as float metadata in Pinecone.
    Queries like "performance_score > 85" are resolved by Pinecone's
    $gte filter, not by embedding similarity.
  - ALL categorical columns stored as title-case string metadata.
    Queries like "role = Sales Manager" use Pinecone's $eq filter.
  - Embedding text = natural-language sentence for semantic understanding.
    The LLM reads this text to form its answer.
  - One summary document per file describes schema + dataset stats.
  - Language auto-detected; stored as metadata for language filtering.
"""

import asyncio
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")


# ── Language detection ────────────────────────────────────────────────────────

_SPANISH_MARKERS = {
    "el", "la", "los", "las", "un", "una", "de", "que", "en", "y",
    "es", "por", "con", "para", "del", "al", "se", "no", "lo", "le",
    "su", "como", "más", "pero", "sus", "muy", "sin", "sobre",
}


def _detect_language_from_df(df: pd.DataFrame, col_types: Dict[str, str]) -> str:
    """Sample text/categorical columns and vote on language."""
    sample_cols = [c for c, t in col_types.items() if t in ("text", "categorical")]
    words: List[str] = []
    for col in sample_cols[:5]:
        for val in df[col].dropna().head(30).astype(str):
            words.extend(val.lower().split())
        if len(words) >= 200:
            break
    if not words:
        return "en"
    hits = len(set(words[:300]) & _SPANISH_MARKERS)
    return "es" if hits >= 4 else "en"


# ── Column classification ─────────────────────────────────────────────────────

_DATE_KEYWORDS = {
    "date", "time", "year", "month", "day", "week", "quarter", "period",
    "at", "on", "ts", "timestamp", "created", "updated", "modified",
    "start", "end", "fecha", "año", "mes", "dia", "día", "hora",
    "inicio", "fin", "creado", "actualizado", "periodo",
}
_ID_KEYWORDS = {
    "id", "uuid", "guid", "pk", "fk", "key", "code", "ref", "index",
    "serial", "no", "num", "number", "nr", "codigo", "código", "clave",
    "numero", "número",
}
_ID_VALUE_PATTERN = re.compile(
    r"^[A-Z]{1,6}-?\d+$|^[0-9a-f]{8}-[0-9a-f]{4}-|^\d{1,8}$", re.I
)


def _content_looks_like_date(s: pd.Series, thr: float = 0.80) -> bool:
    sample = s.dropna().head(40).astype(str)
    return len(sample) > 0 and float(pd.to_datetime(sample, errors="coerce").notna().mean()) >= thr


def _content_looks_like_id(s: pd.Series, thr: float = 0.85) -> bool:
    sample = s.dropna().head(40).astype(str)
    return len(sample) > 0 and float(sample.str.fullmatch(_ID_VALUE_PATTERN).mean()) >= thr


def _numeric_is_sequential_id(s: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(s):
        return False
    vals = s.dropna()
    return (
        len(vals) > 0
        and s.nunique() == len(vals)
        and sorted(vals.tolist()) == list(range(1, len(vals) + 1))
    )


def _classify_columns(df: pd.DataFrame) -> Dict[str, str]:
    col_types: Dict[str, str] = {}
    n = max(len(df), 1)
    for col in df.columns:
        tokens = set(re.split(r"[_\s\-]+", col.lower().strip()))
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s.dtype):
            col_types[col] = "date"
            continue

        if pd.api.types.is_numeric_dtype(s.dtype):
            col_types[col] = "id" if _numeric_is_sequential_id(s) else "numeric"
            continue

        unique_ratio = s.nunique() / n

        if tokens & _DATE_KEYWORDS and _content_looks_like_date(s):
            col_types[col] = "date"
            continue

        if tokens & _ID_KEYWORDS and unique_ratio > 0.8:
            col_types[col] = "id"
            continue

        if _content_looks_like_date(s):
            col_types[col] = "date"
            continue

        if unique_ratio > 0.8 and _content_looks_like_id(s):
            col_types[col] = "id"
            continue

        avg_len = s.dropna().astype(str).str.len().mean() if len(s.dropna()) else 0
        # "text" only when BOTH long AND highly unique — prevents short label columns
        # (role, region, status) from being mis-tagged as text and excluded from metadata.
        col_types[col] = "text" if (avg_len > 60 and unique_ratio > 0.5) else "categorical"

    return col_types


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(val: Any) -> str:
    return f"{val:.6g}" if isinstance(val, float) else str(val)


def _safe_meta_key(col: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", col.lower())[:40]


# ── Embedding text builder ────────────────────────────────────────────────────

def _build_row_text(
    row: pd.Series,
    col_types: Dict[str, str],
    id_cols: List[str],
    language: str = "en",
) -> str:
    """
    Build a concise natural-language sentence from one CSV row.

    English example:
      EMP-1049. Role is Sales Manager. Region is North. Evaluation date: 2023-04-20.
      Revenue usd is 368792. Field visits is 45. Customer satisfaction is 3.94.
      Performance score is 87.3. Target achievement pct is 80.72.
      Operational cost usd is 105917. Evaluation notes: Field inspections improved customer retention rates.

    Spanish example:
      EMP-1049. Role es Sales Manager. Region es North. ...
      Valor de performance score es 87.3. ...
    """
    is_es = (language == "es")
    sentences: List[str] = []

    # ID prefix (e.g. "EMP-1049")
    id_vals = [str(row[c]) for c in id_cols if pd.notna(row.get(c))]
    if id_vals:
        sentences.append(" / ".join(id_vals) + ".")

    # Categorical columns
    for col, t in col_types.items():
        if t == "categorical" and pd.notna(row.get(col)):
            label = col.replace("_", " ").capitalize()
            connector = "es" if is_es else "is"
            sentences.append(f"{label} {connector} {row[col]}.")

    # Date columns
    for col, t in col_types.items():
        if t == "date" and pd.notna(row.get(col)):
            label = col.replace("_", " ").capitalize()
            sentences.append(f"{label}: {_fmt(row[col])}.")

    # Numeric KPI columns — include exact value so LLM can cite them
    for col, t in col_types.items():
        if t == "numeric" and pd.notna(row.get(col)):
            label = col.replace("_", " ").capitalize()
            val_str = _fmt(row[col])
            if is_es:
                sentences.append(f"Valor de {label.lower()} es {val_str}.")
            else:
                sentences.append(f"{label} is {val_str}.")

    # Free-text columns
    for col, t in col_types.items():
        if t == "text" and col not in id_cols and pd.notna(row.get(col)):
            label = col.replace("_", " ").capitalize()
            sentences.append(f"{label}: {row[col]}.")

    return " ".join(sentences).strip()


# ── Main ingestion ─────────────────────────────────────────────────────────────

def _ingest_csv_sync(file_path: str) -> List[Dict[str, Any]]:
    """
    Synchronous CSV ingestion.
    Returns a flat list of documents (one per row + one summary).
    """
    df = pd.read_csv(file_path)
    source_name = Path(file_path).name
    col_types = _classify_columns(df)
    language = _detect_language_from_df(df, col_types)

    numeric_cols     = [c for c, t in col_types.items() if t == "numeric"]
    date_cols        = [c for c, t in col_types.items() if t == "date"]
    text_cols        = [c for c, t in col_types.items() if t == "text"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]
    id_cols          = [c for c, t in col_types.items() if t == "id"]

    documents: List[Dict[str, Any]] = []

    # ── 1. Summary document ───────────────────────────────────────────────────
    lines = [
        f"Dataset: {source_name}",
        f"Language: {language}",
        f"Total rows: {len(df)}",
        f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}",
    ]
    if numeric_cols:
        lines.append("\nNumeric / KPI columns:")
        for col in numeric_cols:
            s = df[col].dropna()
            if len(s):
                lines.append(
                    f"  {col}: count={len(s)}, min={_fmt(s.min())}, "
                    f"max={_fmt(s.max())}, mean={s.mean():.4g}, std={s.std():.4g}"

                )
    if categorical_cols:
        lines.append("\nCategorical columns:")
        for col in categorical_cols:
            top = df[col].value_counts().head(8)
            lines.append(f"  {col}: {', '.join(f'{k}({v})' for k, v in top.items())}")
    if text_cols:
        lines.append(f"\nFree-text columns: {', '.join(text_cols)}")
    if date_cols:
        lines.append(f"\nDate columns: {', '.join(date_cols)}")
    if id_cols:
        lines.append(f"\nID/key columns (metadata only): {', '.join(id_cols)}")

    documents.append({
        "text": "\n".join(lines),
        "metadata": {
            "source":              source_name,
            "source_type":         "csv",
            "doc_type":            "summary",
            "language":            language,
            "chunk_index":         0,
            "rows":                len(df),
            "columns":             ", ".join(df.columns.tolist()),
            "numeric_columns":     ", ".join(numeric_cols),
            "date_columns":        ", ".join(date_cols),
            "text_columns":        ", ".join(text_cols),
            "categorical_columns": ", ".join(categorical_cols),
            "id_columns":          ", ".join(id_cols),
        },
    })

    # ── 2. Per-row documents ──────────────────────────────────────────────────
    for idx, row in df.iterrows():
        row_number = int(idx) + 1

        meta: Dict[str, Any] = {
            "source":      source_name,
            "source_type": "csv",
            "doc_type":    "row",
            "language":    language,
            "row_number":  row_number,
            "chunk_id":    f"{source_name}::row::{row_number}",
            # chunk_index=0 → semantic_chunk() returns this doc immediately as-is
            "chunk_index": 0,
        }

        # Numeric KPIs as floats → enables Pinecone $gte / $lte / $gt / $lt
        for col in numeric_cols:
            if pd.notna(row.get(col)):
                meta[_safe_meta_key(col)] = float(row[col])

        # Categoricals as title-case strings → enables Pinecone $eq
        for col in categorical_cols:
            if pd.notna(row.get(col)):
                meta[_safe_meta_key(col)] = str(row[col]).strip().title()

        # Dates as ISO strings
        for col in date_cols:
            if pd.notna(row.get(col)):
                try:
                    meta[_safe_meta_key(col)] = str(pd.to_datetime(row[col]).date())
                except Exception:
                    meta[_safe_meta_key(col)] = str(row[col])

        # IDs as strings
        for col in id_cols:
            if pd.notna(row.get(col)):
                meta[_safe_meta_key(col)] = str(row[col])

        documents.append({
            "text":     _build_row_text(row, col_types, id_cols, language),
            "metadata": meta,
        })

    return documents


async def ingest_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Async CSV ingestion (offloads pandas/DataFrame work to a thread).
    """
    return await asyncio.to_thread(_ingest_csv_sync, file_path)
