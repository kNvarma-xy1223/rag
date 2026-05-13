"""
ingestion/csv_ingestor.py  — Upgrade #4: language detection + Spanish support

Column classification: two-layer (keyword hints → content probes).
Embedding text: natural-language sentence template (English or Spanish).
Language detection: keyword-probe on free-text/categorical values → "en" | "es".
Metadata: every document (summary + row) carries a `language` field for
          downstream Pinecone filtering via {"language": "es"} / {"language": "en"}.
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")

# ── Language detection ────────────────────────────────────────────────────────
_SPANISH_MARKERS = {
    "el", "la", "los", "las", "un", "una", "de", "que", "en", "y",
    "es", "por", "con", "para", "del", "al", "se", "no", "lo", "le",
    "su", "una", "como", "más", "pero", "sus", "muy", "sin", "sobre",
}

def _detect_language_from_df(df: pd.DataFrame, col_types: Dict[str, str]) -> str:
    """
    Sample up to 200 words from text + categorical columns and vote on language.
    Returns "es" if Spanish markers dominate, else "en".
    """
    sample_cols = [c for c, t in col_types.items() if t in ("text", "categorical")]
    words: List[str] = []
    for col in sample_cols[:5]:                          # cap columns sampled
        for val in df[col].dropna().head(30).astype(str):
            words.extend(val.lower().split())
        if len(words) >= 200:
            break
    if not words:
        return "en"
    word_set = set(words[:300])
    hits = len(word_set & _SPANISH_MARKERS)
    return "es" if hits >= 4 else "en"


# ── Keyword hints ─────────────────────────────────────────────────────────────
_DATE_KEYWORDS = {
    "date", "time", "year", "month", "day", "week", "quarter", "period",
    "at", "on", "ts", "timestamp", "created", "updated", "modified",
    "expired", "start", "end",
    # Spanish equivalents
    "fecha", "año", "mes", "dia", "día", "hora", "inicio", "fin", "creado",
    "actualizado", "periodo", "período",
}
_ID_KEYWORDS = {
    "id", "uuid", "guid", "pk", "fk", "key", "code",
    "ref", "index", "serial", "no", "num", "number", "nr",
    # Spanish equivalents
    "codigo", "código", "clave", "numero", "número", "indice", "índice",
}
_ID_VALUE_PATTERN = re.compile(
    r"^[A-Z]{1,6}-?\d+$|^[0-9a-f]{8}-[0-9a-f]{4}-|^\d{1,8}$", re.I
)


# ── Content probes ────────────────────────────────────────────────────────────
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
    return len(vals) > 0 and s.nunique() == len(vals) and sorted(vals.tolist()) == list(range(1, len(vals) + 1))


# ── Classifier ────────────────────────────────────────────────────────────────
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
        if tokens & _DATE_KEYWORDS:
            if _content_looks_like_date(s):
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
        # FIX: use AND instead of OR.
        # Old logic (OR): short columns like "role" / "region" with many unique
        #   values (unique_ratio > 0.7) were mis-tagged as "text" → never stored
        #   as Pinecone metadata → filter {"role": {"$eq": "Sales Manager"}}
        #   matched nothing → fallback fired → completely wrong records returned.
        # New logic (AND): a column is "text" only when BOTH conditions hold:
        #   • avg value length > 60 chars  (actual free-text length), AND
        #   • unique_ratio > 0.5           (values are not repeated descriptions)
        # Short label columns (role names, regions, statuses, departments) have
        #   avg_len ≤ 60 → always classified as "categorical" → stored in
        #   metadata → Pinecone filters work correctly.
        col_types[col] = "text" if (avg_len > 60 and unique_ratio > 0.5) else "categorical"
    return col_types


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt(val: Any) -> str:
    return f"{val:.6g}" if isinstance(val, float) else str(val)

def _safe_meta_key(col: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", col.lower())[:40]


# ── Embedding text builders ───────────────────────────────────────────────────

def _build_embedding_text_en(
    row: pd.Series,
    col_types: Dict[str, str],
    id_cols: List[str],
) -> str:
    """
    English natural-language embedding text.

    Example output:
      EMP-1003. role: HR Manager. department: HR.
      Location is Remote. hire date 2019-11-20. Salary is 95000.
      Performance score is 4.2. Notes: Improved onboarding this year.
    """
    id_vals = {c: str(row[c]) for c in id_cols if pd.notna(row.get(c))}
    sentences: List[str] = []

    id_str = " / ".join(id_vals.values()) if id_vals else None
    cats = {c: str(row[c]) for c, t in col_types.items()
            if t == "categorical" and pd.notna(row.get(c))}
    cat_items = list(cats.items())

    intro_parts = []
    if id_str:
        intro_parts.append(id_str)
    for col, val in cat_items[:2]:
        intro_parts.append(f"{col.replace('_', ' ')}: {val}")
    if intro_parts:
        sentences.append(". ".join(intro_parts) + ".")

    for col, val in cat_items[2:]:
        sentences.append(f"{col.replace('_', ' ').capitalize()} is {val}.")

    date_parts = []
    for col, t in col_types.items():
        if t == "date" and pd.notna(row.get(col)):
            date_parts.append(f"{col.replace('_', ' ')} {_fmt(row[col])}")
    if date_parts:
        sentences.append(". ".join(date_parts) + ".")

    for col, t in col_types.items():
        if t == "numeric" and pd.notna(row.get(col)):
            sentences.append(f"{col.replace('_', ' ').capitalize()} is {_fmt(row[col])}.")

    for col, t in col_types.items():
        if t == "text" and col not in id_cols and pd.notna(row.get(col)):
            sentences.append(f"{col.replace('_', ' ').capitalize()}: {row[col]}")

    return " ".join(sentences).strip()


def _build_embedding_text_es(
    row: pd.Series,
    col_types: Dict[str, str],
    id_cols: List[str],
) -> str:
    """
    Spanish natural-language embedding text.

    Example output:
      EMP-1003. puesto: Gerente de RRHH. departamento: Recursos Humanos.
      Ubicación es Remoto. fecha de contratación 2019-11-20.
      Salario es 95000. Puntuación de desempeño es 4.2.
      Notas: Mejoró el proceso de incorporación este año.
    """
    id_vals = {c: str(row[c]) for c in id_cols if pd.notna(row.get(c))}
    sentences: List[str] = []

    id_str = " / ".join(id_vals.values()) if id_vals else None
    cats = {c: str(row[c]) for c, t in col_types.items()
            if t == "categorical" and pd.notna(row.get(c))}
    cat_items = list(cats.items())

    intro_parts = []
    if id_str:
        intro_parts.append(id_str)
    for col, val in cat_items[:2]:
        intro_parts.append(f"{col.replace('_', ' ')}: {val}")
    if intro_parts:
        sentences.append(". ".join(intro_parts) + ".")

    for col, val in cat_items[2:]:
        sentences.append(f"{col.replace('_', ' ').capitalize()} es {val}.")

    date_parts = []
    for col, t in col_types.items():
        if t == "date" and pd.notna(row.get(col)):
            date_parts.append(f"{col.replace('_', ' ')} {_fmt(row[col])}")
    if date_parts:
        sentences.append(". ".join(date_parts) + ".")

    for col, t in col_types.items():
        if t == "numeric" and pd.notna(row.get(col)):
            sentences.append(f"Valor de {col.replace('_', ' ')} es {_fmt(row[col])}.")

    for col, t in col_types.items():
        if t == "text" and col not in id_cols and pd.notna(row.get(col)):
            sentences.append(f"{col.replace('_', ' ').capitalize()}: {row[col]}")

    return " ".join(sentences).strip()


def _build_embedding_text(
    row: pd.Series,
    col_types: Dict[str, str],
    id_cols: List[str],
    language: str = "en",
) -> str:
    """Dispatch to the correct language builder."""
    if language == "es":
        return _build_embedding_text_es(row, col_types, id_cols)
    return _build_embedding_text_en(row, col_types, id_cols)


# ── Ingestion ─────────────────────────────────────────────────────────────────
def ingest_csv(file_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(file_path)
    source_name = Path(file_path).name
    col_types   = _classify_columns(df)

    # Detect dataset language once — applied to every document produced
    language = _detect_language_from_df(df, col_types)

    numeric_cols     = [c for c, t in col_types.items() if t == "numeric"]
    date_cols        = [c for c, t in col_types.items() if t == "date"]
    text_cols        = [c for c, t in col_types.items() if t == "text"]
    categorical_cols = [c for c, t in col_types.items() if t == "categorical"]
    id_cols          = [c for c, t in col_types.items() if t == "id"]

    documents: List[Dict[str, Any]] = []

    # ── 1. Summary doc ────────────────────────────────────────────────────────
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
                    f"max={_fmt(s.max())}, mean={_fmt(s.mean())}, std={_fmt(s.std())}"
                )
    if categorical_cols:
        lines.append("\nCategorical columns:")
        for col in categorical_cols:
            top = df[col].value_counts().head(5)
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
            "source":               source_name,
            "source_type":          "csv",
            "doc_type":             "summary",
            "language":             language,          # ← filterable
            "rows":                 len(df),
            "columns":              ", ".join(df.columns.tolist()),
            "numeric_columns":      ", ".join(numeric_cols),
            "date_columns":         ", ".join(date_cols),
            "text_columns":         ", ".join(text_cols),
            "categorical_columns":  ", ".join(categorical_cols),
            "id_columns":           ", ".join(id_cols),
        },
    })

    # ── 2. Per-row docs ───────────────────────────────────────────────────────
    for idx, row in df.iterrows():
        row_number = int(idx) + 1
        meta: Dict[str, Any] = {
            "source":       source_name,
            "source_type":  "csv",
            "doc_type":     "row",
            "language":     language,                  # ← filterable
            "row_number":   row_number,
            "chunk_id":     f"{source_name}::row::{row_number}",
        }
        for col in numeric_cols:
            if pd.notna(row.get(col)):
                meta[_safe_meta_key(col)] = float(row[col])
        for col in categorical_cols:
            if pd.notna(row.get(col)):
                meta[_safe_meta_key(col)] = str(row[col]).strip().title()
        for col in date_cols:
            if pd.notna(row.get(col)):
                try:
                    meta[_safe_meta_key(col)] = str(pd.to_datetime(row[col]).date())
                except Exception:
                    meta[_safe_meta_key(col)] = str(row[col])
        for col in id_cols:
            if pd.notna(row.get(col)):
                meta[_safe_meta_key(col)] = str(row[col])

        documents.append({
            "text":     _build_embedding_text(row, col_types, id_cols, language),
            "metadata": meta,
        })

    return documents