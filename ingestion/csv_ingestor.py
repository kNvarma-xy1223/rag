from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def _fmt(val: Any) -> str:
    if isinstance(val, float):
        # Preserve up to 6 significant figures, strip trailing zeros
        return f"{val:.6g}"
    return str(val)


def ingest_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Ingest CSV producing:
      1. A schema + statistics summary document.
      2. Batched row documents (10 rows each) preserving numerical values.
    """
    df = pd.read_csv(file_path)
    source_name = Path(file_path).name
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    documents: List[Dict[str, Any]] = []

    # --- Summary document ---
    lines = [
        f"Dataset: {source_name}",
        f"Total rows: {len(df)}",
        f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}",
        "",
        "Numerical statistics:",
    ]
    for col in numeric_cols:
        series = df[col].dropna()
        lines.append(
            f"  {col}: count={len(series)}, min={_fmt(series.min())}, "
            f"max={_fmt(series.max())}, mean={_fmt(series.mean())}, "
            f"median={_fmt(series.median())}, std={_fmt(series.std())}"
        )

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        lines.append("")
        lines.append("Categorical columns:")
        for col in cat_cols:
            top = df[col].value_counts().head(5)
            lines.append(f"  {col}: {', '.join(f'{k}({v})' for k, v in top.items())}")

    documents.append({
        "text": "\n".join(lines),
        "metadata": {
            "source": source_name,
            "source_type": "csv",
            "doc_type": "summary",
            "rows": len(df),
            "columns": df.columns.tolist(),
        },
    })

    # --- Row batch documents ---
    batch_size = 10
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        row_lines = []
        for idx, row in batch.iterrows():
            parts = [f"Row {int(idx) + 1}"]
            for col, val in row.items():
                if pd.notna(val):
                    parts.append(f"{col}={_fmt(val)}")
            row_lines.append(" | ".join(parts))

        documents.append({
            "text": "\n".join(row_lines),
            "metadata": {
                "source": source_name,
                "source_type": "csv",
                "doc_type": "rows",
                "row_start": i + 1,
                "row_end": min(i + batch_size, len(df)),
            },
        })

    return documents
