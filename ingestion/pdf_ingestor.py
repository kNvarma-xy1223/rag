import asyncio
from pathlib import Path
from typing import List, Dict, Any
import fitz  # pymupdf


_SPANISH_MARKERS = {"el", "la", "los", "las", "un", "una", "de", "que", "en", "y", "es", "por", "con", "para"}


def _detect_language(text: str) -> str:
    words = set(text.lower().split()[:60])
    return "es" if len(words & _SPANISH_MARKERS) >= 4 else "en"


def _extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """Synchronous helper for extracting PDF pages."""
    doc = fitz.open(file_path)
    source_name = Path(file_path).name
    documents: List[Dict[str, Any]] = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": source_name,
                "source_type": "pdf",
                "page": page_num + 1,
                "total_pages": len(doc),
                "language": _detect_language(text),
            },
        })

    doc.close()
    return documents


async def ingest_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract pages from PDF with metadata (async wrapper)."""
    return await asyncio.to_thread(_extract_pdf_pages, file_path)
