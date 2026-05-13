"""
chunking/semantic_chunker.py

Semantic chunking for free-text documents (PDFs, notes, etc.).

CSV ROW BYPASS:
  Documents with doc_type="row" or chunk_index=0 AND source_type="csv" are
  returned as-is without any splitting or merging. Merging CSV rows destroys
  per-row numeric metadata, which breaks Pinecone metadata filters ($gte, $eq).
  Each CSV row must remain its own atomic vector.

For all other documents (PDF pages, free-text notes):
  Uses sentence-transformer embeddings (all-MiniLM-L6-v2) when available,
  falling back to a TF-IDF statistical chunker otherwise.
"""

import asyncio
import re
import math
import numpy as np
from typing import List, Dict, Any, Optional

_model = None
_model_available: Optional[bool] = None


def _get_model():
    global _model, _model_available
    if _model_available is False:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _model_available = True
    except Exception:
        _model_available = False
        _model = None
    return _model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[^\d\.\s][.!?])\s+(?=[A-ZÁÉÍÓÚÑ\"\'(])", text)
    result = [s.strip() for s in parts if s.strip()]
    return result if result else [text]


def _chunk_by_words(text: str, max_chars: int) -> List[str]:
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        if length + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _tfidf_similarity(sent_a: str, sent_b: str) -> float:
    def tokenize(s):
        return set(re.findall(r"[a-záéíóúña-z0-9]+", s.lower()))
    a, b = tokenize(sent_a), tokenize(sent_b)
    if not a or not b:
        return 0.0
    stops = {"the", "a", "an", "is", "in", "of", "and", "to", "for",
             "was", "were", "are", "el", "la", "de", "en", "y", "que"}
    signal = (a & b) - stops
    return len(signal) / math.sqrt(len(a) * len(b))


def _statistical_chunk(sentences, min_chars, max_chars, threshold=0.05):
    if len(sentences) <= 2:
        return [" ".join(sentences)]
    window = 2
    breakpoints = [0]
    for i in range(window, len(sentences) - window):
        left  = " ".join(sentences[max(0, i - window): i])
        right = " ".join(sentences[i: i + window])
        if _tfidf_similarity(left, right) < threshold:
            breakpoints.append(i)
    breakpoints.append(len(sentences))

    raw = []
    for i in range(len(breakpoints) - 1):
        chunk = " ".join(sentences[breakpoints[i]: breakpoints[i + 1]]).strip()
        if chunk:
            raw.append(chunk)

    merged = []
    for chunk in raw:
        if merged and len(chunk) < min_chars:
            merged[-1] += " " + chunk
        else:
            merged.append(chunk)

    final = []
    for chunk in merged:
        final.extend(_chunk_by_words(chunk, max_chars) if len(chunk) > max_chars else [chunk])
    return [c for c in final if c.strip()] or [" ".join(sentences)]


def _is_csv_row(metadata: Dict[str, Any]) -> bool:
    """
    Returns True if this document is an individual CSV row that must not be
    split or merged with other documents.

    Criteria (any one is sufficient):
      - doc_type == "row"            (set by csv_ingestor for every row doc)
      - source_type == "csv"         (all CSV documents including summary)
    """
    return (
        metadata.get("doc_type") == "row"
        or metadata.get("source_type") == "csv"
    )


async def semantic_chunk(
    text: str,
    metadata: Dict[str, Any],
    breakpoint_threshold: float = 0.45,
    min_chars: int = 120,
    max_chars: int = 800,
) -> List[Dict[str, Any]]:
    """
    Chunk a single document.

    CSV rows and summary docs are returned as-is (single-element list).
    All other documents are semantically chunked.
    """
    # CSV documents: return immediately without any splitting
    if _is_csv_row(metadata):
        return [{"text": text.strip(), "metadata": {**metadata, "chunk_index": 0}}]

    # Short texts: no benefit from chunking
    sentences = _split_sentences(text)
    if len(sentences) <= 2 or len(text) < min_chars:
        return [{"text": text.strip(), "metadata": {**metadata, "chunk_index": 0}}]

    model = _get_model()

    if model is not None:
        embeddings = await asyncio.to_thread(
            model.encode,
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        breakpoints = [0]
        for i in range(len(embeddings) - 1):
            if _cosine(embeddings[i], embeddings[i + 1]) < breakpoint_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        raw = []
        for i in range(len(breakpoints) - 1):
            chunk = " ".join(sentences[breakpoints[i]: breakpoints[i + 1]]).strip()
            if chunk:
                raw.append(chunk)

        merged = []
        for chunk in raw:
            if merged and len(chunk) < min_chars:
                merged[-1] += " " + chunk
            else:
                merged.append(chunk)

        final = []
        for chunk in merged:
            final.extend(_chunk_by_words(chunk, max_chars) if len(chunk) > max_chars else [chunk])
    else:
        final = _statistical_chunk(sentences, min_chars, max_chars)

    return [
        {"text": t.strip(), "metadata": {**metadata, "chunk_index": idx}}
        for idx, t in enumerate(final) if t.strip()
    ] or [{"text": text.strip(), "metadata": {**metadata, "chunk_index": 0}}]


async def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of raw documents.

    CSV row documents pass through untouched (one-in, one-out).
    PDF pages and other free-text documents are semantically chunked.
    """
    all_chunks = []
    for doc in documents:
        all_chunks.extend(await semantic_chunk(doc["text"], doc["metadata"]))
    return all_chunks
