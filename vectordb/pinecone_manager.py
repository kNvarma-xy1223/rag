import asyncio
import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from config.settings import settings


class PineconeManager:
    """
    Drop-in replacement for QdrantManager.
    Maintains two Pinecone serverless indexes:
      - rag-openai  (dim=3072, cosine)
      - rag-cohere  (dim=1536, cosine)

    On startup, _ensure_indexes() validates that existing indexes have the
    correct dimensions and auto-recreates them if there is a mismatch.
    This is the fix for the common "Vector dimension X does not match index Y" error.
    """

    def __init__(self):
        self._pc = Pinecone(api_key=settings.pinecone_api_key)

    async def _init_async(self) -> "PineconeManager":
        """Async-safe initialization of indexes. Call once after construction."""
        await self._ensure_indexes()
        return self

    # ── Collection lifecycle ──────────────────────────────────────────────────

    async def _existing_index_names(self) -> set:
        return {idx.name for idx in await asyncio.to_thread(self._pc.list_indexes)}

    async def _get_index_dimension(self, index_name: str) -> Optional[int]:
        """Return the dimension of an existing Pinecone index, or None on error."""
        try:
            desc = await asyncio.to_thread(self._pc.describe_index, index_name)
            return desc.dimension
        except Exception as e:
            print(f"[Pinecone] Could not describe index '{index_name}': {e}")
            return None

    async def _create_index(self, index_name: str, dim: int) -> None:
        """Create a new Pinecone serverless index and wait until it is ready."""
        print(f"[Pinecone] Creating index '{index_name}' with dim={dim} ...")
        await asyncio.to_thread(
            self._pc.create_index,
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        # Wait for the index to become ready (up to 60 s)
        for _ in range(30):
            try:
                desc = await asyncio.to_thread(self._pc.describe_index, index_name)
                if getattr(desc, "status", {}).get("ready", False):
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        print(f"[Pinecone] Index '{index_name}' is ready.")

    async def _delete_and_wait(self, index_name: str) -> None:
        """Delete a Pinecone index and block until it is fully removed."""
        print(f"[Pinecone] Deleting index '{index_name}' ...")
        await asyncio.to_thread(self._pc.delete_index, index_name)
        for _ in range(30):
            if index_name not in await self._existing_index_names():
                break
            await asyncio.sleep(2)
        print(f"[Pinecone] Index '{index_name}' deleted.")

    async def _ensure_indexes(self) -> None:
        """
        Ensure both indexes exist with the correct dimensions.

        Logic:
          - Index missing  →  create it.
          - Index exists with wrong dim  →  delete and recreate.
          - Index exists with correct dim  →  no-op.
        """
        targets = [
            (settings.pinecone_openai_index, settings.openai_embedding_dim),
            (settings.pinecone_cohere_index, settings.cohere_embedding_dim),
        ]

        existing = await self._existing_index_names()

        for index_name, expected_dim in targets:
            if index_name not in existing:
                await self._create_index(index_name, expected_dim)
            else:
                actual_dim = await self._get_index_dimension(index_name)
                if actual_dim is None:
                    print(f"[Pinecone] WARNING: Could not verify dimension of '{index_name}'. Skipping.")
                elif actual_dim != expected_dim:
                    print(
                        f"[Pinecone] Index '{index_name}' has dim={actual_dim} "
                        f"but expected dim={expected_dim}. Recreating..."
                    )
                    await self._delete_and_wait(index_name)
                    await self._create_index(index_name, expected_dim)
                else:
                    print(f"[Pinecone] Index '{index_name}' OK (dim={actual_dim}).")

    async def reset_collection(self, index_name: str, vector_size: int) -> None:
        """Delete and recreate a Pinecone index (mirrors QdrantManager.reset_collection)."""
        if index_name in await self._existing_index_names():
            await self._delete_and_wait(index_name)
        await self._create_index(index_name, vector_size)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        index_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        index = self._pc.Index(index_name)

        vectors = []
        for chunk, vec in zip(chunks, embeddings):
            metadata = {"text": chunk["text"], **chunk["metadata"]}
            # Pinecone metadata values must be str | int | float | bool | List[str]
            clean: Dict[str, Any] = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = [str(i) for i in v]
                else:
                    clean[k] = str(v)

            # Use a deterministic ID so re-ingesting the same file UPDATES existing
            # vectors instead of accumulating duplicates (upsert = insert-or-replace).
            # Format: <source>__chunk<chunk_index> — stable across runs.
            meta = chunk["metadata"]
            source_slug = meta.get("source", "unknown").replace(" ", "_")
            chunk_idx   = meta.get("chunk_index", 0)
            # For CSV row documents include row_start to distinguish batches
            row_suffix  = f"__rows{meta['row_start']}" if "row_start" in meta else ""
            page_suffix = f"__p{meta['page']}" if "page" in meta else ""
            det_id = f"{source_slug}{page_suffix}{row_suffix}__chunk{chunk_idx}"
            # Truncate to 512 chars (Pinecone ID limit) and replace any illegal chars
            det_id = det_id[:512].replace("/", "_").replace("\\", "_")

            vectors.append({"id": det_id, "values": vec, "metadata": clean})

        # Pinecone recommends upsert batches of ≤ 100 vectors
        for i in range(0, len(vectors), 100):
            await asyncio.to_thread(index.upsert, vectors=vectors[i: i + 100])

        return len(vectors)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        index = self._pc.Index(index_name)

        # Pass the filter dict as-is — query_parser already builds proper
        # Pinecone syntax ($eq / $gte / $and etc.).  Simple dict of plain
        # string values (legacy callers) still works because Pinecone accepts
        # {"key": "value"} as an implicit $eq match.
        pinecone_filter = filters if filters else None

        response = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=True,
        )

        results = []
        for match in response.matches:
            if match.score < score_threshold:
                continue
            meta = dict(match.metadata or {})
            text = meta.pop("text", "")
            results.append(
                {
                    "id": match.id,
                    "score": round(match.score, 6),
                    "text": text,
                    "metadata": meta,
                }
            )
        return results

    # ── Info ──────────────────────────────────────────────────────────────────

    async def get_collection_info(self, index_name: str) -> Dict[str, Any]:
        """Mirror QdrantManager.get_collection_info for drop-in compatibility."""
        try:
            index = self._pc.Index(index_name)
            stats = await asyncio.to_thread(index.describe_index_stats)
            total = stats.total_vector_count or 0
            return {
                "name": index_name,
                "points_count": total,
                "vectors_count": total,
                "status": "active",
                "error": None,
            }
        except Exception as exc:
            return {
                "name": index_name,
                "points_count": 0,
                "vectors_count": 0,
                "status": "error",
                "error": str(exc),
            }


# Singleton – import this everywhere instead of qdrant_manager
pinecone_manager = PineconeManager()