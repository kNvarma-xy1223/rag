from typing import List
from openai import AsyncAzureOpenAI
from .base import BaseEmbedder
from config.settings import settings


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        # Use the embedding-specific API version (2024-02-01), NOT the
        # Responses API version (2025-04-01-preview) which is for gpt-5.4-pro only.
        self._client = AsyncAzureOpenAI(
            api_key=settings.openai_api_key,
            azure_endpoint=settings.openai_endpoint,
            api_version=settings.openai_embedding_api_version,
        )
        self._model = settings.openai_embedding_model
        self._dim = settings.openai_embedding_dim

    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), 100):
            batch = [t.replace("\n", " ") for t in texts[i: i + 100]]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_embeddings.extend(e.embedding for e in response.data)
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model
