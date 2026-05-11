from typing import List
from openai import AsyncOpenAI
from .base import BaseEmbedder
from config.settings import settings


class CohereEmbedder(BaseEmbedder):
    def __init__(self):
        # Regular OpenAI client — Cohere endpoint doesn't use AzureOpenAI client
        self._client = AsyncOpenAI(
            api_key=settings.cohere_api_key,
            base_url=settings.cohere_endpoint,
        )
        self._model = settings.cohere_embedding_model
        self._dim = settings.cohere_embedding_dim

    async def embed(
        self, texts: List[str], input_type: str = "search_document", **kwargs
    ) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), 96):
            batch = [t.replace("\n", " ") for t in texts[i: i + 96]]
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
