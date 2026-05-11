from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed a list of texts. Returns list of float vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this embedder."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Canonical model identifier."""
