from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI Resource (AzureOpenAI client) ──────────────────────────────────
    openai_endpoint: str = ""
    openai_api_key: str = ""
    openai_api_version: str = "2025-04-01-preview"
    openai_embedding_api_version: str = "2024-02-01"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dim: int = 3072
    openai_chat_model: str = "gpt-5.4-pro"

    # ── Cohere Resource ───────────────────────────────────────────────────────
    cohere_endpoint: str = ""
    cohere_api_key: str = ""
    cohere_embedding_model: str = "embed-v-4-0"
    cohere_embedding_dim: int = 1536

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_openai_index: str = "rag-openai"
    pinecone_cohere_index: str = "rag-cohere"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── RAG ───────────────────────────────────────────────────────────────────
    top_k: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 80
    similarity_threshold: float = 0.3

    # ── Retrieval pool (production) ───────────────────────────────────────────
    # pinecone_retrieval_k: how many candidates to pull from Pinecone BEFORE
    #   post-filtering. Set large enough to cover your biggest dataset.
    #   Pinecone hard limit is 10 000; 500 is safe and fast for most datasets.
    # max_final_k: hard ceiling on chunks sent to the LLM (token-budget guard).
    #   gpt-5.4-pro context = 128 k tokens; 1 chunk ≈ 200 tokens → 100 chunks
    #   uses ~20 k tokens, well within budget.
    pinecone_retrieval_k: int = 500
    max_final_k: int = 100

    # ── DeepEval Evaluation ───────────────────────────────────────────────────
    # Judge model for all DeepEval metric calls.
    # Override via DEEPEVAL_JUDGE_MODEL in .env; defaults to gpt-5.4-nano.
    deepeval_judge_model: str = "gpt-5.4-nano"
    deepeval_judge_api_version: str = "2024-12-01-preview"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def resolved_deepeval_judge_model(self) -> str:
        return self.deepeval_judge_model or self.openai_chat_model


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()