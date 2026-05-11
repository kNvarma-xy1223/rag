from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI Resource (AzureOpenAI client) ──────────────────────────────────
    openai_endpoint: str = ""
    openai_api_key: str = ""

    # Responses API (gpt-5.4-pro) needs 2025-03-01-preview or later
    openai_api_version: str = "2025-04-01-preview"

    # Embedding API works best on the stable 2024-02-01 version
    openai_embedding_api_version: str = "2024-02-01"

    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dim: int = 3072
    openai_chat_model: str = "gpt-5.4-pro"

    # ── Cohere Resource (regular OpenAI client) ───────────────────────────────
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
    # TOP_K is intentionally configured here in code, not in .env.
    # Change this value directly in this file to adjust the default retrieval depth.
    top_k: int = 5
    chunk_size: int = 600
    chunk_overlap: int = 80
    similarity_threshold: float = 0.3

    # ── RAGAS Evaluation ──────────────────────────────────────────────────────
    # The LLM used for RAGAS judge calls (faithfulness, relevancy, etc.)
    # Defaults to the same Azure chat model; override via RAGAS_JUDGE_MODEL in .env.
    ragas_judge_model: str = ""          # falls back to openai_chat_model when empty
    ragas_judge_api_version: str = "2025-04-01-preview"

    # Maximum tokens RAGAS judge is allowed to generate per call
    ragas_max_tokens: int = 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    # Convenience: resolve the actual judge model name at runtime
    @property
    def resolved_ragas_judge_model(self) -> str:
        return self.ragas_judge_model or self.openai_chat_model


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()