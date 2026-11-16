"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by ingestion and query commands."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "rag-lab"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int | None = 768
    embedding_batch_size: int = 32
    model_request_timeout_seconds: float = 60.0
    model_max_retries: int = 2
    ollama_url: str = "http://localhost:11434"


def get_settings() -> Settings:
    """Return runtime settings from the environment."""
    return Settings()
