"""Application settings loaded from environment variables."""

from urllib.parse import quote

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by ingestion and query commands."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "rag-lab"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    postgres_user: str = "raglab"
    postgres_password: str = "raglab"
    postgres_db: str = "raglab"
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    provenance_schema: str = "rag"

    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    embedding_trust_remote_code: bool = False
    model_request_timeout_seconds: float = 60.0
    model_max_retries: int = 2
    ollama_url: str = "http://localhost:11434"
    cache_max_age_days: int = 30

    @property
    def postgres_dsn(self) -> str:
        """Connection string for the PostgreSQL provenance store."""
        return (
            f"postgresql://{quote(self.postgres_user)}:"
            f"{quote(self.postgres_password)}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )


def get_settings() -> Settings:
    """Return runtime settings from the environment."""
    return Settings()
