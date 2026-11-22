"""Embedding providers."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from rag_lab.settings import Settings
from rag_lab.setups import EmbeddingProfile


class EmbeddingError(RuntimeError):
    """Raised when a provider cannot return embeddings."""


class Embedder(Protocol):
    """Turns texts into vectors, one vector per text, in order."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class OllamaEmbedder:
    """Embeds texts through Ollama's `/api/embed` endpoint."""

    base_url: str
    model: str
    batch_size: int = 32
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_delay_seconds: float = 0.5
    transport: httpx.BaseTransport | None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
        ) as client:
            for start in range(0, len(texts), self.batch_size):
                vectors.extend(
                    self._embed_batch(client, texts[start : start + self.batch_size])
                )
        return vectors

    def _embed_batch(
        self, client: httpx.Client, batch: Sequence[str]
    ) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(
                    "/api/embed", json={"model": self.model, "input": list(batch)}
                )
            except httpx.TransportError as error:
                failure = f"cannot reach Ollama at {self.base_url} ({error})"
            else:
                if response.status_code == 404:
                    raise EmbeddingError(
                        f"Ollama has no model '{self.model}'; "
                        f"run `ollama pull {self.model}`."
                    )
                if response.is_success:
                    embeddings: list[list[float]] = response.json()["embeddings"]
                    if len(embeddings) != len(batch):
                        raise EmbeddingError(
                            f"Ollama returned {len(embeddings)} vectors "
                            f"for {len(batch)} texts."
                        )
                    return embeddings
                failure = f"Ollama returned HTTP {response.status_code}"
            if attempt < self.max_retries:
                time.sleep(self.retry_delay_seconds * 2**attempt)
        raise EmbeddingError(
            f"Embedding failed after {self.max_retries + 1} attempts: {failure}"
        )


@dataclass(frozen=True)
class HuggingFaceEmbedder:
    """Embeds texts locally with a Hugging Face model (mean pooling, L2-normalized).

    Needs the optional `hf` extra (`uv sync --extra hf`). The model is downloaded
    from the Hugging Face Hub on first use and cached.
    """

    model: str
    batch_size: int = 32
    device: str = "cpu"
    trust_remote_code: bool = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            import torch
            import torch.nn.functional as functional
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise EmbeddingError(
                "Hugging Face embeddings need the 'hf' extra: uv sync --extra hf"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        model.to(self.device).eval()
        vectors: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vectors.extend(functional.normalize(pooled, dim=1).tolist())
        return vectors


def get_embedder(settings: Settings, profile: EmbeddingProfile) -> Embedder:
    """Build the embedder for an embedding profile (provider and model)."""
    if profile.provider == "ollama":
        return OllamaEmbedder(
            base_url=settings.ollama_url,
            model=profile.model,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.model_request_timeout_seconds,
            max_retries=settings.model_max_retries,
        )
    if profile.provider == "huggingface":
        return HuggingFaceEmbedder(
            model=profile.model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
            trust_remote_code=settings.embedding_trust_remote_code,
        )
    raise ValueError(f"Unknown embedding provider '{profile.provider}'.")
