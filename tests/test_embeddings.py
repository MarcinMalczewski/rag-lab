import json

import httpx
import pytest

from rag_lab.embeddings import EmbeddingError, OllamaEmbedder, get_embedder
from rag_lab.settings import Settings
from rag_lab.setups import EmbeddingProfile


def embedder(
    transport: httpx.MockTransport, batch_size: int = 32, max_retries: int = 2
) -> OllamaEmbedder:
    return OllamaEmbedder(
        "http://ollama.test",
        "m",
        batch_size=batch_size,
        max_retries=max_retries,
        retry_delay_seconds=0,
        transport=transport,
    )


def test_embeds_in_batches_and_preserves_order() -> None:
    sizes: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        sizes.append(len(texts))
        return httpx.Response(200, json={"embeddings": [[float(t)] for t in texts]})

    texts = [str(i) for i in range(70)]
    vectors = embedder(httpx.MockTransport(handle), batch_size=32).embed(texts)

    assert sizes == [32, 32, 6]
    assert vectors == [[float(i)] for i in range(70)]


def test_retries_then_succeeds() -> None:
    calls: list[int] = []

    def handle(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    assert embedder(httpx.MockTransport(handle)).embed(["a"]) == [[1.0]]
    assert len(calls) == 3


def test_gives_up_after_retries() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(500))

    with pytest.raises(
        EmbeddingError, match="after 3 attempts: Ollama returned HTTP 500"
    ):
        embedder(transport).embed(["a"])


def test_unreachable_server_is_reported() -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(EmbeddingError, match="cannot reach Ollama"):
        embedder(httpx.MockTransport(handle), max_retries=0).embed(["a"])


def test_missing_model_suggests_pull() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(404))

    with pytest.raises(EmbeddingError, match="ollama pull m"):
        embedder(transport).embed(["a"])


def test_wrong_vector_count_is_rejected() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"embeddings": []})
    )

    with pytest.raises(EmbeddingError, match="0 vectors for 1 texts"):
        embedder(transport).embed(["a"])


def test_get_embedder_uses_profile_and_settings() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    profile = EmbeddingProfile("nomic", "ollama", "nomic-embed-text", 768)

    assert get_embedder(settings, profile) == OllamaEmbedder(
        settings.ollama_url,
        "nomic-embed-text",
        settings.embedding_batch_size,
        settings.model_request_timeout_seconds,
        settings.model_max_retries,
    )


def test_get_embedder_rejects_unknown_provider() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="Unknown embedding provider 'x'"):
        get_embedder(settings, EmbeddingProfile("p", "x", "m", None))
