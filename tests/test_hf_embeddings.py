import pytest

from rag_lab.embeddings import HuggingFaceEmbedder, get_embedder
from rag_lab.settings import Settings
from rag_lab.setups import EmbeddingProfile

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

WORDS = ["hello", "world", "late", "chunking", "embeds", "text", "the", "a"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tiny randomly initialised BERT saved locally, so no download is needed."""
    path = tmp_path_factory.mktemp("tiny-bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (path / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    transformers.BertTokenizer(str(path / "vocab.txt")).save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    transformers.BertModel(config).save_pretrained(path)
    return str(path)


TEXTS = ["hello world", "late chunking embeds the text", "a"]


def test_vectors_have_one_row_per_text_and_unit_length(model_dir: str) -> None:
    vectors = HuggingFaceEmbedder(model_dir).embed(TEXTS)

    assert [len(v) for v in vectors] == [16, 16, 16]
    assert all(sum(x * x for x in v) == pytest.approx(1.0, abs=1e-5) for v in vectors)


def test_padding_and_batch_size_do_not_change_vectors(model_dir: str) -> None:
    together = HuggingFaceEmbedder(model_dir, batch_size=8).embed(TEXTS)
    one_by_one = HuggingFaceEmbedder(model_dir, batch_size=1).embed(TEXTS)

    for a, b in zip(together, one_by_one, strict=True):
        assert a == pytest.approx(b, abs=1e-5)


def test_different_texts_give_different_vectors(model_dir: str) -> None:
    first, second, _ = HuggingFaceEmbedder(model_dir).embed(TEXTS)

    assert first != pytest.approx(second, abs=1e-3)


def test_get_embedder_builds_huggingface_provider() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    profile = EmbeddingProfile("x", "huggingface", "org/model", 16)

    assert get_embedder(settings, profile) == HuggingFaceEmbedder(
        "org/model", settings.embedding_batch_size, "cpu", False
    )
