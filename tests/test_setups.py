import pytest

from rag_lab.setups import get_setup, list_setups


def test_list_setups_is_stable() -> None:
    assert [setup.id for setup in list_setups()] == [
        "late-chunking-v1",
        "naive-100-0",
        "naive-500-100",
    ]


def test_get_setup_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown setup 'missing'"):
        get_setup("missing")
