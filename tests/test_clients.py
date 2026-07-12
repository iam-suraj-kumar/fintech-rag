from unittest.mock import patch

import pytest

import core.clients as mod


@pytest.fixture
def reset_client_singletons():
    mod._qdrant_client = None
    mod._sparse_model = None
    yield
    mod._qdrant_client = None
    mod._sparse_model = None


def test_get_qdrant_client_returns_same_instance_on_repeated_calls(reset_client_singletons):
    with patch.object(mod, "QdrantClient") as mock_cls:
        first = mod.get_qdrant_client()
        second = mod.get_qdrant_client()
    assert first is second
    mock_cls.assert_called_once()


def test_get_voyage_client_is_not_exposed():
    assert not hasattr(mod, "get_voyage_client")


def test_get_sparse_model_returns_same_instance_on_repeated_calls(reset_client_singletons):
    with patch.object(mod, "SparseTextEmbedding") as mock_cls:
        first = mod.get_sparse_model()
        second = mod.get_sparse_model()
    assert first is second
    mock_cls.assert_called_once_with(model_name=mod.SPARSE_MODEL)


def test_module_constants():
    assert mod.COLLECTION_NAME == "sec_filings"
    assert mod.SPARSE_MODEL == "Qdrant/bm25"
    assert mod.QDRANT_URL == "http://localhost:6333"
