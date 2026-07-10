from unittest.mock import patch

import pytest

import core.clients as mod


@pytest.fixture
def reset_client_singletons():
    mod._qdrant_client = None
    mod._voyage_client = None
    mod._sparse_model = None
    yield
    mod._qdrant_client = None
    mod._voyage_client = None
    mod._sparse_model = None


def test_get_qdrant_client_returns_same_instance_on_repeated_calls(reset_client_singletons):
    with patch.object(mod, "QdrantClient") as mock_cls:
        first = mod.get_qdrant_client()
        second = mod.get_qdrant_client()
    assert first is second
    mock_cls.assert_called_once()


def test_get_voyage_client_uses_api_key_from_env(reset_client_singletons, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key-123")
    with patch.object(mod, "voyageai") as mock_voyageai:
        mod.get_voyage_client()
    mock_voyageai.Client.assert_called_once_with(api_key="test-key-123")


def test_get_sparse_model_returns_same_instance_on_repeated_calls(reset_client_singletons):
    with patch.object(mod, "SparseTextEmbedding") as mock_cls:
        first = mod.get_sparse_model()
        second = mod.get_sparse_model()
    assert first is second
    mock_cls.assert_called_once_with(model_name=mod.SPARSE_MODEL)


def test_module_constants():
    assert mod.COLLECTION_NAME == "sec_filings"
    assert mod.DENSE_MODEL == "voyage-finance-2"
    assert mod.SPARSE_MODEL == "Qdrant/bm25"
    assert mod.QDRANT_URL == "http://localhost:6333"
