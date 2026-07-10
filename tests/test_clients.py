from unittest.mock import patch

import core.clients as mod


def test_get_qdrant_client_returns_same_instance_on_repeated_calls():
    mod._qdrant_client = None
    with patch.object(mod, "QdrantClient") as mock_cls:
        first = mod.get_qdrant_client()
        second = mod.get_qdrant_client()
    assert first is second
    mock_cls.assert_called_once()


def test_get_voyage_client_uses_api_key_from_env(monkeypatch):
    mod._voyage_client = None
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key-123")
    with patch.object(mod, "voyageai") as mock_voyageai:
        mod.get_voyage_client()
    mock_voyageai.Client.assert_called_once_with(api_key="test-key-123")
