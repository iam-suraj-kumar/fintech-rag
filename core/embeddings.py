import os

from core.retry import with_retry

EMBEDDING_DIM = 1536  # must match the Qdrant "dense" vector size; see ingestion/index_to_qdrant.py
MODEL = os.environ.get("EMBEDDING_MODEL") or "@demo-fintech/text-embedding-3-small"

_clients: dict[str, object] = {}


def _get_portkey_client():
    if "portkey" not in _clients:
        from portkey_ai import Portkey

        _clients["portkey"] = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    return _clients["portkey"]


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed texts via Portkey.

    These embeddings are symmetric, so the caller does not need to pass an
    input type. The model can be overridden via the EMBEDDING_MODEL env var.
    Retries transient API errors with exponential backoff.
    """
    client = _get_portkey_client()
    response = with_retry(
        client.embeddings.create,
        model=MODEL,
        input=texts,
        dimensions=EMBEDDING_DIM,
        encoding_format="float",
    )
    return [item.embedding for item in response.data]
