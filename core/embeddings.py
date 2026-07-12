import os

EMBEDDING_DIM = 1536  # must match the Qdrant "dense" vector size; see ingestion/index_to_qdrant.py
MODEL = os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"

_clients: dict[str, object] = {}


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Embed texts with the OpenAI embeddings API.

    OpenAI embeddings are symmetric, so the caller does not need to pass an
    input type. The model can be overridden via the EMBEDDING_MODEL env var.
    """
    client = _get_openai_client()
    response = client.embeddings.create(model=MODEL, input=texts, dimensions=EMBEDDING_DIM)
    return [item.embedding for item in response.data]
