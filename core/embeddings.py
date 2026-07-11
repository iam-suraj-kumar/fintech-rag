import os

from core.clients import DENSE_MODEL, get_voyage_client

EMBEDDING_DIM = 1024  # must match the Qdrant "dense" vector size; see ingestion/index_to_qdrant.py

DEFAULT_MODELS = {
    "voyage": DENSE_MODEL,
    "openai": "text-embedding-3-small",
    "local": "nomic-embed-text",
}

_clients: dict[str, object] = {}


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def _get_local_client():
    if "local" not in _clients:
        import openai

        base_url = os.environ.get("LOCAL_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
        _clients["local"] = openai.OpenAI(api_key="not-needed", base_url=base_url)
    return _clients["local"]


def embed_dense(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed texts with the configured dense embedding provider.

    Provider is chosen via the EMBEDDING_PROVIDER env var (default: "voyage").
    "local" targets any OpenAI-compatible local embedding server (e.g.
    Ollama) via LOCAL_EMBEDDING_BASE_URL, no API key required. input_type is
    "query" or "document" -- Voyage uses this for asymmetric embeddings;
    OpenAI-compatible providers ignore it.

    Switching to a provider whose model outputs a different dimension than
    EMBEDDING_DIM (1024, matching voyage-finance-2) requires recreating the
    Qdrant collection with a matching vector size -- this function does not
    do that for you.
    """
    provider = os.environ.get("EMBEDDING_PROVIDER", "voyage").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: {provider!r}. Supported: {sorted(DEFAULT_MODELS)}"
        )
    model = os.environ.get("EMBEDDING_MODEL") or DEFAULT_MODELS[provider]

    if provider == "voyage":
        return get_voyage_client().embed(texts, model=model, input_type=input_type).embeddings

    client = _get_openai_client() if provider == "openai" else _get_local_client()
    kwargs = {"model": model, "input": texts}
    if provider == "openai":
        kwargs["dimensions"] = EMBEDDING_DIM
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]
