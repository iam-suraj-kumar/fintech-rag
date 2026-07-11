import os

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    "local": "llama3.1",
}

_OPENAI_COMPATIBLE = {"openai", "local"}

_clients: dict[str, object] = {}


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic

        _clients["anthropic"] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _clients["anthropic"]


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def _get_local_client():
    if "local" not in _clients:
        import openai

        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        _clients["local"] = openai.OpenAI(api_key="not-needed", base_url=base_url)
    return _clients["local"]


_CLIENT_FACTORIES = {
    "anthropic": _get_anthropic_client,
    "openai": _get_openai_client,
    "local": _get_local_client,
}


def complete(prompt: str, max_tokens: int = 1024) -> str:
    """Call the configured LLM provider and return its raw text response.

    Provider is chosen via the LLM_PROVIDER env var (default: "anthropic").
    "local" targets any OpenAI-compatible local server (Ollama, LM Studio,
    vLLM) via LOCAL_LLM_BASE_URL, no API key required. Model can be
    overridden via LLM_MODEL; otherwise each provider's default in
    DEFAULT_MODELS is used. Adding a new provider means adding one entry to
    DEFAULT_MODELS/_CLIENT_FACTORIES and, if it isn't OpenAI-wire-compatible,
    one branch below.
    """
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider not in _CLIENT_FACTORIES:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. Supported: {sorted(_CLIENT_FACTORIES)}"
        )
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider]
    client = _CLIENT_FACTORIES[provider]()

    if provider in _OPENAI_COMPATIBLE:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
