import os

MODEL = os.environ.get("LLM_MODEL") or "gpt-4o"

_clients: dict[str, object] = {}


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def complete(prompt: str, max_tokens: int = 1024) -> str:
    """Call the OpenAI API and return its raw text response.

    Model can be overridden via the LLM_MODEL env var.
    """
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
