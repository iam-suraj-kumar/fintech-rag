import os
from dataclasses import dataclass

from core.retry import with_retry

MODEL = os.environ.get("LLM_MODEL") or "gpt-4o"

# USD per 1M tokens; gpt-4o pricing used as the default fallback for unlisted models.
_PRICE_PER_1M_INPUT = {"gpt-4o": 2.50, "gpt-4o-mini": 0.15}
_PRICE_PER_1M_OUTPUT = {"gpt-4o": 10.00, "gpt-4o-mini": 0.60}

_clients: dict[str, object] = {}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _get_openai_client():
    if "openai" not in _clients:
        import openai

        _clients["openai"] = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


def complete_with_usage(prompt: str, max_tokens: int = 1024) -> LLMResponse:
    """Call the OpenAI API and return the response text plus token usage and estimated cost.

    Retries transient API errors with exponential backoff. Model can be overridden
    via the LLM_MODEL env var.
    """
    client = _get_openai_client()
    response = with_retry(
        client.chat.completions.create,
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    cost_usd = (
        input_tokens / 1_000_000 * _PRICE_PER_1M_INPUT.get(MODEL, _PRICE_PER_1M_INPUT["gpt-4o"])
        + output_tokens / 1_000_000 * _PRICE_PER_1M_OUTPUT.get(MODEL, _PRICE_PER_1M_OUTPUT["gpt-4o"])
    )
    return LLMResponse(
        text=response.choices[0].message.content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )


def complete(prompt: str, max_tokens: int = 1024) -> str:
    """Call the OpenAI API and return its raw text response.

    Model can be overridden via the LLM_MODEL env var.
    """
    return complete_with_usage(prompt, max_tokens=max_tokens).text
