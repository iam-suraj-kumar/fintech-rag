import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[..., T], *args, max_retries: int = 3, backoff_base: float = 1.0, **kwargs
) -> T:
    """Call fn(*args, **kwargs), retrying on transient OpenAI API errors with exponential backoff."""
    import openai

    retryable = (
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except retryable as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base * (2**attempt))
    assert last_exc is not None
    raise last_exc
