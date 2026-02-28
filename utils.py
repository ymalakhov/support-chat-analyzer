"""
utils.py — Shared utilities for the support chat analyzer.
"""

import asyncio
import logging

from openai import APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError)


async def retry_with_backoff(
    coro_factory,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable: tuple = RETRYABLE_EXCEPTIONS,
):
    """Retry an async callable with exponential backoff on transient errors."""
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt + 1,
                max_retries,
                e,
                delay,
            )
            await asyncio.sleep(delay)
