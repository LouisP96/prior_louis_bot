"""Gemini grounded search research provider.

Uses the `google-genai` SDK directly (NOT via OpenRouter) so we get real
first-party Google Search grounding rather than OpenRouter's Exa-backed web
plugin. This adds a genuinely distinct search index to the ensemble — the
Metaculus Fall 2025 writeup identified research breadth as the single
strongest predictor of winning bots.

Mirrors `_native_search_provider` in `research_providers.py` for consistency.
"""

import asyncio
import functools
import logging
import os
import random
import time
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion
from google import genai
from google.genai import types as genai_types

from metaculus_bot.constants import (
    GEMINI_SEARCH_DEFAULT_MODEL,
    GEMINI_SEARCH_FALLBACK_MODEL,
    GEMINI_SEARCH_MAX_CONCURRENCY,
    GEMINI_SEARCH_MODEL_ENV,
    GEMINI_SEARCH_TIMEOUT,
    GEMINI_SEARCH_TOTAL_BUDGET,
    GOOGLE_API_KEY_ENV,
)
from metaculus_bot.prompts import web_research_prompt
from metaculus_bot.research.providers import ResearchCallable

logger = logging.getLogger(__name__)

__all__ = ["build_gemini_client", "gemini_search_provider", "invoke_gemini_grounded"]

# Header-only initializer for the sources list. Checking `len(sources_lines) > _SOURCES_HEADER_LEN`
# against this named constant keeps the sources-present gate tied to the init block.
_SOURCES_HEADER_LEN = 3

# Process-wide cap on concurrent grounded-search calls. The provider + gap-fill
# otherwise burst ~6 at once and trip Google's per-model concurrency quota
# (surfaced as 503 "high demand"). Lazily created so it binds to the running
# event loop rather than import-time. See GEMINI_SEARCH_MAX_CONCURRENCY.
_grounded_semaphore: asyncio.Semaphore | None = None

# Retry config for the 503 throttle. Single calls always succeed, so a couple of
# short backed-off retries mop up any stragglers the semaphore doesn't fully
# serialize away. Only 503/UNAVAILABLE is retried; other errors propagate.
_GROUNDED_RETRY_ATTEMPTS = 3
_GROUNDED_RETRY_BASE_DELAY = 4.0
# Uniform 0..N seconds added to each backoff so parallel gap-fill searches that
# 503 in the same wave don't retry in lockstep.
_GROUNDED_RETRY_JITTER = 2.0


def _get_grounded_semaphore() -> asyncio.Semaphore:
    """Return the process-wide grounded-call semaphore, creating it on first use."""
    global _grounded_semaphore
    if _grounded_semaphore is None:
        _grounded_semaphore = asyncio.Semaphore(GEMINI_SEARCH_MAX_CONCURRENCY)
    return _grounded_semaphore


def _is_throttle_503(exc: Exception) -> bool:
    """Whether an exception is Google's 503 UNAVAILABLE concurrency/demand throttle."""
    msg = str(exc).lower()
    return "503" in msg or "unavailable" in msg


@functools.lru_cache(maxsize=1)
def _cached_client_for_key(api_key: str) -> genai.Client:
    """Process-global cached genai.Client keyed on API key.

    SDK clients are designed to be long-lived; keeping one across a backtest
    lets TLS connections and HTTP/2 multiplexing be reused across the ~thousands
    of calls the Gemini provider + gap-fill make per round. Keyed on api_key so
    a rotated key (rare) produces a fresh client.
    """
    return genai.Client(api_key=api_key)


def build_gemini_client() -> genai.Client:
    """Return the cached google-genai Client for the operator's personal Gemini key.

    Reads GOOGLE_API_KEY (the operator's personal Google AI Studio key — in CI
    populated from ``secrets.GEMINI_API_KEY``). There is no Metaculus-donated
    Gemini key on the google-genai side; the donated path only exists for
    OpenRouter-routed Gemini models. Raises ValueError if the key is missing
    so misconfiguration is loud.
    """
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{GOOGLE_API_KEY_ENV} must be set to use the Gemini search provider")
    return _cached_client_for_key(api_key)


def _resolve_model(model_slug: str | None) -> str:
    return model_slug or os.getenv(GEMINI_SEARCH_MODEL_ENV, GEMINI_SEARCH_DEFAULT_MODEL)


def _format_grounded_response(response: genai_types.GenerateContentResponse) -> str:
    """Stitch response text with inline citations from grounding metadata.

    Output format:
        <response text>

        ### Sources
        [1] <title> — <uri>
        [2] <title> — <uri>
        ...

    Inline citation markers are inserted per-segment using
    grounding_metadata.grounding_supports, iterating in reverse end_index order
    so index offsets stay valid while we mutate the string. Falls back to a
    plain-text + sources-list if supports are missing.
    """
    text = response.text or ""
    if not text:
        return ""

    candidates = response.candidates
    if not candidates:
        return text

    metadata = candidates[0].grounding_metadata
    if metadata is None:
        # Grounding didn't fire — return plain text.
        return text

    chunks = metadata.grounding_chunks
    supports = metadata.grounding_supports

    if not chunks:
        # Grounding enabled but model didn't cite anything — return the text as-is.
        return text

    # Insert inline citation markers based on supports, iterating right-to-left
    # so earlier offsets remain stable as we mutate the string.
    annotated = text
    if supports:
        try:
            sorted_supports = sorted(
                supports,
                key=lambda s: s.segment.end_index if s.segment and s.segment.end_index is not None else 0,
                reverse=True,
            )
            for support in sorted_supports:
                segment = support.segment
                if segment is None or segment.end_index is None:
                    continue
                chunk_indices = support.grounding_chunk_indices
                if not chunk_indices:
                    continue
                # Convert to 1-indexed markers, dedup, sort for readability.
                markers = sorted({int(i) + 1 for i in chunk_indices})
                marker_str = "[" + ", ".join(str(m) for m in markers) + "]"
                end_index = segment.end_index
                annotated = annotated[:end_index] + marker_str + annotated[end_index:]
        except (AttributeError, TypeError, ValueError) as exc:
            # Malformed supports shouldn't kill the whole response.
            logger.warning(f"GeminiSearch: could not splice inline citations ({type(exc).__name__}): {exc}")
            annotated = text

    # Append a Sources section
    sources_lines = ["", "", "### Sources"]
    for idx, chunk in enumerate(chunks, start=1):
        web = chunk.web
        if web and web.uri:
            if web.title:
                sources_lines.append(f"[{idx}] {web.title} — {web.uri}")
            else:
                sources_lines.append(f"[{idx}] {web.uri}")

    return annotated + "\n".join(sources_lines) if len(sources_lines) > _SOURCES_HEADER_LEN else annotated


async def _generate_grounded_with_retries(
    client: genai.Client,
    model: str,
    prompt: str,
    config: genai_types.GenerateContentConfig,
    semaphore: asyncio.Semaphore,
    *,
    deadline: float,
    max_attempts: int,
) -> genai_types.GenerateContentResponse:
    """Run one model's grounded call with 503-retry under a wall-clock deadline.

    Each attempt's timeout is clamped to the remaining budget; retrying stops
    once the budget (or ``max_attempts``) is spent. Raises ``asyncio.TimeoutError``
    on a per-attempt timeout, or the last 503 ``ServerError`` when exhausted.
    Non-503 errors propagate immediately.
    """
    last_throttle: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(f"GeminiSearch: {model} time budget exhausted before attempt {attempt}")
            break
        attempt_timeout = min(GEMINI_SEARCH_TIMEOUT, remaining)
        try:
            # Hold the concurrency slot only for the in-flight request, not the
            # backoff sleep, so a throttled call doesn't block a healthy one.
            async with semaphore:
                return await asyncio.wait_for(
                    client.aio.models.generate_content(model=model, contents=prompt, config=config),
                    timeout=attempt_timeout,
                )
        except asyncio.TimeoutError:
            logger.warning(f"GeminiSearch: {model} timed out after {attempt_timeout:.0f}s")
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised below unless a retryable 503
            if not _is_throttle_503(exc):
                raise
            last_throttle = exc
            if attempt >= max_attempts:
                break
            delay = _GROUNDED_RETRY_BASE_DELAY * attempt + random.uniform(0, _GROUNDED_RETRY_JITTER)
            if time.monotonic() + delay >= deadline:
                logger.warning(f"GeminiSearch: {model} time budget exhausted; not retrying after attempt {attempt}")
                break
            logger.warning(
                f"GeminiSearch: {model} 503 throttle (attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    if last_throttle is not None:
        raise last_throttle
    raise asyncio.TimeoutError(f"GeminiSearch: {model} exhausted time budget before any attempt completed")


async def invoke_gemini_grounded(
    prompt: str,
    *,
    model_slug: str | None = None,
    include_url_context: bool = True,
) -> str:
    """Invoke Gemini with Google Search grounding and return formatted text.

    Shared by the first-pass Gemini provider and the second-pass gap-fill
    searches. Enables the URL context tool alongside Google Search by default
    so the model can directly read specific URLs (e.g., resolution sources
    named in question fine print).

    On sustained 503 throttling the primary model retries under a wall-clock
    budget, then falls back once to GEMINI_SEARCH_FALLBACK_MODEL. Raises on SDK
    errors — callers decide whether to fail hard or soft.
    """
    client = build_gemini_client()
    model = _resolve_model(model_slug)

    tools: list[Any] = [{"google_search": {}}]
    if include_url_context:
        tools.append({"url_context": {}})

    config = genai_types.GenerateContentConfig(tools=tools)
    semaphore = _get_grounded_semaphore()

    logger.info(f"GeminiSearch: calling {model} with grounding")
    try:
        response = await _generate_grounded_with_retries(
            client,
            model,
            prompt,
            config,
            semaphore,
            deadline=time.monotonic() + GEMINI_SEARCH_TOTAL_BUDGET,
            max_attempts=_GROUNDED_RETRY_ATTEMPTS,
        )
        served_by = model
    except Exception as exc:
        if not (_is_throttle_503(exc) and GEMINI_SEARCH_FALLBACK_MODEL and model != GEMINI_SEARCH_FALLBACK_MODEL):
            raise
        logger.warning(f"GeminiSearch: {model} exhausted 503 retries; falling back to {GEMINI_SEARCH_FALLBACK_MODEL}")
        response = await _generate_grounded_with_retries(
            client,
            GEMINI_SEARCH_FALLBACK_MODEL,
            prompt,
            config,
            semaphore,
            deadline=time.monotonic() + GEMINI_SEARCH_TIMEOUT,
            max_attempts=1,
        )
        served_by = GEMINI_SEARCH_FALLBACK_MODEL

    formatted = _format_grounded_response(response)
    n_chunks = 0
    candidates = response.candidates
    if candidates:
        metadata = candidates[0].grounding_metadata
        if metadata is not None and metadata.grounding_chunks:
            n_chunks = len(metadata.grounding_chunks)
    logger.info(f"GeminiSearch: got {len(formatted)} chars, {n_chunks} grounding chunks from {served_by}")
    return formatted


def gemini_search_provider(
    model_slug: str | None = None,
    is_benchmarking: bool = False,
) -> ResearchCallable:
    """Research provider using Gemini with Google Search grounding.

    Mirrors the `_native_search_provider` contract (`MetaculusQuestion -> str`).
    """

    async def _fetch(question: MetaculusQuestion) -> str:  # noqa: D401
        prompt = web_research_prompt(
            question.question_text,
            is_benchmarking=is_benchmarking,
            citation_style="auto_annotated",
            allow_resolution_source_reading=True,
        )
        return await invoke_gemini_grounded(prompt, model_slug=model_slug)

    return _fetch
