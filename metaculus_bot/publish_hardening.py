"""Wall-clock hardening for the synchronous Metaculus publish path.

Stock forecasting-tools makes the four publish POSTs against
``https://www.metaculus.com/api/`` via blocking ``requests.post`` calls (see
``forecasting_tools/helpers/metaculus_client.py``; ``MetaculusApi`` is now a
deprecated shim delegating to ``MetaculusClient``):

- ``MetaculusClient.post_binary_question_prediction``      -> ``requests.post``
- ``MetaculusClient.post_numeric_question_prediction``     -> ``requests.post``
- ``MetaculusClient.post_multiple_choice_question_prediction`` -> ``requests.post``
- ``MetaculusClient.post_question_comment``                -> ``requests.post``

If the Metaculus API hangs mid-tournament, those calls block the asyncio event
loop (they're invoked synchronously from inside the `async def
publish_report_to_metaculus` methods on each report type) and block every other
Q in the batch from publishing. ``apply_publish_hardening()`` monkey-patches
each of those four instance methods at startup with two layers of defense:

1. Request-side socket timeout (primary): for the duration of the wrapped
   call, ``requests.post`` on the forecasting-tools module is patched to
   inject ``timeout=PUBLISH_POST_TIMEOUT`` if the caller didn't supply one.
   This makes the underlying socket actually close when the server stalls,
   so the worker thread terminates instead of leaking.

2. ``concurrent.futures.Future.result(timeout=...)`` cap (belt-and-suspenders):
   covers pathological cases where a request might somehow ignore the socket
   timeout (e.g. unbounded DNS resolution before connect). Note that
   ``Future.cancel()`` does NOT interrupt a running thread; without layer (1)
   the worker would keep running until socket close, risking duplicate
   publishes on retry. Layer (1) makes that scenario unreachable.

Each wrapper retries once on timeout / connection error.

We use ``concurrent.futures.ThreadPoolExecutor`` (rather than asyncio.to_thread)
because the patched callsite remains synchronous — calling code is
``MetaculusApi.post_*(...)`` without await — so we can't return a coroutine.

The wrappers are attached as plain functions, so they rebind as instance
methods on ``MetaculusClient`` (``self`` flows through as the wrapper's first
positional arg), preserving the original calling convention.

Idempotent: calling ``apply_publish_hardening()`` more than once is a no-op
(checked via a sentinel attribute on ``MetaculusClient``).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import logging
from typing import Any, Callable, Iterator

import requests
from forecasting_tools.helpers import metaculus_client as _ft_metaculus_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import PUBLISH_POST_RETRIES, PUBLISH_POST_TIMEOUT

assert PUBLISH_POST_RETRIES >= 0, "PUBLISH_POST_RETRIES must be non-negative"

logger = logging.getLogger(__name__)

_SENTINEL = "_publish_hardening_applied"

# Method names to patch. Each is a synchronous instance method on MetaculusClient
# that wraps a single requests.post call.
_PATCHED_METHODS: tuple[str, ...] = (
    "post_binary_question_prediction",
    "post_numeric_question_prediction",
    "post_multiple_choice_question_prediction",
    "post_question_comment",
)

# Single shared executor across the four wrappers. Publish calls are infrequent
# and serialized within a single Q's publish_report_to_metaculus().
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="publish-hardening")
    return _executor


@contextlib.contextmanager
def _inject_socket_timeout(timeout_s: float) -> Iterator[None]:
    """Patch ``forecasting_tools.helpers.metaculus_client.requests.post`` to inject ``timeout=`` if absent."""
    original_post = _ft_metaculus_client.requests.post

    @functools.wraps(original_post)
    def post_with_timeout(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout_s)
        return original_post(*args, **kwargs)

    _ft_metaculus_client.requests.post = post_with_timeout
    try:
        yield
    finally:
        _ft_metaculus_client.requests.post = original_post


def _wrap_with_timeout_retry(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a sync wrapper that runs ``original`` on a worker thread with timeout + retry.

    The wrapper layers two timeout mechanisms:
    - Request-side: ``requests.post`` on forecasting-tools is monkey-patched to
      inject ``timeout=PUBLISH_POST_TIMEOUT``, bounding the underlying socket.
    - Caller-side: ``Future.result(timeout=...)`` provides a final ceiling.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        executor = _get_executor()
        attempts = PUBLISH_POST_RETRIES + 1  # read at call time so tests' monkeypatch is honored

        def _run_with_socket_timeout() -> Any:
            with _inject_socket_timeout(PUBLISH_POST_TIMEOUT):
                return original(*args, **kwargs)

        last_exc: BaseException = RuntimeError(f"PUBLISH_HARDENING: {method_name} loop exited without running")
        for attempt in range(1, attempts + 1):
            future = executor.submit(_run_with_socket_timeout)
            try:
                return future.result(timeout=PUBLISH_POST_TIMEOUT)
            except concurrent.futures.TimeoutError as exc:
                last_exc = exc
                future.cancel()
                logger.warning(
                    "PUBLISH_HARDENING: %s attempt %d/%d timed out after %ds",
                    method_name,
                    attempt,
                    attempts,
                    PUBLISH_POST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "PUBLISH_HARDENING: %s attempt %d/%d failed (%s: %s)",
                    method_name,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
        raise last_exc

    return wrapper


def apply_publish_hardening() -> None:
    """Patch ``MetaculusClient.post_*`` to add timeout + retry. Idempotent."""
    if getattr(MetaculusClient, _SENTINEL, False):
        return

    for method_name in _PATCHED_METHODS:
        # Resolve the raw function (unwrapping classmethod/staticmethod if present)
        # and reattach the wrapper as a plain instance method.
        descriptor = MetaculusClient.__dict__[method_name]
        if isinstance(descriptor, (classmethod, staticmethod)):
            original_func = descriptor.__func__
        else:
            original_func = descriptor

        wrapped = _wrap_with_timeout_retry(method_name, original_func)
        setattr(MetaculusClient, method_name, wrapped)

    setattr(MetaculusClient, _SENTINEL, True)
    logger.info(
        "Publish hardening applied: %d MetaculusClient.post_* methods wrapped with %ds timeout + %d retry",
        len(_PATCHED_METHODS),
        PUBLISH_POST_TIMEOUT,
        PUBLISH_POST_RETRIES,
    )
