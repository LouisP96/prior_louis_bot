"""Tests for the Gemini grounded search research provider.

These tests mock the google-genai SDK at the module level; no live API calls.
Patterns mirror ``tests/test_native_search_provider.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract. Tests only care about question_text on this path."""
    q = MagicMock()
    q.question_text = text
    return q


# ---------------------------------------------------------------------------
# Canned response helpers (for grounding metadata tests)
# ---------------------------------------------------------------------------


class CannedWebChunk:
    def __init__(self, uri: str, title: str | None) -> None:
        self.web = SimpleNamespace(uri=uri, title=title)


class CannedSegment:
    def __init__(self, end_index: int, text: str) -> None:
        self.end_index = end_index
        self.text = text


class CannedSupport:
    def __init__(self, seg: CannedSegment, indices: list[int]) -> None:
        self.segment = seg
        self.grounding_chunk_indices = indices


def _make_response(
    text: str,
    chunks: list[CannedWebChunk] | None = None,
    supports: list[CannedSupport] | None = None,
) -> SimpleNamespace:
    metadata = SimpleNamespace(
        grounding_chunks=chunks,
        grounding_supports=supports,
    )
    candidate = SimpleNamespace(grounding_metadata=metadata)
    return SimpleNamespace(text=text, candidates=[candidate])


def _make_client_with_response(response: object) -> MagicMock:
    """Build a MagicMock Client whose aio.models.generate_content awaits to ``response``."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# build_gemini_client
# ---------------------------------------------------------------------------


def test_builder_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing GOOGLE_API_KEY should raise. The grounded-search side has no
    donated/shared key path — Google AI Studio doesn't offer one — so this is
    the only key gate to test here.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    from metaculus_bot.research.gemini_search import build_gemini_client

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_gemini_client()


# ---------------------------------------------------------------------------
# gemini_search_provider: model selection & tool wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_uses_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GEMINI_SEARCH_MODEL env set, default slug is used."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    response = _make_response("some research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    assert fake_client.aio.models.generate_content.await_count == 1
    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-3-flash-preview"
    # The question_text must actually reach the SDK (guard against broken f-string interpolation).
    assert "Will X happen?" in call_kwargs["contents"]


@pytest.mark.asyncio
async def test_provider_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMINI_SEARCH_MODEL env var overrides the default."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_provider_uses_explicit_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``model_slug=`` param takes precedence over env var."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "gemini-2.5-flash")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(model_slug="gemini-explicit-override")
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-explicit-override"


@pytest.mark.asyncio
async def test_provider_attaches_google_search_and_url_context_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate_content config must include both the GoogleSearch and url_context tools."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider()
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    config = call_kwargs["config"]
    tools = list(config.tools)
    assert len(tools) == 2
    # The SDK normalizes the {"google_search": {}} / {"url_context": {}} dicts into
    # pydantic Tool objects with the corresponding attribute populated.
    google_search_configured = any(getattr(t, "google_search", None) is not None for t in tools)
    url_context_configured = any(getattr(t, "url_context", None) is not None for t in tools)
    assert google_search_configured
    assert url_context_configured


# ---------------------------------------------------------------------------
# gemini_search_provider: prompt content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benchmarking_carve_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=True: prompt contains 'benchmarking run' and NOT 'Prediction market'."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=True)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "benchmarking run" in prompt
    assert "Prediction market" not in prompt


@pytest.mark.asyncio
async def test_non_benchmarking_includes_prediction_markets(monkeypatch: pytest.MonkeyPatch) -> None:
    """is_benchmarking=False: prompt includes 'Prediction market' line."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = _make_response("research text")
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import gemini_search_provider

        provider = gemini_search_provider(is_benchmarking=False)
        await provider(_make_q("Will X happen?"))

    call_kwargs = fake_client.aio.models.generate_content.await_args.kwargs
    prompt = call_kwargs["contents"]
    assert "Prediction market" in prompt
    assert "benchmarking run" not in prompt


# ---------------------------------------------------------------------------
# _format_grounded_response behavior (via invoke_gemini_grounded)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_citations_appended_to_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response with grounding chunks ends with a '### Sources' block listing both URIs/titles."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [
        CannedWebChunk(uri="https://example.com/1", title="Example One"),
        CannedWebChunk(uri="https://example.com/2", title="Example Two"),
    ]
    response = _make_response("body text", chunks=chunks, supports=None)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "### Sources" in out
    assert "https://example.com/1" in out
    assert "Example One" in out
    assert "https://example.com/2" in out
    assert "Example Two" in out
    # Sources comes after body text
    assert out.index("body text") < out.index("### Sources")


@pytest.mark.asyncio
async def test_inline_citation_markers_inserted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A support mapping a segment end_index to chunk 0 produces a ``[1]`` marker after that offset.

    With multiple supports, reverse-iteration must preserve earlier offsets.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    text = "Alpha fact. Beta fact."
    end_alpha = text.index("Alpha fact.") + len("Alpha fact.")
    end_beta = text.index("Beta fact.") + len("Beta fact.")

    chunks = [
        CannedWebChunk(uri="https://example.com/a", title="A"),
        CannedWebChunk(uri="https://example.com/b", title="B"),
    ]
    supports = [
        CannedSupport(seg=CannedSegment(end_index=end_alpha, text="Alpha fact."), indices=[0]),
        CannedSupport(seg=CannedSegment(end_index=end_beta, text="Beta fact."), indices=[1]),
    ]
    response = _make_response(text, chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "Alpha fact.[1]" in out
    assert "Beta fact.[2]" in out
    # The sources block must also be appended whenever chunks are present — this is
    # the common production path (chunks + supports together), not chunks-only.
    assert "### Sources" in out
    assert "https://example.com/a" in out


@pytest.mark.asyncio
async def test_missing_grounding_metadata_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response with no grounding metadata still returns its plain text."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    response = SimpleNamespace(text="plain response body", candidates=[])
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == "plain response body"


@pytest.mark.asyncio
async def test_empty_response_text_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """response.text == '' short-circuits to empty, even when candidates + grounding metadata are populated.

    Isolating the ``not text`` guard: if that early-return regresses (e.g. moved below the candidates
    check), the chunks/supports path would produce a non-empty "\\n\\n### Sources\\n[1] ..." string and
    this assertion would fail.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    chunks = [CannedWebChunk(uri="https://example.com/1", title="Example One")]
    supports = [CannedSupport(seg=CannedSegment(end_index=0, text=""), indices=[0])]
    response = _make_response("", chunks=chunks, supports=supports)
    fake_client = _make_client_with_response(response)

    with patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == ""


# ---------------------------------------------------------------------------
# 503 throttle handling: retry-under-budget + fallback model
# ---------------------------------------------------------------------------


def _throttle_503() -> Exception:
    """A stand-in for google.genai's ServerError that _is_throttle_503 recognizes."""
    return Exception("503 UNAVAILABLE. This model is currently experiencing high demand.")


def _make_dispatching_client(handler) -> MagicMock:
    """Client whose generate_content runs ``handler(**kwargs)`` (raise or return) per call."""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock(side_effect=handler)
    return client


@pytest.mark.asyncio
async def test_retries_on_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single 503 is retried; the next attempt's success is returned."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    response = _make_response("recovered body", chunks=[CannedWebChunk("https://x/1", "X")])
    fake_client = _make_client_with_response(response)
    fake_client.aio.models.generate_content = AsyncMock(side_effect=[_throttle_503(), response])

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert "recovered body" in out
    assert fake_client.aio.models.generate_content.await_count == 2
    # Backoff was applied between the failed attempt and the retry.
    assert sleep_mock.await_count == 1
    # Stayed on the primary model — no fallback needed.
    models_called = [c.kwargs["model"] for c in fake_client.aio.models.generate_content.await_args_list]
    assert models_called == ["gemini-3-flash-preview", "gemini-3-flash-preview"]


@pytest.mark.asyncio
async def test_falls_back_to_secondary_model_on_503_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary model 503s on every attempt; the GA fallback model is tried once and wins."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    fallback_response = _make_response("fallback body")
    calls: list[str] = []

    def handler(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini-3-flash-preview":
            raise _throttle_503()
        return fallback_response

    fake_client = _make_dispatching_client(handler)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.asyncio.sleep", new=AsyncMock()),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        out = await invoke_gemini_grounded("prompt")

    assert out == "fallback body"
    # 3 primary attempts (all 503), then 1 fallback attempt.
    assert calls == ["gemini-3-flash-preview"] * 3 + ["gemini-2.5-flash"]


@pytest.mark.asyncio
async def test_fallback_503_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the fallback model also 503s, the error propagates (callers soft-fail)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    def handler(**kwargs):
        raise _throttle_503()

    fake_client = _make_dispatching_client(handler)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.asyncio.sleep", new=AsyncMock()),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        with pytest.raises(Exception, match="503"):
            await invoke_gemini_grounded("prompt")

    # 3 primary attempts + 1 fallback attempt, then give up.
    assert fake_client.aio.models.generate_content.await_count == 4


@pytest.mark.asyncio
async def test_non_503_error_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-throttle error propagates immediately without retry or fallback."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)

    fake_client = _make_dispatching_client(ValueError("boom"))

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.asyncio.sleep", new=AsyncMock()) as sleep_mock,
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        with pytest.raises(ValueError, match="boom"):
            await invoke_gemini_grounded("prompt")

    assert fake_client.aio.models.generate_content.await_count == 1
    assert sleep_mock.await_count == 0


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the wall-clock budget is spent, retrying stops before max_attempts."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.delenv("GEMINI_SEARCH_MODEL", raising=False)
    # Zero the primary budget AND disable the fallback so we isolate the budget gate.
    monkeypatch.setattr("metaculus_bot.research.gemini_search.GEMINI_SEARCH_TOTAL_BUDGET", 0)
    monkeypatch.setattr("metaculus_bot.research.gemini_search.GEMINI_SEARCH_FALLBACK_MODEL", "")

    def handler(**kwargs):
        raise _throttle_503()

    fake_client = _make_dispatching_client(handler)

    with (
        patch("metaculus_bot.research.gemini_search.genai.Client", return_value=fake_client),
        patch("metaculus_bot.research.gemini_search.asyncio.sleep", new=AsyncMock()),
    ):
        from metaculus_bot.research.gemini_search import invoke_gemini_grounded

        with pytest.raises(Exception, match="budget|503"):
            await invoke_gemini_grounded("prompt")

    # Budget <= 0 means not even the first attempt runs.
    assert fake_client.aio.models.generate_content.await_count == 0


# ---------------------------------------------------------------------------
# Parallel provider selection in main.py
# ---------------------------------------------------------------------------


class TestParallelProviderSelectionGemini:
    """Tests for Gemini gating via GEMINI_SEARCH_ENABLED in ``_select_research_providers``."""

    def test_select_research_providers_includes_gemini_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" in provider_names

    def test_select_research_providers_excludes_gemini_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "gemini_search" not in provider_names
