"""Tests for date-question support: parse dates as UNIX timestamps, build a PCHIP
CDF over timestamps, aggregate via the numeric path, and render readable dates."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from forecasting_tools import DatePercentile, DateQuestion, NumericDistribution
from forecasting_tools.data_models.numeric_report import NumericReport, Percentile

from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.forecaster_runners import run_date_forecast
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import numeric_view_of_date_question, to_utc_timestamp

_PCTS = [0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975]


def _make_date_question() -> DateQuestion:
    return DateQuestion(
        question_text="When will X happen?",
        id_of_question=1,
        id_of_post=1,
        page_url="u",
        background_info="",
        resolution_criteria="",
        fine_print="",
        lower_bound=datetime(2030, 1, 1, tzinfo=timezone.utc),
        upper_bound=datetime(2060, 1, 1, tzinfo=timezone.utc),
        open_lower_bound=False,
        open_upper_bound=False,
        open_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        scheduled_resolution_time=datetime(2060, 1, 1, tzinfo=timezone.utc),
    )


def _date_percentiles(years: list[int]) -> list[DatePercentile]:
    return [DatePercentile(percentile=p, value=datetime(y, 1, 1, tzinfo=timezone.utc)) for p, y in zip(_PCTS, years)]


def _build_date_prediction(question: DateQuestion, years: list[int]) -> NumericDistribution:
    percentile_list = [
        Percentile(percentile=d.percentile, value=to_utc_timestamp(d.value)) for d in _date_percentiles(years)
    ]
    ts_question = numeric_view_of_date_question(question)
    sanitized, zero_point = sanitize_percentiles(percentile_list, ts_question)
    return build_numeric_distribution(sanitized, ts_question, zero_point)


class DummyLLM:
    def __init__(self, reasoning: str = "reasoning"):
        self._reasoning = reasoning
        self.model = "dummy-test-model"

    async def invoke(self, prompt: str) -> str:
        return self._reasoning


def test_to_utc_timestamp_treats_naive_as_utc():
    naive = datetime(2040, 1, 1)
    aware = datetime(2040, 1, 1, tzinfo=timezone.utc)
    assert to_utc_timestamp(naive) == to_utc_timestamp(aware) == aware.timestamp()


def test_numeric_view_has_timestamp_bounds_and_is_date():
    q = _make_date_question()
    view = numeric_view_of_date_question(q)
    assert view.lower_bound == q.lower_bound.timestamp()
    assert view.upper_bound == q.upper_bound.timestamp()
    assert isinstance(view.lower_bound, float)
    assert view.is_date is True
    assert view.cdf_size == q.cdf_size


@pytest.mark.asyncio
async def test_run_date_forecast_builds_pchip_over_timestamps():
    q = _make_date_question()
    parsed = _date_percentiles([2031, 2032, 2033, 2035, 2037, 2038, 2040, 2043, 2046, 2049, 2052])

    with patch("metaculus_bot.forecaster_runners.structure_output", return_value=parsed):
        result = await run_date_forecast(q, "research", DummyLLM(), DummyLLM())

    pred = result.prediction_value
    assert isinstance(pred, NumericDistribution)
    assert pred.is_date is True
    assert len(pred.get_cdf()) == 201
    # Declared percentile values are timestamps within the (timestamp) bounds.
    values = [p.value for p in pred.declared_percentiles]
    assert values == sorted(values)
    assert q.lower_bound.timestamp() <= values[0]
    assert values[-1] <= q.upper_bound.timestamp()


@pytest.mark.asyncio
async def test_run_date_forecast_renders_readable_dates_not_timestamps():
    q = _make_date_question()
    parsed = _date_percentiles([2031, 2032, 2033, 2035, 2037, 2038, 2040, 2043, 2046, 2049, 2052])

    with patch("metaculus_bot.forecaster_runners.structure_output", return_value=parsed):
        result = await run_date_forecast(q, "research", DummyLLM(), DummyLLM())

    readable = NumericReport.make_readable_prediction(result.prediction_value)
    # is_date=True makes the library format CDF values as dates, not raw 1.7e9 timestamps.
    assert "20" in readable  # contains year-like dates
    assert "e+0" not in readable.lower()
    assert "chance of value below 2" in readable


def test_date_predictions_aggregate_via_combine_by_type():
    q = _make_date_question()
    p1 = _build_date_prediction(q, [2031, 2032, 2033, 2035, 2037, 2038, 2040, 2043, 2046, 2049, 2052])
    p2 = _build_date_prediction(q, [2032, 2033, 2034, 2036, 2038, 2039, 2041, 2044, 2047, 2050, 2053])

    pipe = AggregationPipeline(strategy=AggregationStrategy.MEDIAN, parser_llm=None, stacker_llm=None)
    combined = pipe._combine_by_type([p1, p2], q, AggregationStrategy.MEDIAN, error_context="date test")

    assert isinstance(combined, NumericDistribution)
    assert combined.is_date is True
    assert len(combined.get_cdf()) == 201
