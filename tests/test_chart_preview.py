from __future__ import annotations

import pandas as pd

from chart_preview import (
    BENCHMARK_NAME,
    BENCHMARK_SIGNAL_OPACITY,
    BENCHMARK_SIGNAL_WIDTH,
    BENCHMARK_TICKER,
    CANDLE_DOWN_COLOR,
    CANDLE_UP_COLOR,
    CANDLE_WIDTH_RATIO,
    INDICATOR_BAR_OPACITY,
    INDICATOR_BAR_WIDTH_RATIO,
    MACD_SIGNAL_COLOR,
    MA_STYLES,
    NEGATIVE_BAR_COLOR,
    OSCILLATOR_LINE_COLOR,
    OSCILLATOR_CENTER,
    OSCILLATOR_LOWER,
    OSCILLATOR_UPPER,
    POSITIVE_BAR_COLOR,
    VOLUME_MA_STYLE,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    comparison_view_indices,
    cycle_view_dates,
    cycle_view_indices,
    expanded_comparison_width,
)


def test_sp500_is_the_single_comparison_benchmark() -> None:
    assert BENCHMARK_TICKER == "^GSPC"
    assert BENCHMARK_NAME == "S&P 500"
    assert BENCHMARK_SIGNAL_OPACITY == 0.78
    assert BENCHMARK_SIGNAL_WIDTH == 2.2


def test_comparison_view_uses_the_stock_visible_date_range() -> None:
    benchmark_index = pd.date_range("2024-01-01", "2024-12-30", freq="W-MON")

    start, end = comparison_view_indices(
        benchmark_index,
        pd.Timestamp("2024-03-06"),
        pd.Timestamp("2024-06-14"),
    )

    assert benchmark_index[start] == pd.Timestamp("2024-03-11")
    assert benchmark_index[end] == pd.Timestamp("2024-06-10")


def test_comparison_expands_to_the_right_without_shrinking_primary_chart() -> None:
    assert expanded_comparison_width(1600, 1440) == 3044


def test_chart_colors_and_oscillator_thresholds_match_the_reference() -> None:
    assert OSCILLATOR_UPPER == 70.0
    assert OSCILLATOR_CENTER == 50.0
    assert OSCILLATOR_LOWER == 30.0
    assert CANDLE_UP_COLOR == "#ef4444"
    assert CANDLE_DOWN_COLOR == "#2563eb"
    assert INDICATOR_BAR_OPACITY == 0.5
    assert POSITIVE_BAR_COLOR == "#f7a2a2"
    assert NEGATIVE_BAR_COLOR == "#92b1f5"
    assert CANDLE_WIDTH_RATIO == 0.76
    assert INDICATOR_BAR_WIDTH_RATIO == 0.84
    assert ZOOM_IN_FACTOR == 0.88
    assert ZOOM_OUT_FACTOR == 1.14
    assert [MA_STYLES[column][0] for column in MA_STYLES] == [
        "#6aa84f",
        "#00d4d8",
        "#1428e8",
        "#9b1010",
        "#f00078",
    ]
    assert [MA_STYLES[column][1] for column in MA_STYLES] == [
        2.0,
        2.4,
        2.8,
        3.2,
        3.6,
    ]
    assert MACD_SIGNAL_COLOR == MA_STYLES["MA_50"][0]
    assert VOLUME_MA_STYLE == MA_STYLES["MA_50"]
    assert OSCILLATOR_LINE_COLOR == CANDLE_UP_COLOR


def test_completed_cycle_view_includes_one_year_before_and_after() -> None:
    index = pd.date_range("2020-01-06", "2025-12-29", freq="W-MON")
    cycle = pd.Series(
        {
            "FirstSignalDate": pd.Timestamp("2022-06-06"),
            "SecondSignalDate": pd.Timestamp("2022-10-03"),
            "ThirdDecisionDate": pd.Timestamp("2023-02-06"),
            "Outcome": "매수 성공",
        }
    )

    start, end = cycle_view_dates(index, cycle)
    start_position, end_position = cycle_view_indices(index, cycle)

    assert start == pd.Timestamp("2021-06-07")
    assert end == pd.Timestamp("2024-02-05")
    assert index[start_position] == start
    assert index[end_position] == end


def test_waiting_cycle_view_extends_through_latest_available_week() -> None:
    index = pd.date_range("2020-01-06", "2025-12-29", freq="W-MON")
    cycle = pd.Series(
        {
            "FirstSignalDate": pd.Timestamp("2021-03-01"),
            "SecondSignalDate": pd.NaT,
            "ThirdDecisionDate": pd.NaT,
            "Outcome": "2차 신호 대기",
        }
    )

    start, end = cycle_view_dates(index, cycle)

    assert start == pd.Timestamp("2020-03-02")
    assert end == index[-1]


def test_chart_without_cycle_defaults_to_latest_three_years() -> None:
    index = pd.date_range("2018-01-01", "2025-12-29", freq="W-MON")

    start, end = cycle_view_dates(index, None)

    assert start == pd.Timestamp("2023-01-02")
    assert end == index[-1]
