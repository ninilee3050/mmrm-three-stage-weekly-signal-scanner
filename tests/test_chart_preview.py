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
    DARK_MA_COLOR_OVERRIDES,
    INDICATOR_BAR_OPACITY,
    INDICATOR_BAR_WIDTH_RATIO,
    LIGHT_MA_COLOR_OVERRIDES,
    MACD_SIGNAL_COLOR,
    MA_STYLES,
    NEGATIVE_BAR_COLOR,
    OSCILLATOR_LINE_COLOR,
    OSCILLATOR_CENTER,
    OSCILLATOR_LOWER,
    OSCILLATOR_UPPER,
    Panel,
    POSITIVE_BAR_COLOR,
    VOLUME_MA_STYLE,
    ZOOM_IN_FACTOR,
    ZOOM_OUT_FACTOR,
    ChartPreviewWindow,
    comparison_view_indices,
    cycle_return_summary,
    cycle_view_dates,
    cycle_view_indices,
    expanded_comparison_width,
    format_cycle_return,
    moving_average_styles,
    price_crosshair_y,
)


class _FakeVariable:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _FakeButton:
    def __init__(self) -> None:
        self.state = "normal"

    def configure(self, *, state: str) -> None:
        self.state = state

    def cget(self, option: str) -> str:
        assert option == "state"
        return self.state


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


def test_synchronized_crosshair_maps_each_chart_close_to_its_own_price_scale() -> None:
    data = pd.DataFrame(
        {
            "Low": [90.0, 100.0],
            "High": [110.0, 130.0],
            "MA_5": [98.0, 105.0],
            "MA_20": [97.0, 103.0],
            "MA_50": [96.0, 101.0],
            "MA_150": [92.0, 95.0],
            "MA_200": [91.0, 93.0],
        }
    )
    panel = Panel("가격", 10.0, 210.0)

    lower_close_y = price_crosshair_y(data, panel, 100.0)
    higher_close_y = price_crosshair_y(data, panel, 120.0)

    assert lower_close_y is not None
    assert higher_close_y is not None
    assert panel.top < higher_close_y < lower_close_y < panel.bottom
    assert price_crosshair_y(data, panel, float("nan")) is None


def test_cycle_return_summary_uses_stored_returns_and_statuses() -> None:
    cycle = pd.Series(
        {
            "Return3M": 12.345,
            "Return3MStatus": "확정",
            "Return6M": -4.5,
            "Return6MStatus": "확정",
            "Return9M": float("nan"),
            "Return9MStatus": "진행 중",
            "Return12M": float("nan"),
            "Return12MStatus": "데이터 없음",
        }
    )

    assert cycle_return_summary(cycle) == (
        "3M +12.35% · 6M -4.50% · 9M 진행 중 · 12M 데이터 없음"
    )
    assert format_cycle_return(float("nan"), "해당 없음") == "해당 없음"
    assert cycle_return_summary(None) == ""


def test_navigation_state_disables_only_the_unavailable_direction() -> None:
    window = object.__new__(ChartPreviewWindow)
    window.navigation_var = _FakeVariable()
    window.previous_button = _FakeButton()
    window.next_button = _FakeButton()

    ChartPreviewWindow._set_navigation_state(window, 0, 3)
    assert window.navigation_var.value == "1 / 3"
    assert window.previous_button.state == "disabled"
    assert window.next_button.state == "normal"

    ChartPreviewWindow._set_navigation_state(window, 1, 3)
    assert window.navigation_var.value == "2 / 3"
    assert window.previous_button.state == "normal"
    assert window.next_button.state == "normal"

    ChartPreviewWindow._set_navigation_state(window, 2, 3)
    assert window.navigation_var.value == "3 / 3"
    assert window.previous_button.state == "normal"
    assert window.next_button.state == "disabled"


def test_navigation_request_calls_back_without_wrapping() -> None:
    calls: list[int] = []
    window = object.__new__(ChartPreviewWindow)
    window.previous_button = _FakeButton()
    window.next_button = _FakeButton()
    window._on_navigate_callback = calls.append

    window.previous_button.state = "disabled"
    assert ChartPreviewWindow._request_navigation(window, -1) == "break"
    assert calls == []

    window.previous_button.state = "normal"
    assert ChartPreviewWindow._request_navigation(window, -1) == "break"
    assert calls == [-1]

    window.next_button.state = "normal"
    assert ChartPreviewWindow._request_navigation(window, 1) == "break"
    assert calls == [-1, 1]


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


def test_theme_specific_ma_colors_keep_periods_and_widths() -> None:
    light = moving_average_styles("light")
    dark = moving_average_styles("dark")

    assert light["MA_20"] == ("#00e5ff", 2.4)
    assert dark["MA_20"] == ("#00d4d8", 2.4)
    assert light["MA_150"] == ("#9b1010", 3.2)
    assert dark["MA_150"] == ("#c77832", 3.2)
    assert LIGHT_MA_COLOR_OVERRIDES == {"MA_20": "#00e5ff"}
    assert DARK_MA_COLOR_OVERRIDES == {"MA_150": "#c77832"}


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
