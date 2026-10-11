from __future__ import annotations

import pandas as pd
import pytest

from signal_trend import (
    POINT_COLUMNS,
    change_from_first,
    range_positions,
    signal_points,
    time_fractions,
    trend_lines,
    weeks_between,
)


def weekly_data(weeks: int = 60) -> pd.DataFrame:
    index = pd.date_range("2025-01-06", periods=weeks, freq="W-MON")
    steps = pd.Series(range(weeks), index=index, dtype=float)
    return pd.DataFrame(
        {
            "Close": 100 + steps,  # 100 .. 159
            "Volume": 1_000_000 * (1 + steps),
            "Volume_MA_50": 1_000_000 * (1 + steps / 2),
            "Momentum": steps - 10,
            "MACD": steps - 30,  # crosses zero
            "RSI": 40 + steps / 2,
            "MFI": 50.0,  # flat: has no range to scale by
        }
    )


def cycle(data: pd.DataFrame, *positions: int | None) -> pd.Series:
    columns = ("FirstSignalDate", "SecondSignalDate", "ThirdDecisionDate")
    return pd.Series(
        {
            column: data.index[position] if position is not None else pd.NaT
            for column, position in zip(columns, positions)
        }
    )


def test_points_are_the_indicator_values_on_each_signal_week() -> None:
    data = weekly_data()

    points = signal_points(data, cycle(data, 55, 57, 59))

    assert list(points.index) == ["1차", "2차", "3차"]
    assert list(points["Close"]) == [155.0, 157.0, 159.0]
    assert points.loc["2차", "Date"] == data.index[57]
    assert list(points.columns) == ["Date", *POINT_COLUMNS]
    # The moving baseline is read on the same weeks as the indicator.
    assert points.loc["1차", "Volume_MA_50"] == 1_000_000 * (1 + 55 / 2)


def test_signals_that_have_not_happened_are_left_out() -> None:
    data = weekly_data()

    assert list(signal_points(data, cycle(data, 55, 57, None)).index) == ["1차", "2차"]
    assert signal_points(data, None).empty
    assert signal_points(pd.DataFrame(), cycle(data, 55, 57, 59)).empty


def test_positions_scale_each_indicator_to_its_own_range() -> None:
    data = weekly_data()
    points = signal_points(data, cycle(data, 55, 57, 59))

    positions = range_positions(data, points)

    # The window runs from 52 weeks before the first signal (row 3) to row 59.
    assert positions.loc["3차", "Close"] == pytest.approx(100.0)
    assert positions.loc["1차", "Close"] == pytest.approx((55 - 3) / (59 - 3) * 100)
    # Different units end up on the same scale.
    assert positions["Volume"].tolist() == pytest.approx(positions["Close"].tolist())
    assert positions["MACD"].tolist() == pytest.approx(positions["Close"].tolist())
    assert positions["MFI"].isna().all()


def test_change_starts_every_line_at_zero() -> None:
    data = weekly_data()
    positions = range_positions(data, signal_points(data, cycle(data, 55, 57, 59)))

    change = change_from_first(positions)

    assert change.loc["1차", "Close"] == 0
    assert change.loc["3차", "Close"] == pytest.approx(4 / 56 * 100)


def test_average_skips_indicators_without_a_value() -> None:
    data = weekly_data()
    points = signal_points(data, cycle(data, 55, 57, 59))

    change = trend_lines(data, points)

    # MFI is flat, so the average is over the five indicators that moved.
    assert change["평균"].tolist() == pytest.approx([0.0, 2 / 56 * 100, 4 / 56 * 100])


def test_spread_is_the_gap_between_the_highest_and_lowest_indicator() -> None:
    data = weekly_data()
    data["RSI"] = 40.0
    data.loc[data.index[59], "RSI"] = 20.0  # falls while the others rise
    points = signal_points(data, cycle(data, 55, 57, 59))

    change = trend_lines(data, points)

    assert change.loc["1차", "벌어짐"] == 0
    assert change.loc["3차", "벌어짐"] == pytest.approx(4 / 56 * 100 + 100)


def test_points_are_spaced_by_elapsed_weeks() -> None:
    dates = pd.to_datetime(["2026-06-29", "2026-09-07", "2026-10-05"])

    assert weeks_between(dates) == [10, 4]
    assert time_fractions(dates) == pytest.approx([0, 10 / 14, 1])


def test_close_signals_keep_room_for_their_labels() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-12", "2026-10-05"])

    fractions = time_fractions(dates, min_gap=0.2)

    assert fractions[0] == 0 and fractions[-1] == pytest.approx(1)
    assert fractions[1] == pytest.approx(0.2 / (0.2 + 38 / 39))
    assert time_fractions(dates[:1]) == [0.5]
