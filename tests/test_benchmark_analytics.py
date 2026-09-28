from __future__ import annotations

import math

import numpy as np
import pandas as pd

from benchmark_analytics import (
    add_benchmark_returns,
    benchmark_metrics,
    nearby_return_column,
    nearby_win_rate_column,
    sp500_return_column,
)
from performance_analytics import build_signal_validation

AFTER_DATA = pd.Timestamp("2030-01-01")


def weekly_frame(closes: list[float], start: str = "2020-01-06") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="W-MON")
    return pd.DataFrame({"Close": closes}, index=index)


def buy_cycle(signal_date: pd.Timestamp, return_3m: float, status: str = "확정") -> dict:
    return {
        "ThirdDecisionDate": signal_date,
        "Outcome": "매수 성공",
        "Return3M": return_3m,
        "Return3MStatus": status,
        "Return6M": np.nan,
        "Return6MStatus": "진행 중",
        "Return9M": np.nan,
        "Return9MStatus": "진행 중",
        "Return12M": np.nan,
        "Return12MStatus": "진행 중",
    }


def test_sp500_return_uses_the_same_weeks_as_the_signal() -> None:
    stock = weekly_frame([100.0] * 120)
    sp500 = weekly_frame([100.0] * 60 + [110.0] * 60)
    signal_date = stock.index[50]  # 13 weeks later is row 63 -> S&P +10%
    cycles = pd.DataFrame([buy_cycle(signal_date, 5.0)])

    result = add_benchmark_returns(cycles, stock, sp500, now=AFTER_DATA)

    assert math.isclose(result.loc[0, sp500_return_column(3)], 10.0)


def test_nearby_baseline_uses_same_stock_within_one_year() -> None:
    # Price rises 1% per week, so every nearby 13-week return is positive.
    stock = weekly_frame([100.0 * 1.01**i for i in range(200)])
    signal_date = stock.index[80]
    cycles = pd.DataFrame([buy_cycle(signal_date, 20.0)])

    result = add_benchmark_returns(cycles, stock, None, now=AFTER_DATA)

    expected = (1.01**13 - 1) * 100
    assert math.isclose(result.loc[0, nearby_return_column(3)], expected)
    assert result.loc[0, nearby_win_rate_column(3)] == 100.0
    assert math.isnan(result.loc[0, sp500_return_column(3)])


def test_baselines_are_skipped_for_unconfirmed_or_failed_cycles() -> None:
    stock = weekly_frame([100.0] * 120)
    pending = buy_cycle(stock.index[50], np.nan, status="진행 중")
    failed = {**buy_cycle(stock.index[60], np.nan), "Outcome": "실패"}

    result = add_benchmark_returns(
        pd.DataFrame([pending, failed]), stock, stock, now=AFTER_DATA
    )

    assert result[sp500_return_column(3)].isna().all()
    assert result[nearby_return_column(3)].isna().all()


def test_nearby_baseline_needs_enough_weeks() -> None:
    stock = weekly_frame([100.0] * 30)
    cycles = pd.DataFrame([buy_cycle(stock.index[5], 1.0)])

    result = add_benchmark_returns(cycles, stock, None, now=AFTER_DATA)

    assert math.isnan(result.loc[0, nearby_return_column(3)])


def test_benchmark_metrics_compare_only_paired_cycles() -> None:
    bought = pd.DataFrame(
        {
            sp500_return_column(3): [5.0, 5.0, np.nan],
            nearby_return_column(3): [2.0, 2.0, 2.0],
            nearby_win_rate_column(3): [60.0, 40.0, 50.0],
        }
    )
    signal_returns = pd.Series([10.0, 0.0, 4.0])

    metrics = benchmark_metrics(bought, signal_returns, 3)

    assert metrics["S&P 비교 표본"] == 2
    assert metrics["S&P 이긴 건수"] == 1
    assert metrics["S&P 이긴 비율"] == 50.0
    assert metrics["S&P 대비 초과"] == 0.0
    assert metrics["기준 비교 표본"] == 3
    assert metrics["기준 승률"] == 50.0
    assert math.isclose(metrics["기준 대비 초과"], (8.0 - 2.0 + 2.0) / 3)


def test_signal_validation_splits_by_chart_strength_grade() -> None:
    dates = pd.to_datetime(["2024-01-08", "2024-02-05", "2024-03-04"])
    cycles = pd.DataFrame([buy_cycle(date, value) for date, value in zip(dates, [10.0, -5.0, 3.0])])
    grades = {
        ("AAA", "2024-01-08"): "우선검토",
        ("AAA", "2024-02-05"): "일반검토",
    }

    validation = build_signal_validation({"AAA": cycles}, grades)
    three_month = validation[validation["분석 기간"] == "3개월"].set_index("구분")

    assert three_month.loc["전체 매수 성공", "분석 표본"] == 3
    assert three_month.loc["차트 강도 산정분", "분석 표본"] == 2
    assert three_month.loc["우선검토", "분석 표본"] == 1
    assert three_month.loc["우선검토", "승률"] == 100.0
    assert three_month.loc["일반검토", "승률"] == 0.0
    assert len(validation) == 16
