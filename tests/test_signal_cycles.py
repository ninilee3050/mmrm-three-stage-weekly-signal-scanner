from __future__ import annotations

import math

import pandas as pd

from scanner import scan_signal_cycles


def make_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    defaults = {
        "Open": 100.0,
        "High": 105.0,
        "Low": 95.0,
        "Close": 100.0,
        "Volume": 1_000_000.0,
        "MA_5": 95.0,
        "MA_20": 90.0,
        "MA_50": 100.0,
        "MA_150": 110.0,
        "MA_200": 100.0,
        "MACD": -0.5,
        "Signal": -1.0,
        "Momentum": 1.0,
        "RSI": 55.0,
        "MFI": 56.0,
    }
    normalized = [{**defaults, **row} for row in rows]
    index = pd.date_range("2024-01-01", periods=len(normalized), freq="W-MON")
    return pd.DataFrame(normalized, index=index)


def first_signal_setup() -> list[dict[str, float]]:
    return [
        {
            "MACD": -2.0,
            "Signal": -1.0,
            "Momentum": -1.0,
            "RSI": 40.0,
            "MFI": 40.0,
        },
        {
            "Open": 100.0,
            "Close": 105.0,
            "MA_5": 100.0,
            "MACD": -0.8,
            "Signal": -1.2,
        },
    ]


def successful_cycle_setup() -> list[dict[str, float]]:
    return [
        *first_signal_setup(),
        {
            "Open": 98.0,
            "Close": 99.0,
            "MA_5": 100.0,
        },
        {
            "Open": 100.0,
            "Close": 98.0,
            "MA_5": 99.0,
        },
        {
            "Open": 102.0,
            "Close": 101.0,
            "MA_5": 100.0,
            "MA_20": 99.0,
        },
        {
            "Open": 103.0,
            "Close": 103.0,
            "MA_5": 101.0,
            "MA_20": 102.0,
        },
    ]


def test_first_signal_filters_the_fixed_original_mmrm_date() -> None:
    rows = first_signal_setup()
    rows[1]["MA_20"] = 101.0
    rows[1]["MA_50"] = 100.0
    rows.append(
        {
            "Close": 106.0,
            "MA_5": 101.0,
            "MA_20": 99.0,
            "MA_50": 100.0,
        }
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert bool(full.iloc[1]["original_mmrm_point"]) is True
    assert bool(full.iloc[1]["first_signal"]) is False
    assert bool(full.iloc[2]["first_signal"]) is False
    assert cycles.empty


def test_first_signal_below_five_week_average_is_discarded() -> None:
    rows = first_signal_setup()
    rows[1]["Close"] = 99.0
    rows[1]["MA_5"] = 100.0
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert bool(full.iloc[1]["original_mmrm_point"]) is True
    assert bool(full.iloc[1]["first_signal"]) is False
    assert cycles.empty


def test_signal_cycle_ignores_wrong_candles_and_accepts_doji_third_signal() -> None:
    data = make_frame(successful_cycle_setup())

    cycles, full = scan_signal_cycles(data)

    assert len(cycles) == 1
    cycle = cycles.iloc[0]
    assert cycle["FirstSignalDate"] == data.index[1]
    assert cycle["SecondSignalDate"] == data.index[3]
    assert cycle["ThirdDecisionDate"] == data.index[5]
    assert cycle["Outcome"] == "매수 성공"
    assert bool(full.iloc[2]["second_signal"]) is False
    assert bool(full.iloc[3]["second_signal"]) is True
    assert bool(full.iloc[4]["third_signal"]) is False
    assert bool(full.iloc[5]["third_signal"]) is True


def test_third_candidate_at_or_below_twenty_week_average_ends_as_failure() -> None:
    rows = successful_cycle_setup()[:4]
    rows.append(
        {
            "Open": 100.0,
            "Close": 101.0,
            "MA_5": 100.0,
            "MA_20": 101.0,
        }
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert len(cycles) == 1
    cycle = cycles.iloc[0]
    assert cycle["ThirdDecisionDate"] == data.index[4]
    assert cycle["Outcome"] == "실패"
    assert pd.isna(cycle["Return3M"])
    assert cycle["Return3MStatus"] == "해당 없음"
    assert bool(full.iloc[4]["third_failure"]) is True


def test_second_candidate_with_five_percent_ma20_lead_discards_cycle() -> None:
    rows = first_signal_setup()
    rows.append(
        {
            "Open": 100.0,
            "Close": 90.0,
            "MA_5": 95.0,
            "MA_20": 105.0,
            "MA_50": 100.0,
        }
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert len(cycles) == 1
    cycle = cycles.iloc[0]
    assert cycle["SecondSignalDate"] == data.index[2]
    assert pd.isna(cycle["ThirdDecisionDate"])
    assert cycle["Outcome"] == "2차 이격 과다 폐기"
    assert cycle["Return3MStatus"] == "해당 없음"
    assert bool(full.iloc[2]["second_signal"]) is False
    assert bool(full.iloc[2]["second_rejection"]) is True


def test_second_candidate_below_five_percent_ma20_lead_is_accepted() -> None:
    rows = first_signal_setup()
    rows.extend(
        [
            {
                "Open": 100.0,
                "Close": 90.0,
                "MA_5": 95.0,
                "MA_20": 104.99,
                "MA_50": 100.0,
                "MA_200": 99.0,
            },
            {
                "Open": 105.0,
                "Close": 106.0,
                "MA_5": 100.0,
                "MA_20": 104.0,
            },
        ]
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert cycles.iloc[0]["Outcome"] == "매수 성공"
    assert bool(full.iloc[2]["second_signal"]) is True
    assert bool(full.iloc[2]["second_rejection"]) is False


def test_reversed_short_mas_require_ma50_strictly_above_ma200_at_second() -> None:
    rows = first_signal_setup()
    rows.append(
        {
            "Open": 100.0,
            "Close": 90.0,
            "MA_5": 95.0,
            "MA_20": 102.0,
            "MA_50": 100.0,
            "MA_200": 100.0,
        }
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert cycles.iloc[0]["Outcome"] == "2차 중기 구조 미충족 폐기"
    assert pd.isna(cycles.iloc[0]["ThirdDecisionDate"])
    assert bool(full.iloc[2]["second_signal"]) is False
    assert bool(full.iloc[2]["second_structure_rejection"]) is True


def test_reversed_short_mas_are_accepted_when_ma50_is_above_ma200() -> None:
    rows = first_signal_setup()
    rows.extend(
        [
            {
                "Open": 100.0,
                "Close": 90.0,
                "MA_5": 95.0,
                "MA_20": 102.0,
                "MA_50": 100.0,
                "MA_200": 99.0,
            },
            {
                "Open": 105.0,
                "Close": 106.0,
                "MA_5": 100.0,
                "MA_20": 104.0,
            },
        ]
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert cycles.iloc[0]["Outcome"] == "매수 성공"
    assert bool(full.iloc[2]["second_signal"]) is True
    assert bool(full.iloc[2]["second_structure_rejection"]) is False


def test_ma50_above_ma200_is_not_required_while_ma50_remains_above_ma20() -> None:
    rows = first_signal_setup()
    rows.extend(
        [
            {
                "Open": 100.0,
                "Close": 90.0,
                "MA_5": 95.0,
                "MA_20": 90.0,
                "MA_50": 100.0,
                "MA_200": 110.0,
            },
            {
                "Open": 105.0,
                "Close": 106.0,
                "MA_5": 100.0,
                "MA_20": 104.0,
            },
        ]
    )
    data = make_frame(rows)

    cycles, full = scan_signal_cycles(data)

    assert cycles.iloc[0]["Outcome"] == "매수 성공"
    assert bool(full.iloc[2]["second_signal"]) is True
    assert bool(full.iloc[2]["second_structure_rejection"]) is False


def test_successful_cycle_returns_use_13_26_39_and_52_week_closes() -> None:
    rows = successful_cycle_setup()
    for _ in range(52):
        rows.append(
            {
                "Open": 103.0,
                "Close": 103.0,
                "MA_5": 102.0,
                "MA_20": 101.0,
            }
        )

    third_position = 5
    rows[third_position + 13]["Close"] = 113.3
    rows[third_position + 26]["Close"] = 92.7
    rows[third_position + 39]["Close"] = 103.0
    rows[third_position + 52]["Close"] = 123.6
    data = make_frame(rows)

    cycles, _full = scan_signal_cycles(data)

    cycle = cycles.iloc[0]
    assert math.isclose(cycle["Return3M"], 10.0)
    assert math.isclose(cycle["Return6M"], -10.0)
    assert math.isclose(cycle["Return9M"], 0.0)
    assert math.isclose(cycle["Return12M"], 20.0)
    assert cycle["Return3MStatus"] == "확정"
    assert cycle["Return6MStatus"] == "확정"
    assert cycle["Return9MStatus"] == "확정"
    assert cycle["Return12MStatus"] == "확정"


def test_open_cycle_is_included_in_selected_ticker_history() -> None:
    data = make_frame(first_signal_setup())

    cycles, _full = scan_signal_cycles(data)

    assert len(cycles) == 1
    assert cycles.iloc[0]["Outcome"] == "2차 신호 대기"
    assert pd.isna(cycles.iloc[0]["SecondSignalDate"])
    assert pd.isna(cycles.iloc[0]["ThirdDecisionDate"])
    assert cycles.iloc[0]["Return3MStatus"] == "해당 없음"


def test_successful_cycle_marks_unreached_returns_as_in_progress() -> None:
    data = make_frame(successful_cycle_setup())

    cycles, _full = scan_signal_cycles(data)

    cycle = cycles.iloc[0]
    assert cycle["Outcome"] == "매수 성공"
    assert pd.isna(cycle["Return3M"])
    assert cycle["Return3MStatus"] == "진행 중"
    assert cycle["Return12MStatus"] == "진행 중"


def test_successful_cycle_marks_missing_future_price_as_data_unavailable() -> None:
    rows = successful_cycle_setup()
    for _ in range(13):
        rows.append(
            {
                "Open": 103.0,
                "Close": 103.0,
                "MA_5": 102.0,
                "MA_20": 101.0,
            }
        )
    rows[5 + 13]["Close"] = float("nan")
    data = make_frame(rows)

    cycles, _full = scan_signal_cycles(data)

    cycle = cycles.iloc[0]
    assert pd.isna(cycle["Return3M"])
    assert cycle["Return3MStatus"] == "데이터 없음"
