from __future__ import annotations

import math

import pandas as pd

from market_cap_provider import MarketCapCompany
from scanner import scan_signal_cycles
from scenario_tracker import (
    ACTIVE_SCENARIO_COLUMNS,
    latest_scan_date,
    load_active_scenarios,
    market_today,
    merge_scan_universe,
    preserve_failed_active_rows,
    save_active_scenarios,
    summarize_ticker_cycles,
)
from test_signal_cycles import make_frame, first_signal_setup, successful_cycle_setup


def company(ticker: str = "TEST", rank: int = 1) -> MarketCapCompany:
    return MarketCapCompany(
        rank=rank,
        ticker=ticker,
        company=f"{ticker} Company",
        market_cap="1.00T",
    )


def test_active_scenario_state_round_trip(tmp_path) -> None:
    path = tmp_path / "active.csv"
    active = pd.DataFrame(
        [
            {
                "순위": 1,
                "티커": "TEST",
                "회사명": "Test Company",
                "시가총액": "1.00T",
                "현재상태": "2차 신호 대기",
                "1차신호일": pd.Timestamp("2024-01-08"),
                "2차신호일": pd.NaT,
                "마지막확인일": pd.Timestamp("2024-01-12"),
                "데이터기준일": pd.Timestamp("2024-01-08"),
                "데이터상태": "현재 주봉 포함",
            }
        ],
        columns=ACTIVE_SCENARIO_COLUMNS,
    )

    save_active_scenarios(active, path)
    loaded = load_active_scenarios(path)

    assert loaded.loc[0, "티커"] == "TEST"
    assert loaded.loc[0, "1차신호일"] == pd.Timestamp("2024-01-08")
    assert loaded.loc[0, "현재상태"] == "2차 신호 대기"


def test_merge_scan_universe_keeps_active_ticker_outside_current_top100() -> None:
    top100 = [company("TOP", rank=1)]
    active = pd.DataFrame(
        [
            {
                "순위": 120,
                "티커": "OLD",
                "회사명": "Old Company",
                "시가총액": "10.00B",
            }
        ]
    )

    merged = merge_scan_universe(top100, active)

    assert [item.ticker for item in merged] == ["TOP", "OLD"]


def test_current_first_signal_is_reported_and_saved_as_active() -> None:
    data = make_frame(first_signal_setup())
    cycles, full = scan_signal_cycles(data)

    events, active, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=data.index[-1],
    )

    assert [event["단계"] for event in events] == ["1차 신호"]
    assert active is not None
    assert active["현재상태"] == "2차 신호 대기"
    assert active["1차신호일"] == data.index[1]
    assert closed == []


def test_previous_active_scenario_reports_missed_second_and_third_transitions() -> None:
    data = make_frame(successful_cycle_setup())
    cycles, full = scan_signal_cycles(data)
    previous = pd.Series(
        {
            "현재상태": "2차 신호 대기",
            "1차신호일": data.index[1],
            "2차신호일": pd.NaT,
            "마지막확인일": data.index[1],
        }
    )

    events, active, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=data.index[-1],
        previous_active=previous,
    )

    assert [event["단계"] for event in events] == ["2차 신호", "3차 신호"]
    assert events[-1]["결과"] == "매수 성공"
    assert active is None
    assert len(closed) == 1
    assert closed[0]["결과"] == "매수 성공"


def test_previous_active_scenario_reports_rule_reassessment_rejection() -> None:
    rows = first_signal_setup()
    rows.append(
        {
            "Open": 100.0,
            "Close": 90.0,
            "MA_5": 95.0,
            "MA_20": 115.0,
            "MA_50": 100.0,
        }
    )
    data = make_frame(rows)
    cycles, full = scan_signal_cycles(data)
    previous = pd.Series(
        {
            "현재상태": "3차 신호 대기",
            "1차신호일": data.index[1],
            "2차신호일": data.index[2],
            "마지막확인일": data.index[2],
        }
    )

    events, active, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=data.index[2] + pd.Timedelta(days=7),
        previous_active=previous,
    )

    assert [event["단계"] for event in events] == ["2차 폐기"]
    assert events[0]["결과"] == "2차 이격 과다 폐기"
    assert events[0]["신호구분"] == "규칙 재평가"
    assert math.isclose(events[0]["MA20_50이격률"], 15.0)
    assert active is None
    assert len(closed) == 1
    assert closed[0]["결과"] == "2차 이격 과다 폐기"


def test_previous_active_scenario_reports_medium_structure_rejection() -> None:
    rows = first_signal_setup()
    rows.append(
        {
            "Open": 100.0,
            "Close": 90.0,
            "MA_5": 95.0,
            "MA_20": 102.0,
            "MA_50": 100.0,
            "MA_200": 101.0,
        }
    )
    data = make_frame(rows)
    cycles, full = scan_signal_cycles(data)
    previous = pd.Series(
        {
            "현재상태": "3차 신호 대기",
            "1차신호일": data.index[1],
            "2차신호일": data.index[2],
            "마지막확인일": data.index[2],
        }
    )

    events, active, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=data.index[2] + pd.Timedelta(days=7),
        previous_active=previous,
    )

    assert [event["단계"] for event in events] == ["2차 폐기"]
    assert events[0]["결과"] == "2차 중기 구조 미충족 폐기"
    assert events[0]["신호구분"] == "규칙 재평가"
    assert events[0]["MA_50"] == 100.0
    assert events[0]["MA_200"] == 101.0
    assert active is None
    assert len(closed) == 1


def test_failed_active_ticker_is_preserved_for_next_scan() -> None:
    active = pd.DataFrame(
        [
            {
                "순위": 1,
                "티커": "TEST",
                "회사명": "Test Company",
                "시가총액": "1.00T",
                "현재상태": "3차 신호 대기",
            }
        ],
        columns=ACTIVE_SCENARIO_COLUMNS,
    )

    preserved = preserve_failed_active_rows(active, {"TEST"})

    assert len(preserved) == 1
    assert preserved[0]["현재상태"] == "3차 신호 대기"
    assert preserved[0]["데이터상태"] == "갱신 실패"


def test_third_signal_after_midweek_scan_is_reported_on_next_week_scan() -> None:
    # Previous scan ran on Wednesday of the third-signal week, before Friday's
    # close completed the signal.  The next scan runs in the following week.
    data = make_frame(successful_cycle_setup())
    cycles, full = scan_signal_cycles(data)
    third_week = pd.Timestamp("2024-02-05")
    previous = pd.Series(
        {
            "현재상태": "3차 신호 대기",
            "1차신호일": pd.Timestamp("2024-01-08"),
            "2차신호일": pd.Timestamp("2024-01-22"),
            "마지막확인일": third_week + pd.Timedelta(days=2),
        }
    )

    events, active, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=third_week + pd.Timedelta(days=9),
        previous_active=previous,
    )

    assert [event["단계"] for event in events] == ["3차 신호"]
    assert events[0]["결과"] == "매수 성공"
    assert events[0]["신호구분"] == "미확인 기간"
    assert active is None
    assert len(closed) == 1


def test_weekend_scan_after_same_week_scan_reports_completed_third_signal() -> None:
    # Last scan was Monday morning in Korea (still the prior US week); this
    # scan is the following weekend in US time.
    data = make_frame(successful_cycle_setup())
    cycles, full = scan_signal_cycles(data)
    previous = pd.Series(
        {
            "현재상태": "3차 신호 대기",
            "1차신호일": pd.Timestamp("2024-01-08"),
            "2차신호일": pd.Timestamp("2024-01-22"),
            "마지막확인일": pd.Timestamp("2024-02-04"),
        }
    )

    events, _, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=pd.Timestamp("2024-02-11"),
        previous_active=previous,
    )

    assert [event["단계"] for event in events] == ["3차 신호"]
    assert len(closed) == 1


def test_known_first_signal_is_not_repeated_in_a_later_week() -> None:
    data = make_frame(first_signal_setup())
    cycles, full = scan_signal_cycles(data)
    first_week = data.index[1]
    previous = pd.Series(
        {
            "현재상태": "2차 신호 대기",
            "1차신호일": first_week,
            "2차신호일": pd.NaT,
            "마지막확인일": first_week + pd.Timedelta(days=4),
        }
    )

    events, active, _ = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=first_week + pd.Timedelta(days=9),
        previous_active=previous,
    )

    assert events == []
    assert active is not None
    assert active["현재상태"] == "2차 신호 대기"


def test_untracked_ticker_reports_cycle_that_began_after_last_scan() -> None:
    data = make_frame(first_signal_setup())
    cycles, full = scan_signal_cycles(data)
    first_week = data.index[1]

    events, active, _ = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=first_week + pd.Timedelta(days=8),
        last_scan_date=first_week + pd.Timedelta(days=2),
    )

    assert [event["단계"] for event in events] == ["1차 신호"]
    assert events[0]["신호구분"] == "미확인 기간"
    assert active is not None


def test_untracked_ticker_does_not_repeat_cycle_older_than_last_scan() -> None:
    data = make_frame(successful_cycle_setup())
    cycles, full = scan_signal_cycles(data)
    third_week = pd.Timestamp("2024-02-05")

    events, active, closed = summarize_ticker_cycles(
        company(),
        cycles,
        full,
        scan_date=third_week + pd.Timedelta(days=8),
        last_scan_date=third_week + pd.Timedelta(days=4),
    )

    assert events == []
    assert active is None
    assert closed == []


def test_latest_scan_date_uses_most_recent_check() -> None:
    active = pd.DataFrame(
        {
            "티커": ["A", "B", "C"],
            "마지막확인일": [
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-12"),
                pd.NaT,
            ],
        }
    )

    assert latest_scan_date(active) == pd.Timestamp("2024-01-12")
    assert latest_scan_date(pd.DataFrame(columns=ACTIVE_SCENARIO_COLUMNS)) is None


def test_market_today_is_naive_normalized_date() -> None:
    today = market_today()

    assert today.tzinfo is None
    assert today == today.normalize()
