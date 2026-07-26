from __future__ import annotations

import pandas as pd

from app import (
    ACTIVE_SCENARIO_DISPLAY_COLUMNS,
    CLOSED_SCENARIO_DISPLAY_COLUMNS,
    CLOSED_RESULT_DISPLAY_COLUMNS,
    FIELD_DISPLAY_COLUMNS,
    RANKING_DISPLAY_COLUMNS,
    SCAN_EVENT_DISPLAY_COLUMNS,
    SCAN_FAILURE_COLUMNS,
    SIGNAL_HISTORY_DISPLAY_COLUMNS,
    _column_width,
    _format_value,
    _table_required_width,
    active_scenario_tag,
    add_scan_performance_columns,
    add_sector_column,
    build_closed_scenario_history,
    field_performance_for_display,
    history_cycle_tag,
    load_closed_scenarios,
    prioritize_active_scenarios,
    prioritize_scan_events,
    ranking_for_display,
    save_analytics_outputs,
    save_closed_scenarios,
    scan_event_tag,
    scanner_table_for_display,
    signal_cycles_for_display,
)
from market_cap_provider import MarketCapCompany


def test_table_panel_widths_cover_every_visible_column() -> None:
    history_width = _table_required_width(SIGNAL_HISTORY_DISPLAY_COLUMNS)
    scanner_tables = (
        SCAN_EVENT_DISPLAY_COLUMNS,
        ACTIVE_SCENARIO_DISPLAY_COLUMNS,
        CLOSED_RESULT_DISPLAY_COLUMNS,
        CLOSED_SCENARIO_DISPLAY_COLUMNS,
        FIELD_DISPLAY_COLUMNS,
        RANKING_DISPLAY_COLUMNS,
        SCAN_FAILURE_COLUMNS,
    )
    scanner_width = max(_table_required_width(columns) for columns in scanner_tables)

    assert history_width > sum(_column_width(column) for column in SIGNAL_HISTORY_DISPLAY_COLUMNS)
    assert all(scanner_width >= _table_required_width(columns) for columns in scanner_tables)
    assert _column_width("회사명") >= 220
    assert _column_width("결과") >= 200


def test_closed_scenario_history_collects_current_top100_and_sorts_by_first_signal() -> None:
    companies = [
        MarketCapCompany(1, "AAA", "Alpha", "1T"),
        MarketCapCompany(2, "BBB", "Beta", "900B"),
    ]
    cycles_by_ticker = {
        "AAA": pd.DataFrame(
            [
                {
                    "FirstSignalDate": "2026-01-05",
                    "SecondSignalDate": "2026-01-12",
                    "ThirdDecisionDate": "2026-01-19",
                    "Outcome": "매수 성공",
                    "Return3M": 12.0,
                    "Return3MStatus": "확정",
                },
                {
                    "FirstSignalDate": "2026-07-20",
                    "Outcome": "2차 신호 대기",
                },
            ]
        ),
        "BBB": pd.DataFrame(
            [
                {
                    "FirstSignalDate": "2026-03-02",
                    "SecondSignalDate": "2026-03-09",
                    "ThirdDecisionDate": None,
                    "Outcome": "2차 이격 과다 폐기",
                }
            ]
        ),
        "OLD": pd.DataFrame(
            [{"FirstSignalDate": "2026-06-01", "Outcome": "실패"}]
        ),
    }
    classifications = pd.DataFrame(
        [
            {"티커": "AAA", "섹터": "정보기술"},
            {"티커": "BBB", "섹터": "금융"},
        ]
    )

    history = build_closed_scenario_history(
        companies,
        cycles_by_ticker,
        classifications,
    )

    assert history.columns.tolist() == CLOSED_SCENARIO_DISPLAY_COLUMNS
    assert history["현재 시총순위"].tolist() == [2, 1]
    assert history["티커"].tolist() == ["BBB", "AAA"]
    assert history["섹터"].tolist() == ["금융", "정보기술"]
    assert "2차 신호 대기" not in history["결과"].tolist()
    assert "OLD" not in history["티커"].tolist()


def test_closed_scenario_history_preserves_failed_current_ticker_and_round_trips(
    tmp_path,
) -> None:
    previous = pd.DataFrame(
        [
            {
                "순위": 1,
                "티커": "AAA",
                "회사명": "Alpha",
                "섹터": "정보기술",
                "1차신호일": "2025-01-06",
                "결과": "실패",
            },
            {
                "순위": 3,
                "티커": "OLD",
                "회사명": "Old",
                "섹터": "금융",
                "1차신호일": "2024-01-01",
                "결과": "실패",
            },
        ]
    )
    history = build_closed_scenario_history(
        [MarketCapCompany(1, "AAA", "Alpha", "1T")],
        {},
        pd.DataFrame(),
        previous=previous,
        failed_tickers={"AAA"},
    )

    path = save_closed_scenarios(history, tmp_path / "closed.csv")
    loaded = load_closed_scenarios(path)

    assert loaded["티커"].tolist() == ["AAA"]
    assert loaded.loc[0, "현재 시총순위"] == 1
    assert loaded.loc[0, "1차신호일"] == pd.Timestamp("2025-01-06")


def test_closed_scenario_loader_migrates_legacy_rank_column(tmp_path) -> None:
    path = tmp_path / "legacy_closed.csv"
    pd.DataFrame(
        [{"순위": 7, "티커": "AAA", "1차신호일": "2025-01-06"}]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    loaded = load_closed_scenarios(path)

    assert "순위" not in loaded.columns
    assert loaded.loc[0, "현재 시총순위"] == 7


def test_signal_history_displays_progress_and_unavailable_states() -> None:
    cycles = pd.DataFrame(
        [
            {
                "FirstSignalDate": pd.Timestamp("2026-01-05"),
                "SecondSignalDate": pd.Timestamp("2026-01-12"),
                "ThirdDecisionDate": pd.Timestamp("2026-01-19"),
                "Outcome": "매수 성공",
                "Return3M": float("nan"),
                "Return3MStatus": "진행 중",
                "Return6M": 12.345,
                "Return6MStatus": "확정",
                "Return9M": float("nan"),
                "Return9MStatus": "데이터 없음",
                "Return12M": float("nan"),
                "Return12MStatus": "진행 중",
            },
            {
                "Outcome": "실패",
                "Return3M": float("nan"),
                "Return3MStatus": "해당 없음",
                "Return6M": float("nan"),
                "Return6MStatus": "해당 없음",
                "Return9M": float("nan"),
                "Return9MStatus": "해당 없음",
                "Return12M": float("nan"),
                "Return12MStatus": "해당 없음",
            },
        ]
    )

    display = signal_cycles_for_display(cycles)

    assert display.loc[0, "3개월후 수익률"] == "진행 중"
    assert display.loc[0, "6개월후 수익률"] == 12.345
    assert display.loc[0, "9개월후 수익률"] == "데이터 없음"
    assert display.loc[1, "3개월후 수익률"] == "해당 없음"


def test_second_spread_columns_are_readable_and_formatted_as_percent() -> None:
    assert _format_value(14.778206, "MA20_50이격률") == "+14.78%"
    assert _column_width("MA20_50이격률") >= 120
    assert _column_width("결과") >= 150


def test_sector_column_is_inserted_after_ticker_without_changing_source() -> None:
    source = pd.DataFrame([{"순위": 1, "티커": "NVDA", "회사명": "NVIDIA"}])
    classifications = pd.DataFrame(
        [{"티커": "NVDA", "섹터": "정보기술", "산업": "반도체"}]
    )

    display = add_sector_column(source, classifications)

    assert display.columns.tolist() == ["순위", "티커", "섹터", "회사명"]
    assert display.loc[0, "섹터"] == "정보기술"
    assert "섹터" not in source.columns


def test_scanner_display_hides_diagnostic_columns_without_changing_source() -> None:
    source = pd.DataFrame(
        [
            {
                "티커": "NVDA",
                "회사명": "NVIDIA",
                "단계": "2차 신호",
                "Close": 100.0,
                "MA_5": 99.0,
                "MA20_50이격률": 1.2,
            }
        ]
    )
    columns = ["티커", "회사명", "단계", "데이터기준일"]

    display = scanner_table_for_display(source, columns)

    assert display.columns.tolist() == columns
    assert "Close" not in display.columns
    assert "MA_5" not in display.columns
    assert "MA20_50이격률" not in display.columns
    assert "Close" in source.columns


def test_scan_events_prioritize_actionable_third_then_second_then_first() -> None:
    events = pd.DataFrame(
        [
            {"순위": 1, "단계": "1차 신호", "결과": "2차 신호 대기", "신호일": pd.Timestamp("2026-07-20")},
            {"순위": 2, "단계": "2차 신호", "결과": "3차 신호 대기", "신호일": pd.Timestamp("2026-07-20")},
            {"순위": 3, "단계": "3차 신호", "결과": "매수 성공", "신호일": pd.Timestamp("2026-07-13")},
            {"순위": 4, "단계": "3차 신호", "결과": "실패", "신호일": pd.Timestamp("2026-07-20")},
        ]
    )

    sorted_events = prioritize_scan_events(events)

    assert sorted_events[["단계", "결과"]].values.tolist() == [
        ["3차 신호", "매수 성공"],
        ["2차 신호", "3차 신호 대기"],
        ["1차 신호", "2차 신호 대기"],
        ["3차 신호", "실패"],
    ]
    assert scan_event_tag("3차 신호", "매수 성공") == "signal_third"
    assert scan_event_tag("2차 신호", "3차 신호 대기") == "signal_second"
    assert scan_event_tag("1차 신호", "2차 신호 대기") == "signal_first"
    assert scan_event_tag("3차 신호", "실패") == ""


def test_active_scenarios_prioritize_third_waiting_and_use_completed_stage_colors() -> None:
    active = pd.DataFrame(
        [
            {
                "순위": 1,
                "티커": "AAA",
                "현재상태": "2차 신호 대기",
                "1차신호일": "2026-07-20",
                "2차신호일": pd.NaT,
            },
            {
                "순위": 5,
                "티커": "BBB",
                "현재상태": "3차 신호 대기",
                "1차신호일": "2026-07-13",
                "2차신호일": "2026-07-06",
            },
            {
                "순위": 2,
                "티커": "CCC",
                "현재상태": "3차 신호 대기",
                "1차신호일": "2026-06-01",
                "2차신호일": "2026-07-20",
            },
            {
                "순위": 3,
                "티커": "DDD",
                "현재상태": "2차 신호 대기",
                "1차신호일": "2026-06-01",
                "2차신호일": pd.NaT,
            },
        ]
    )

    sorted_active = prioritize_active_scenarios(active)

    assert sorted_active["티커"].tolist() == ["CCC", "BBB", "AAA", "DDD"]
    assert active_scenario_tag("3차 신호 대기") == "signal_second"
    assert active_scenario_tag("2차 신호 대기") == "signal_first"
    assert active_scenario_tag("매수 성공") == ""


def test_history_cycle_colors_separate_process_result_and_three_month_return() -> None:
    assert history_cycle_tag("2차 이격 과다 폐기", "해당 없음") == "history_discard"
    assert history_cycle_tag("실패", "해당 없음") == "history_failure"
    assert history_cycle_tag("매수 성공", "진행 중") == "history_success_pending"
    assert history_cycle_tag("매수 성공", -3.0) == "history_loss"
    assert history_cycle_tag("매수 성공", 0.0) == "history_flat"
    assert history_cycle_tag("매수 성공", 5.0) == "history_success_low"
    assert history_cycle_tag("매수 성공", 15.0) == "history_success_medium"
    assert history_cycle_tag("매수 성공", 30.0) == "history_success_high"


def test_scan_events_show_ticker_and_sector_win_rates_with_samples() -> None:
    companies = [
        MarketCapCompany(1, "AAA", "AAA Company", "1.00T"),
        MarketCapCompany(2, "BBB", "BBB Company", "900B"),
    ]
    cycles_by_ticker = {
        "AAA": pd.DataFrame(
            [{"Outcome": "매수 성공", "Return3M": 8.0, "Return3MStatus": "확정"}]
        ),
        "BBB": pd.DataFrame(
            [{"Outcome": "매수 성공", "Return3M": -2.0, "Return3MStatus": "확정"}]
        ),
    }
    classifications = pd.DataFrame(
        [
            {"티커": "AAA", "섹터": "정보기술", "산업": "반도체"},
            {"티커": "BBB", "섹터": "정보기술", "산업": "소프트웨어"},
        ]
    )
    sector_output = pd.DataFrame(
        [
            {
                "분석 기간": "3개월",
                "분야": "정보기술",
                "승률": 50.0,
                "승리": 1,
                "분석 표본": 2,
            }
        ]
    )
    events = pd.DataFrame([{"티커": "AAA", "섹터": "정보기술"}])

    display = add_scan_performance_columns(
        events,
        companies,
        cycles_by_ticker,
        classifications,
        sector_output,
    )

    assert display.loc[0, "종목 3개월 승률"] == "100.0% (1/1)"
    assert display.loc[0, "섹터 3개월 승률"] == "50.0% (1/2)"


def test_field_and_ranking_views_show_denominators_without_hiding_small_samples() -> None:
    field = pd.DataFrame(
        [
            {
                "분야": "정보기술",
                "종목 수": 2,
                "종료 사이클": 5,
                "매수 건수": 3,
                "매수 도달률": 60.0,
                "승리": 1,
                "분석 표본": 1,
                "승률": 100.0,
                "평균 손익률": 10.0,
                "중앙값": 10.0,
            }
        ]
    )
    ranking = pd.DataFrame(
        [
            {
                "순위": 1,
                "티커": "NVDA",
                "회사명": "NVIDIA",
                "종료 사이클": 5,
                "매수 건수": 3,
                "매수 도달률": 60.0,
                "승리": 1,
                "분석 표본": 1,
                "승률": 100.0,
                "평균 손익률": 10.0,
                "중앙값": 10.0,
                "최고": 10.0,
                "최저": 10.0,
                "종합점수": 100.0,
            }
        ]
    )

    field_display = field_performance_for_display(field)
    ranking_display = ranking_for_display(ranking)

    assert field_display.loc[0, "승률"] == "100.0% (1/1)"
    assert field_display.loc[0, "매수 도달률"] == "60.0% (3/5)"
    assert ranking_display.loc[0, "승률"] == "100.0% (1/1)"


def test_analytics_outputs_use_stable_and_dated_names(tmp_path) -> None:
    data = pd.DataFrame([{"분야": "정보기술"}])

    stable = save_analytics_outputs(data, data, data, output_dir=tmp_path)
    dated = save_analytics_outputs(
        data,
        data,
        data,
        output_dir=tmp_path,
        date_suffix="2026-07-26",
    )

    assert [path.name for path in stable] == [
        "mmrm_sector_performance.csv",
        "mmrm_industry_performance.csv",
        "mmrm_field_stock_rankings.csv",
    ]
    assert [path.name for path in dated] == [
        "MMRM_sector_performance_2026-07-26.csv",
        "MMRM_industry_performance_2026-07-26.csv",
        "MMRM_field_stock_rankings_2026-07-26.csv",
    ]
    assert all(path.exists() for path in (*stable, *dated))
