from __future__ import annotations

import pandas as pd

from app import (
    ACTIVE_SCENARIO_DISPLAY_COLUMNS,
    CLOSED_RESULT_DISPLAY_COLUMNS,
    FIELD_DISPLAY_COLUMNS,
    RANKING_DISPLAY_COLUMNS,
    SCAN_EVENT_DISPLAY_COLUMNS,
    SCAN_FAILURE_COLUMNS,
    SIGNAL_HISTORY_DISPLAY_COLUMNS,
    _column_width,
    _format_value,
    _table_required_width,
    add_sector_column,
    field_performance_for_display,
    ranking_for_display,
    save_analytics_outputs,
    scanner_table_for_display,
    signal_cycles_for_display,
)


def test_table_panel_widths_cover_every_visible_column() -> None:
    history_width = _table_required_width(SIGNAL_HISTORY_DISPLAY_COLUMNS)
    scanner_tables = (
        SCAN_EVENT_DISPLAY_COLUMNS,
        ACTIVE_SCENARIO_DISPLAY_COLUMNS,
        CLOSED_RESULT_DISPLAY_COLUMNS,
        FIELD_DISPLAY_COLUMNS,
        RANKING_DISPLAY_COLUMNS,
        SCAN_FAILURE_COLUMNS,
    )
    scanner_width = max(_table_required_width(columns) for columns in scanner_tables)

    assert history_width > sum(_column_width(column) for column in SIGNAL_HISTORY_DISPLAY_COLUMNS)
    assert all(scanner_width >= _table_required_width(columns) for columns in scanner_tables)
    assert _column_width("회사명") >= 220
    assert _column_width("결과") >= 200


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
