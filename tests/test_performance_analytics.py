from __future__ import annotations

import math

import pandas as pd

from market_cap_provider import MarketCapCompany
from performance_analytics import (
    build_field_performance,
    build_stock_ranking,
    build_ticker_performance,
    format_rate,
    format_reach_rate,
)


def company(rank: int, ticker: str) -> MarketCapCompany:
    return MarketCapCompany(rank, ticker, f"{ticker} Company", "1.00T")


def classifications() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"티커": "AAA", "섹터": "정보기술", "산업": "반도체"},
            {"티커": "BBB", "섹터": "정보기술", "산업": "소프트웨어"},
            {"티커": "CCC", "섹터": "금융", "산업": "은행"},
        ]
    )


def cycles(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_ticker_metrics_separate_buy_reach_from_matured_win_rate() -> None:
    companies = [company(1, "AAA")]
    history = {
        "AAA": cycles(
            {
                "Outcome": "매수 성공",
                "Return3M": 10.0,
                "Return3MStatus": "확정",
            },
            {
                "Outcome": "매수 성공",
                "Return3M": 0.0,
                "Return3MStatus": "확정",
            },
            {
                "Outcome": "매수 성공",
                "Return3M": float("nan"),
                "Return3MStatus": "진행 중",
            },
            {"Outcome": "실패"},
            {"Outcome": "2차 이격 과다 폐기"},
            {"Outcome": "2차 신호 대기"},
        )
    }

    result = build_ticker_performance(companies, history, classifications(), 3).iloc[0]

    assert result["종료 사이클"] == 5
    assert result["매수 건수"] == 3
    assert math.isclose(result["매수 도달률"], 60.0)
    assert result["분석 표본"] == 2
    assert result["승리"] == 1
    assert math.isclose(result["승률"], 50.0)
    assert math.isclose(result["평균 손익률"], 5.0)
    assert math.isclose(result["중앙값"], 5.0)


def test_small_samples_are_kept_and_zero_sample_is_unranked() -> None:
    companies = [company(1, "AAA"), company(2, "BBB")]
    history = {
        "AAA": cycles(
            {"Outcome": "매수 성공", "Return6M": 20.0, "Return6MStatus": "확정"}
        ),
        "BBB": cycles(
            {"Outcome": "매수 성공", "Return6M": float("nan"), "Return6MStatus": "진행 중"}
        ),
    }

    ranking = build_stock_ranking(
        companies,
        history,
        classifications(),
        "섹터",
        "정보기술",
        6,
    )

    assert ranking["티커"].tolist() == ["AAA", "BBB"]
    assert ranking.loc[0, "순위"] == 1
    assert pd.isna(ranking.loc[1, "순위"])
    assert ranking.loc[0, "분석 표본"] == 1
    assert ranking.loc[1, "분석 표본"] == 0
    assert format_rate(100.0, 1, 1) == "100.0% (1/1)"
    assert format_rate(float("nan"), 0, 0) == "미산출 (0건)"


def test_field_metrics_pool_returns_instead_of_averaging_ticker_rates() -> None:
    companies = [company(1, "AAA"), company(2, "BBB"), company(3, "CCC")]
    history = {
        "AAA": cycles(
            {"Outcome": "매수 성공", "Return3M": 10.0, "Return3MStatus": "확정"},
        ),
        "BBB": cycles(
            {"Outcome": "매수 성공", "Return3M": -10.0, "Return3MStatus": "확정"},
            {"Outcome": "매수 성공", "Return3M": 20.0, "Return3MStatus": "확정"},
        ),
        "CCC": cycles(
            {"Outcome": "실패", "Return3M": float("nan"), "Return3MStatus": "해당 없음"},
        ),
    }

    fields = build_field_performance(
        companies, history, classifications(), "섹터", 3
    ).set_index("분야")

    technology = fields.loc["정보기술"]
    assert technology["종목 수"] == 2
    assert technology["분석 표본"] == 3
    assert technology["승리"] == 2
    assert math.isclose(technology["승률"], 200 / 3)
    assert math.isclose(technology["평균 손익률"], 20 / 3)
    assert math.isclose(technology["중앙값"], 10.0)


def test_field_performance_is_sorted_by_win_rate_before_sample_size() -> None:
    companies = [company(1, "AAA"), company(2, "BBB"), company(3, "CCC")]
    history = {
        "AAA": cycles(
            {"Outcome": "매수 성공", "Return3M": 10.0, "Return3MStatus": "확정"},
        ),
        "BBB": cycles(
            {"Outcome": "매수 성공", "Return3M": 20.0, "Return3MStatus": "확정"},
        ),
        "CCC": cycles(
            {"Outcome": "매수 성공", "Return3M": 5.0, "Return3MStatus": "확정"},
            {"Outcome": "매수 성공", "Return3M": -5.0, "Return3MStatus": "확정"},
            {"Outcome": "매수 성공", "Return3M": -10.0, "Return3MStatus": "확정"},
        ),
    }

    fields = build_field_performance(
        companies, history, classifications(), "섹터", 3
    )

    assert fields["분야"].tolist() == ["정보기술", "금융"]
    assert fields.loc[0, "승률"] == 100.0
    assert fields.loc[0, "분석 표본"] == 2
    assert fields.loc[1, "분석 표본"] == 3


def test_composite_score_uses_within_field_percentile_ranks() -> None:
    companies = [company(1, "AAA"), company(2, "BBB")]
    history = {
        "AAA": cycles(
            {"Outcome": "매수 성공", "Return12M": 10.0, "Return12MStatus": "확정"},
            {"Outcome": "매수 성공", "Return12M": -5.0, "Return12MStatus": "확정"},
        ),
        "BBB": cycles(
            {"Outcome": "매수 성공", "Return12M": 50.0, "Return12MStatus": "확정"},
        ),
    }

    ranking = build_stock_ranking(
        companies,
        history,
        classifications(),
        "섹터",
        "정보기술",
        12,
    )

    assert ranking.iloc[0]["티커"] == "BBB"
    assert math.isclose(ranking.iloc[0]["종합점수"], 100.0)
    assert ranking.iloc[1]["종합점수"] < ranking.iloc[0]["종합점수"]


def test_reach_rate_formatter_includes_raw_denominator() -> None:
    assert format_reach_rate(40.0, 2, 5) == "40.0% (2/5)"
    assert format_reach_rate(float("nan"), 0, 0) == "미산출 (0건)"
