from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from market_cap_provider import MarketCapCompany


HORIZON_MONTHS = (3, 6, 9, 12)
HORIZON_RETURN_COLUMNS = {
    3: ("Return3M", "Return3MStatus"),
    6: ("Return6M", "Return6MStatus"),
    9: ("Return9M", "Return9MStatus"),
    12: ("Return12M", "Return12MStatus"),
}
CLOSED_OUTCOMES = {
    "매수 성공",
    "실패",
    "2차 이격 과다 폐기",
    "2차 중기 구조 미충족 폐기",
}

TICKER_PERFORMANCE_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "섹터",
    "산업",
    "종료 사이클",
    "매수 건수",
    "매수 도달률",
    "승리",
    "분석 표본",
    "승률",
    "평균 손익률",
    "중앙값",
    "최고",
    "최저",
]

FIELD_PERFORMANCE_COLUMNS = [
    "분야",
    "종목 수",
    "종료 사이클",
    "매수 건수",
    "매수 도달률",
    "승리",
    "분석 표본",
    "승률",
    "평균 손익률",
    "중앙값",
]

RANKING_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "종료 사이클",
    "매수 건수",
    "승리",
    "분석 표본",
    "승률",
    "평균 손익률",
    "중앙값",
    "최고",
    "최저",
    "매수 도달률",
    "종합점수",
]


def build_ticker_performance(
    companies: Sequence[MarketCapCompany],
    cycles_by_ticker: Mapping[str, pd.DataFrame],
    classifications: pd.DataFrame,
    horizon_months: int,
) -> pd.DataFrame:
    _validate_horizon(horizon_months)
    classification_by_ticker = _classification_lookup(classifications)
    rows = []
    for company in companies:
        ticker = company.ticker.upper()
        cycles = cycles_by_ticker.get(ticker, pd.DataFrame())
        classification = classification_by_ticker.get(ticker, {})
        metrics = _cycle_metrics(cycles, horizon_months)
        rows.append(
            {
                "순위": company.rank,
                "티커": company.ticker,
                "회사명": company.company,
                "섹터": classification.get("섹터", "미분류"),
                "산업": classification.get("산업", "미분류"),
                **metrics,
            }
        )
    return pd.DataFrame(rows, columns=TICKER_PERFORMANCE_COLUMNS)


def build_field_performance(
    companies: Sequence[MarketCapCompany],
    cycles_by_ticker: Mapping[str, pd.DataFrame],
    classifications: pd.DataFrame,
    level: str,
    horizon_months: int,
) -> pd.DataFrame:
    _validate_level(level)
    ticker_performance = build_ticker_performance(
        companies,
        cycles_by_ticker,
        classifications,
        horizon_months,
    )
    rows = []
    for field, members in ticker_performance.groupby(level, dropna=False, sort=True):
        tickers = members["티커"].astype(str).str.upper().tolist()
        combined = _combine_cycles(cycles_by_ticker, tickers)
        metrics = _cycle_metrics(combined, horizon_months)
        rows.append(
            {
                "분야": _classification_value(field),
                "종목 수": len(members),
                **metrics,
            }
        )
    result = pd.DataFrame(rows, columns=FIELD_PERFORMANCE_COLUMNS)
    if result.empty:
        return result
    return result.sort_values(
        by=["승률", "분석 표본", "평균 손익률", "분야"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_stock_ranking(
    companies: Sequence[MarketCapCompany],
    cycles_by_ticker: Mapping[str, pd.DataFrame],
    classifications: pd.DataFrame,
    level: str,
    field: str,
    horizon_months: int,
    sort_by: str = "종합점수",
) -> pd.DataFrame:
    _validate_level(level)
    ticker_performance = build_ticker_performance(
        companies,
        cycles_by_ticker,
        classifications,
        horizon_months,
    )
    selected = ticker_performance[ticker_performance[level] == field].copy()
    if selected.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)

    eligible = selected["분석 표본"] > 0
    selected["종합점수"] = float("nan")
    if eligible.any():
        win_score = selected.loc[eligible, "승률"].rank(
            pct=True, method="average", ascending=True
        ) * 100
        return_score = selected.loc[eligible, "평균 손익률"].rank(
            pct=True, method="average", ascending=True
        ) * 100
        selected.loc[eligible, "종합점수"] = (win_score + return_score) / 2

    sort_column = {
        "종합점수": "종합점수",
        "승률": "승률",
        "평균 손익률": "평균 손익률",
        "매수 도달률": "매수 도달률",
    }.get(sort_by, "종합점수")
    selected["_정렬값"] = selected[sort_column].where(selected["분석 표본"] > 0)
    selected = selected.sort_values(
        by=["_정렬값", "분석 표본", "평균 손익률", "티커"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    selected["순위"] = pd.Series(pd.NA, index=selected.index, dtype="Int64")
    ranked = selected["_정렬값"].notna()
    selected.loc[ranked, "순위"] = range(1, int(ranked.sum()) + 1)
    return selected.reindex(columns=RANKING_COLUMNS)


def build_all_field_outputs(
    companies: Sequence[MarketCapCompany],
    cycles_by_ticker: Mapping[str, pd.DataFrame],
    classifications: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sector_frames = []
    industry_frames = []
    ranking_frames = []
    for horizon in HORIZON_MONTHS:
        sector = build_field_performance(
            companies, cycles_by_ticker, classifications, "섹터", horizon
        )
        sector.insert(0, "분석 기간", f"{horizon}개월")
        sector_frames.append(sector)

        industry = build_field_performance(
            companies, cycles_by_ticker, classifications, "산업", horizon
        )
        industry.insert(0, "분석 기간", f"{horizon}개월")
        industry_frames.append(industry)

        for level, fields in (("섹터", sector), ("산업", industry)):
            for field in fields["분야"].tolist():
                ranking = build_stock_ranking(
                    companies,
                    cycles_by_ticker,
                    classifications,
                    level,
                    field,
                    horizon,
                )
                if ranking.empty:
                    continue
                ranking.insert(0, "분야", field)
                ranking.insert(0, "구분", level)
                ranking.insert(0, "분석 기간", f"{horizon}개월")
                ranking_frames.append(ranking)

    sector_output = pd.concat(sector_frames, ignore_index=True)
    industry_output = pd.concat(industry_frames, ignore_index=True)
    if ranking_frames:
        ranking_output = pd.concat(ranking_frames, ignore_index=True)
    else:
        ranking_output = pd.DataFrame(
            columns=["분석 기간", "구분", "분야", *RANKING_COLUMNS]
        )
    return sector_output, industry_output, ranking_output


def format_rate(value: object, wins: object, sample: object) -> str:
    sample_count = _safe_int(sample)
    win_count = _safe_int(wins)
    if sample_count <= 0 or pd.isna(value):
        return "미산출 (0건)"
    return f"{float(value):.1f}% ({win_count}/{sample_count})"


def format_reach_rate(value: object, buys: object, closed: object) -> str:
    closed_count = _safe_int(closed)
    buy_count = _safe_int(buys)
    if closed_count <= 0 or pd.isna(value):
        return "미산출 (0건)"
    return f"{float(value):.1f}% ({buy_count}/{closed_count})"


def _cycle_metrics(cycles: pd.DataFrame, horizon_months: int) -> dict[str, object]:
    _validate_horizon(horizon_months)
    if cycles.empty or "Outcome" not in cycles.columns:
        return _empty_metrics()

    closed = cycles[cycles["Outcome"].isin(CLOSED_OUTCOMES)]
    bought = closed[closed["Outcome"] == "매수 성공"]
    return_column, status_column = HORIZON_RETURN_COLUMNS[horizon_months]
    if return_column in bought.columns:
        numeric_returns = pd.to_numeric(bought[return_column], errors="coerce")
    else:
        numeric_returns = pd.Series(float("nan"), index=bought.index)
    if status_column in bought.columns:
        confirmed = bought[status_column].eq("확정")
    else:
        confirmed = pd.Series(False, index=bought.index)
    values = numeric_returns[confirmed & numeric_returns.notna()]

    closed_count = len(closed)
    buy_count = len(bought)
    sample_count = len(values)
    win_count = int((values > 0).sum())
    return {
        "종료 사이클": closed_count,
        "매수 건수": buy_count,
        "매수 도달률": _percentage(buy_count, closed_count),
        "승리": win_count,
        "분석 표본": sample_count,
        "승률": _percentage(win_count, sample_count),
        "평균 손익률": values.mean() if sample_count else float("nan"),
        "중앙값": values.median() if sample_count else float("nan"),
        "최고": values.max() if sample_count else float("nan"),
        "최저": values.min() if sample_count else float("nan"),
    }


def _empty_metrics() -> dict[str, object]:
    return {
        "종료 사이클": 0,
        "매수 건수": 0,
        "매수 도달률": float("nan"),
        "승리": 0,
        "분석 표본": 0,
        "승률": float("nan"),
        "평균 손익률": float("nan"),
        "중앙값": float("nan"),
        "최고": float("nan"),
        "최저": float("nan"),
    }


def _combine_cycles(
    cycles_by_ticker: Mapping[str, pd.DataFrame], tickers: Sequence[str]
) -> pd.DataFrame:
    frames = [
        cycles_by_ticker[ticker]
        for ticker in tickers
        if ticker in cycles_by_ticker and not cycles_by_ticker[ticker].empty
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _classification_lookup(classifications: pd.DataFrame) -> dict[str, dict[str, str]]:
    if classifications.empty or "티커" not in classifications.columns:
        return {}
    lookup = {}
    for _, row in classifications.iterrows():
        ticker = str(row.get("티커", "")).strip().upper()
        if not ticker:
            continue
        lookup[ticker] = {
            "섹터": _classification_value(row.get("섹터")),
            "산업": _classification_value(row.get("산업")),
        }
    return lookup


def _classification_value(value: object) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "미분류"
    return str(value).strip()


def _percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator * 100


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _validate_horizon(horizon_months: int) -> None:
    if horizon_months not in HORIZON_RETURN_COLUMNS:
        raise ValueError(f"지원하지 않는 분석 기간입니다: {horizon_months}")


def _validate_level(level: str) -> None:
    if level not in {"섹터", "산업"}:
        raise ValueError(f"지원하지 않는 분야 구분입니다: {level}")
