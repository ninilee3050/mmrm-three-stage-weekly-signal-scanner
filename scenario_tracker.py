from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_cap_provider import MarketCapCompany


ACTIVE_STATE_PATH = Path("outputs") / "mmrm_active_scenarios.csv"

ACTIVE_SCENARIO_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "시가총액",
    "현재상태",
    "1차신호일",
    "2차신호일",
    "마지막확인일",
    "데이터기준일",
    "데이터상태",
]

SCAN_EVENT_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "단계",
    "신호일",
    "결과",
    "1차신호일",
    "2차신호일",
    "3차판정일",
    "신호구분",
    "Close",
    "MA_5",
    "MA_20",
    "MA_50",
    "MA_200",
    "MA20_50이격률",
    "데이터기준일",
]

CLOSED_RESULT_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "1차신호일",
    "2차신호일",
    "3차판정일",
    "종료일",
    "결과",
    "신호구분",
    "Close",
    "MA_5",
    "MA_20",
    "MA_50",
    "MA_200",
    "MA20_50이격률",
    "데이터기준일",
]

_DATE_COLUMNS = {
    "1차신호일",
    "2차신호일",
    "3차판정일",
    "종료일",
    "신호일",
    "마지막확인일",
    "데이터기준일",
}


def load_active_scenarios(
    path: Path | str = ACTIVE_STATE_PATH,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ACTIVE_SCENARIO_COLUMNS)

    data = pd.read_csv(path)
    data = data.reindex(columns=ACTIVE_SCENARIO_COLUMNS)
    for column in _DATE_COLUMNS.intersection(data.columns):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


def save_active_scenarios(
    active_scenarios: pd.DataFrame,
    path: Path | str = ACTIVE_STATE_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = active_scenarios.reindex(columns=ACTIVE_SCENARIO_COLUMNS)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    normalized.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)
    return path


def merge_scan_universe(
    top100_companies: list[MarketCapCompany],
    active_scenarios: pd.DataFrame,
) -> list[MarketCapCompany]:
    companies_by_ticker = {
        company.ticker: company
        for company in top100_companies
    }

    for _, row in active_scenarios.iterrows():
        ticker = str(row.get("티커", "")).strip().upper()
        if not ticker or ticker in companies_by_ticker:
            continue
        companies_by_ticker[ticker] = MarketCapCompany(
            rank=_rank_or_default(row.get("순위")),
            ticker=ticker,
            company=str(row.get("회사명", "")).strip(),
            market_cap=str(row.get("시가총액", "")).strip(),
        )

    return list(companies_by_ticker.values())


def summarize_ticker_cycles(
    company: MarketCapCompany,
    cycles: pd.DataFrame,
    full_table: pd.DataFrame,
    scan_date: pd.Timestamp | str,
    previous_active: pd.Series | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None, list[dict[str, object]]]:
    if full_table.empty:
        return [], None, []

    week_start = _week_start(scan_date)
    data_date = pd.Timestamp(full_table.index[-1]).normalize()
    previous_first = _as_timestamp(
        previous_active.get("1차신호일") if previous_active is not None else None
    )
    last_checked = _as_timestamp(
        previous_active.get("마지막확인일") if previous_active is not None else None
    )

    events: list[dict[str, object]] = []
    closed_results: list[dict[str, object]] = []
    event_keys: set[tuple[str, pd.Timestamp]] = set()

    for _, cycle in cycles.iterrows():
        first_date = _as_timestamp(cycle.get("FirstSignalDate"))
        if previous_first is not None and first_date is not None and first_date < previous_first:
            continue

        outcome = str(cycle.get("Outcome", ""))
        second_rejection_outcomes = {
            "2차 이격 과다 폐기",
            "2차 중기 구조 미충족 폐기",
        }
        second_stage = "2차 폐기" if outcome in second_rejection_outcomes else "2차 신호"
        signal_fields = (
            ("1차 신호", "FirstSignalDate"),
            (second_stage, "SecondSignalDate"),
            ("3차 신호", "ThirdDecisionDate"),
        )
        for stage, field in signal_fields:
            signal_date = _as_timestamp(cycle.get(field))
            should_report = _should_report_signal(
                signal_date,
                week_start=week_start,
                last_checked=last_checked,
                has_previous=previous_active is not None,
            )
            rule_reassessment = (
                not should_report
                and stage == "2차 폐기"
                and previous_active is not None
                and str(previous_active.get("현재상태", "")) == "3차 신호 대기"
                and previous_first == first_date
            )
            if not should_report and not rule_reassessment:
                continue

            key = (stage, signal_date)
            if key in event_keys:
                continue
            event_keys.add(key)

            event = _event_row(
                company=company,
                cycle=cycle,
                full_table=full_table,
                stage=stage,
                signal_date=signal_date,
                week_start=week_start,
                data_date=data_date,
            )
            if rule_reassessment:
                event["신호구분"] = "규칙 재평가"
            events.append(event)
            if stage in {"2차 폐기", "3차 신호"}:
                closed_results.append(_closed_result_row(event))

    active_cycles = cycles[
        cycles["Outcome"].isin(["2차 신호 대기", "3차 신호 대기"])
    ]
    active_row = None
    if not active_cycles.empty:
        active_cycle = active_cycles.iloc[-1]
        active_row = {
            "순위": company.rank,
            "티커": company.ticker,
            "회사명": company.company,
            "시가총액": company.market_cap,
            "현재상태": active_cycle["Outcome"],
            "1차신호일": _as_timestamp(active_cycle.get("FirstSignalDate")),
            "2차신호일": _as_timestamp(active_cycle.get("SecondSignalDate")),
            "마지막확인일": pd.Timestamp(scan_date).normalize(),
            "데이터기준일": data_date,
            "데이터상태": (
                "현재 주봉 포함" if data_date == week_start else "확정 주봉"
            ),
        }

    return events, active_row, closed_results


def preserve_failed_active_rows(
    active_scenarios: pd.DataFrame,
    failed_tickers: set[str],
) -> list[dict[str, object]]:
    if active_scenarios.empty or not failed_tickers:
        return []

    preserved = []
    for _, row in active_scenarios.iterrows():
        if str(row.get("티커", "")).upper() not in failed_tickers:
            continue
        item = row.reindex(ACTIVE_SCENARIO_COLUMNS).to_dict()
        item["데이터상태"] = "갱신 실패"
        preserved.append(item)
    return preserved


def _event_row(
    company: MarketCapCompany,
    cycle: pd.Series,
    full_table: pd.DataFrame,
    stage: str,
    signal_date: pd.Timestamp,
    week_start: pd.Timestamp,
    data_date: pd.Timestamp,
) -> dict[str, object]:
    price_row = _row_at(full_table, signal_date)
    outcome = str(cycle.get("Outcome", ""))
    if stage == "1차 신호":
        result = "2차 신호 대기"
    elif stage == "2차 신호":
        result = "3차 신호 대기"
    else:
        result = outcome

    ma_20 = price_row.get("MA_20")
    ma_50 = price_row.get("MA_50")
    ma_200 = price_row.get("MA_200")
    spread_pct = _ma20_over_ma50_spread_pct(ma_20, ma_50)

    return {
        "순위": company.rank,
        "티커": company.ticker,
        "회사명": company.company,
        "단계": stage,
        "신호일": signal_date,
        "결과": result,
        "1차신호일": _as_timestamp(cycle.get("FirstSignalDate")),
        "2차신호일": _as_timestamp(cycle.get("SecondSignalDate")),
        "3차판정일": _as_timestamp(cycle.get("ThirdDecisionDate")),
        "신호구분": "이번 주" if signal_date == week_start else "미확인 기간",
        "Close": price_row.get("Close"),
        "MA_5": price_row.get("MA_5"),
        "MA_20": ma_20,
        "MA_50": ma_50,
        "MA_200": ma_200,
        "MA20_50이격률": spread_pct,
        "데이터기준일": data_date,
    }


def _closed_result_row(event: dict[str, object]) -> dict[str, object]:
    row = {
        column: event.get(column)
        for column in CLOSED_RESULT_COLUMNS
    }
    row["종료일"] = event.get("신호일")
    return row


def _ma20_over_ma50_spread_pct(ma_20: object, ma_50: object) -> float:
    if pd.isna(ma_20) or pd.isna(ma_50) or float(ma_50) <= 0:
        return float("nan")
    return (float(ma_20) / float(ma_50) - 1) * 100


def _should_report_signal(
    signal_date: pd.Timestamp | None,
    week_start: pd.Timestamp,
    last_checked: pd.Timestamp | None,
    has_previous: bool,
) -> bool:
    if signal_date is None:
        return False
    if signal_date == week_start:
        return True
    if not has_previous:
        return False
    return last_checked is None or signal_date > last_checked


def _row_at(data: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    if date not in data.index:
        return pd.Series(dtype=object)
    row = data.loc[date]
    if isinstance(row, pd.DataFrame):
        return row.iloc[-1]
    return row


def _as_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


def _week_start(value: pd.Timestamp | str) -> pd.Timestamp:
    date = pd.Timestamp(value).normalize()
    return date - pd.Timedelta(days=date.weekday())


def _rank_or_default(value: object) -> int:
    try:
        if pd.isna(value):
            raise ValueError
        return int(value)
    except (TypeError, ValueError):
        return 9999
