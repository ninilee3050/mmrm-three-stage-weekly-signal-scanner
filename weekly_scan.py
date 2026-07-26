from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from data_provider import load_weekly_data
from indicators import calculate_indicators
from market_cap_provider import MarketCapCompany, fetch_us_top_market_cap
from scanner import scan_signal_cycles
from scenario_tracker import (
    ACTIVE_SCENARIO_COLUMNS,
    CLOSED_RESULT_COLUMNS,
    SCAN_EVENT_COLUMNS,
    load_active_scenarios,
    merge_scan_universe,
    preserve_failed_active_rows,
    save_active_scenarios,
    summarize_ticker_cycles,
)


OUTPUT_DIR = Path("outputs")
TOP100_LIMIT = 100
RETRY_DELAY_SECONDS = 2
SCAN_FAILURE_COLUMNS = ["순위", "티커", "회사명", "시가총액", "오류"]


def main() -> int:
    try:
        scan_date = pd.Timestamp.today().normalize()
        previous_active = load_active_scenarios()
        print("미국 시가총액 Top 100 목록과 활성 시나리오를 불러옵니다...")
        top100_companies = fetch_us_top_market_cap(limit=TOP100_LIMIT)
        companies = merge_scan_universe(top100_companies, previous_active)

        events, active_rows, closed_results, failures = scan_companies(
            companies,
            scan_date,
            previous_active,
            progress_label="스캔 중",
        )
        if failures:
            print(f"실패한 {len(failures)}개 종목을 한 번 더 시도합니다...")
            time.sleep(RETRY_DELAY_SECONDS)
            retry_companies = [failure["company"] for failure in failures]
            retry_events, retry_active, retry_closed, retry_failures = scan_companies(
                retry_companies,
                scan_date,
                previous_active,
                progress_label="재시도 중",
            )
            events.extend(retry_events)
            active_rows.extend(retry_active)
            closed_results.extend(retry_closed)
            failures = retry_failures

        failed_tickers = {failure["company"].ticker for failure in failures}
        active_rows.extend(
            preserve_failed_active_rows(previous_active, failed_tickers)
        )

        events_df = _sorted_frame(
            events,
            SCAN_EVENT_COLUMNS,
            by=["신호일", "순위"],
            ascending=[False, True],
        )
        active_df = _sorted_frame(
            active_rows,
            ACTIVE_SCENARIO_COLUMNS,
            by=["순위", "티커"],
            ascending=[True, True],
        ).drop_duplicates(subset=["티커"], keep="last")
        closed_df = _sorted_frame(
            closed_results,
            CLOSED_RESULT_COLUMNS,
            by=["종료일", "순위"],
            ascending=[False, True],
        )
        failures_df = pd.DataFrame(
            [_failure_row(failure["company"], failure["error"]) for failure in failures],
            columns=SCAN_FAILURE_COLUMNS,
        )

        save_active_scenarios(active_df)
        saved_paths = save_scan_outputs(
            events_df,
            active_df,
            closed_df,
            failures_df,
            scan_date,
        )
    except Exception as exc:
        print(f"스캔 실패: {exc}", file=sys.stderr)
        return 1

    first_count = int((events_df["단계"] == "1차 신호").sum()) if not events_df.empty else 0
    second_count = int((events_df["단계"] == "2차 신호").sum()) if not events_df.empty else 0
    second_rejection_count = int(
        (events_df["단계"] == "2차 폐기").sum()
    ) if not events_df.empty else 0
    third_count = int(
        ((events_df["단계"] == "3차 신호") & (events_df["결과"] == "매수 성공")).sum()
    ) if not events_df.empty else 0
    signal_failure_count = int(
        ((events_df["단계"] == "3차 신호") & (events_df["결과"] == "실패")).sum()
    ) if not events_df.empty else 0

    print("")
    print(
        f"스캔 완료: 신규 1차 {first_count}개 / 2차 {second_count}개 / "
        f"2차 폐기 {second_rejection_count}개 / 3차 매수 {third_count}개 / "
        f"신호 실패 {signal_failure_count}개 / "
        f"계속 관찰 {len(active_df)}개 / 데이터 오류 {len(failures_df)}개"
    )
    for path in saved_paths:
        print(f"저장: {path}")
    return 0


def scan_companies(
    companies: list[MarketCapCompany],
    scan_date: pd.Timestamp,
    previous_active: pd.DataFrame,
    progress_label: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    events: list[dict[str, object]] = []
    active_rows: list[dict[str, object]] = []
    closed_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    previous_by_ticker = {
        str(row["티커"]).upper(): row
        for _, row in previous_active.iterrows()
    }

    for index, company in enumerate(companies, start=1):
        print(f"{progress_label}... {index}/{len(companies)} {company.ticker}")
        try:
            raw_data = load_weekly_data(
                company.ticker,
                include_current_week=True,
                force_refresh=True,
            )
            calculated = calculate_indicators(raw_data)
            cycles, full_table = scan_signal_cycles(calculated)
            ticker_events, active_row, ticker_closed = summarize_ticker_cycles(
                company,
                cycles,
                full_table,
                scan_date,
                previous_active=previous_by_ticker.get(company.ticker.upper()),
            )
            events.extend(ticker_events)
            closed_results.extend(ticker_closed)
            if active_row is not None:
                active_rows.append(active_row)
        except Exception as exc:
            failures.append({"company": company, "error": str(exc)})

    return events, active_rows, closed_results, failures


def save_scan_outputs(
    events: pd.DataFrame,
    active_scenarios: pd.DataFrame,
    closed_results: pd.DataFrame,
    failures: pd.DataFrame,
    scan_date: pd.Timestamp,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_text = scan_date.strftime("%Y-%m-%d")
    event_path = output_dir / f"signal_events_{date_text}.csv"
    active_path = output_dir / f"active_scenarios_{date_text}.csv"
    closed_path = output_dir / f"closed_results_{date_text}.csv"
    failure_path = output_dir / f"scan_failures_{date_text}.csv"

    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    active_scenarios.to_csv(active_path, index=False, encoding="utf-8-sig")
    closed_results.to_csv(closed_path, index=False, encoding="utf-8-sig")
    failures.to_csv(failure_path, index=False, encoding="utf-8-sig")
    return event_path, active_path, closed_path, failure_path


def _failure_row(company: MarketCapCompany, error: str) -> dict[str, object]:
    return {
        "순위": company.rank,
        "티커": company.ticker,
        "회사명": company.company,
        "시가총액": company.market_cap,
        "오류": error,
    }


def _sorted_frame(
    rows: list[dict[str, object]],
    columns: list[str],
    by: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=columns)
    if data.empty:
        return data
    return data.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
