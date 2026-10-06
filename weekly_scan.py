from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from chart_strength import (
    ChartStrengthReferenceError,
    annotate_scan_events,
    load_chart_strength_reference,
)
from data_provider import load_weekly_data, weekly_bar_in_progress
from indicators import calculate_indicators
from market_cap_provider import (
    MarketCapCompany,
    MarketCapLoadError,
    fetch_us_top_market_cap_result,
)
from market_context import annotate_sp500_status, load_sp500_context
from notifier import (
    NotificationError,
    build_failure_message,
    build_scan_message,
    notify_from_environment,
)
from scanner import scan_signal_cycles
from scenario_tracker import (
    ACTIVE_SCENARIO_COLUMNS,
    CLOSED_RESULT_COLUMNS,
    SCAN_EVENT_COLUMNS,
    latest_scan_date,
    load_active_scenarios,
    market_today,
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
    scan_date = market_today()
    warnings: list[str] = []
    try:
        previous_active = load_active_scenarios()
        print("미국 시가총액 Top 100 목록과 활성 시나리오를 불러옵니다...")
        try:
            top100 = fetch_us_top_market_cap_result(limit=TOP100_LIMIT)
            top100_companies = top100.companies
            if top100.warning:
                print(f"경고: {top100.warning}")
                warnings.append(top100.warning)
        except MarketCapLoadError as exc:
            # Keep following scenarios that are already active even when the
            # ranking cannot be read; only new first signals are missed.
            if previous_active.empty:
                raise
            top100_companies = []
            print(f"경고: {exc} 활성 시나리오 종목만 스캔합니다.")
            warnings.append(f"{exc} 활성 시나리오 종목만 스캔했습니다.")
        companies = merge_scan_universe(top100_companies, previous_active)
        print("S&P500 주봉 데이터를 한 번 갱신합니다...")
        try:
            sp500_data, sp500_load = load_sp500_context(
                force_refresh=True,
                max_cache_age_seconds=None,
            )
            sp500_warning = sp500_load.warning
        except Exception as exc:
            sp500_data = pd.DataFrame()
            sp500_warning = f"S&P500 상태 확인 실패: {exc}"

        full_tables_by_ticker: dict[str, pd.DataFrame] = {}
        events, active_rows, closed_results, failures = scan_companies(
            companies,
            scan_date,
            previous_active,
            progress_label="스캔 중",
            full_tables_by_ticker=full_tables_by_ticker,
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
                full_tables_by_ticker=full_tables_by_ticker,
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
        try:
            chart_strength_reference = load_chart_strength_reference()
            reference_error = None
        except ChartStrengthReferenceError as exc:
            chart_strength_reference = None
            reference_error = str(exc)
        events_df, _details = annotate_scan_events(
            events_df,
            full_tables_by_ticker,
            chart_strength_reference,
            reference_error=reference_error,
        )
        events_df = annotate_sp500_status(
            events_df,
            sp500_data,
            reference_date_columns=("신호일",),
        )
        active_df = _sorted_frame(
            active_rows,
            ACTIVE_SCENARIO_COLUMNS,
            by=["순위", "티커"],
            ascending=[True, True],
        ).drop_duplicates(subset=["티커"], keep="last")
        active_df = annotate_sp500_status(active_df, sp500_data, use_latest=True)
        closed_df = _sorted_frame(
            closed_results,
            CLOSED_RESULT_COLUMNS,
            by=["종료일", "순위"],
            ascending=[False, True],
        )
        closed_df = annotate_sp500_status(
            closed_df,
            sp500_data,
            reference_date_columns=("3차판정일", "종료일", "2차신호일"),
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
        _send_notification(build_failure_message(exc, scan_date))
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
    if sp500_warning:
        print(sp500_warning)
    for path in saved_paths:
        print(f"저장: {path}")
    if sp500_warning:
        warnings.append(sp500_warning)
    week_start = scan_date - pd.Timedelta(days=scan_date.weekday())
    _send_notification(
        build_scan_message(
            events_df,
            active_df,
            len(failures_df),
            scan_date,
            warnings,
            provisional=weekly_bar_in_progress(week_start),
        )
    )
    return 0


def _send_notification(text: str) -> None:
    """Send the Telegram summary; a delivery problem never fails the scan."""
    try:
        sent = notify_from_environment(text)
    except NotificationError as exc:
        print(f"경고: {exc}", file=sys.stderr)
        return
    print("텔레그램 알림을 보냈습니다." if sent else "텔레그램 설정이 없어 알림을 건너뜁니다.")


def scan_companies(
    companies: list[MarketCapCompany],
    scan_date: pd.Timestamp,
    previous_active: pd.DataFrame,
    progress_label: str,
    full_tables_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Scan each company; ``full_tables_by_ticker`` collects indicator tables."""
    events: list[dict[str, object]] = []
    active_rows: list[dict[str, object]] = []
    closed_results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    previous_by_ticker = {
        str(row["티커"]).upper(): row
        for _, row in previous_active.iterrows()
    }
    last_scan_date = latest_scan_date(previous_active)

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
            if full_tables_by_ticker is not None:
                full_tables_by_ticker[company.ticker.upper()] = full_table
            ticker_events, active_row, ticker_closed = summarize_ticker_cycles(
                company,
                cycles,
                full_table,
                scan_date,
                previous_active=previous_by_ticker.get(company.ticker.upper()),
                last_scan_date=last_scan_date,
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
