"""Telegram notification for the headless weekly scan.

The bot token and chat id come from the TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID environment variables (GitHub Actions secrets).  Without
them the scan runs as before and nothing is sent.
"""

from __future__ import annotations

import os

import pandas as pd

from telegram_api import (
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_TOKEN_ENV,
    send_telegram_message,
)
from watchlist import is_weekend_traded


MAX_LISTED_TICKERS = 15
PRIORITY_GRADE = "우선검토"


def build_scan_message(
    events: pd.DataFrame,
    active_scenarios: pd.DataFrame,
    failure_count: int,
    scan_date: pd.Timestamp,
    warnings: list[str] | None = None,
    provisional: bool = False,
) -> str:
    """Summarize one scan as a short Telegram message, buy signals first.

    ``provisional`` marks a scan run before the week's Friday close, whose
    current-week signals can still change.
    """
    lines = [f"📈 MMRM 주간 스캔 ({pd.Timestamp(scan_date):%Y-%m-%d})", ""]
    if provisional:
        lines += ["※ 이번 주 장 마감 전의 잠정 결과입니다.", ""]

    buys = _stage_rows(events, "3차 신호", "매수 성공")
    if buys.empty:
        lines.append("3차 매수 신호 없음")
    else:
        grades = (
            buys["검토등급"]
            if "검토등급" in buys.columns
            else pd.Series("", index=buys.index, dtype=object)
        )
        priority = int((grades == PRIORITY_GRADE).sum())
        suffix = f" (우선검토 {priority}건)" if priority else ""
        lines.append(f"🔴 3차 매수 신호 {len(buys)}건{suffix}")
        # Priority-review signals first, then the stronger chart score.
        ordered = buys.assign(
            _priority=(grades == PRIORITY_GRADE),
            _score=pd.to_numeric(
                buys.get("차트 강도", pd.Series(dtype=object))
                .astype(str)
                .str.replace("점", "", regex=False),
                errors="coerce",
            ),
        ).sort_values(["_priority", "_score"], ascending=False, na_position="last")
        for _, row in ordered.head(MAX_LISTED_TICKERS).iterrows():
            lines.append(f"• {_buy_line(row)}")
        if len(buys) > MAX_LISTED_TICKERS:
            lines.append(f"• 외 {len(buys) - MAX_LISTED_TICKERS}건")

    other_stages = (
        ("🟠 2차 신호", _stage_rows(events, "2차 신호")),
        ("🟢 1차 신호", _stage_rows(events, "1차 신호")),
        ("⚪ 2차 폐기", _stage_rows(events, "2차 폐기")),
        ("⚪ 3차 실패", _stage_rows(events, "3차 신호", "실패")),
    )
    stage_lines = [
        f"{label} {len(rows)}건: {_ticker_list(rows)}"
        for label, rows in other_stages
        if not rows.empty
    ]
    if stage_lines:
        lines += ["", *stage_lines]

    states = (
        active_scenarios["현재상태"]
        if not active_scenarios.empty and "현재상태" in active_scenarios.columns
        else pd.Series(dtype=object)
    )
    lines += [
        "",
        f"계속 관찰 {len(active_scenarios)}건 "
        f"(3차 대기 {int((states == '3차 신호 대기').sum())} · "
        f"2차 대기 {int((states == '2차 신호 대기').sum())})",
    ]
    if failure_count:
        lines.append(f"데이터 오류 {failure_count}건")
    if not events.empty and "티커" in events.columns:
        weekend_tickers = sorted(
            {str(ticker) for ticker in events["티커"] if is_weekend_traded(ticker)}
        )
        if weekend_tickers:
            # Crypto keeps trading after the Friday scan, so its week is not over.
            lines.append(
                f"※ {', '.join(weekend_tickers)}: 주말 거래 반영 전 결과입니다."
            )
    for warning in warnings or []:
        if warning:
            lines.append(f"⚠️ {warning}")
    return "\n".join(lines)


def build_failure_message(error: object, scan_date: pd.Timestamp) -> str:
    return (
        f"⚠️ MMRM 주간 스캔 실패 ({pd.Timestamp(scan_date):%Y-%m-%d})\n\n"
        f"{error}\n\nGitHub Actions 실행 기록을 확인해 주세요."
    )


def notify_from_environment(text: str) -> bool:
    """Send ``text`` when Telegram is configured; return whether it was sent."""
    token = os.environ.get(TELEGRAM_TOKEN_ENV, "").strip()
    chat_id = os.environ.get(TELEGRAM_CHAT_ID_ENV, "").strip()
    if not token or not chat_id:
        return False
    send_telegram_message(text, token, chat_id)
    return True


def _stage_rows(
    events: pd.DataFrame,
    stage: str,
    result: str | None = None,
) -> pd.DataFrame:
    if events.empty or "단계" not in events.columns:
        return events.iloc[0:0]
    mask = events["단계"] == stage
    if result is not None:
        mask &= events["결과"] == result
    return events[mask]


def _buy_line(row: pd.Series) -> str:
    text = f"{row.get('티커', '')} {row.get('회사명', '')}".strip()
    grade = str(row.get("검토등급", "") or "").strip()
    score = str(row.get("차트 강도", "") or "").strip()
    detail = " ".join(
        part for part in (grade, score) if part and part not in {"nan", "해당 없음"}
    )
    if detail:
        text += f" — {detail}"
    if row.get("신호구분") == "미확인 기간":
        signal_date = pd.to_datetime(row.get("신호일"), errors="coerce")
        if not pd.isna(signal_date):
            text += f" ({signal_date:%m/%d} 주 신호)"
    return text


def _ticker_list(rows: pd.DataFrame) -> str:
    tickers = [str(ticker) for ticker in rows["티커"].tolist()]
    listed = ", ".join(tickers[:MAX_LISTED_TICKERS])
    if len(tickers) > MAX_LISTED_TICKERS:
        listed += f" 외 {len(tickers) - MAX_LISTED_TICKERS}건"
    return listed
