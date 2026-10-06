"""Telegram notification for the headless weekly scan.

The bot token and chat id come from the TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID environment variables (GitHub Actions secrets).  Without
them the scan runs as before and nothing is sent.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
MAX_LISTED_TICKERS = 15
PRIORITY_GRADE = "우선검토"


class NotificationError(RuntimeError):
    """Raised when a Telegram message could not be delivered."""


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
        priority = int((buys.get("검토등급") == PRIORITY_GRADE).sum())
        suffix = f" (우선검토 {priority}건)" if priority else ""
        lines.append(f"🔴 3차 매수 신호 {len(buys)}건{suffix}")
        # Priority-review signals first, then the stronger chart score.
        ordered = buys.assign(
            _priority=(buys.get("검토등급") == PRIORITY_GRADE),
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


def send_telegram_message(text: str, token: str, chat_id: str) -> None:
    request = Request(
        TELEGRAM_API_URL.format(token=token),
        data=json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Error text never includes the request URL, because the URL holds the token.
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise NotificationError(
            f"텔레그램 전송 실패: HTTP {exc.code} ({_telegram_error_hint(exc.code)})"
        ) from None
    except (URLError, TimeoutError, ValueError) as exc:
        raise NotificationError(
            f"텔레그램 전송 실패: {type(exc).__name__}"
        ) from None
    if not payload.get("ok"):
        raise NotificationError("텔레그램 전송 실패: 응답이 올바르지 않습니다.")


def _telegram_error_hint(status: int) -> str:
    return {
        400: "채팅 ID가 틀렸거나 봇에게 먼저 말을 걸지 않았습니다",
        401: "봇 토큰이 틀렸습니다",
        403: "봇을 차단했거나 대화를 시작하지 않았습니다",
        404: "봇 토큰이 틀렸습니다",
    }.get(status, "잠시 후 다시 시도해 주세요")


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
    detail = " ".join(part for part in (grade, score) if part and part != "nan")
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
