"""Telegram notification for the headless weekly scan.

The bot token and chat id come from the TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID environment variables (GitHub Actions secrets).  Without
them the scan runs as before and nothing is sent.
"""

from __future__ import annotations

import http.client
import json
import os
from html import escape
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd

from watchlist import WATCHLIST_LABEL, is_weekend_traded, market_cap_rank_text


TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
MAX_LISTED_TICKERS = 15
ITEM_MARKER = "▸"
MAX_COMPANY_NAME_LENGTH = 16
COMPANY_SUFFIXES = (
    " & Co.",
    " Group",
    " Incorporated",
    " Corporation",
    " Company",
    " Limited",
    " Holdings",
    " Holding",
    " Inc.",
    " Inc",
    " Corp.",
    " Corp",
    " Co.",
    " Ltd.",
    " Ltd",
    " plc",
    " N.V.",
    " S.A.",
)
PRIORITY_GRADE = "우선검토"
NEW_LABEL = "신규"
WEEKDAY_NAMES = "월화수목금토일"
# (heading, 현재상태 of the active scenario, signal that puts a stock there)
WAITING_STAGES = (
    ("🟠 2차 신호 · 3차 대기 중", "3차 신호 대기", "2차 신호"),
    ("🟢 1차 신호 · 2차 대기 중", "2차 신호 대기", "1차 신호"),
)


class NotificationError(RuntimeError):
    """Raised when a Telegram message could not be delivered."""


def build_scan_message(
    events: pd.DataFrame,
    active_scenarios: pd.DataFrame,
    failure_count: int,
    scan_date: pd.Timestamp,
    warnings: list[str] | None = None,
    provisional: bool = False,
    data_basis: str = "",
) -> str:
    """Summarize one scan as a short Telegram message (HTML formatting).

    The message is the active-scenario list grouped by stage, with this
    week's changes marked in it: buy signals first, then every scenario still
    waiting for its 3차 or 2차 signal ("신규" for the ones that got there this
    week), then the scenarios that ended this week.

    Each stage has a heading followed by one line per stock:
    "▸ <rank> <ticker> (company) — score".  The rank is monospace and the
    ticker bold so rank, ticker, company and score read as separate items,
    and lines stay short enough not to wrap on a phone.

    ``provisional`` marks a scan run before the week's Friday close, whose
    current-week signals can still change; ``data_basis`` then says which
    session the latest prices come from.
    """
    lines = [f"📈 MMRM 주간 스캔 · {_date_label(scan_date)} · {'잠정' if provisional else '확정'}"]
    if provisional:
        lines.append(f"※ {escape(data_basis) if data_basis else '이번 주 장 마감 전'}")

    buys = _stage_rows(events, "3차 신호", "매수 성공")
    lines.append("")
    if buys.empty:
        lines.append("3차 신호 · 매수 신호 없음")
    else:
        grades = (
            buys["검토등급"]
            if "검토등급" in buys.columns
            else pd.Series("", index=buys.index, dtype=object)
        )
        # Priority-review signals first, then the stronger chart score.
        ordered = buys.assign(
            _priority=(grades == PRIORITY_GRADE),
            _score=pd.to_numeric(
                buys.get("차트 강도", pd.Series("", index=buys.index, dtype=object))
                .astype(str)
                .str.replace("점", "", regex=False),
                errors="coerce",
            ),
        ).sort_values(["_priority", "_score"], ascending=False, na_position="last")
        lines.append(_heading("🔴 3차 신호 · 매수 신호 발생", len(buys)))
        lines += _stock_lines(ordered, with_score=True)

    sections = [
        (label, _waiting_rows(active_scenarios, events, state, entry_stage))
        for label, state, entry_stage in WAITING_STAGES
    ]
    sections.append(("종료", _closed_rows(events)))
    for label, rows in sections:
        if not rows.empty:
            lines += ["", _heading(label, len(rows)), *_stock_lines(rows, with_score=False)]

    notes = []
    if failure_count:
        notes.append(f"데이터 오류 {failure_count}건")
    if not events.empty and "티커" in events.columns:
        weekend_tickers = sorted(
            {str(ticker) for ticker in events["티커"] if is_weekend_traded(ticker)}
        )
        if weekend_tickers:
            # Crypto keeps trading after the Friday scan, so its week is not over.
            notes.append(f"※ {escape(', '.join(weekend_tickers))}: 주말 거래 반영 전")
    for warning in warnings or []:
        if warning:
            notes.append(f"⚠️ {escape(warning)}")
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)


def build_failure_message(error: object, scan_date: pd.Timestamp) -> str:
    return (
        f"⚠️ MMRM 주간 스캔 실패 · {_date_label(scan_date)}\n\n"
        f"{escape(str(error))}\n\nGitHub Actions 실행 기록을 확인해 주세요."
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
        # Messages use Telegram's HTML formatting (<b>, <code>).
        data=json.dumps(
            {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        ).encode("utf-8"),
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
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # URLError, timeouts and dropped connections all land here.
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


def _date_label(date: pd.Timestamp) -> str:
    """"10/16 (금)" for the scan date."""
    date = pd.Timestamp(date)
    return f"{date:%m/%d} ({WEEKDAY_NAMES[date.weekday()]})"


def _heading(label: str, count: int) -> str:
    """Stage heading with the count in bold at the end, where it stands out."""
    return f"{label} <b>{count}건</b>"


def _waiting_rows(
    active_scenarios: pd.DataFrame,
    events: pd.DataFrame,
    state: str,
    entry_stage: str,
) -> pd.DataFrame:
    """Active scenarios in ``state``: this week's arrivals first, then by rank."""
    if active_scenarios.empty or not {"현재상태", "티커"} <= set(active_scenarios.columns):
        return active_scenarios.iloc[0:0]
    rows = active_scenarios[active_scenarios["현재상태"] == state]
    entered = _stage_rows(events, entry_stage)
    is_new = rows["티커"].isin(set(entered["티커"]) if "티커" in entered.columns else set())
    ranks = (
        pd.to_numeric(rows["순위"], errors="coerce")
        if "순위" in rows.columns
        else pd.Series(float("nan"), index=rows.index)
    )
    return rows.assign(
        _note=is_new.map({True: NEW_LABEL, False: ""}), _new=is_new, _rank=ranks
    ).sort_values(["_new", "_rank"], ascending=[False, True], na_position="last")


def _closed_rows(events: pd.DataFrame) -> pd.DataFrame:
    """Scenarios that ended this week, each noted with how it ended."""
    closed = [
        rows.assign(_note=note)
        for note, rows in (
            ("3차 실패", _stage_rows(events, "3차 신호", "실패")),
            ("2차 폐기", _stage_rows(events, "2차 폐기")),
        )
        if not rows.empty
    ]
    return pd.concat(closed) if closed else events.iloc[0:0]


def _stock_lines(rows: pd.DataFrame, with_score: bool) -> list[str]:
    lines = [
        f"{ITEM_MARKER} {_stock_line(row, with_score)}"
        for _, row in rows.head(MAX_LISTED_TICKERS).iterrows()
    ]
    if len(rows) > MAX_LISTED_TICKERS:
        lines.append(f"{ITEM_MARKER} 외 {len(rows) - MAX_LISTED_TICKERS}건")
    return lines


def _stock_line(row: pd.Series, with_score: bool) -> str:
    """"<code>59</code> <b>SAP</b> (SAP SE) — 79점 · 우선검토" for one stock."""
    parts = []
    rank = _rank_label(row.get("순위"))
    if rank:
        parts.append(f"<code>{escape(rank)}</code>")
    parts.append(f"<b>{escape(str(row.get('티커', '')))}</b>")
    company = short_company_name(row.get("회사명"))
    if company:
        parts.append(f"({escape(company)})")
    text = " ".join(parts)

    details = []
    if with_score:
        score = row.get("_score")
        if score is not None and not pd.isna(score):
            details.append(f"{int(float(score) + 0.5)}점")  # 30.5 -> 31
        if row.get("_priority"):
            details.append(PRIORITY_GRADE)
    note = row.get("_note")
    if isinstance(note, str) and note:
        details.append(note)
    if details:
        text += " — " + " · ".join(details)
    if row.get("신호구분") == "미확인 기간":
        signal_date = pd.to_datetime(row.get("신호일"), errors="coerce")
        if not pd.isna(signal_date):
            text += f" ({signal_date:%m/%d} 주)"
    return text


def _rank_label(rank: object) -> str:
    """Bare rank number, or "관심" / "순위 밖" for unranked tickers."""
    text = market_cap_rank_text(rank)
    return text[:-1] if text.endswith("위") else text


def short_company_name(company: object) -> str:
    """Company name short enough for one phone line.

    Drops the watchlist label and legal suffixes ("Mastercard Incorporated" ->
    "Mastercard") and cuts what is still too long.
    """
    if company is None or (isinstance(company, float) and pd.isna(company)):
        return ""
    name = str(company).strip()
    if name.startswith(WATCHLIST_LABEL):
        name = name[len(WATCHLIST_LABEL) :].strip()
    if name == "nan":
        return ""
    shortened = True
    while shortened:
        shortened = False
        for suffix in COMPANY_SUFFIXES:
            if name.lower().endswith(suffix.lower()) and len(name) > len(suffix):
                name = name[: -len(suffix)].rstrip(" ,")
                shortened = True
    if len(name) > MAX_COMPANY_NAME_LENGTH:
        name = name[: MAX_COMPANY_NAME_LENGTH - 1].rstrip() + "…"
    return name
