from __future__ import annotations

import json

import pandas as pd
import pytest

import notifier
import telegram_api
from notifier import (
    build_failure_message,
    build_scan_message,
    notify_from_environment,
)
from telegram_api import NotificationError, send_telegram_message

SCAN_DATE = pd.Timestamp("2026-10-09")


def event(ticker: str, stage: str, result: str, **extra: object) -> dict[str, object]:
    return {
        "티커": ticker,
        "회사명": f"{ticker} Inc.",
        "단계": stage,
        "결과": result,
        "신호일": pd.Timestamp("2026-10-05"),
        "신호구분": "이번 주",
        "차트 강도": "",
        "검토등급": "",
        **extra,
    }


def test_message_lists_buy_signals_with_priority_first() -> None:
    events = pd.DataFrame(
        [
            event("V", "3차 신호", "매수 성공", **{"차트 강도": "12.3점", "검토등급": "일반검토"}),
            event("NVDA", "3차 신호", "매수 성공", **{"차트 강도": "82.5점", "검토등급": "우선검토"}),
            event("META", "2차 신호", "3차 신호 대기"),
            event("AAPL", "1차 신호", "2차 신호 대기"),
            event("KO", "3차 신호", "실패"),
        ]
    )
    active = pd.DataFrame({"현재상태": ["3차 신호 대기", "3차 신호 대기", "2차 신호 대기"]})

    message = build_scan_message(events, active, failure_count=1, scan_date=SCAN_DATE)

    assert message.splitlines() == [
        "📈 MMRM 주간 스캔 (2026-10-09)",
        "",
        "🔴 3차 매수 신호 2건 (우선검토 1건)",
        "• NVDA NVDA Inc. — 우선검토 82.5점",
        "• V V Inc. — 일반검토 12.3점",
        "",
        "🟠 2차 신호 1건: META",
        "🟢 1차 신호 1건: AAPL",
        "⚪ 3차 실패 1건: KO",
        "",
        "계속 관찰 3건 (3차 대기 2 · 2차 대기 1)",
        "데이터 오류 1건",
    ]


def test_message_without_signals_and_with_warning() -> None:
    message = build_scan_message(
        pd.DataFrame(columns=["티커", "단계", "결과"]),
        pd.DataFrame(columns=["현재상태"]),
        failure_count=0,
        scan_date=SCAN_DATE,
        warnings=["저장한 목록을 사용합니다.", ""],
    )

    assert message.splitlines() == [
        "📈 MMRM 주간 스캔 (2026-10-09)",
        "",
        "3차 매수 신호 없음",
        "",
        "계속 관찰 0건 (3차 대기 0 · 2차 대기 0)",
        "⚠️ 저장한 목록을 사용합니다.",
    ]


def test_signal_from_a_missed_week_shows_its_week() -> None:
    events = pd.DataFrame(
        [event("MA", "3차 신호", "매수 성공", 신호구분="미확인 기간", 신호일=pd.Timestamp("2026-09-28"))]
    )

    message = build_scan_message(events, pd.DataFrame(), 0, SCAN_DATE)

    assert "• MA MA Inc. (09/28 주 신호)" in message


def test_failure_message_names_the_error() -> None:
    message = build_failure_message("목록을 찾지 못했습니다.", SCAN_DATE)

    assert message.startswith("⚠️ MMRM 주간 스캔 실패 (2026-10-09)")
    assert "목록을 찾지 못했습니다." in message


def test_nothing_is_sent_without_telegram_settings(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(
        notifier, "send_telegram_message", lambda *args: pytest.fail("must not send")
    )

    assert notify_from_environment("hello") is False


def test_message_is_posted_to_the_configured_chat(monkeypatch) -> None:
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(telegram_api, "urlopen", fake_urlopen)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", " TEST-TOKEN ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "4242")

    assert notify_from_environment("스캔 결과") is True
    assert sent["url"] == "https://api.telegram.org/botTEST-TOKEN/sendMessage"
    assert sent["body"] == {"chat_id": "4242", "text": "스캔 결과"}


def test_delivery_errors_do_not_reveal_the_token(monkeypatch) -> None:
    from urllib.error import HTTPError

    def failing_urlopen(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(telegram_api, "urlopen", failing_urlopen)

    with pytest.raises(NotificationError) as error:
        send_telegram_message("text", "SECRET-TOKEN", "1")

    assert "SECRET-TOKEN" not in str(error.value)
    assert "봇 토큰이 틀렸습니다" in str(error.value)
    assert error.value.__cause__ is None


def test_scan_before_friday_close_is_marked_provisional() -> None:
    message = build_scan_message(
        pd.DataFrame(), pd.DataFrame(), 0, SCAN_DATE, provisional=True
    )

    assert message.splitlines()[2] == "※ 이번 주 장 마감 전의 잠정 결과입니다."
