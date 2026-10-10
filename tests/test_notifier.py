from __future__ import annotations

import json

import pandas as pd
import pytest

import notifier
from notifier import (
    NotificationError,
    build_failure_message,
    build_scan_message,
    notify_from_environment,
    send_telegram_message,
    short_company_name,
)

SCAN_DATE = pd.Timestamp("2026-10-16")


def event(ticker: str, company: str, stage: str, result: str, rank: int, **extra: object) -> dict:
    return {
        "순위": rank,
        "티커": ticker,
        "회사명": company,
        "단계": stage,
        "결과": result,
        "신호일": pd.Timestamp("2026-10-12"),
        "신호구분": "이번 주",
        "차트 강도": "",
        "검토등급": "",
        **extra,
    }


def buy(ticker: str, company: str, rank: int, score: str, grade: str, **extra: object) -> dict:
    return event(
        ticker, company, "3차 신호", "매수 성공", rank,
        **{"차트 강도": score, "검토등급": grade, **extra},
    )


def test_each_stock_line_has_rank_ticker_company_and_score() -> None:
    events = pd.DataFrame(
        [
            buy("V", "Visa Inc.", 18, "25.6점", "일반검토"),
            buy("SAP", "SAP SE", 59, "79.2점", "우선검토"),
            buy("BLK", "BlackRock, Inc.", 84, "22.9점", "일반검토"),
            buy("MA", "Mastercard Incorporated", 23, "30.5점", "일반검토"),
            event("META", "Meta Platforms, Inc.", "2차 신호", "3차 신호 대기", 8),
            event("AAPL", "Apple Inc.", "1차 신호", "2차 신호 대기", 2),
            event("KO", "The Coca-Cola Company", "3차 신호", "실패", 41),
        ]
    )
    active = pd.DataFrame({"현재상태": ["3차 신호 대기", "3차 신호 대기", "2차 신호 대기"]})

    message = build_scan_message(events, active, failure_count=1, scan_date=SCAN_DATE)

    assert message.splitlines() == [
        "📈 MMRM 주간 스캔 · 10/16",
        "",
        "🔴 3차 매수 신호 4건",
        "▸ <code>59</code> <b>SAP</b> (SAP SE) — 79점 · 우선검토",
        "▸ <code>23</code> <b>MA</b> (Mastercard) — 31점",
        "▸ <code>18</code> <b>V</b> (Visa) — 26점",
        "▸ <code>84</code> <b>BLK</b> (BlackRock) — 23점",
        "",
        "🟠 2차 신호 1건",
        "▸ <code>8</code> <b>META</b> (Meta Platforms)",
        "",
        "🟢 1차 신호 1건",
        "▸ <code>2</code> <b>AAPL</b> (Apple)",
        "",
        "⚪ 3차 실패 1건",
        "▸ <code>41</code> <b>KO</b> (The Coca-Cola)",
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
        "📈 MMRM 주간 스캔 · 10/16",
        "",
        "3차 매수 신호 없음",
        "",
        "계속 관찰 0건 (3차 대기 0 · 2차 대기 0)",
        "⚠️ 저장한 목록을 사용합니다.",
    ]


def test_watchlist_and_unranked_tickers_show_a_label_instead_of_a_rank() -> None:
    events = pd.DataFrame(
        [
            buy("BTC-USD", "[관심] 비트코인", 9000, "해당 없음", ""),
            buy("OLD", "Old Co.", 9999, "40.0점", "일반검토"),
        ]
    )

    lines = build_scan_message(events, pd.DataFrame(), 0, SCAN_DATE).splitlines()

    assert "▸ <code>순위 밖</code> <b>OLD</b> (Old) — 40점" in lines
    assert "▸ <code>관심</code> <b>BTC-USD</b> (비트코인)" in lines
    assert lines[-1] == "※ BTC-USD: 주말 거래 반영 전"


def test_signal_from_a_missed_week_shows_its_week() -> None:
    events = pd.DataFrame(
        [buy("MA", "Mastercard Incorporated", 23, "30.5점", "일반검토",
             신호구분="미확인 기간", 신호일=pd.Timestamp("2026-09-28"))]
    )

    message = build_scan_message(events, pd.DataFrame(), 0, SCAN_DATE)

    assert "▸ <code>23</code> <b>MA</b> (Mastercard) — 31점 (09/28 주)" in message.splitlines()


def test_events_without_rank_grade_or_score_columns_are_still_listed() -> None:
    events = pd.DataFrame(
        [{"티커": "MA", "회사명": "Mastercard", "단계": "3차 신호", "결과": "매수 성공"}]
    )

    lines = build_scan_message(events, pd.DataFrame(), 0, SCAN_DATE).splitlines()

    assert lines[2:4] == ["🔴 3차 매수 신호 1건", "▸ <b>MA</b> (Mastercard)"]


def test_provisional_message_has_a_one_line_notice() -> None:
    with_basis = build_scan_message(
        pd.DataFrame(), pd.DataFrame(), 0, SCAN_DATE, provisional=True, data_basis="목요일 종가까지 반영"
    )
    without_basis = build_scan_message(pd.DataFrame(), pd.DataFrame(), 0, SCAN_DATE, provisional=True)

    assert with_basis.splitlines()[:3] == [
        "📈 MMRM 주간 스캔 · 10/16",
        "※ 잠정 결과 · 목요일 종가까지 반영",
        "",
    ]
    assert without_basis.splitlines()[1] == "※ 잠정 결과 · 이번 주 장 마감 전"


def test_data_basis_follows_the_new_york_clock() -> None:
    from weekly_scan import data_basis_text

    friday_before_open = pd.Timestamp("2026-10-16 01:00", tz="America/New_York")
    friday_session = pd.Timestamp("2026-10-16 11:00", tz="America/New_York")
    monday_before_open = pd.Timestamp("2026-10-19 08:00", tz="America/New_York")
    saturday = pd.Timestamp("2026-10-17 10:00", tz="America/New_York")

    assert data_basis_text(friday_before_open) == "목요일 종가까지 반영"
    assert data_basis_text(friday_session) == "오늘 미국 장중 가격 포함"
    assert data_basis_text(monday_before_open) == "금요일 종가까지 반영"
    assert data_basis_text(saturday) == "최근 거래일 종가까지 반영"


def test_company_names_are_shortened_to_fit_one_line() -> None:
    assert short_company_name("Mastercard Incorporated") == "Mastercard"
    assert short_company_name("Meta Platforms, Inc.") == "Meta Platforms"
    assert short_company_name("JPMorgan Chase & Co.") == "JPMorgan Chase"
    assert short_company_name("UnitedHealth Group Incorporated") == "UnitedHealth"
    assert short_company_name("SAP SE") == "SAP SE"
    assert short_company_name("[관심] 비트코인") == "비트코인"
    assert short_company_name("Taiwan Semiconductor Manufacturing Company Limited") == "Taiwan Semicond…"
    assert short_company_name(None) == ""
    assert short_company_name(float("nan")) == ""


def test_text_from_outside_is_escaped_for_html() -> None:
    message = build_scan_message(
        pd.DataFrame([event("A&B", "AT&T Inc.", "2차 신호", "3차 신호 대기", 60)]),
        pd.DataFrame(),
        0,
        SCAN_DATE,
        warnings=["<목록> 오류 & 재시도"],
    )
    failure = build_failure_message("HTTP <500> & retry", SCAN_DATE)

    assert "▸ <code>60</code> <b>A&amp;B</b> (AT&amp;T)" in message.splitlines()
    assert "⚠️ &lt;목록&gt; 오류 &amp; 재시도" in message
    assert failure.startswith("⚠️ MMRM 주간 스캔 실패 · 10/16")
    assert "HTTP &lt;500&gt; &amp; retry" in failure


def test_nothing_is_sent_without_telegram_settings(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(
        notifier, "send_telegram_message", lambda *args: pytest.fail("must not send")
    )

    assert notify_from_environment("hello") is False


def test_message_is_posted_as_html_to_the_configured_chat(monkeypatch) -> None:
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

    monkeypatch.setattr(notifier, "urlopen", fake_urlopen)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", " TEST-TOKEN ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "4242")

    assert notify_from_environment("<b>SAP</b>") is True
    assert sent["url"] == "https://api.telegram.org/botTEST-TOKEN/sendMessage"
    assert sent["body"] == {"chat_id": "4242", "text": "<b>SAP</b>", "parse_mode": "HTML"}


def test_delivery_errors_do_not_reveal_the_token(monkeypatch) -> None:
    from urllib.error import HTTPError

    def failing_urlopen(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(notifier, "urlopen", failing_urlopen)

    with pytest.raises(NotificationError) as error:
        send_telegram_message("text", "SECRET-TOKEN", "1")

    assert "SECRET-TOKEN" not in str(error.value)
    assert "봇 토큰이 틀렸습니다" in str(error.value)
    assert error.value.__cause__ is None
