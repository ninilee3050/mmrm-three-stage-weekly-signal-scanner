"""Regression tests for the second full review."""

from __future__ import annotations

import http.client
import math

import pandas as pd
import pytest

import market_cap_provider
import notifier
import weekly_scan
from data_provider import is_weekend_traded, weekly_bar_in_progress
from gui.tables import restored_window_placement
from market_cap_provider import MarketCapLoadError
from notifier import NotificationError, build_scan_message, send_telegram_message
from scanner import scan_signal_cycles
from scenario_tracker import merge_scan_universe
from test_signal_cycles import make_frame, successful_cycle_setup


def test_crypto_week_closes_monday_utc_not_friday() -> None:
    week = pd.Timestamp("2026-10-05")
    friday_evening_ny = pd.Timestamp("2026-10-09 17:00")  # stocks are done
    sunday_night_utc = pd.Timestamp("2026-10-11 23:30", tz="UTC")
    monday_utc = pd.Timestamp("2026-10-12 00:00", tz="UTC")

    assert not weekly_bar_in_progress(week, now=friday_evening_ny)
    assert weekly_bar_in_progress(week, now=friday_evening_ny, weekend_traded=True)
    assert weekly_bar_in_progress(week, now=sunday_night_utc, weekend_traded=True)
    assert not weekly_bar_in_progress(week, now=monday_utc, weekend_traded=True)
    assert is_weekend_traded("BTC-USD") and not is_weekend_traded("SCHD")


def test_crypto_return_landing_on_the_weekend_bar_stays_in_progress() -> None:
    rows = successful_cycle_setup() + [{"Close": 110.0} for _ in range(13)]
    data = make_frame(rows)
    saturday = data.index[-1] + pd.Timedelta(days=5, hours=12)

    stock_cycles, _ = scan_signal_cycles(data, now=saturday)
    crypto_cycles, _ = scan_signal_cycles(data, now=saturday, weekend_traded=True)

    assert stock_cycles.loc[0, "Return3MStatus"] == "확정"
    assert crypto_cycles.loc[0, "Return3MStatus"] == "진행 중"
    assert math.isnan(crypto_cycles.loc[0, "Return3M"])


def test_window_position_is_kept_on_a_second_monitor() -> None:
    saved = {"width": 1600, "height": 900, "x": 2100, "y": 80}
    two_monitors = (0, 0, 3840, 1080)

    kept = restored_window_placement(saved, 3501, 820, 1920, 1080, virtual_bounds=two_monitors)
    primary_only = restored_window_placement(saved, 3501, 820, 1920, 1080)
    left_monitor = restored_window_placement(
        {**saved, "x": -1800}, 3501, 820, 1920, 1080, virtual_bounds=(-1920, 0, 1920, 1080)
    )

    assert kept == (1600, 900, 2100, 80)
    assert primary_only[2] == 320  # pulled back onto the primary monitor
    assert left_monitor[2] == -1800


def test_dropped_connections_fall_back_like_other_ranking_errors(monkeypatch) -> None:
    def dropped(request, timeout):
        raise http.client.RemoteDisconnected("Remote end closed connection")

    monkeypatch.setattr(market_cap_provider, "urlopen", dropped)

    with pytest.raises(MarketCapLoadError, match="RemoteDisconnected"):
        market_cap_provider._download_list_page(market_cap_provider.US_EXCHANGE_LIST_URLS[0])


def test_connection_reset_during_telegram_send_is_a_notification_error(monkeypatch) -> None:
    def reset(request, timeout):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(notifier, "urlopen", reset)

    with pytest.raises(NotificationError, match="ConnectionResetError"):
        send_telegram_message("text", "TOKEN", "1")


def test_alert_message_works_without_a_grade_column() -> None:
    events = pd.DataFrame(
        [{"티커": "MA", "회사명": "Mastercard", "단계": "3차 신호", "결과": "매수 성공"}]
    )

    message = build_scan_message(events, pd.DataFrame(), 0, pd.Timestamp("2026-10-09"))

    assert "🔴 3차 매수 신호 1건" in message
    assert "• MA Mastercard" in message


def test_active_rows_without_market_cap_do_not_become_nan_text() -> None:
    active = pd.DataFrame(
        [{"순위": 9000, "티커": "BTC-USD", "회사명": "[관심] 비트코인", "시가총액": float("nan")}]
    )

    merged = merge_scan_universe([], active)

    assert merged[0].market_cap == ""
    assert merged[0].company == "[관심] 비트코인"


def test_lost_state_is_reported_only_when_the_workflow_says_so(monkeypatch) -> None:
    monkeypatch.delenv(weekly_scan.STATE_RESTORED_ENV, raising=False)
    assert weekly_scan.state_restore_warning() == ""

    monkeypatch.setenv(weekly_scan.STATE_RESTORED_ENV, "true")
    assert weekly_scan.state_restore_warning() == ""

    monkeypatch.setenv(weekly_scan.STATE_RESTORED_ENV, "False")
    assert weekly_scan.state_restore_warning() == weekly_scan.STATE_LOST_WARNING
