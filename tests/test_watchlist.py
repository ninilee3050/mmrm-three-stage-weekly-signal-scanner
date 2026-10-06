from __future__ import annotations

import pandas as pd
import pytest

from market_cap_provider import MarketCapCompany
from notifier import build_scan_message
from watchlist import (
    WATCHLIST_RANK_START,
    WatchlistItem,
    add_watchlist_item,
    clear_chart_strength_for_weekend_assets,
    is_weekend_traded,
    load_watchlist,
    remove_watchlist_item,
    save_watchlist,
    watchlist_companies,
)


def test_watchlist_file_round_trip_keeps_names_and_skips_comments(tmp_path) -> None:
    path = tmp_path / "watchlist.txt"
    path.write_text(
        "# 설명 줄\n\nbtc-usd, 비트코인\nSCHD\n  o , 리얼티인컴 \nSCHD, 중복\n",
        encoding="utf-8",
    )

    items = load_watchlist(path)

    assert items == [
        WatchlistItem("BTC-USD", "비트코인"),
        WatchlistItem("SCHD"),
        WatchlistItem("O", "리얼티인컴"),
    ]
    save_watchlist(items, path)
    assert load_watchlist(path) == items
    assert path.read_text(encoding="utf-8").startswith("# 관심종목")


def test_missing_watchlist_file_is_an_empty_list(tmp_path) -> None:
    assert load_watchlist(tmp_path / "none.txt") == []


def test_adding_and_removing_watchlist_tickers() -> None:
    items = add_watchlist_item([], " jepq ", " 인컴 ETF ")

    assert items == [WatchlistItem("JEPQ", "인컴 ETF")]
    with pytest.raises(ValueError, match="이미 관심종목"):
        add_watchlist_item(items, "JEPQ")
    with pytest.raises(ValueError, match="티커를 입력"):
        add_watchlist_item(items, "   ")
    with pytest.raises(ValueError, match="빈칸이나 쉼표"):
        add_watchlist_item(items, "BTC USD")
    assert remove_watchlist_item(items, "jepq") == []


def test_watchlist_companies_skip_ranked_tickers_and_are_labelled() -> None:
    ranked = [MarketCapCompany(1, "NVDA", "NVIDIA Corporation", "5.77T")]
    items = [
        WatchlistItem("NVDA", "엔비디아"),
        WatchlistItem("BTC-USD", "비트코인"),
        WatchlistItem("SCHD"),
    ]

    companies = watchlist_companies(items, ranked)

    assert [(c.rank, c.ticker, c.company) for c in companies] == [
        (WATCHLIST_RANK_START + 1, "BTC-USD", "[관심] 비트코인"),
        (WATCHLIST_RANK_START + 2, "SCHD", "[관심] SCHD"),
    ]


def test_crypto_pairs_are_weekend_traded() -> None:
    assert is_weekend_traded("btc-usd")
    assert not is_weekend_traded("SCHD")
    assert not is_weekend_traded("BRK.B")


def test_chart_strength_is_not_scored_for_weekend_assets() -> None:
    events = pd.DataFrame(
        {
            "티커": ["BTC-USD", "BTC-USD", "NVDA"],
            "차트 강도": ["97.0점", "산정 대기", "82.5점"],
            "검토등급": ["우선검토", "", "우선검토"],
        }
    )

    cleared = clear_chart_strength_for_weekend_assets(events)

    assert cleared["차트 강도"].tolist() == ["해당 없음", "산정 대기", "82.5점"]
    assert cleared["검토등급"].tolist() == ["", "", "우선검토"]
    assert events.loc[0, "차트 강도"] == "97.0점"  # the input is left unchanged


def test_alert_notes_that_crypto_week_is_not_finished() -> None:
    events = pd.DataFrame(
        [
            {
                "티커": "BTC-USD",
                "회사명": "[관심] 비트코인",
                "단계": "3차 신호",
                "결과": "매수 성공",
                "신호일": pd.Timestamp("2026-10-05"),
                "신호구분": "이번 주",
                "차트 강도": "해당 없음",
                "검토등급": "",
            }
        ]
    )

    message = build_scan_message(events, pd.DataFrame(), 0, pd.Timestamp("2026-10-09"))

    assert "• BTC-USD [관심] 비트코인" in message.splitlines()
    assert message.splitlines()[-1] == "※ BTC-USD: 주말 거래 반영 전 결과입니다."
