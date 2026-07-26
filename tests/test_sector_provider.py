from __future__ import annotations

import pandas as pd

from sector_provider import load_sector_classifications


def test_sector_metadata_is_fetched_translated_and_cached(tmp_path) -> None:
    cache_path = tmp_path / "sectors.csv"
    calls = []

    def fake_fetcher(ticker: str) -> dict[str, str]:
        calls.append(ticker)
        return {
            "섹터": "정보기술",
            "산업": "반도체",
            "원본섹터": "Technology",
            "원본산업": "Semiconductors",
            "출처": "test",
        }

    first = load_sector_classifications(
        ["NVDA"], cache_path=cache_path, fetcher=fake_fetcher
    )
    second = load_sector_classifications(
        ["NVDA"], cache_path=cache_path, fetcher=fake_fetcher
    )

    assert calls == ["NVDA"]
    assert first.loc[0, "섹터"] == "정보기술"
    assert first.loc[0, "산업"] == "반도체"
    assert second.loc[0, "출처"] == "test"
    assert cache_path.exists()


def test_metadata_failure_is_visible_and_does_not_poison_cache(tmp_path) -> None:
    cache_path = tmp_path / "sectors.csv"

    def failing_fetcher(_ticker: str) -> dict[str, str]:
        raise RuntimeError("network down")

    result = load_sector_classifications(
        ["TEST"], cache_path=cache_path, fetcher=failing_fetcher
    )

    assert result.loc[0, "섹터"] == "미분류"
    assert result.loc[0, "산업"] == "미분류"
    assert result.loc[0, "출처"] == "조회 실패"
    assert not cache_path.exists()


def test_stale_cache_is_used_when_refresh_fails(tmp_path) -> None:
    cache_path = tmp_path / "sectors.csv"
    pd.DataFrame(
        [
            {
                "티커": "OLD",
                "섹터": "금융",
                "산업": "은행",
                "원본섹터": "Financial Services",
                "원본산업": "Banks - Diversified",
                "출처": "old source",
                "갱신일": "2020-01-01",
            }
        ]
    ).to_csv(cache_path, index=False, encoding="utf-8-sig")

    result = load_sector_classifications(
        ["OLD"],
        cache_path=cache_path,
        max_age_days=1,
        fetcher=lambda _ticker: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    assert result.loc[0, "섹터"] == "금융"
    assert result.loc[0, "산업"] == "은행"
