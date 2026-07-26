from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from data_provider import normalize_ticker


SECTOR_CACHE_PATH = Path("data") / "sector_classification.csv"
SECTOR_CACHE_COLUMNS = [
    "티커",
    "섹터",
    "산업",
    "원본섹터",
    "원본산업",
    "출처",
    "갱신일",
]
YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
DEFAULT_CACHE_MAX_AGE_DAYS = 180

_CACHE_LOCK = threading.Lock()

_SECTOR_TRANSLATIONS = {
    "Technology": "정보기술",
    "Financial Services": "금융",
    "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재",
    "Communication Services": "커뮤니케이션 서비스",
    "Healthcare": "헬스케어",
    "Industrials": "산업재",
    "Energy": "에너지",
    "Basic Materials": "소재",
    "Real Estate": "부동산",
    "Utilities": "유틸리티",
}

_INDUSTRY_TRANSLATIONS = {
    "Semiconductors": "반도체",
    "Semiconductor Equipment & Materials": "반도체 장비",
    "Software - Infrastructure": "인프라 소프트웨어",
    "Software - Application": "응용 소프트웨어",
    "Consumer Electronics": "소비자 전자제품",
    "Computer Hardware": "컴퓨터 하드웨어",
    "Information Technology Services": "IT 서비스",
    "Internet Content & Information": "인터넷 콘텐츠",
    "Telecom Services": "통신 서비스",
    "Banks - Diversified": "종합 은행",
    "Banks - Regional": "지역 은행",
    "Credit Services": "신용 서비스",
    "Capital Markets": "자본시장",
    "Insurance - Diversified": "종합 보험",
    "Insurance - Life": "생명보험",
    "Insurance - Property & Casualty": "손해보험",
    "Asset Management": "자산운용",
    "Auto Manufacturers": "자동차 제조",
    "Auto Parts": "자동차 부품",
    "Discount Stores": "할인점",
    "Specialty Retail": "전문 소매",
    "Internet Retail": "온라인 소매",
    "Restaurants": "외식",
    "Beverages - Non-Alcoholic": "비알코올 음료",
    "Household & Personal Products": "생활용품",
    "Drug Manufacturers - General": "종합 제약",
    "Biotechnology": "바이오테크",
    "Medical Devices": "의료기기",
    "Aerospace & Defense": "항공우주·방산",
    "Farm & Heavy Construction Machinery": "중장비",
    "Oil & Gas Integrated": "종합 에너지",
    "Oil & Gas E&P": "석유·가스 탐사",
    "Specialty Chemicals": "특수 화학",
    "Copper": "구리",
    "Real Estate Services": "부동산 서비스",
    "REIT - Specialty": "특수 리츠",
}


class SectorMetadataError(RuntimeError):
    """Raised when Yahoo cannot provide sector metadata for a ticker."""


def load_sector_classifications(
    tickers: Iterable[str],
    cache_path: Path | str = SECTOR_CACHE_PATH,
    max_age_days: int = DEFAULT_CACHE_MAX_AGE_DAYS,
    fetcher: Callable[[str], dict[str, str]] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    normalized_tickers = list(dict.fromkeys(normalize_ticker(ticker) for ticker in tickers))
    if not normalized_tickers:
        return pd.DataFrame(columns=SECTOR_CACHE_COLUMNS)

    cache_path = Path(cache_path)
    cached = _read_cache(cache_path)
    cached_by_ticker = {
        str(row["티커"]).upper(): row.to_dict()
        for _, row in cached.iterrows()
    }
    today = pd.Timestamp.today().normalize()
    fetcher = fetch_sector_metadata if fetcher is None else fetcher
    fetched_rows: list[dict[str, object]] = []
    result_by_ticker: dict[str, dict[str, object]] = {}
    needs_fetch: list[str] = []
    completed = 0

    for ticker in normalized_tickers:
        cached_row = cached_by_ticker.get(ticker)
        if cached_row is not None and not _is_stale(cached_row.get("갱신일"), today, max_age_days):
            result_by_ticker[ticker] = cached_row
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(normalized_tickers), ticker)
        else:
            needs_fetch.append(ticker)

    if needs_fetch:
        worker_count = min(6, len(needs_fetch))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(fetcher, ticker): ticker for ticker in needs_fetch}
            for future in as_completed(futures):
                ticker = futures[future]
                cached_row = cached_by_ticker.get(ticker)
                try:
                    metadata = future.result()
                    row = _cache_row(ticker, metadata, today)
                    fetched_rows.append(row)
                    result_by_ticker[ticker] = row
                except Exception:
                    result_by_ticker[ticker] = (
                        cached_row if cached_row is not None else _unclassified_row(ticker)
                    )
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(normalized_tickers), ticker)

    if fetched_rows:
        _merge_and_save_cache(cache_path, fetched_rows)
    result_rows = [result_by_ticker[ticker] for ticker in normalized_tickers]
    return pd.DataFrame(result_rows, columns=SECTOR_CACHE_COLUMNS)


def fetch_sector_metadata(ticker: str) -> dict[str, str]:
    ticker = normalize_ticker(ticker)
    query_ticker = ticker.replace(".", "-")
    url = (
        f"{YAHOO_SEARCH_URL}?q={quote(query_ticker, safe='')}"
        "&quotesCount=8&newsCount=0&enableFuzzyQuery=false"
    )
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise SectorMetadataError(f"Yahoo 분야 조회 실패: HTTP {exc.code}") from exc
    except URLError as exc:
        raise SectorMetadataError(f"Yahoo 분야 조회 실패: {exc.reason}") from exc

    quotes = payload.get("quotes") or []
    match = next(
        (
            item
            for item in quotes
            if str(item.get("symbol", "")).upper().replace("-", ".") == ticker
        ),
        None,
    )
    if match is None:
        raise SectorMetadataError(f"{ticker} 분야 정보를 찾지 못했습니다.")

    raw_sector = str(match.get("sector") or match.get("sectorDisp") or "").strip()
    raw_industry = str(match.get("industry") or match.get("industryDisp") or "").strip()
    return {
        "섹터": _translate_sector(raw_sector),
        "산업": _translate_industry(raw_industry),
        "원본섹터": raw_sector,
        "원본산업": raw_industry,
        "출처": "Yahoo 검색 API",
    }


def _cache_row(ticker: str, metadata: dict[str, str], today: pd.Timestamp) -> dict[str, object]:
    return {
        "티커": ticker,
        "섹터": _classification_value(metadata.get("섹터")),
        "산업": _classification_value(metadata.get("산업")),
        "원본섹터": str(metadata.get("원본섹터", "")).strip(),
        "원본산업": str(metadata.get("원본산업", "")).strip(),
        "출처": str(metadata.get("출처", "Yahoo 검색 API")).strip(),
        "갱신일": today,
    }


def _unclassified_row(ticker: str) -> dict[str, object]:
    return {
        "티커": ticker,
        "섹터": "미분류",
        "산업": "미분류",
        "원본섹터": "",
        "원본산업": "",
        "출처": "조회 실패",
        "갱신일": pd.NaT,
    }


def _read_cache(path: Path) -> pd.DataFrame:
    with _CACHE_LOCK:
        if not path.exists():
            return pd.DataFrame(columns=SECTOR_CACHE_COLUMNS)
        try:
            data = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            return pd.DataFrame(columns=SECTOR_CACHE_COLUMNS)
    data = data.reindex(columns=SECTOR_CACHE_COLUMNS)
    data["갱신일"] = pd.to_datetime(data["갱신일"], errors="coerce")
    return data


def _merge_and_save_cache(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        if path.exists():
            try:
                existing = pd.read_csv(path).reindex(columns=SECTOR_CACHE_COLUMNS)
            except (OSError, pd.errors.ParserError):
                existing = pd.DataFrame(columns=SECTOR_CACHE_COLUMNS)
        else:
            existing = pd.DataFrame(columns=SECTOR_CACHE_COLUMNS)
        updated = pd.concat(
            [existing, pd.DataFrame(rows, columns=SECTOR_CACHE_COLUMNS)],
            ignore_index=True,
        )
        updated = updated.drop_duplicates(subset=["티커"], keep="last").sort_values("티커")
        temporary = path.with_name(
            f"{path.name}.{threading.get_ident()}.tmp"
        )
        updated.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)


def _is_stale(value: object, today: pd.Timestamp, max_age_days: int) -> bool:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return True
    return bool((today - pd.Timestamp(date).normalize()).days > max_age_days)


def _translate_sector(value: str) -> str:
    return _SECTOR_TRANSLATIONS.get(value, _classification_value(value))


def _translate_industry(value: str) -> str:
    return _INDUSTRY_TRANSLATIONS.get(value, _classification_value(value))


def _classification_value(value: object) -> str:
    if value is None or not str(value).strip():
        return "미분류"
    return str(value).strip()
