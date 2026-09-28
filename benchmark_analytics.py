"""Compare MMRM buy-signal returns with two baselines.

* S&P 500: buying the index in the same week and holding it for the same
  number of weeks.  Answers "did the signal beat simply holding the market?"
* Nearby weeks: buying the same stock in any other week within one year
  before or after the signal.  Same stock and same market period, so what
  remains is the effect of the signal's timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_provider import weekly_bar_in_progress


HORIZON_WEEKS = {3: 13, 6: 26, 9: 39, 12: 52}
NEARBY_WINDOW_WEEKS = 52
MIN_NEARBY_WEEKS = 26
SUCCESS_OUTCOME = "매수 성공"


def sp500_return_column(horizon_months: int) -> str:
    return f"SP500Return{horizon_months}M"


def nearby_return_column(horizon_months: int) -> str:
    return f"NearbyReturn{horizon_months}M"


def nearby_win_rate_column(horizon_months: int) -> str:
    return f"NearbyWinRate{horizon_months}M"


def add_benchmark_returns(
    cycles: pd.DataFrame,
    full_table: pd.DataFrame,
    sp500: pd.DataFrame | None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add S&P 500 and nearby-week baseline returns to confirmed buy cycles.

    Baselines are filled only where the cycle's own return is confirmed
    ("확정"), so both sides always cover the same holding period.
    """
    result = cycles.copy()
    for horizon in HORIZON_WEEKS:
        result[sp500_return_column(horizon)] = np.nan
        result[nearby_return_column(horizon)] = np.nan
        result[nearby_win_rate_column(horizon)] = np.nan
    if result.empty or full_table.empty or "Outcome" not in result.columns:
        return result

    stock_close = _weekly_close(full_table)
    sp500_close = (
        _weekly_close(sp500)
        if sp500 is not None and not sp500.empty and "Close" in sp500.columns
        else pd.Series(dtype=float)
    )
    stock_forward = {
        horizon: _forward_returns(stock_close, weeks, now)
        for horizon, weeks in HORIZON_WEEKS.items()
    }
    sp500_forward = {
        horizon: _forward_returns(sp500_close, weeks, now)
        for horizon, weeks in HORIZON_WEEKS.items()
    }

    for index, cycle in result.iterrows():
        if cycle.get("Outcome") != SUCCESS_OUTCOME:
            continue
        signal_date = pd.to_datetime(cycle.get("ThirdDecisionDate"), errors="coerce")
        if pd.isna(signal_date):
            continue
        signal_date = pd.Timestamp(signal_date).normalize()
        for horizon in HORIZON_WEEKS:
            if cycle.get(f"Return{horizon}MStatus") != "확정":
                continue
            sp500_value = sp500_forward[horizon].get(signal_date, np.nan)
            result.at[index, sp500_return_column(horizon)] = sp500_value
            nearby = _nearby_returns(stock_forward[horizon], signal_date)
            if len(nearby) >= MIN_NEARBY_WEEKS:
                result.at[index, nearby_return_column(horizon)] = float(nearby.mean())
                result.at[index, nearby_win_rate_column(horizon)] = float(
                    (nearby > 0).mean() * 100
                )
    return result


def benchmark_metrics(
    bought: pd.DataFrame,
    signal_returns: pd.Series,
    horizon_months: int,
) -> dict[str, object]:
    """Summarize confirmed buy returns against both baselines.

    ``signal_returns`` holds the confirmed signal returns, indexed like
    ``bought``.  Each comparison only uses cycles that have both values.
    """
    sp500_column = sp500_return_column(horizon_months)
    nearby_column = nearby_return_column(horizon_months)
    win_rate_column = nearby_win_rate_column(horizon_months)

    sp500 = _numeric_column(bought, sp500_column)
    sp500_pairs = pd.DataFrame(
        {"signal": signal_returns, "benchmark": sp500}
    ).dropna()
    sp500_excess = sp500_pairs["signal"] - sp500_pairs["benchmark"]

    nearby = _numeric_column(bought, nearby_column)
    nearby_win = _numeric_column(bought, win_rate_column)
    nearby_pairs = pd.DataFrame(
        {"signal": signal_returns, "benchmark": nearby, "win_rate": nearby_win}
    ).dropna()
    nearby_excess = nearby_pairs["signal"] - nearby_pairs["benchmark"]

    sp500_count = len(sp500_pairs)
    sp500_wins = int((sp500_excess > 0).sum())
    nearby_count = len(nearby_pairs)
    return {
        "S&P 비교 표본": sp500_count,
        "S&P 이긴 건수": sp500_wins,
        "S&P 이긴 비율": sp500_wins / sp500_count * 100 if sp500_count else np.nan,
        "S&P 대비 초과": float(sp500_excess.mean()) if sp500_count else np.nan,
        "평소 매수 비교 표본": nearby_count,
        "평소 매수 승률": float(nearby_pairs["win_rate"].mean()) if nearby_count else np.nan,
        "평소 매수 대비 초과": float(nearby_excess.mean()) if nearby_count else np.nan,
    }


def empty_benchmark_metrics() -> dict[str, object]:
    return {
        "S&P 비교 표본": 0,
        "S&P 이긴 건수": 0,
        "S&P 이긴 비율": np.nan,
        "S&P 대비 초과": np.nan,
        "평소 매수 비교 표본": 0,
        "평소 매수 승률": np.nan,
        "평소 매수 대비 초과": np.nan,
    }


def _weekly_close(data: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(data["Close"], errors="coerce")
    close.index = pd.DatetimeIndex(data.index).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close


def _forward_returns(
    close: pd.Series,
    weeks: int,
    now: pd.Timestamp | None,
) -> pd.Series:
    """Percent return from each week's close to the close ``weeks`` later."""
    if close.empty:
        return close
    forward = (close.shift(-weeks) / close - 1) * 100
    if weekly_bar_in_progress(close.index[-1], now=now):
        # The last bar is not final yet, so returns ending on it are not either.
        last_position = len(close) - 1
        forward.iloc[max(0, last_position - weeks)] = np.nan
    return forward.replace([np.inf, -np.inf], np.nan)


def _nearby_returns(forward: pd.Series, signal_date: pd.Timestamp) -> pd.Series:
    positions = np.flatnonzero(forward.index == signal_date)
    if not len(positions):
        return pd.Series(dtype=float)
    position = int(positions[0])
    start = max(0, position - NEARBY_WINDOW_WEEKS)
    window = forward.iloc[start : position + NEARBY_WINDOW_WEEKS + 1]
    return window.drop(index=signal_date).dropna()


def _numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce")
