from __future__ import annotations

import numpy as np
import pandas as pd

from chart_strength import (
    FEATURE_KEYS,
    annotate_scan_events,
    chart_strength_detail_key,
    extract_chart_strength_features,
    load_chart_strength_reference,
    score_chart_strength,
)


def sample_full_table() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    close = pd.Series(np.arange(100.0, 160.0), index=index)
    return pd.DataFrame(
        {
            "Close": close,
            "Low": close - 5.0,
            "Momentum": close - close.shift(14),
            "MFI": 62.0,
        },
        index=index,
    )


def ordered_reference(count: int = 220) -> pd.DataFrame:
    values = np.arange(float(count))
    data: dict[str, object] = {
        "available_date": pd.Timestamp("2020-01-01"),
    }
    for key in FEATURE_KEYS:
        data[key] = values
    return pd.DataFrame(data)


def test_extract_chart_strength_features_matches_frozen_research_formulas() -> None:
    full = sample_full_table()
    date = full.index[-1]

    features = extract_chart_strength_features(full, date)

    assert features["close_from_52w_low_pct"] == pytest_approx(
        (159.0 / 103.0 - 1.0) * 100.0
    )
    assert features["prior_13w_return_pct"] == pytest_approx(
        (159.0 / 146.0 - 1.0) * 100.0
    )
    assert features["momentum_pct"] == pytest_approx(14.0 / 159.0 * 100.0)
    assert features["mfi"] == 62.0
    assert features["volatility_26w_pct"] > 0


def test_score_uses_only_reference_cases_available_by_signal_date() -> None:
    reference = ordered_reference()
    reference.loc[200:, "available_date"] = pd.Timestamp("2030-01-01")
    features = {key: 150.0 for key in FEATURE_KEYS}

    detail = score_chart_strength(
        features,
        "2026-07-27",
        reference,
        minimum_events=100,
    )

    assert detail["available"] is True
    assert detail["reference_count"] == 200
    assert detail["score"] == pytest_approx(75.5)
    assert detail["grade"] == "우선검토"


def test_annotation_adds_priority_grade_and_hover_detail_only_to_successful_third() -> None:
    full = sample_full_table()
    signal_date = full.index[-1]
    raw_features = extract_chart_strength_features(full, signal_date)
    reference = ordered_reference()
    events = pd.DataFrame(
        [
            {
                "티커": "AAA",
                "단계": "3차 신호",
                "결과": "매수 성공",
                "신호일": signal_date,
            },
            {
                "티커": "BBB",
                "단계": "2차 신호",
                "결과": "3차 신호 대기",
                "신호일": signal_date,
            },
        ]
    )
    # Center the synthetic reference on the actual feature scales while
    # retaining a monotonic percentile order.
    for key, value in raw_features.items():
        reference[key] = np.linspace(value * 0.25, value * 1.10, len(reference))

    annotated, details = annotate_scan_events(
        events,
        {"AAA": full},
        reference,
    )

    assert annotated.loc[0, "검토등급"] == "우선검토"
    assert annotated.loc[0, "차트 강도"].endswith("점")
    assert annotated.loc[1, "차트 강도"] == "산정 대기"
    key = chart_strength_detail_key("AAA", signal_date)
    assert len(details[key]["components"]) == 5
    assert details[key]["reasons"]


def test_bundled_reference_is_valid_and_large_enough_for_live_scoring() -> None:
    load_chart_strength_reference.cache_clear()
    reference = load_chart_strength_reference()

    assert len(reference) == 2175
    assert reference["available_date"].max() <= pd.Timestamp("2026-07-30")
    assert reference[list(FEATURE_KEYS)].notna().all().all()


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
