from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd


REFERENCE_PATH = Path(__file__).resolve().parent / "resources" / "chart_strength_reference.json"
MIN_REFERENCE_EVENTS = 200
PRIORITY_THRESHOLD = 70.0

FEATURE_SPECS = (
    ("close_from_52w_low_pct", "52주 저점 대비 상승폭", "percent"),
    ("volatility_26w_pct", "최근 26주 변동성", "percent_unsigned"),
    ("prior_13w_return_pct", "최근 13주 수익률", "percent"),
    ("momentum_pct", "Momentum(종가 대비)", "percent"),
    ("mfi", "MFI", "number"),
)
FEATURE_KEYS = tuple(spec[0] for spec in FEATURE_SPECS)


class ChartStrengthReferenceError(RuntimeError):
    pass


def extract_chart_strength_features(
    full_table: pd.DataFrame,
    signal_date: pd.Timestamp | str,
) -> dict[str, float]:
    """Return the five frozen research features at a third-signal week."""
    if full_table.empty:
        return _missing_features()

    data = full_table.sort_index()
    date = pd.Timestamp(signal_date)
    positions = np.flatnonzero(pd.DatetimeIndex(data.index) == date)
    if not len(positions):
        return _missing_features()

    position = int(positions[-1])
    row = data.iloc[position]
    close = _number(row.get("Close"))
    if not math.isfinite(close) or close == 0:
        return _missing_features()

    trailing_52 = data.iloc[max(0, position - 51) : position + 1]
    low_52 = _number(pd.to_numeric(trailing_52.get("Low"), errors="coerce").min())

    trailing_26 = data.iloc[max(0, position - 25) : position + 1]
    weekly_returns = pd.to_numeric(
        trailing_26.get("Close"), errors="coerce"
    ).pct_change(fill_method=None).dropna()
    volatility = (
        float(weekly_returns.std(ddof=1) * math.sqrt(52) * 100.0)
        if len(weekly_returns) >= 2
        else math.nan
    )

    prior_13_close = (
        _number(data.iloc[position - 13].get("Close"))
        if position >= 13
        else math.nan
    )
    momentum = _number(row.get("Momentum"))

    return {
        "close_from_52w_low_pct": _pct_ratio(close, low_52),
        "volatility_26w_pct": volatility,
        "prior_13w_return_pct": _pct_ratio(close, prior_13_close),
        "momentum_pct": momentum / close * 100.0 if math.isfinite(momentum) else math.nan,
        "mfi": _number(row.get("MFI")),
    }


@lru_cache(maxsize=4)
def load_chart_strength_reference(
    path: str | Path = REFERENCE_PATH,
) -> pd.DataFrame:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ChartStrengthReferenceError(
            f"차트 강도 기준자료를 읽을 수 없습니다: {source}"
        ) from exc

    if payload.get("version") != 1 or payload.get("feature_keys") != list(FEATURE_KEYS):
        raise ChartStrengthReferenceError("차트 강도 기준자료의 형식이 올바르지 않습니다.")

    records = payload.get("records")
    if not isinstance(records, list):
        raise ChartStrengthReferenceError("차트 강도 기준자료에 과거 사례가 없습니다.")

    columns = ["available_date", *FEATURE_KEYS]
    reference = pd.DataFrame(records, columns=columns)
    reference["available_date"] = pd.to_datetime(
        reference["available_date"], errors="coerce"
    )
    for key in FEATURE_KEYS:
        reference[key] = pd.to_numeric(reference[key], errors="coerce")
    reference = reference.dropna(subset=columns).reset_index(drop=True)
    if len(reference) < MIN_REFERENCE_EVENTS:
        raise ChartStrengthReferenceError(
            f"차트 강도 기준자료가 부족합니다: {len(reference)}건"
        )
    return reference


def score_chart_strength(
    features: dict[str, float],
    signal_date: pd.Timestamp | str,
    reference: pd.DataFrame,
    minimum_events: int = MIN_REFERENCE_EVENTS,
) -> dict[str, object]:
    date = pd.Timestamp(signal_date).normalize()
    eligible = reference.loc[
        pd.to_datetime(reference["available_date"], errors="coerce") <= date
    ].copy()
    eligible = eligible.dropna(subset=list(FEATURE_KEYS))

    missing = [key for key in FEATURE_KEYS if not math.isfinite(_number(features.get(key)))]
    if missing:
        return unavailable_chart_strength(
            "3차 신호 주의 평가 데이터가 부족합니다.",
            reference_count=len(eligible),
        )
    if len(eligible) < minimum_events:
        return unavailable_chart_strength(
            f"해당 시점 이전의 확정 과거 사례가 {len(eligible)}건으로 부족합니다.",
            reference_count=len(eligible),
        )

    component_scores: dict[str, float] = {}
    reference_percentiles: list[np.ndarray] = []
    for key in FEATURE_KEYS:
        values = pd.to_numeric(eligible[key], errors="coerce").to_numpy(float)
        component_scores[key] = 100.0 * _percentile_against(
            values, np.array([float(features[key])])
        )[0]
        reference_percentiles.append(_percentile_against(values, values))

    reference_composite = np.mean(reference_percentiles, axis=0)
    candidate_composite = float(
        np.mean([component_scores[key] / 100.0 for key in FEATURE_KEYS])
    )
    final_score = 100.0 * _percentile_against(
        reference_composite, np.array([candidate_composite])
    )[0]
    grade = "우선검토" if final_score >= PRIORITY_THRESHOLD else "일반검토"

    components = []
    for key, label, value_kind in FEATURE_SPECS:
        components.append(
            {
                "key": key,
                "label": label,
                "value": float(features[key]),
                "value_text": _format_feature_value(float(features[key]), value_kind),
                "score": component_scores[key],
                "score_text": f"{component_scores[key]:.0f}점",
            }
        )

    return {
        "available": True,
        "score": final_score,
        "score_text": f"{final_score:.1f}점",
        "grade": grade,
        "reference_count": int(len(eligible)),
        "components": components,
        "reasons": _build_reasons(final_score, components),
        "note": "점수는 수익 확률이 아니라 과거 차트 강도 분포에서의 상대 위치입니다.",
    }


def unavailable_chart_strength(
    reason: str,
    reference_count: int = 0,
) -> dict[str, object]:
    return {
        "available": False,
        "score": math.nan,
        "score_text": "계산 불가",
        "grade": "확인 필요",
        "reference_count": int(reference_count),
        "components": [],
        "reasons": [reason],
        "note": "기존 1·2·3차 신호 판정에는 영향을 주지 않습니다.",
    }


def annotate_scan_events(
    events: pd.DataFrame,
    full_tables_by_ticker: dict[str, pd.DataFrame],
    reference: pd.DataFrame | None,
    reference_error: str | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    annotated = events.copy()
    annotated["차트 강도"] = ""
    annotated["검토등급"] = ""
    details: dict[tuple[str, str], dict[str, object]] = {}

    for index, row in annotated.iterrows():
        ticker = str(row.get("티커", "")).strip().upper()
        signal_date = pd.to_datetime(row.get("신호일"), errors="coerce")
        key = chart_strength_detail_key(ticker, signal_date)
        is_successful_third = (
            row.get("단계") == "3차 신호" and row.get("결과") == "매수 성공"
        )

        if not is_successful_third:
            waiting = row.get("단계") in {"1차 신호", "2차 신호"}
            annotated.at[index, "차트 강도"] = "산정 대기" if waiting else "해당 없음"
            annotated.at[index, "검토등급"] = ""
            continue

        detail = _evaluate_chart_strength(
            ticker,
            signal_date,
            full_tables_by_ticker,
            reference,
            reference_error,
        )

        annotated.at[index, "차트 강도"] = detail["score_text"]
        annotated.at[index, "검토등급"] = detail["grade"]
        details[key] = detail

    return annotated, details


def annotate_completed_scenarios(
    scenarios: pd.DataFrame,
    full_tables_by_ticker: dict[str, pd.DataFrame],
    reference: pd.DataFrame | None,
    reference_error: str | None = None,
    signal_date_column: str = "3차판정일",
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    """Add the same third-signal strength result to completed scenario rows."""
    annotated = scenarios.copy()
    annotated["차트 강도"] = ""
    annotated["검토등급"] = ""
    details: dict[tuple[str, str], dict[str, object]] = {}

    for index, row in annotated.iterrows():
        if row.get("결과") != "매수 성공":
            annotated.at[index, "차트 강도"] = "해당 없음"
            continue

        ticker = str(row.get("티커", "")).strip().upper()
        signal_date = pd.to_datetime(row.get(signal_date_column), errors="coerce")
        detail = _evaluate_chart_strength(
            ticker,
            signal_date,
            full_tables_by_ticker,
            reference,
            reference_error,
        )
        annotated.at[index, "차트 강도"] = detail["score_text"]
        annotated.at[index, "검토등급"] = detail["grade"]
        details[chart_strength_detail_key(ticker, signal_date)] = detail

    return annotated, details


def annotate_pending_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Mark active first/second-stage scenarios as awaiting a third signal."""
    annotated = scenarios.copy()
    annotated["차트 강도"] = "산정 대기"
    annotated["검토등급"] = ""
    return annotated


def chart_strength_detail_key(
    ticker: object,
    signal_date: object,
) -> tuple[str, str]:
    date = pd.to_datetime(signal_date, errors="coerce")
    date_text = "" if pd.isna(date) else pd.Timestamp(date).strftime("%Y-%m-%d")
    return str(ticker).strip().upper(), date_text


def _evaluate_chart_strength(
    ticker: str,
    signal_date: pd.Timestamp,
    full_tables_by_ticker: dict[str, pd.DataFrame],
    reference: pd.DataFrame | None,
    reference_error: str | None,
) -> dict[str, object]:
    if reference is None:
        return unavailable_chart_strength(
            reference_error or "차트 강도 기준자료를 불러오지 못했습니다."
        )
    if pd.isna(signal_date) or ticker not in full_tables_by_ticker:
        return unavailable_chart_strength("3차 신호 주봉 데이터를 찾지 못했습니다.")
    features = extract_chart_strength_features(
        full_tables_by_ticker[ticker], pd.Timestamp(signal_date)
    )
    return score_chart_strength(features, signal_date, reference)


def _build_reasons(
    final_score: float,
    components: list[dict[str, object]],
) -> list[str]:
    ordered = sorted(components, key=lambda item: float(item["score"]), reverse=True)
    reasons = [
        f"{item['label']}: 과거 대비 {float(item['score']):.0f}점으로 강합니다."
        for item in ordered[:2]
    ]
    weakest = ordered[-1]
    if float(weakest["score"]) < 50.0:
        reasons.append(
            f"{weakest['label']}: {float(weakest['score']):.0f}점으로 상대적으로 약합니다."
        )
    if final_score >= PRIORITY_THRESHOLD:
        top_fraction = max(1.0, 100.0 - final_score)
        reasons.append(
            f"다섯 항목의 조합이 과거 분포 상위 약 {top_fraction:.0f}%로 우선검토 기준을 충족합니다."
        )
    else:
        reasons.append(
            f"최종 {final_score:.0f}점으로 상위 30% 우선검토 기준에는 미달합니다."
        )
    return reasons


def _percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite_reference = np.sort(reference[np.isfinite(reference)])
    if not len(finite_reference):
        return np.full(len(values), 0.5)
    return np.searchsorted(finite_reference, values, side="right") / len(
        finite_reference
    )


def _format_feature_value(value: float, value_kind: str) -> str:
    if value_kind == "number":
        return f"{value:.1f}"
    if value_kind == "percent_unsigned":
        return f"{value:.2f}%"
    return f"{value:+.2f}%"


def _pct_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
        return math.nan
    return (numerator / denominator - 1.0) * 100.0


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _missing_features() -> dict[str, float]:
    return {key: math.nan for key in FEATURE_KEYS}
