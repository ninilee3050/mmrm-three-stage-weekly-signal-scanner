from __future__ import annotations

import pandas as pd


SECOND_SIGNAL_MAX_MA20_OVER_MA50_SPREAD = 0.05


BUY_POINT_COLUMNS = [
    "Close",
    "MA_20",
    "MA_50",
    "MA_150",
    "MA_200",
    "MACD",
    "Signal",
    "Momentum",
    "RSI",
    "MFI",
    "ConditionSummary",
]

SIGNAL_CYCLE_COLUMNS = [
    "FirstSignalDate",
    "SecondSignalDate",
    "ThirdDecisionDate",
    "Outcome",
    "Return3M",
    "Return3MStatus",
    "Return6M",
    "Return6MStatus",
    "Return9M",
    "Return9MStatus",
    "Return12M",
    "Return12MStatus",
]


def add_signal_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["macd_area"] = result["MACD"].apply(_macd_area)
    result["macd_flow"] = result.apply(_macd_flow, axis=1)

    prev_macd = result["MACD"].shift(1)
    prev_signal = result["Signal"].shift(1)
    result["macd_bullish_start"] = (
        (prev_signal > prev_macd) & (result["MACD"] > result["Signal"])
    ).fillna(False)
    result["lower_area_bullish_start"] = (
        result["macd_bullish_start"] & ((result["MACD"] < 0) | (prev_macd < 0))
    ).fillna(False)
    result["macd_bearish_start"] = (
        (prev_macd > prev_signal) & (result["Signal"] > result["MACD"])
    ).fillna(False)

    result["momentum_ok"] = (result["Momentum"] > 0).fillna(False)
    result["rsi_ok"] = (result["RSI"] > 50).fillna(False)
    result["mfi_ok"] = (result["MFI"] > 50).fillna(False)
    result["short_ma_inverted_ok"] = (result["MA_50"] > result["MA_20"]).fillna(False)
    result["long_ma_ok"] = (result["MA_150"] > result["MA_200"]).fillna(False)
    result["all_three_ok"] = result["momentum_ok"] & result["rsi_ok"] & result["mfi_ok"]
    result["all_buy_conditions_ok"] = (
        result["all_three_ok"]
        & result["long_ma_ok"]
        & result["short_ma_inverted_ok"]
    )
    return result


def scan_buy_points(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = add_signal_columns(data)
    full["buy_point"] = False
    full["observation_active"] = False
    full["observation_start_date"] = pd.NaT

    buy_rows = []
    observing = False
    observation_start_date = None

    for date, row in full.iterrows():
        if observing and bool(row["macd_bearish_start"]):
            observing = False
            observation_start_date = None

        if (
            not observing
            and bool(row["lower_area_bullish_start"])
        ):
            observing = True
            observation_start_date = date

        if observing:
            full.at[date, "observation_active"] = True
            full.at[date, "observation_start_date"] = observation_start_date

            if row["MACD"] > row["Signal"] and bool(row["all_buy_conditions_ok"]):
                full.at[date, "buy_point"] = True
                buy_rows.append(_make_buy_point_row(date, row))
                observing = False
                observation_start_date = None

    buy_points = pd.DataFrame(buy_rows, columns=["Date", *BUY_POINT_COLUMNS])
    buy_points = buy_points.set_index("Date")
    return buy_points, full


def scan_signal_cycles(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = add_signal_columns(data)
    full["original_mmrm_point"] = False
    full["first_signal"] = False
    full["second_signal"] = False
    full["second_rejection"] = False
    full["second_structure_rejection"] = False
    full["third_signal"] = False
    full["third_failure"] = False

    _mark_original_mmrm_points(full)

    cycle_rows: list[dict[str, object]] = []
    state = "idle"
    first_date: pd.Timestamp | None = None
    second_date: pd.Timestamp | None = None

    for date, row in full.iterrows():
        date = pd.Timestamp(date)

        if state == "idle":
            if bool(row["original_mmrm_point"]) and _first_signal_conditions_met(row):
                full.at[date, "first_signal"] = True
                state = "await_second"
                first_date = date
            continue

        if state == "await_second":
            if row["Close"] < row["MA_5"] and row["Close"] < row["Open"]:
                second_date = date
                if _second_signal_spread_too_wide(row):
                    full.at[date, "second_rejection"] = True
                    cycle_rows.append(
                        _make_signal_cycle_row(
                            full,
                            first_date=first_date,
                            second_date=second_date,
                            third_decision_date=None,
                            outcome="2차 이격 과다 폐기",
                        )
                    )
                    state = "idle"
                    first_date = None
                    second_date = None
                    continue

                if _second_signal_medium_structure_invalid(row):
                    full.at[date, "second_structure_rejection"] = True
                    cycle_rows.append(
                        _make_signal_cycle_row(
                            full,
                            first_date=first_date,
                            second_date=second_date,
                            third_decision_date=None,
                            outcome="2차 중기 구조 미충족 폐기",
                        )
                    )
                    state = "idle"
                    first_date = None
                    second_date = None
                    continue

                full.at[date, "second_signal"] = True
                state = "await_third"
            continue

        if row["Close"] > row["MA_5"] and row["Close"] >= row["Open"]:
            if row["Close"] > row["MA_20"]:
                full.at[date, "third_signal"] = True
                outcome = "매수 성공"
            else:
                full.at[date, "third_failure"] = True
                outcome = "실패"

            cycle_rows.append(
                _make_signal_cycle_row(
                    full,
                    first_date=first_date,
                    second_date=second_date,
                    third_decision_date=date,
                    outcome=outcome,
                )
            )
            state = "idle"
            first_date = None
            second_date = None

    if state != "idle":
        outcome = "2차 신호 대기" if state == "await_second" else "3차 신호 대기"
        cycle_rows.append(
            _make_signal_cycle_row(
                full,
                first_date=first_date,
                second_date=second_date,
                third_decision_date=None,
                outcome=outcome,
            )
        )

    cycles = pd.DataFrame(cycle_rows, columns=SIGNAL_CYCLE_COLUMNS)
    return cycles, full


def _mark_original_mmrm_points(full: pd.DataFrame) -> None:
    observing = False

    for date, row in full.iterrows():
        if observing and bool(row["macd_bearish_start"]):
            observing = False

        if not observing and bool(row["lower_area_bullish_start"]):
            observing = True

        if observing and row["MACD"] > row["Signal"] and bool(row["all_three_ok"]):
            full.at[date, "original_mmrm_point"] = True
            observing = False


def _first_signal_conditions_met(row: pd.Series) -> bool:
    return bool(
        row["long_ma_ok"]
        and row["short_ma_inverted_ok"]
        and row["Close"] >= row["MA_5"]
    )


def _second_signal_spread_too_wide(row: pd.Series) -> bool:
    ma_20 = row.get("MA_20")
    ma_50 = row.get("MA_50")
    if pd.isna(ma_20) or pd.isna(ma_50) or ma_50 <= 0:
        return False
    return bool((ma_20 / ma_50 - 1) >= SECOND_SIGNAL_MAX_MA20_OVER_MA50_SPREAD)


def _second_signal_medium_structure_invalid(row: pd.Series) -> bool:
    ma_20 = row.get("MA_20")
    ma_50 = row.get("MA_50")
    ma_200 = row.get("MA_200")
    if pd.isna(ma_20) or pd.isna(ma_50) or pd.isna(ma_200):
        return False
    return bool(ma_20 > ma_50 and ma_50 <= ma_200)


def _make_signal_cycle_row(
    full: pd.DataFrame,
    first_date: pd.Timestamp | None,
    second_date: pd.Timestamp | None,
    third_decision_date: pd.Timestamp | None,
    outcome: str,
) -> dict[str, object]:
    successful = outcome == "매수 성공"
    return_3m, return_3m_status = _forward_return_result(
        full,
        third_decision_date,
        13,
    ) if successful else (float("nan"), "해당 없음")
    return_6m, return_6m_status = _forward_return_result(
        full,
        third_decision_date,
        26,
    ) if successful else (float("nan"), "해당 없음")
    return_9m, return_9m_status = _forward_return_result(
        full,
        third_decision_date,
        39,
    ) if successful else (float("nan"), "해당 없음")
    return_12m, return_12m_status = _forward_return_result(
        full,
        third_decision_date,
        52,
    ) if successful else (float("nan"), "해당 없음")

    return {
        "FirstSignalDate": first_date,
        "SecondSignalDate": second_date,
        "ThirdDecisionDate": third_decision_date,
        "Outcome": outcome,
        "Return3M": return_3m,
        "Return3MStatus": return_3m_status,
        "Return6M": return_6m,
        "Return6MStatus": return_6m_status,
        "Return9M": return_9m,
        "Return9MStatus": return_9m_status,
        "Return12M": return_12m,
        "Return12MStatus": return_12m_status,
    }


def _forward_return_result(
    full: pd.DataFrame,
    signal_date: pd.Timestamp | None,
    weeks: int,
) -> tuple[float, str]:
    if signal_date is None:
        return float("nan"), "해당 없음"

    position = full.index.get_indexer([signal_date])[0]
    future_position = position + weeks
    if position < 0:
        return float("nan"), "데이터 없음"
    if future_position >= len(full):
        return float("nan"), "진행 중"

    signal_close = full.iloc[position]["Close"]
    future_close = full.iloc[future_position]["Close"]
    if pd.isna(signal_close) or pd.isna(future_close) or signal_close == 0:
        return float("nan"), "데이터 없음"
    return (future_close / signal_close - 1) * 100, "확정"


def current_week_buy_point(
    buy_points: pd.DataFrame,
    scan_date: pd.Timestamp | str | None = None,
) -> pd.Series | None:
    week_start = _week_start(scan_date)
    if buy_points.empty or week_start not in buy_points.index:
        return None

    row = buy_points.loc[week_start]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _week_start(scan_date: pd.Timestamp | str | None = None) -> pd.Timestamp:
    date = pd.Timestamp.today() if scan_date is None else pd.Timestamp(scan_date)
    normalized = date.normalize()
    return normalized - pd.Timedelta(days=normalized.weekday())


def _macd_area(macd: float) -> str:
    if pd.isna(macd):
        return ""
    if macd > 0:
        return "MACD 상승영역"
    if macd < 0:
        return "MACD 하락영역"
    return "기준선"


def _macd_flow(row: pd.Series) -> str:
    macd = row["MACD"]
    signal = row["Signal"]
    if pd.isna(macd) or pd.isna(signal):
        return ""
    if macd > signal:
        return "MACD 상승흐름"
    if signal > macd:
        return "MACD 하락흐름"
    return "교차 지점 / 중립"


def _make_buy_point_row(
    date: pd.Timestamp,
    row: pd.Series,
) -> dict[str, object]:
    return {
        "Date": date,
        "Close": row.get("Close"),
        "MA_20": row.get("MA_20"),
        "MA_50": row.get("MA_50"),
        "MA_150": row.get("MA_150"),
        "MA_200": row.get("MA_200"),
        "MACD": row["MACD"],
        "Signal": row["Signal"],
        "Momentum": row["Momentum"],
        "RSI": row["RSI"],
        "MFI": row["MFI"],
        "ConditionSummary": (
            "MACD 상승흐름 유지 + Momentum > 0 + RSI > 50 + MFI > 50"
            " + 150주선 > 200주선 + 50주선 > 20주선"
        ),
    }
