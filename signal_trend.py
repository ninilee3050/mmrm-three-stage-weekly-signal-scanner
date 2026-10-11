"""Signal-point trend view (experimental).

Takes one cycle's 1차/2차/3차 weeks and compares how the close, volume,
momentum, MACD, RSI and MFI moved between them, plus the average of those moves.  Opened from
a temporary button in the chart preview; kept in its own module until its
final place in the UI is decided.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from ui_theme import configure_ui_fonts, normalize_theme, theme_palette


SIGNAL_POINTS = (
    ("1차", "FirstSignalDate", "#16a34a"),
    ("2차", "SecondSignalDate", "#f59e0b"),
    ("3차", "ThirdDecisionDate", "#dc2626"),
)
SIGNAL_COLORS = {label: color for label, _, color in SIGNAL_POINTS}
# (name, data column, show value in K/M/B, fixed 0-100 scale,
#  baseline: a fixed level or the data column that holds it, baseline name)
TREND_INDICATORS = (
    ("종가", "Close", False, False, None, ""),
    ("거래량", "Volume", True, False, "Volume_MA_50", "50주 평균"),
    ("Momentum", "Momentum", False, False, 0.0, "0"),
    ("MACD", "MACD", False, False, 0.0, "0"),
    ("RSI", "RSI", False, True, 50.0, "50"),
    ("MFI", "MFI", False, True, 50.0, "50"),
)
# Brighter than the 1차 signal green, so the baseline is not taken for it.
BASELINE_COLORS = {"dark": "#39ff14", "light": "#00c853"}
TREND_COLUMNS = [indicator[1] for indicator in TREND_INDICATORS]
BASELINE_COLUMNS = [
    indicator[4] for indicator in TREND_INDICATORS if isinstance(indicator[4], str)
]
POINT_COLUMNS = [*TREND_COLUMNS, *BASELINE_COLUMNS]
RANGE_LOOKBACK_WEEKS = 52
SCALE_STEPS = (25, 10, 5, 2, 1)
OSCILLATOR_GUIDES = (30.0, 70.0)
MIN_POINT_GAP = 0.18
RISE_COLOR = "#ef4444"
FALL_COLOR = "#3b82f6"
MODE_CHANGE = "change"
MODE_POSITION = "position"
MODE_LABELS = {
    MODE_CHANGE: "1차 대비 변화",
    MODE_POSITION: "1년 범위 안 위치",
}


def signal_points(data: pd.DataFrame, cycle: pd.Series | None) -> pd.DataFrame:
    """Indicator values on each signal week of ``cycle``, one row per signal."""
    rows: dict[str, dict] = {}
    if cycle is not None and not data.empty:
        index = pd.DatetimeIndex(data.index)
        for label, column, _ in SIGNAL_POINTS:
            value = cycle.get(column)
            if value is None or pd.isna(value):
                continue
            position = int(index.searchsorted(pd.Timestamp(value), side="left"))
            if position >= len(index):
                continue
            row = data.iloc[position]
            rows[label] = {"Date": index[position], **{c: row[c] for c in POINT_COLUMNS}}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Date", *POINT_COLUMNS])


def range_bounds(
    data: pd.DataFrame,
    points: pd.DataFrame,
    lookback_weeks: int = RANGE_LOOKBACK_WEEKS,
) -> tuple[pd.Series, pd.Series]:
    """Lowest and highest weekly value of each indicator around the cycle.

    The window runs from a year before the first signal up to the last one.
    """
    index = pd.DatetimeIndex(data.index)
    start = max(0, int(index.searchsorted(points["Date"].iloc[0])) - lookback_weeks)
    end = int(index.searchsorted(points["Date"].iloc[-1]))
    window = data.iloc[start : end + 1][TREND_COLUMNS].astype(float)
    return window.min(), window.max()


def range_positions(
    data: pd.DataFrame,
    points: pd.DataFrame,
    lookback_weeks: int = RANGE_LOOKBACK_WEEKS,
) -> pd.DataFrame:
    """Put every indicator on the same 0-100 scale.

    Each value becomes its place between the indicator's lowest (0) and
    highest (100) value in ``range_bounds``.  That makes dollars, share counts
    and oscillators comparable.
    """
    if points.empty:
        return points.reindex(columns=TREND_COLUMNS).astype(float)
    low, high = range_bounds(data, points, lookback_weeks)
    span = (high - low).where(high > low)
    return (points[TREND_COLUMNS].astype(float) - low) / span * 100


def change_from_first(positions: pd.DataFrame) -> pd.DataFrame:
    """Scaled values as a move from the first signal, which becomes 0."""
    if positions.empty:
        return positions
    return positions - positions.iloc[0]


def trend_lines(data: pd.DataFrame, points: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Scaled indicator lines for ``mode``, their average and their spread.

    "벌어짐" is the gap between the highest and lowest indicator: a small
    average with a large spread means the indicators disagree.
    """
    lines = range_positions(data, points)
    if mode == MODE_CHANGE:
        lines = change_from_first(lines)
    return lines.assign(
        평균=lines.mean(axis=1, skipna=True),
        벌어짐=lines.max(axis=1, skipna=True) - lines.min(axis=1, skipna=True),
    )


def weeks_between(dates) -> list[int]:
    """Whole weeks from each signal to the next."""
    dates = pd.DatetimeIndex(dates)
    return [int(round((later - earlier).days / 7)) for earlier, later in zip(dates, dates[1:])]


def time_fractions(dates, min_gap: float = MIN_POINT_GAP) -> list[float]:
    """Horizontal place (0-1) of each signal, spaced by elapsed time.

    A gap never shrinks below ``min_gap`` of the width, so signals a week
    apart still have room for their labels.
    """
    gaps = np.asarray(weeks_between(dates), dtype=float)
    if gaps.size == 0:
        return [0.5]
    if gaps.sum() <= 0:
        gaps = np.ones_like(gaps)
    shares = np.maximum(gaps / gaps.sum(), min(min_gap, 1 / gaps.size))
    shares = shares / shares.sum()
    return [0.0, *np.cumsum(shares).clip(0, 1).tolist()]


class SignalTrendWindow(tk.Toplevel):
    """Small panels with each indicator's signal points, and a combined panel."""

    def __init__(self, master: tk.Misc, theme_mode: str = "light") -> None:
        super().__init__(master)
        self.ui_font_family = configure_ui_fonts(self)
        self.theme_mode = normalize_theme(theme_mode)
        self.palette = theme_palette(self.theme_mode)
        self.configure(background=self.palette["window"])
        self.title("신호 지점 흐름 (임시)")
        self.geometry("1240x700")
        self.minsize(820, 560)

        self.status_var = tk.StringVar(value="")
        self.mode = MODE_CHANGE
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(
            header,
            textvariable=self.status_var,
            font=(self.ui_font_family, 11, "bold"),
            padding=(10, 7, 10, 4),
        ).pack(side="left")
        self.mode_button = ttk.Button(header, command=self._toggle_mode)
        self.mode_button.pack(side="right", padx=(0, 10), pady=(5, 2))
        self._update_mode_button()

        self.canvas = tk.Canvas(
            self, background=self.palette["chart_background"], highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

        self.data = pd.DataFrame()
        self.points = signal_points(self.data, None)

    def show(self, ticker: str, data: pd.DataFrame, cycle: pd.Series | None) -> None:
        self.data = data
        self.points = signal_points(data, cycle)
        dates = "  ·  ".join(
            f"{label} {row['Date']:%Y-%m-%d}" for label, row in self.points.iterrows()
        )
        self.status_var.set(f"{ticker}  |  {dates or '신호 지점 없음'}")
        self.title(f"{ticker} · 신호 지점 흐름 (임시)")
        self._redraw()

    def set_theme(self, mode: str) -> None:
        self.theme_mode = normalize_theme(mode)
        self.palette = theme_palette(self.theme_mode)
        self.configure(background=self.palette["window"])
        self.canvas.configure(background=self.palette["chart_background"])
        self._redraw()

    def _toggle_mode(self) -> None:
        self.mode = MODE_POSITION if self.mode == MODE_CHANGE else MODE_CHANGE
        self._update_mode_button()
        self._redraw()

    def _update_mode_button(self) -> None:
        other = MODE_POSITION if self.mode == MODE_CHANGE else MODE_CHANGE
        self.mode_button.configure(text=f"종합 기준 바꾸기 → {MODE_LABELS[other]}")

    def _redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 200 or height < 200:
            return
        if self.points.empty:
            canvas.create_text(
                width / 2,
                height / 2,
                text="이 기록에는 신호 지점이 없습니다.",
                fill=self.palette["chart_text"],
                font=(self.ui_font_family, 11),
            )
            return

        margin, gap = 14.0, 10.0
        small_bottom = margin + (height - 2 * margin) * 0.44
        small_width = (width - 2 * margin - gap * (len(TREND_INDICATORS) - 1)) / len(
            TREND_INDICATORS
        )
        self._fractions = time_fractions(self.points["Date"])
        low, high = range_bounds(self.data, self.points)
        for number, indicator in enumerate(TREND_INDICATORS):
            column, fixed_scale = indicator[1], indicator[3]
            left = margin + number * (small_width + gap)
            scale = (0.0, 100.0) if fixed_scale else (float(low[column]), float(high[column]))
            self._draw_indicator_panel(
                (left, margin, left + small_width, small_bottom), indicator, scale
            )
        self._draw_combined_panel((margin, small_bottom + gap, width - margin, height - margin))

    def _text(self, x: float, y: float, text: str, anchor: str, color: str, size: int = 9,
              bold: bool = False) -> int:
        font = (self.ui_font_family, size, "bold") if bold else (self.ui_font_family, size)
        return self.canvas.create_text(x, y, text=text, anchor=anchor, fill=color, font=font)

    def _text_on_panel(self, x: float, y: float, text: str, anchor: str, color: str,
                       size: int = 9, bold: bool = False) -> None:
        """Text on a panel-colored patch, so lines behind it do not cut through."""
        item = self._text(x, y, text, anchor, color, size, bold)
        left, top, right, bottom = self.canvas.bbox(item)
        self.canvas.create_rectangle(
            left - 3, top, right + 3, bottom, fill=self.palette["chart_panel"], outline=""
        )
        self.canvas.tag_raise(item)

    def _draw_frame(self, box: tuple[float, float, float, float], title: str, color: str) -> None:
        left, top, right, bottom = box
        self.canvas.create_rectangle(
            left, top, right, bottom,
            outline=self.palette["chart_border"],
            fill=self.palette["chart_panel"],
        )
        self._text(left + 8, top + 6, title, "nw", color, bold=True)

    def _point_xs(self, left: float, right: float) -> list[float]:
        return [left + fraction * (right - left) for fraction in self._fractions]

    def _draw_points_line(
        self, xs: list[float], ys: list[float | None], color: str, width: float, radius: float,
        signal_colored: bool,
    ) -> None:
        """Connect the points that have a value and draw a dot on each.

        A rising stretch is red and a falling one blue, as on the candles.
        """
        drawn = [(x, y) for x, y in zip(xs, ys) if y is not None]
        for (x1, y1), (x2, y2) in zip(drawn, drawn[1:]):
            # Canvas y grows downward, so a smaller y is a higher value.
            segment = RISE_COLOR if y2 < y1 else FALL_COLOR if y2 > y1 else color
            self.canvas.create_line(x1, y1, x2, y2, fill=segment, width=width)
        for label, x, y in zip(self.points.index, xs, ys):
            if y is None:
                continue
            fill = SIGNAL_COLORS[label] if signal_colored else color
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius, fill=fill, outline=fill
            )

    def _draw_signal_labels(self, xs: list[float], y: float, with_dates: bool) -> None:
        for (label, row), x in zip(self.points.iterrows(), xs):
            text = f"{label} {row['Date']:%m-%d}" if with_dates else label
            self._text(x, y, text, "s", SIGNAL_COLORS[label], bold=True)

    def _draw_indicator_panel(
        self, box: tuple[float, float, float, float], indicator: tuple,
        scale: tuple[float, float],
    ) -> None:
        """One indicator's signal points on a scale that does not exaggerate.

        The oscillators use their full 0-100 scale; the others use the range
        of the past year, so a small move looks small.  A dashed baseline
        shows which side of its reference level each point is on.
        """
        name, column, compact, fixed_scale, baseline, baseline_name = indicator
        left, top, right, bottom = box
        muted = self.palette["chart_text"]
        # Lines are colored by direction only; the signal dots carry the rest.
        color = self.palette["text"]
        self._draw_frame(box, name, color)
        if baseline is None:
            baseline_values = []
        elif isinstance(baseline, str):
            baseline_values = self.points[baseline].astype(float).tolist()
        else:
            baseline_values = [float(baseline)] * len(self.points)
        low, high = scale
        if not np.isfinite([low, high]).all() or high <= low:
            low, high = _padded_range(self.points[column].to_numpy(), padding=0.25)
        # The baseline must be on the scale even when the year never touched it.
        finite_baseline = [value for value in baseline_values if np.isfinite(value)]
        low, high = min([low, *finite_baseline]), max([high, *finite_baseline])
        plot_left, plot_right = left + 30, right - 52
        plot_top, plot_bottom = top + 48, bottom - 48
        y = lambda value: _map(value, low, high, plot_top, plot_bottom)

        # The scale's top and bottom, so the reader sees what the panel spans.
        for value, edge_y in ((high, plot_top), (low, plot_bottom)):
            self.canvas.create_line(
                left + 8, edge_y, right - 8, edge_y, fill=self.palette["chart_grid"]
            )
            edge_text = f"{value:.0f}" if fixed_scale else _value_text(value, compact)
            self._text(right - 8, edge_y - 2, edge_text, "se", muted, size=8)
        if fixed_scale:
            for guide in OSCILLATOR_GUIDES:
                self.canvas.create_line(
                    left + 8, y(guide), right - 8, y(guide),
                    fill=self.palette["chart_grid"], dash=(2, 4),
                )
                self._text(right - 8, y(guide) - 2, f"{guide:.0f}", "se", muted, size=8)
        note = "0~100 눈금" if fixed_scale else "최근 1년 범위"
        if baseline_name:
            # Named up here: inside the plot the name would sit on the points.
            note += f" · 점선 {baseline_name}"
        self._text(right - 8, top + 7, note, "ne", muted, size=8)

        values = self.points[column].astype(float).tolist()
        xs = self._point_xs(plot_left, plot_right)
        self._draw_baseline(left + 8, right - 8, xs, baseline_values, y)
        ys = [None if pd.isna(value) else y(value) for value in values]
        self._draw_points_line(xs, ys, color, width=2.5, radius=5, signal_colored=True)
        for number, (x, point_y, value) in enumerate(zip(xs, ys, values)):
            if point_y is None:
                continue
            # A point lower than its neighbours has lines leaving upward.
            neighbours = [values[n] for n in (number - 1, number + 1) if 0 <= n < len(values)]
            below = bool(neighbours) and all(value <= other for other in neighbours)
            self._text(
                x, point_y + 9 if below else point_y - 9, _value_text(value, compact),
                "n" if below else "s", self.palette["text"],
            )
        self._draw_signal_labels(xs, bottom - 6, with_dates=False)

    def _draw_baseline(self, left: float, right: float, xs: list[float],
                       values: list[float], y) -> None:
        """Dashed reference level across the panel, following it if it moves."""
        drawn = [(x, y(value)) for x, value in zip(xs, values) if np.isfinite(value)]
        if not drawn:
            return
        path = [(left, drawn[0][1]), *drawn, (right, drawn[-1][1])]
        self.canvas.create_line(
            *[coordinate for point in path for coordinate in point],
            fill=BASELINE_COLORS[self.theme_mode], dash=(6, 4), width=2,
        )

    def _draw_combined_panel(self, box: tuple[float, float, float, float]) -> None:
        left, top, right, bottom = box
        mode = self.mode
        muted = self.palette["chart_text"]
        average_color = self.palette["text"]
        lines = trend_lines(self.data, self.points, mode)
        self._draw_frame(box, f"종합 · {MODE_LABELS[mode]}", average_color)
        self._text(
            right - 8, top + 7,
            "눈금: 각 지표의 최근 1년 최저~최고를 0~100으로 환산"
            + ("  ·  1차 신호 때 = 0" if mode == MODE_CHANGE else ""),
            "ne", muted, size=8,
        )

        plot_left, plot_right = left + 96, right - 110
        plot_top, plot_bottom = top + 62, bottom - 46
        if mode == MODE_POSITION:
            low, high = 0.0, 100.0
        else:
            low, high = _padded_range(lines["평균"].to_numpy(), padding=0.35, include_zero=True)
        y = lambda value: _map(value, low, high, plot_top, plot_bottom)
        xs = self._point_xs(plot_left, plot_right)

        step = _scale_step(low, high)
        for tick in np.arange(np.ceil(low / step) * step, high + 0.01, step):
            self.canvas.create_line(
                left + 46, y(tick), right - 100, y(tick), fill=self.palette["chart_grid"]
            )
            self._text(left + 40, y(tick), _scaled_text(tick, mode), "e", muted, size=8)
        if mode == MODE_CHANGE:
            self.canvas.create_line(
                left + 46, y(0), right - 100, y(0), fill=muted, dash=(4, 3)
            )

        # Only the average is drawn; "벌어짐" tells how far the indicators spread.
        average_ys = [None if pd.isna(value) else y(value) for value in lines["평균"]]
        self._draw_points_line(xs, average_ys, average_color, width=4, radius=6, signal_colored=False)
        if len(xs) > 1 and average_ys[-1] is not None:
            self._text(xs[-1] + 14, average_ys[-1], "평균", "w", average_color, bold=True)

        for x, point_y, value, spread in zip(xs, average_ys, lines["평균"], lines["벌어짐"]):
            if point_y is None:
                continue
            self._text_on_panel(
                x, point_y - 26, _scaled_text(value, mode), "s", average_color, 11, True
            )
            if spread > 0:
                self._text_on_panel(x, point_y - 12, f"벌어짐 {spread:.0f}", "s", muted, size=8)

        self._draw_signal_labels(xs, bottom - 8, with_dates=True)
        for weeks, earlier, later in zip(weeks_between(self.points["Date"]), xs, xs[1:]):
            self._text((earlier + later) / 2, bottom - 9, f"{weeks}주", "s", muted, size=8)


def _scale_step(low: float, high: float) -> float:
    """Largest round step that still puts at least three ticks on the scale."""
    for step in SCALE_STEPS:
        if (high - low) / step >= 3:
            return step
    return SCALE_STEPS[-1]


def _padded_range(
    values, padding: float, include_zero: bool = False
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    if include_zero:
        low, high = min(0.0, low), max(0.0, high)
    margin = (high - low) * padding if high > low else max(abs(low) * 0.05, 1.0)
    return low - margin, high + margin


def _map(value: float, low: float, high: float, top: float, bottom: float) -> float:
    return bottom - (float(value) - low) / (high - low) * (bottom - top)


def _value_text(value: float, compact: bool) -> str:
    absolute = abs(value)
    if compact and absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if compact and absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1000:
        return f"{value:,.0f}"
    return f"{value:.1f}" if absolute >= 10 else f"{value:.2f}"


def _scaled_text(value: float, mode: str) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:+.0f}" if mode == MODE_CHANGE else f"{value:.0f}"
