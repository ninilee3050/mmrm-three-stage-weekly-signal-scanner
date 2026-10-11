"""Signal-point trend view (experimental).

Takes one cycle's 1차/2차/3차 weeks and compares how the close, volume, MACD,
RSI and MFI moved between them, plus the average of those moves.  Opened from
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
# (name, data column, line color, show value in K/M/B)
TREND_INDICATORS = (
    ("종가", "Close", "#3b82f6", False),
    ("거래량", "Volume", "#8b5cf6", True),
    ("MACD", "MACD", "#f59e0b", False),
    ("RSI", "RSI", "#10b981", False),
    ("MFI", "MFI", "#ec4899", False),
)
TREND_COLUMNS = [column for _, column, _, _ in TREND_INDICATORS]
RANGE_LOOKBACK_WEEKS = 52
SCALE_STEP = 25
INDICATOR_LINE_OPACITY = 0.6
BAND_OPACITY = 0.13
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
            rows[label] = {"Date": index[position], **{c: row[c] for c in TREND_COLUMNS}}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Date", *TREND_COLUMNS])


def range_positions(
    data: pd.DataFrame,
    points: pd.DataFrame,
    lookback_weeks: int = RANGE_LOOKBACK_WEEKS,
) -> pd.DataFrame:
    """Put every indicator on the same 0-100 scale.

    Each value becomes its place between the indicator's lowest (0) and
    highest (100) weekly value, from a year before the first signal up to the
    last one.  That makes dollars, share counts and oscillators comparable.
    """
    if points.empty:
        return points.reindex(columns=TREND_COLUMNS).astype(float)
    index = pd.DatetimeIndex(data.index)
    start = max(0, int(index.searchsorted(points["Date"].iloc[0])) - lookback_weeks)
    end = int(index.searchsorted(points["Date"].iloc[-1]))
    window = data.iloc[start : end + 1][TREND_COLUMNS].astype(float)
    low, high = window.min(), window.max()
    span = (high - low).where(high > low)
    return (points[TREND_COLUMNS].astype(float) - low) / span * 100


def change_from_first(positions: pd.DataFrame) -> pd.DataFrame:
    """Scaled values as a move from the first signal, which becomes 0."""
    if positions.empty:
        return positions
    return positions - positions.iloc[0]


def trend_lines(data: pd.DataFrame, points: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Scaled indicator lines for ``mode`` with their average in "평균"."""
    lines = range_positions(data, points)
    if mode == MODE_CHANGE:
        lines = change_from_first(lines)
    return lines.assign(평균=lines.mean(axis=1, skipna=True))


class SignalTrendWindow(tk.Toplevel):
    """Small panels with each indicator's signal points, and a combined panel."""

    def __init__(self, master: tk.Misc, theme_mode: str = "light") -> None:
        super().__init__(master)
        self.ui_font_family = configure_ui_fonts(self)
        self.theme_mode = normalize_theme(theme_mode)
        self.palette = theme_palette(self.theme_mode)
        self.configure(background=self.palette["window"])
        self.title("신호 지점 흐름 (임시)")
        self.geometry("1100x760")
        self.minsize(820, 560)

        self.status_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=MODE_CHANGE)
        header = ttk.Frame(self)
        header.pack(fill="x")
        ttk.Label(
            header,
            textvariable=self.status_var,
            font=(self.ui_font_family, 11, "bold"),
            padding=(10, 7, 10, 2),
        ).pack(side="left")
        for mode in (MODE_POSITION, MODE_CHANGE):
            ttk.Radiobutton(
                header,
                text=MODE_LABELS[mode],
                value=mode,
                variable=self.mode_var,
                command=self._redraw,
            ).pack(side="right", padx=(0, 10))
        ttk.Label(header, text="종합 기준:").pack(side="right", padx=(0, 6))

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
        small_bottom = margin + (height - 2 * margin) * 0.36
        small_width = (width - 2 * margin - gap * (len(TREND_INDICATORS) - 1)) / len(
            TREND_INDICATORS
        )
        for number, (name, column, color, compact) in enumerate(TREND_INDICATORS):
            left = margin + number * (small_width + gap)
            self._draw_indicator_panel(
                (left, margin, left + small_width, small_bottom), name, column, color, compact
            )
        self._draw_combined_panel((margin, small_bottom + gap, width - margin, height - margin))

    def _draw_frame(self, box: tuple[float, float, float, float], title: str, color: str) -> None:
        left, top, right, bottom = box
        self.canvas.create_rectangle(
            left, top, right, bottom,
            outline=self.palette["chart_border"],
            fill=self.palette["chart_panel"],
        )
        self.canvas.create_text(
            left + 8, top + 6, text=title, anchor="nw", fill=color,
            font=(self.ui_font_family, 9, "bold"),
        )

    def _point_xs(self, left: float, right: float, padding: float) -> list[float]:
        count = len(self.points)
        if count == 1:
            return [(left + right) / 2]
        step = (right - left - 2 * padding) / (count - 1)
        return [left + padding + number * step for number in range(count)]

    def _draw_points_line(
        self, xs: list[float], ys: list[float | None], color: str, width: float, radius: float,
        signal_colored: bool,
    ) -> None:
        """Connect the points that have a value and draw a dot on each."""
        drawn = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if len(drawn) >= 2:
            self.canvas.create_line(
                *[coordinate for point in drawn for coordinate in point], fill=color, width=width
            )
        signal_colors = {label: signal_color for label, _, signal_color in SIGNAL_POINTS}
        for label, x, y in zip(self.points.index, xs, ys):
            if y is None:
                continue
            fill = signal_colors[label] if signal_colored else color
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius, fill=fill, outline=fill
            )

    def _draw_signal_labels(self, xs: list[float], y: float, with_dates: bool) -> None:
        signal_colors = {label: color for label, _, color in SIGNAL_POINTS}
        for (label, row), x in zip(self.points.iterrows(), xs):
            text = f"{label}  {row['Date']:%Y-%m-%d}" if with_dates else label
            self.canvas.create_text(
                x, y, text=text, anchor="s", fill=signal_colors[label],
                font=(self.ui_font_family, 9, "bold"),
            )

    def _draw_indicator_panel(
        self, box: tuple[float, float, float, float], name: str, column: str, color: str,
        compact: bool,
    ) -> None:
        left, top, right, bottom = box
        self._draw_frame(box, name, color)
        values = self.points[column].astype(float)
        low, high = _padded_range(values.to_numpy(), padding=0.25)
        plot_top, plot_bottom = top + 46, bottom - 46
        xs = self._point_xs(left, right, padding=34)
        ys = [
            None if pd.isna(value) else _map(value, low, high, plot_top, plot_bottom)
            for value in values
        ]
        self._draw_points_line(xs, ys, color, width=2.5, radius=5, signal_colored=True)
        lowest = values.min()
        for x, y, value in zip(xs, ys, values):
            if y is None:
                continue
            # The lowest point has lines leaving upward, so its label goes below.
            below = len(values) > 1 and value == lowest
            self.canvas.create_text(
                x, y + 9 if below else y - 9, text=_value_text(value, compact),
                anchor="n" if below else "s",
                fill=self.palette["text"], font=(self.ui_font_family, 9),
            )
        self._draw_signal_labels(xs, bottom - 6, with_dates=False)

    def _draw_combined_panel(self, box: tuple[float, float, float, float]) -> None:
        left, top, right, bottom = box
        mode = self.mode_var.get()
        lines = trend_lines(self.data, self.points, mode)
        title = (
            "종합 · 1차 신호 때를 0으로 본 변화 (각 지표의 최근 1년 범위를 100으로 환산)"
            if mode == MODE_CHANGE
            else "종합 · 각 지표가 최근 1년 최저(0)~최고(100) 중 어디에 있는지"
        )
        self._draw_frame(box, title, self.palette["text"])

        plot_left, plot_right = left + 54, right - 20
        plot_top, plot_bottom = top + 58, bottom - 34
        low, high = (0.0, 100.0) if mode == MODE_POSITION else _change_range(lines.to_numpy())
        y = lambda value: _map(value, low, high, plot_top, plot_bottom)
        xs = self._point_xs(plot_left, plot_right, padding=70)
        panel_color = self.palette["chart_panel"]
        average_color = self.palette["text"]
        band_color = _blend(average_color, panel_color, BAND_OPACITY)
        self._draw_spread_band(xs, lines[TREND_COLUMNS], y, band_color)

        for tick in np.arange(low, high + 1, SCALE_STEP):
            self.canvas.create_line(
                plot_left, y(tick), plot_right, y(tick), fill=self.palette["chart_grid"]
            )
            self.canvas.create_text(
                plot_left - 8, y(tick), text=_scaled_text(tick, mode), anchor="e",
                fill=self.palette["chart_text"], font=(self.ui_font_family, 8),
            )
        if mode == MODE_CHANGE:
            self.canvas.create_line(
                plot_left, y(0), plot_right, y(0), fill=self.palette["chart_text"], dash=(4, 3)
            )

        # The average is the main line; the indicators stay faint behind it.
        legend_x = left + 10
        for name, column, color, _ in TREND_INDICATORS:
            faint = _blend(color, panel_color, INDICATOR_LINE_OPACITY)
            ys = [None if pd.isna(value) else y(value) for value in lines[column]]
            self._draw_points_line(xs, ys, faint, width=1.3, radius=2.5, signal_colored=False)
            legend_x = self._draw_legend_item(legend_x, top + 34, name, faint, width=2)
        legend_x = self._draw_legend_item(legend_x, top + 34, "최고~최저 범위", band_color, width=10)
        average_ys = [None if pd.isna(value) else y(value) for value in lines["평균"]]
        self._draw_points_line(xs, average_ys, average_color, width=4, radius=6, signal_colored=False)
        self._draw_legend_item(legend_x, top + 34, "평균", average_color, width=4)
        for x, point_y, value in zip(xs, average_ys, lines["평균"]):
            if point_y is None:
                continue
            self.canvas.create_text(
                x, point_y - 12, text=_scaled_text(value, mode), anchor="s",
                fill=average_color, font=(self.ui_font_family, 10, "bold"),
            )
        self._draw_signal_labels(xs, bottom - 8, with_dates=True)

    def _draw_spread_band(self, xs: list[float], indicators: pd.DataFrame, y, color: str) -> None:
        """Shade between the highest and lowest indicator at each signal.

        The lines start together and drift apart, so the band widens like a
        forecast cone and shows how much the indicators disagree.
        """
        upper = [
            (x, y(value)) for x, value in zip(xs, indicators.max(axis=1)) if not pd.isna(value)
        ]
        lower = [
            (x, y(value)) for x, value in zip(xs, indicators.min(axis=1)) if not pd.isna(value)
        ]
        if len(upper) < 2:
            return
        outline = [coordinate for point in upper + lower[::-1] for coordinate in point]
        self.canvas.create_polygon(*outline, fill=color, outline="")

    def _draw_legend_item(self, x: float, y: float, name: str, color: str, width: int) -> float:
        self.canvas.create_line(x, y, x + 22, y, fill=color, width=width)
        text = self.canvas.create_text(
            x + 28, y, text=name, anchor="w", fill=self.palette["chart_text"],
            font=(self.ui_font_family, 9),
        )
        return self.canvas.bbox(text)[2] + 18


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


def _change_range(values) -> tuple[float, float]:
    """Range around 0 that ends on scale steps, with room for the point labels."""
    low, high = _padded_range(values, padding=0.12, include_zero=True)
    return (
        float(np.floor(low / SCALE_STEP) * SCALE_STEP),
        float(np.ceil(high / SCALE_STEP) * SCALE_STEP),
    )


def _blend(color: str, background: str, opacity: float) -> str:
    """``color`` at ``opacity`` over ``background``; a Tk canvas has no alpha."""
    front = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    back = [int(background[index : index + 2], 16) for index in (1, 3, 5)]
    return "#" + "".join(
        f"{round(f * opacity + b * (1 - opacity)):02x}" for f, b in zip(front, back)
    )


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
