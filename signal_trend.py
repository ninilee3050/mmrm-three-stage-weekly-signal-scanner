"""Signal-point trend view.

Takes one cycle's 1차/2차/3차 weeks and compares how the close, volume,
momentum, MACD, RSI and MFI moved between them, plus the average of those moves.  Opened in
its own window from a button in the chart preview.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

from chart_preview import render_header_items
from market_context import SP500_NAME, SP500_TICKER, load_sp500_context
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
BENCHMARK_TICKER = SP500_TICKER
BENCHMARK_NAME = SP500_NAME
BENCHMARK_OPEN_TEXT = "S&P 500 비교 열기"
BENCHMARK_CLOSE_TEXT = "S&P 500 비교 닫기"
END_LABEL_GAP = 16.0
SCALE_STEPS = (25, 10, 5, 2, 1)
OSCILLATOR_GUIDES = (30.0, 70.0)
MIN_POINT_GAP = 0.18
WEEK_LINE_SPACING = 6.0
RISE_COLOR = "#ef4444"
FALL_COLOR = "#3b82f6"


def signal_points(data: pd.DataFrame, cycle: pd.Series | None) -> pd.DataFrame:
    """Indicator values on each signal week of ``cycle``, one row per signal."""
    if cycle is None:
        return points_on_dates(data, {})
    return points_on_dates(
        data, {label: cycle.get(column) for label, column, _ in SIGNAL_POINTS}
    )


def points_on_dates(data: pd.DataFrame, dates: dict) -> pd.DataFrame:
    """Indicator values of ``data`` on the week of each labelled date."""
    rows: dict[str, dict] = {}
    if not data.empty:
        index = pd.DatetimeIndex(data.index)
        for label, value in dates.items():
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


def trend_lines(data: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Each indicator's move since the first signal, with their average in "평균"."""
    lines = change_from_first(range_positions(data, points))
    return lines.assign(평균=lines.mean(axis=1, skipna=True))


def average_range(*averages: pd.Series) -> tuple[float, float]:
    """Vertical range of the combined panel that holds every given average and 0."""
    values = np.concatenate([np.asarray(average, dtype=float) for average in averages])
    return _padded_range(values, padding=0.35, include_zero=True)


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


def week_line_xs(xs: list[float], weeks: list[int], min_spacing: float = WEEK_LINE_SPACING) -> list[float]:
    """Horizontal places of the weeks between the signals, signal weeks left out.

    When the weeks of a stretch would be closer than ``min_spacing`` pixels,
    only every second, third, ... week is kept.
    """
    places: list[float] = []
    for count, start, end in zip(weeks, xs, xs[1:]):
        if count < 2:
            continue
        width = (end - start) / count
        stride = max(1, int(np.ceil(min_spacing / width))) if width > 0 else count
        places += [start + week * width for week in range(stride, count, stride)]
    return places


class SignalTrendWindow(tk.Toplevel):
    """Signal-point trend of one record, with the S&P 500 beside it on request."""

    def __init__(
        self,
        master: tk.Misc,
        theme_mode: str = "light",
        on_navigate=None,
        on_close=None,
    ) -> None:
        super().__init__(master)
        self.ui_font_family = configure_ui_fonts(self)
        self._on_navigate_callback = on_navigate
        self._on_close_callback = on_close
        self._navigating = False
        self.theme_mode = normalize_theme(theme_mode)
        self.configure(background=theme_palette(self.theme_mode)["window"])
        self.title("신호 지점 흐름")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.geometry("1240x750")
        self.minsize(820, 560)

        self.status_var = tk.StringVar(value="")
        self.primary_frame = ttk.Frame(self)
        self.primary_frame.pack(side="left", fill="both", expand=True)
        header = self.header = ttk.Frame(self.primary_frame)
        header.pack(fill="x")
        ttk.Label(
            header,
            textvariable=self.status_var,
            font=(self.ui_font_family, 12, "bold"),
            padding=(10, 7, 10, 4),
        ).pack(side="left")
        self.benchmark_button = ttk.Button(
            header, text=BENCHMARK_OPEN_TEXT, command=self._toggle_benchmark
        )
        self.benchmark_button.pack(side="right", padx=(0, 10), pady=(5, 2))
        # The same previous/next controls as the chart preview, which does the moving.
        self.next_button = ttk.Button(
            header, text="다음 →", command=lambda: self._request_navigation(1), state="disabled"
        )
        self.next_button.pack(side="right", padx=(0, 16), pady=(5, 2))
        self.navigation_var = tk.StringVar(value="- / -")
        ttk.Label(header, textvariable=self.navigation_var, anchor="center", width=8).pack(
            side="right"
        )
        self.previous_button = ttk.Button(
            header, text="← 이전", command=lambda: self._request_navigation(-1), state="disabled"
        )
        self.previous_button.pack(side="right", pady=(5, 2))
        self.bind("<Left>", lambda _event: self._request_navigation(-1))
        self.bind("<Right>", lambda _event: self._request_navigation(1))
        # The same record values as the chart preview's header.
        self.info_frame = ttk.Frame(self.primary_frame, padding=(10, 0, 10, 8))
        self.info_frame.pack(fill="x")
        self.header_items: list[tuple[str, str, str]] = []
        self.canvas = TrendCanvas(self.primary_frame, self.ui_font_family, self.theme_mode)
        self.canvas.pack(fill="both", expand=True)

        self.benchmark_frame = ttk.Frame(self)
        # Same height as the stock's header, so the two charts line up.
        self.benchmark_header = ttk.Frame(self.benchmark_frame)
        self.benchmark_header.pack(fill="x")
        self.benchmark_header.pack_propagate(False)
        ttk.Label(
            self.benchmark_header,
            text=f"{BENCHMARK_NAME} ({BENCHMARK_TICKER})  |  같은 주의 값",
            font=(self.ui_font_family, 11, "bold"),
            padding=(10, 7, 10, 4),
        ).pack(side="left")
        self.benchmark_canvas = TrendCanvas(
            self.benchmark_frame, self.ui_font_family, self.theme_mode
        )
        self.benchmark_canvas.pack(fill="both", expand=True)

        self.ticker = ""
        self.data = pd.DataFrame()
        self.points = signal_points(self.data, None)
        self.benchmark_data = pd.DataFrame()
        self.benchmark_visible = False
        self.benchmark_loading = False
        self.benchmark_error = ""
        self._geometry_before_benchmark = ""

    def show(
        self,
        ticker: str,
        data: pd.DataFrame,
        cycle: pd.Series | None,
        navigation_index: int | None = None,
        navigation_total: int = 0,
        heading: str = "",
        header_items: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Show one record; ``heading`` and ``header_items`` fill the header."""
        self._set_navigation_state(navigation_index, navigation_total)
        self.ticker = ticker
        self.data = data
        self.points = signal_points(data, cycle)
        self.status_var.set(heading or ticker)
        self.header_items = list(header_items or [])
        self._render_header()
        self.title(f"{ticker} · 신호 지점 흐름")
        self._ensure_benchmark()
        self._show_charts()
        if self._navigating:
            # The chart preview raised itself to show the record; come back on top.
            self._navigating = False
            self.lift()
            self.focus_force()

    def _render_header(self) -> None:
        render_header_items(
            self.info_frame, self.header_items, theme_palette(self.theme_mode), self.ui_font_family
        )

    def _close(self) -> None:
        self.destroy()
        if self._on_close_callback is not None:
            self._on_close_callback()

    def _request_navigation(self, direction: int) -> str:
        button = self.previous_button if direction < 0 else self.next_button
        if self._on_navigate_callback is not None and str(button.cget("state")) != "disabled":
            self._navigating = True
            self._on_navigate_callback(direction)
            self._navigating = False
        return "break"

    def _set_navigation_state(self, index: int | None, total: int) -> None:
        if index is None or total <= 0:
            self.navigation_var.set("- / -")
            self.previous_button.configure(state="disabled")
            self.next_button.configure(state="disabled")
            return
        self.navigation_var.set(f"{index + 1} / {total}")
        self.previous_button.configure(state="normal" if index > 0 else "disabled")
        self.next_button.configure(state="normal" if index < total - 1 else "disabled")

    def set_theme(self, mode: str) -> None:
        self.theme_mode = normalize_theme(mode)
        self.configure(background=theme_palette(self.theme_mode)["window"])
        self._render_header()
        self.canvas.set_theme(self.theme_mode)
        self.benchmark_canvas.set_theme(self.theme_mode)

    def _toggle_benchmark(self) -> None:
        if self.benchmark_visible:
            self.benchmark_visible = False
            self.benchmark_frame.pack_forget()
            self.benchmark_button.configure(text=BENCHMARK_OPEN_TEXT)
            if self._geometry_before_benchmark and self.state() == "normal":
                self.geometry(self._geometry_before_benchmark)
            self._show_charts()
            return

        self.update_idletasks()
        self.benchmark_visible = True
        self._geometry_before_benchmark = self.geometry()
        width, height = self.winfo_width(), self.winfo_height()
        if self.state() == "normal":
            # Double the window so the stock keeps its width, as far as the screen allows.
            width = min(self.winfo_screenwidth(), width * 2)
            x = max(0, min(self.winfo_x(), self.winfo_screenwidth() - width))
            self.geometry(f"{width}x{height}+{x}+{self.winfo_y()}")
        self.benchmark_header.configure(
            height=self.header.winfo_height() + self.info_frame.winfo_height()
        )
        self.benchmark_frame.configure(width=width // 2)
        self.benchmark_frame.pack_propagate(False)
        self.benchmark_frame.pack(side="right", fill="both")
        self.benchmark_button.configure(text=BENCHMARK_CLOSE_TEXT)

        self.benchmark_error = ""  # opening the comparison retries a failed load
        self._ensure_benchmark()
        self._show_charts()

    def _ensure_benchmark(self) -> None:
        """Get the S&P 500 series once; a failed load is not retried by itself."""
        if not self.benchmark_data.empty or self.benchmark_loading or self.benchmark_error:
            return
        loaded = getattr(self.master, "benchmark_data", None)
        if loaded is not None and not loaded.empty:
            # The chart preview already has the series.
            self.benchmark_data = loaded
        else:
            self.benchmark_loading = True
            threading.Thread(target=self._load_benchmark_worker, daemon=True).start()

    def _load_benchmark_worker(self) -> None:
        try:
            expected_latest = self.data.index[-1] if not self.data.empty else None
            prepared, _ = load_sp500_context(expected_latest_date=expected_latest)
            error = "" if not prepared.empty else "S&P 500 데이터가 비어 있습니다."
        except Exception as exc:  # Network/provider errors are shown in the window.
            prepared, error = pd.DataFrame(), str(exc)
        try:
            self.after(0, self._finish_benchmark_load, prepared, error)
        except tk.TclError:
            pass  # The window was closed while loading.

    def _finish_benchmark_load(self, data: pd.DataFrame, error: str) -> None:
        if not self.winfo_exists():
            return
        self.benchmark_loading = False
        self.benchmark_error = error
        if not error:
            index = pd.DatetimeIndex(pd.to_datetime(data.index))
            data = data.set_axis(index.tz_localize(None) if index.tz is not None else index)
            self.benchmark_data = data[~data.index.duplicated(keep="last")].sort_index()
        self._show_charts()

    def _show_charts(self) -> None:
        """Draw the record and the S&P 500 on the same signal weeks.

        With the comparison closed, the S&P 500's average is laid over the
        record's.  Side by side, each chart shows its own average and the two
        share one vertical scale, so the same slope means the same move.
        """
        if not self.benchmark_visible:
            overlay = None
            if not self.benchmark_data.empty and not self.points.empty:
                points = points_on_dates(self.benchmark_data, self.points["Date"].to_dict())
                overlay = trend_lines(self.benchmark_data, points)["평균"]
            self.canvas.show(self.data, self.points, overlay=overlay, name=self.ticker)
            return
        if self.benchmark_loading:
            message = "S&P 500 주봉 데이터를 불러오는 중입니다..."
        elif self.benchmark_error:
            message = "S&P 500 데이터를 불러오지 못했습니다.\n" + self.benchmark_error
        else:
            message = ""
        if message or self.benchmark_data.empty:
            self.canvas.show(self.data, self.points)
            self.benchmark_canvas.show(pd.DataFrame(), signal_points(pd.DataFrame(), None), message)
            return
        points = points_on_dates(self.benchmark_data, self.points["Date"].to_dict())
        shared = average_range(
            trend_lines(self.data, self.points)["평균"],
            trend_lines(self.benchmark_data, points)["평균"],
        )
        self.canvas.show(self.data, self.points, scale=shared)
        self.benchmark_canvas.show(self.benchmark_data, points, scale=shared)


class TrendCanvas(tk.Canvas):
    """Six indicator panels and the combined average for one price series."""

    def __init__(self, master: tk.Misc, font_family: str, theme_mode: str) -> None:
        self.ui_font_family = font_family
        self.theme_mode = theme_mode
        self.palette = theme_palette(theme_mode)
        super().__init__(
            master, background=self.palette["chart_background"], highlightthickness=0
        )
        self.data = pd.DataFrame()
        self.points = signal_points(self.data, None)
        self.message = ""
        self.average_scale: tuple[float, float] | None = None
        self.overlay: pd.Series | None = None
        self.name = ""
        self.bind("<Configure>", lambda _event: self._redraw())

    def show(
        self,
        data: pd.DataFrame,
        points: pd.DataFrame,
        message: str = "",
        scale: tuple[float, float] | None = None,
        overlay: pd.Series | None = None,
        name: str = "",
    ) -> None:
        """Draw ``points`` of ``data``.

        ``message`` replaces the empty-state text; ``scale`` fixes the range
        of the combined panel instead of fitting it to this average.
        ``overlay`` is the S&P 500's average on the same signals, drawn over
        the average of ``name`` in the combined panel.
        """
        self.data, self.points, self.message = data, points, message
        self.average_scale = scale
        self.overlay = overlay
        self.name = name
        self._redraw()

    def set_theme(self, mode: str) -> None:
        self.theme_mode = mode
        self.palette = theme_palette(mode)
        self.configure(background=self.palette["chart_background"])
        self._redraw()

    def _redraw(self) -> None:
        canvas = self
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 200 or height < 200:
            return
        if self.points.empty:
            canvas.create_text(
                width / 2,
                height / 2,
                text=self.message or "이 기록에는 신호 지점이 없습니다.",
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
        return self.create_text(x, y, text=text, anchor=anchor, fill=color, font=font)

    def _draw_frame(self, box: tuple[float, float, float, float], title: str, color: str) -> None:
        left, top, right, bottom = box
        self.create_rectangle(
            left, top, right, bottom,
            outline=self.palette["chart_border"],
            fill=self.palette["chart_panel"],
        )
        self._text(left + 8, top + 6, title, "nw", color, bold=True)

    def _point_xs(self, left: float, right: float) -> list[float]:
        return [left + fraction * (right - left) for fraction in self._fractions]

    def _draw_points_line(
        self, xs: list[float], ys: list[float | None], color: str, width: float, radius: float,
        signal_colored: bool, dash: tuple[int, int] | None = None, hollow: bool = False,
    ) -> None:
        """Connect the points that have a value and draw a dot on each.

        A rising stretch is red and a falling one blue, as on the candles.
        ``dash`` and ``hollow`` set a second series apart from the main one.
        """
        drawn = [(x, y) for x, y in zip(xs, ys) if y is not None]
        for (x1, y1), (x2, y2) in zip(drawn, drawn[1:]):
            # Canvas y grows downward, so a smaller y is a higher value.
            segment = RISE_COLOR if y2 < y1 else FALL_COLOR if y2 > y1 else color
            self.create_line(x1, y1, x2, y2, fill=segment, width=width, dash=dash)
        for label, x, y in zip(self.points.index, xs, ys):
            if y is None:
                continue
            fill = SIGNAL_COLORS[label] if signal_colored else color
            self.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=self.palette["chart_panel"] if hollow else fill, outline=fill, width=2,
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
        plot_top, plot_bottom = top + 62, bottom - 48
        y = lambda value: _map(value, low, high, plot_top, plot_bottom)

        # The scale's top and bottom, so the reader sees what the panel spans.
        for value, edge_y in ((high, plot_top), (low, plot_bottom)):
            self.create_line(
                left + 8, edge_y, right - 8, edge_y, fill=self.palette["chart_grid"]
            )
            edge_text = f"{value:.0f}" if fixed_scale else _value_text(value, compact)
            self._text(right - 8, edge_y - 2, edge_text, "se", muted, size=8)
        if fixed_scale:
            for guide in OSCILLATOR_GUIDES:
                self.create_line(
                    left + 8, y(guide), right - 8, y(guide),
                    fill=self.palette["chart_grid"], dash=(2, 4),
                )
                self._text(right - 8, y(guide) - 2, f"{guide:.0f}", "se", muted, size=8)
        note = "0~100 눈금" if fixed_scale else "최근 1년 범위"
        if baseline_name:
            # Named up here: inside the plot the name would sit on the points.
            note += f" · 점선 {baseline_name}"
        # Under the title: beside it the two overlap in a narrow panel.
        self._text(left + 8, top + 24, note, "nw", muted, size=8)

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
        self.create_line(
            *[coordinate for point in path for coordinate in point],
            fill=BASELINE_COLORS[self.theme_mode], dash=(6, 4), width=2,
        )

    def _draw_combined_panel(self, box: tuple[float, float, float, float]) -> None:
        left, top, right, bottom = box
        muted = self.palette["chart_text"]
        average_color = self.palette["text"]
        lines = trend_lines(self.data, self.points)
        overlay = None if self.overlay is None else self.overlay.reindex(self.points.index)
        self._draw_frame(box, "종합 · 1차 대비 변화", average_color)
        note = "1차 신호 때 = 0  ·  여섯 지표가 평균적으로 오르면 +, 내리면 −"
        if overlay is not None:
            note = f"굵은 실선 {self.name}  ·  가는 점선 {BENCHMARK_NAME} (같은 주)  ·  " + note
        self._text(right - 8, top + 7, note, "ne", muted, size=8)

        plot_left, plot_right = left + 96, right - 110
        plot_top, plot_bottom = top + 62, bottom - 46
        if overlay is not None:
            low, high = average_range(lines["평균"], overlay)
        else:
            low, high = self.average_scale or average_range(lines["평균"])
        y = lambda value: _map(value, low, high, plot_top, plot_bottom)
        xs = self._point_xs(plot_left, plot_right)

        step = _scale_step(low, high)
        for tick in np.arange(np.ceil(low / step) * step, high + 0.01, step):
            self.create_line(
                left + 46, y(tick), right - 100, y(tick), fill=self.palette["chart_grid"]
            )
            self._text(left + 40, y(tick), _change_text(tick), "e", muted, size=8)
        self.create_line(left + 46, y(0), right - 100, y(0), fill=muted, dash=(4, 3))
        # One vertical line per week, signal weeks included.
        for x in [*xs, *week_line_xs(xs, weeks_between(self.points["Date"]))]:
            self.create_line(
                x, plot_top - 18, x, plot_bottom + 8, fill=self.palette["chart_grid"]
            )

        average_ys = [None if pd.isna(value) else y(value) for value in lines["평균"]]
        if overlay is None:
            overlay_ys: list[float | None] = [None] * len(xs)
            overlay_values = [float("nan")] * len(xs)
            ends = [(average_ys[-1], "평균", average_color)]
        else:
            overlay_values = overlay.tolist()
            overlay_ys = [None if pd.isna(value) else y(value) for value in overlay_values]
            self._draw_points_line(
                xs, overlay_ys, muted, width=2, radius=6, signal_colored=True,
                dash=(6, 4), hollow=True,
            )
            ends = [(average_ys[-1], self.name, average_color), (overlay_ys[-1], BENCHMARK_NAME, muted)]
        self._draw_points_line(xs, average_ys, average_color, width=4, radius=9, signal_colored=True)

        # Name each line at its right end, moved apart when the ends are close.
        ends = sorted((end for end in ends if end[0] is not None), key=lambda end: end[0])
        if len(xs) > 1 and ends:
            shift = max(0.0, END_LABEL_GAP - (ends[-1][0] - ends[0][0])) / 2
            offsets = (-shift, shift) if len(ends) == 2 else (0.0,)
            for (end_y, name, color), offset in zip(ends, offsets):
                self._text(xs[-1] + 18, end_y + offset, name, "w", color, bold=True)

        # The higher line's number goes above its point, the lower one's below.
        for x, point_y, value, overlay_y, overlay_value in zip(
            xs, average_ys, lines["평균"], overlay_ys, overlay_values
        ):
            on_top = overlay_y is None or point_y is None or point_y <= overlay_y
            if point_y is not None:
                self._text(
                    x, point_y - 15 if on_top else point_y + 15, _change_text(value),
                    "s" if on_top else "n", average_color, 11, True,
                )
            # Where both lines sit on the same spot, one number is enough.
            if overlay_y is not None and overlay_y != point_y:
                self._text(
                    x, overlay_y + 12 if on_top else overlay_y - 12, _change_text(overlay_value),
                    "n" if on_top else "s", muted, 9,
                )

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


def _change_text(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value + 0.0:+.0f}"  # adding 0.0 turns -0.0 into 0.0
