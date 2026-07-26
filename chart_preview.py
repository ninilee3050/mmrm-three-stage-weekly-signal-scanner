from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

import numpy as np
import pandas as pd

from ui_theme import configure_ui_fonts, normalize_theme, theme_palette


CHART_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_5",
    "MA_20",
    "MA_50",
    "MA_150",
    "MA_200",
    "Volume_MA_50",
    "Momentum",
    "MACD",
    "Signal",
    "Histogram",
    "RSI",
    "MFI",
]

MA_STYLES = {
    "MA_5": ("#6aa84f", 2.0),
    "MA_20": ("#00d4d8", 2.4),
    "MA_50": ("#1428e8", 2.8),
    "MA_150": ("#9b1010", 3.2),
    "MA_200": ("#f00078", 3.6),
}


def _blend_hex(foreground: str, background: str, opacity: float) -> str:
    foreground_rgb = tuple(int(foreground[index : index + 2], 16) for index in (1, 3, 5))
    background_rgb = tuple(int(background[index : index + 2], 16) for index in (1, 3, 5))
    blended = tuple(
        round(front * opacity + back * (1 - opacity))
        for front, back in zip(foreground_rgb, background_rgb)
    )
    return "#" + "".join(f"{channel:02x}" for channel in blended)


OSCILLATOR_UPPER = 70.0
OSCILLATOR_CENTER = 50.0
OSCILLATOR_LOWER = 30.0
CANDLE_UP_COLOR = "#ef4444"
CANDLE_DOWN_COLOR = "#2563eb"
INDICATOR_BAR_OPACITY = 0.5
POSITIVE_BAR_COLOR = _blend_hex(CANDLE_UP_COLOR, "#ffffff", INDICATOR_BAR_OPACITY)
NEGATIVE_BAR_COLOR = _blend_hex(CANDLE_DOWN_COLOR, "#ffffff", INDICATOR_BAR_OPACITY)
MACD_SIGNAL_COLOR = MA_STYLES["MA_50"][0]
VOLUME_MA_STYLE = MA_STYLES["MA_50"]
OSCILLATOR_LINE_COLOR = CANDLE_UP_COLOR
CANDLE_WIDTH_RATIO = 0.76
INDICATOR_BAR_WIDTH_RATIO = 0.84
ZOOM_IN_FACTOR = 0.88
ZOOM_OUT_FACTOR = 1.14

SIGNAL_STYLES = (
    ("1차", "FirstSignalDate", "#16a34a"),
    ("2차", "SecondSignalDate", "#f59e0b"),
    ("3차", "ThirdDecisionDate", "#dc2626"),
)


@dataclass(frozen=True)
class Panel:
    name: str
    top: float
    bottom: float


def cycle_view_dates(
    index: pd.DatetimeIndex,
    cycle: pd.Series | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = _normalized_index(index)
    if dates.empty:
        raise ValueError("차트 데이터가 비어 있습니다.")

    latest = dates[-1]
    if cycle is None or pd.isna(cycle.get("FirstSignalDate")):
        start = max(dates[0], latest - pd.DateOffset(years=3))
        return dates[dates.searchsorted(start, side="left")], latest

    first = _timestamp(cycle.get("FirstSignalDate"))
    second = _timestamp(cycle.get("SecondSignalDate"))
    third = _timestamp(cycle.get("ThirdDecisionDate"))
    anchor = third if third is not None else second if second is not None else first
    start = first - pd.DateOffset(years=1)
    end = anchor + pd.DateOffset(years=1)

    if "대기" in str(cycle.get("Outcome", "")):
        end = max(end, latest)
    start = max(dates[0], start)
    end = min(latest, end)
    start_position = min(len(dates) - 1, int(dates.searchsorted(start, side="left")))
    end_position = max(0, int(dates.searchsorted(end, side="right") - 1))
    return dates[start_position], dates[end_position]


def cycle_view_indices(
    index: pd.DatetimeIndex,
    cycle: pd.Series | None,
) -> tuple[int, int]:
    dates = _normalized_index(index)
    start_date, end_date = cycle_view_dates(dates, cycle)
    start = int(dates.searchsorted(start_date, side="left"))
    end = int(dates.searchsorted(end_date, side="right") - 1)
    return max(0, start), min(len(dates) - 1, max(start, end))


class ChartPreviewWindow(tk.Toplevel):
    """Interactive weekly chart kept in one reusable Tkinter window."""

    def __init__(
        self,
        master: tk.Misc,
        on_close=None,
        theme_mode: str = "light",
    ) -> None:
        super().__init__(master)
        self.ui_font_family = configure_ui_fonts(self)
        self._on_close_callback = on_close
        self.theme_mode = normalize_theme(theme_mode)
        self.palette = theme_palette(self.theme_mode)
        self.configure(background=self.palette["window"])
        self.title("MMRM 시나리오 차트 미리보기")
        self.geometry("1600x920")
        self.minsize(1000, 680)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.status_var = tk.StringVar(value="차트 데이터를 준비해 주세요.")
        self.hover_var = tk.StringVar(
            value="휠: 확대·축소  |  드래그: 좌우 이동  |  더블클릭: 선택 사이클 전체 보기"
        )
        ttk.Label(
            self,
            textvariable=self.status_var,
            font=(self.ui_font_family, 11, "bold"),
            padding=(10, 7, 10, 2),
        ).pack(fill="x")
        ttk.Label(
            self,
            textvariable=self.hover_var,
            padding=(10, 2, 10, 7),
        ).pack(fill="x")

        self.canvas = tk.Canvas(
            self,
            background=self.palette["chart_background"],
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.panel_title_font = tkfont.Font(
            root=self,
            family=self.ui_font_family,
            size=8,
            weight="bold",
        )
        self.panel_value_font = tkfont.Font(
            root=self,
            family=self.ui_font_family,
            size=8,
        )

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._clear_crosshair)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<Double-Button-1>", self._reset_view)

        self.ticker = ""
        self.company = ""
        self.data = pd.DataFrame()
        self.cycle: pd.Series | None = None
        self.view_start = 0
        self.view_end = 0
        self.initial_view = (0, 0)
        self._redraw_job: str | None = None
        self._drag_origin: tuple[int, int, int] | None = None

    def set_theme(self, mode: str) -> None:
        self.theme_mode = normalize_theme(mode)
        self.palette = theme_palette(self.theme_mode)
        self.configure(background=self.palette["window"])
        self.canvas.configure(background=self.palette["chart_background"])
        self._schedule_redraw()

    def show_cycle(
        self,
        ticker: str,
        data: pd.DataFrame,
        cycle: pd.Series | None,
        company: str = "",
    ) -> None:
        missing = [column for column in CHART_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError("차트 필수 컬럼이 없습니다: " + ", ".join(missing))

        prepared = data.loc[:, CHART_COLUMNS].copy()
        prepared.index = _normalized_index(pd.DatetimeIndex(prepared.index))
        prepared = prepared[~prepared.index.duplicated(keep="last")].sort_index()
        if prepared.empty:
            raise ValueError("표시할 차트 데이터가 없습니다.")

        self.ticker = ticker.upper()
        self.company = company
        self.data = prepared
        self.cycle = cycle.copy() if cycle is not None else None
        self.view_start, self.view_end = cycle_view_indices(prepared.index, cycle)
        self.initial_view = (self.view_start, self.view_end)
        self._drag_origin = None

        first = _date_text(cycle.get("FirstSignalDate")) if cycle is not None else "-"
        second = _date_text(cycle.get("SecondSignalDate")) if cycle is not None else "-"
        third = _date_text(cycle.get("ThirdDecisionDate")) if cycle is not None else "-"
        outcome = str(cycle.get("Outcome", "전체 차트")) if cycle is not None else "최근 3년"
        name = f" · {company}" if company and company.upper() != self.ticker else ""
        self.status_var.set(
            f"{self.ticker}{name}  |  1차 {first}  ·  2차 {second}  ·  "
            f"3차 {third}  |  {outcome}"
        )
        self.title(f"{self.ticker} · MMRM 시나리오 차트 미리보기")
        self.deiconify()
        self.lift()
        self.focus_force()
        self._schedule_redraw()

    def _close(self) -> None:
        callback = self._on_close_callback
        self._on_close_callback = None
        self.destroy()
        if callback is not None:
            callback()

    def _on_resize(self, _event=None) -> None:
        self._schedule_redraw()

    def _schedule_redraw(self) -> None:
        if self._redraw_job is not None:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after_idle(self._redraw)

    def _redraw(self) -> None:
        self._redraw_job = None
        self.canvas.delete("all")
        if self.data.empty or self.canvas.winfo_width() < 200:
            return

        panels = self._panels()
        visible = self.data.iloc[self.view_start : self.view_end + 1]
        x_positions = self._x_positions(len(visible))

        self._draw_panel_frames(panels, visible)
        self._draw_price(panels[0], visible, x_positions)
        self._draw_volume(panels[1], visible, x_positions)
        self._draw_momentum(panels[2], visible, x_positions)
        self._draw_macd(panels[3], visible, x_positions)
        self._draw_oscillator(panels[4], visible, x_positions, "RSI")
        self._draw_oscillator(panels[5], visible, x_positions, "MFI")
        self._draw_signal_lines(panels)
        self._draw_date_axis(visible, x_positions, panels[-1].bottom)
        self._show_rightmost_values(panels, visible)

    def _panels(self) -> list[Panel]:
        height = max(500, self.canvas.winfo_height())
        top = 12.0
        bottom = height - 28.0
        gap = 5.0
        weights = (0.39, 0.13, 0.10, 0.14, 0.12, 0.12)
        usable = bottom - top - gap * (len(weights) - 1)
        panels = []
        cursor = top
        names = ("가격", "거래량", "Momentum", "MACD", "RSI", "MFI")
        for name, weight in zip(names, weights):
            panel_bottom = cursor + usable * weight
            panels.append(Panel(name, cursor, panel_bottom))
            cursor = panel_bottom + gap
        return panels

    def _plot_edges(self) -> tuple[float, float]:
        return 64.0, max(180.0, self.canvas.winfo_width() - 92.0)

    def _x_positions(self, count: int) -> np.ndarray:
        left, right = self._plot_edges()
        step = (right - left) / max(1, count)
        return left + (np.arange(count) + 0.5) * step

    def _draw_panel_frames(self, panels: list[Panel], visible: pd.DataFrame) -> None:
        left, right = self._plot_edges()
        for panel in panels:
            self.canvas.create_rectangle(
                left,
                panel.top,
                right,
                panel.bottom,
                outline=self.palette["chart_border"],
                fill=self.palette["chart_panel"],
            )
            self.canvas.create_text(
                left + 6,
                panel.top + 5,
                text=panel.name,
                anchor="nw",
                fill=self.palette["chart_text"],
                font=self.panel_title_font,
            )
            for fraction in (0.25, 0.5, 0.75):
                y = panel.top + (panel.bottom - panel.top) * fraction
                self.canvas.create_line(
                    left,
                    y,
                    right,
                    y,
                    fill=self.palette["chart_grid"],
                )

        count = len(visible)
        for fraction in np.linspace(0, 1, min(9, count), endpoint=True):
            x = left + (right - left) * fraction
            self.canvas.create_line(
                x,
                panels[0].top,
                x,
                panels[-1].bottom,
                fill=self.palette["chart_grid"],
            )

    def _draw_price(self, panel: Panel, data: pd.DataFrame, xs: np.ndarray) -> None:
        values = pd.concat(
            [data[["Low", "High"]], data[list(MA_STYLES)]],
            axis=1,
        ).to_numpy(dtype=float)
        low, high = _finite_range(values, padding=0.06)
        y = lambda value: _map_y(value, low, high, panel)
        self._draw_scale(panel, low, high)

        step = (self._plot_edges()[1] - self._plot_edges()[0]) / max(1, len(data))
        candle_half = max(0.7, step * CANDLE_WIDTH_RATIO / 2)

        # Draw moving averages first so candle wicks and bodies remain fully visible
        # when a moving-average line crosses the same price area.
        for column, (color, width) in MA_STYLES.items():
            self._draw_line(
                xs,
                data[column],
                y,
                color,
                width,
                tag=f"{column.lower()}_line",
            )

        for x, (_, row) in zip(xs, data.iterrows()):
            if any(pd.isna(row[column]) for column in ("Open", "High", "Low", "Close")):
                continue
            rising = row["Close"] >= row["Open"]
            color = CANDLE_UP_COLOR if rising else CANDLE_DOWN_COLOR
            self.canvas.create_line(x, y(row["High"]), x, y(row["Low"]), fill=color)
            top = y(max(row["Open"], row["Close"]))
            bottom = y(min(row["Open"], row["Close"]))
            if abs(bottom - top) < 1:
                bottom = top + 1
            self.canvas.create_rectangle(
                x - candle_half,
                top,
                x + candle_half,
                bottom,
                outline=color,
                fill=self.palette["chart_panel"],
                tags=(
                    "candle_body",
                    "rising_candle" if rising else "falling_candle",
                ),
            )

    def _draw_volume(self, panel: Panel, data: pd.DataFrame, xs: np.ndarray) -> None:
        _, high = _finite_range(data[["Volume", "Volume_MA_50"]].to_numpy(), padding=0.08)
        low = 0.0
        y = lambda value: _map_y(value, low, high, panel)
        self._draw_scale(panel, low, high, compact=True)
        step = (self._plot_edges()[1] - self._plot_edges()[0]) / max(1, len(data))
        half = max(0.7, step * INDICATOR_BAR_WIDTH_RATIO / 2)
        baseline = y(0)
        for x, (_, row) in zip(xs, data.iterrows()):
            color = self._indicator_bar_color(row["Close"] >= row["Open"])
            if pd.notna(row["Volume"]):
                self.canvas.create_rectangle(
                    x - half,
                    y(row["Volume"]),
                    x + half,
                    baseline,
                    outline=color,
                    fill=color,
                )
        self._draw_line(xs, data["Volume_MA_50"], y, *VOLUME_MA_STYLE)

    def _draw_momentum(self, panel: Panel, data: pd.DataFrame, xs: np.ndarray) -> None:
        low, high = _finite_range(data["Momentum"].to_numpy(), include_zero=True, padding=0.10)
        y = lambda value: _map_y(value, low, high, panel)
        self._draw_scale(panel, low, high)
        zero = y(0)
        left, right = self._plot_edges()
        self._draw_zero_bars(data["Momentum"], xs, y, "momentum")
        self.canvas.create_line(
            left,
            zero,
            right,
            zero,
            fill=self.palette["zero_line"],
        )

    def _draw_macd(self, panel: Panel, data: pd.DataFrame, xs: np.ndarray) -> None:
        low, high = _finite_range(
            data[["MACD", "Signal", "Histogram"]].to_numpy(),
            include_zero=True,
            padding=0.10,
        )
        y = lambda value: _map_y(value, low, high, panel)
        self._draw_scale(panel, low, high)
        zero = y(0)
        left, right = self._plot_edges()
        self._draw_zero_bars(data["Histogram"], xs, y, "macd")
        self.canvas.create_line(
            left,
            zero,
            right,
            zero,
            fill=self.palette["zero_line"],
        )
        self._draw_line(xs, data["MACD"], y, "#ff5b5b", 1.5)
        self._draw_line(xs, data["Signal"], y, MACD_SIGNAL_COLOR, 1.5)

    def _draw_zero_bars(
        self,
        series: pd.Series,
        xs: np.ndarray,
        mapper,
        tag_prefix: str,
    ) -> None:
        left, right = self._plot_edges()
        step = (right - left) / max(1, len(series))
        half = max(0.8, step * INDICATOR_BAR_WIDTH_RATIO / 2)
        zero = mapper(0)
        for x, value in zip(xs, series):
            if pd.isna(value):
                continue
            positive = value >= 0
            color = self._indicator_bar_color(positive)
            direction = "positive" if positive else "negative"
            self.canvas.create_rectangle(
                x - half,
                mapper(value),
                x + half,
                zero,
                outline=color,
                fill=color,
                tags=(f"{tag_prefix}_bar", f"{tag_prefix}_{direction}_bar"),
            )

    def _draw_oscillator(
        self,
        panel: Panel,
        data: pd.DataFrame,
        xs: np.ndarray,
        column: str,
    ) -> None:
        y = lambda value: _map_y(value, 0.0, 100.0, panel)
        self._draw_scale(panel, 0.0, 100.0)
        left, right = self._plot_edges()
        self._draw_threshold_fill(
            panel,
            xs,
            data[column],
            OSCILLATOR_UPPER,
            above=True,
            color=self.palette["overbought_fill"],
            tag=f"{column.lower()}_overbought_fill",
        )
        self._draw_threshold_fill(
            panel,
            xs,
            data[column],
            OSCILLATOR_LOWER,
            above=False,
            color=self.palette["oversold_fill"],
            tag=f"{column.lower()}_oversold_fill",
        )
        self.canvas.create_line(
            left,
            y(OSCILLATOR_UPPER),
            right,
            y(OSCILLATOR_UPPER),
            fill="#ef4444",
            dash=(4, 3),
        )
        self.canvas.create_line(
            left,
            y(OSCILLATOR_CENTER),
            right,
            y(OSCILLATOR_CENTER),
            fill="#16a34a",
            width=1,
        )
        self.canvas.create_line(
            left,
            y(OSCILLATOR_LOWER),
            right,
            y(OSCILLATOR_LOWER),
            fill="#3b5bff",
            dash=(4, 3),
        )
        self._draw_line(xs, data[column], y, OSCILLATOR_LINE_COLOR, 1.5)

    def _indicator_bar_color(self, positive: bool) -> str:
        candle_color = CANDLE_UP_COLOR if positive else CANDLE_DOWN_COLOR
        return _blend_hex(
            candle_color,
            self.palette["chart_panel"],
            INDICATOR_BAR_OPACITY,
        )

    def _draw_threshold_fill(
        self,
        panel: Panel,
        xs: np.ndarray,
        series: pd.Series,
        threshold: float,
        above: bool,
        color: str,
        tag: str,
    ) -> None:
        baseline = _map_y(threshold, 0.0, 100.0, panel)
        values = series.to_numpy(dtype=float)
        for position in range(len(values) - 1):
            x1, x2 = float(xs[position]), float(xs[position + 1])
            value1, value2 = values[position], values[position + 1]
            if not np.isfinite(value1) or not np.isfinite(value2):
                continue

            inside1 = value1 > threshold if above else value1 < threshold
            inside2 = value2 > threshold if above else value2 < threshold
            if not inside1 and not inside2:
                continue

            if inside1 and inside2:
                points = (
                    x1,
                    baseline,
                    x1,
                    _map_y(value1, 0.0, 100.0, panel),
                    x2,
                    _map_y(value2, 0.0, 100.0, panel),
                    x2,
                    baseline,
                )
            else:
                crossing = x1 + (threshold - value1) / (value2 - value1) * (x2 - x1)
                if inside1:
                    points = (
                        x1,
                        baseline,
                        x1,
                        _map_y(value1, 0.0, 100.0, panel),
                        crossing,
                        baseline,
                    )
                else:
                    points = (
                        crossing,
                        baseline,
                        x2,
                        _map_y(value2, 0.0, 100.0, panel),
                        x2,
                        baseline,
                    )
            self.canvas.create_polygon(
                *points,
                fill=color,
                outline=color,
                tags=("oscillator_fill", tag),
            )

    def _draw_line(
        self,
        xs,
        series,
        mapper,
        color: str,
        width: float,
        tag: str | None = None,
    ) -> None:
        tags = (tag,) if tag else ()
        points: list[float] = []
        for x, value in zip(xs, series):
            if pd.isna(value):
                if len(points) >= 4:
                    self.canvas.create_line(
                        *points,
                        fill=color,
                        width=width,
                        smooth=False,
                        tags=tags,
                    )
                points = []
                continue
            points.extend((float(x), float(mapper(value))))
        if len(points) >= 4:
            self.canvas.create_line(
                *points,
                fill=color,
                width=width,
                smooth=False,
                tags=tags,
            )

    def _draw_scale(self, panel: Panel, low: float, high: float, compact: bool = False) -> None:
        right = self._plot_edges()[1]
        for fraction in (0.0, 0.5, 1.0):
            value = high - (high - low) * fraction
            raw_y = panel.top + (panel.bottom - panel.top) * fraction
            y = min(panel.bottom - 8, max(panel.top + 8, raw_y))
            text = _compact_number(value) if compact else _number(value)
            self.canvas.create_text(
                right + 6,
                y,
                text=text,
                anchor="w",
                fill=self.palette["chart_text"],
                font=(self.ui_font_family, 8),
                tags="scale_label",
            )

    def _draw_signal_lines(self, panels: list[Panel]) -> None:
        if self.cycle is None:
            return
        left, right = self._plot_edges()
        count = self.view_end - self.view_start + 1
        for label, column, color in SIGNAL_STYLES:
            signal_date = _timestamp(self.cycle.get(column))
            if signal_date is None:
                continue
            position = int(self.data.index.searchsorted(signal_date, side="left"))
            if position < self.view_start or position > self.view_end:
                continue
            x = left + ((position - self.view_start) + 0.5) / max(1, count) * (right - left)
            self.canvas.create_line(
                x,
                panels[0].top,
                x,
                panels[-1].bottom,
                fill=color,
                width=2,
            )
            self.canvas.create_text(
                x + 3,
                panels[0].top + 22,
                text=label,
                anchor="nw",
                fill=color,
                font=(self.ui_font_family, 9, "bold"),
            )

    def _draw_date_axis(self, data: pd.DataFrame, xs: np.ndarray, bottom: float) -> None:
        if data.empty:
            return
        tick_count = min(10, len(data))
        positions = np.unique(np.linspace(0, len(data) - 1, tick_count).astype(int))
        for position in positions:
            date = pd.Timestamp(data.index[position]).strftime("%Y-%m-%d")
            self.canvas.create_text(
                xs[position],
                bottom + 8,
                text=date,
                anchor="n",
                fill=self.palette["chart_text"],
                font=(self.ui_font_family, 8),
            )

    def _on_motion(self, event) -> None:
        if self.data.empty or self._drag_origin is not None:
            return
        left, right = self._plot_edges()
        panels = self._panels()
        if event.x < left or event.x > right or event.y < panels[0].top or event.y > panels[-1].bottom:
            self._clear_crosshair()
            return

        count = self.view_end - self.view_start + 1
        relative = (event.x - left) / max(1.0, right - left)
        offset = min(count - 1, max(0, int(relative * count)))
        position = self.view_start + offset
        x = left + (offset + 0.5) / max(1, count) * (right - left)
        panel = next((item for item in panels if item.top <= event.y <= item.bottom), panels[0])

        self.canvas.delete("crosshair")
        self.canvas.create_line(
            x,
            panels[0].top,
            x,
            panels[-1].bottom,
            fill=self.palette["crosshair"],
            dash=(3, 3),
            tags="crosshair",
        )
        self.canvas.create_line(
            left,
            event.y,
            right,
            event.y,
            fill=self.palette["crosshair"],
            dash=(3, 3),
            tags="crosshair",
        )
        row = self.data.iloc[position]
        date = self.data.index[position].strftime("%Y-%m-%d")
        self._draw_panel_hover_values(panels, row)
        self._draw_crosshair_date(x, panels[-1].bottom, date)
        self.hover_var.set(f"{date}  |  십자선 수치는 각 보조지표 제목 옆에 표시됩니다.")

    def _draw_crosshair_date(self, x: float, bottom: float, date: str) -> None:
        left, right = self._plot_edges()
        half_width = self.panel_value_font.measure(date) / 2 + 6
        center_x = min(right - half_width, max(left + half_width, x))
        top = bottom + 2
        label_bottom = bottom + 21
        self.canvas.create_rectangle(
            center_x - half_width,
            top,
            center_x + half_width,
            label_bottom,
            outline=self.palette["date_label_background"],
            fill=self.palette["date_label_background"],
            tags=("crosshair", "crosshair_date_background"),
        )
        self.canvas.create_text(
            center_x,
            (top + label_bottom) / 2,
            text=date,
            fill=self.palette["date_label_text"],
            font=self.panel_value_font,
            tags=("crosshair", "crosshair_date_label"),
        )

    def _draw_panel_hover_values(self, panels: list[Panel], row: pd.Series) -> None:
        self.canvas.delete("panel_hover_value")
        price_values = [
            ("O", row["Open"], self.palette["muted"], False),
            ("H", row["High"], CANDLE_UP_COLOR, False),
            ("L", row["Low"], CANDLE_DOWN_COLOR, False),
            ("C", row["Close"], self.palette["text"], False),
            *[
                (column.replace("MA_", "MA"), row[column], color, False)
                for column, (color, _width) in MA_STYLES.items()
            ],
        ]
        specifications = (
            (panels[0], "price", price_values),
            (
                panels[1],
                "volume",
                [
                    ("거래량", row["Volume"], self.palette["muted"], True),
                    ("평균", row["Volume_MA_50"], VOLUME_MA_STYLE[0], True),
                ],
            ),
            (
                panels[2],
                "momentum",
                [
                    (
                        "",
                        row["Momentum"],
                        self._indicator_bar_color(row["Momentum"] >= 0),
                        False,
                    )
                ],
            ),
            (
                panels[3],
                "macd",
                [
                    (
                        "Histogram",
                        row["Histogram"],
                        self._indicator_bar_color(row["Histogram"] >= 0),
                        False,
                    ),
                    ("MACD", row["MACD"], "#ff5b5b", False),
                    ("Signal", row["Signal"], MACD_SIGNAL_COLOR, False),
                ],
            ),
            (panels[4], "rsi", [("", row["RSI"], OSCILLATOR_LINE_COLOR, False)]),
            (panels[5], "mfi", [("", row["MFI"], OSCILLATOR_LINE_COLOR, False)]),
        )

        left, _right = self._plot_edges()
        for panel, tag_prefix, values in specifications:
            x = left + 12 + self.panel_title_font.measure(panel.name)
            for label, value, color, compact in values:
                formatted = _compact_number(value) if compact else _number(value)
                text = f"{label} {formatted}".strip()
                self.canvas.create_text(
                    x,
                    panel.top + 5,
                    text=text,
                    anchor="nw",
                    fill=color,
                    font=self.panel_value_font,
                    tags=("panel_hover_value", f"{tag_prefix}_hover_value"),
                )
                x += self.panel_value_font.measure(text) + 9

    def _clear_crosshair(self, _event=None) -> None:
        self.canvas.delete("crosshair")
        if self.data.empty:
            return
        visible = self.data.iloc[self.view_start : self.view_end + 1]
        self._show_rightmost_values(self._panels(), visible)

    def _show_rightmost_values(
        self,
        panels: list[Panel],
        visible: pd.DataFrame,
    ) -> None:
        if visible.empty:
            return
        row = visible.iloc[-1]
        date = visible.index[-1].strftime("%Y-%m-%d")
        self._draw_panel_hover_values(panels, row)
        self.hover_var.set(
            f"{date}  |  현재 화면의 가장 오른쪽 주봉 수치"
        )

    def _on_mousewheel(self, event) -> None:
        if self.data.empty or event.delta == 0:
            return
        left, right = self._plot_edges()
        count = self.view_end - self.view_start + 1
        new_count = int(
            round(count * (ZOOM_IN_FACTOR if event.delta > 0 else ZOOM_OUT_FACTOR))
        )
        new_count = min(len(self.data), max(12, new_count))
        anchor = min(1.0, max(0.0, (event.x - left) / max(1.0, right - left)))
        anchor_index = self.view_start + int(anchor * max(0, count - 1))
        new_start = anchor_index - int(anchor * max(0, new_count - 1))
        self.view_start, self.view_end = _clamped_view(new_start, new_count, len(self.data))
        self._clear_crosshair()
        self._schedule_redraw()

    def _on_drag_start(self, event) -> None:
        if self.data.empty:
            return
        self._drag_origin = (event.x, self.view_start, self.view_end)
        self.canvas.configure(cursor="fleur")

    def _on_drag_motion(self, event) -> None:
        if self._drag_origin is None:
            return
        origin_x, origin_start, origin_end = self._drag_origin
        count = origin_end - origin_start + 1
        left, right = self._plot_edges()
        shift = int(round((origin_x - event.x) / max(1.0, right - left) * count))
        self.view_start, self.view_end = _clamped_view(origin_start + shift, count, len(self.data))
        self._schedule_redraw()

    def _on_drag_end(self, _event=None) -> None:
        self._drag_origin = None
        self.canvas.configure(cursor="crosshair")

    def _reset_view(self, _event=None) -> None:
        if self.data.empty:
            return
        self.view_start, self.view_end = self.initial_view
        self._drag_origin = None
        self._clear_crosshair()
        self._schedule_redraw()


def _normalized_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates


def _timestamp(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _date_text(value) -> str:
    timestamp = _timestamp(value)
    return timestamp.strftime("%Y-%m-%d") if timestamp is not None else "-"


def _finite_range(values, include_zero: bool = False, padding: float = 0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    if include_zero:
        low = min(0.0, low)
        high = max(0.0, high)
    if low == high:
        margin = max(abs(low) * 0.05, 1.0)
    else:
        margin = (high - low) * padding
    return low - margin, high + margin


def _map_y(value: float, low: float, high: float, panel: Panel) -> float:
    if high <= low:
        return (panel.top + panel.bottom) / 2
    ratio = (float(value) - low) / (high - low)
    return panel.bottom - ratio * (panel.bottom - panel.top)


def _clamped_view(start: int, count: int, total: int) -> tuple[int, int]:
    count = min(total, max(1, count))
    start = min(max(0, start), total - count)
    return start, start + count - 1


def _number(value: float) -> str:
    if pd.isna(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _compact_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    return _number(value)
