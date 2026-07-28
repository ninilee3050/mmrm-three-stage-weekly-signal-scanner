from __future__ import annotations

import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from chart_preview import ChartPreviewWindow
from data_provider import DataLoadError, load_weekly_data, normalize_ticker
from indicators import calculate_indicators
from market_cap_provider import MarketCapCompany, MarketCapLoadError, fetch_us_top_market_cap
from performance_analytics import (
    build_all_field_outputs,
    build_field_performance,
    build_stock_ranking,
    build_ticker_performance,
    format_rate,
    format_reach_rate,
)
from scanner import scan_signal_cycles
from sector_provider import load_sector_classifications
from scenario_tracker import (
    ACTIVE_SCENARIO_COLUMNS,
    CLOSED_RESULT_COLUMNS,
    SCAN_EVENT_COLUMNS,
    load_active_scenarios,
    merge_scan_universe,
    preserve_failed_active_rows,
    save_active_scenarios,
    summarize_ticker_cycles,
)
from ui_theme import configure_ui_fonts, load_theme, save_theme, theme_palette


OUTPUT_DIR = Path("outputs")
DOWNLOADS_DIR = Path.home() / "Downloads"
UI_SETTINGS_PATH = OUTPUT_DIR / "ui_settings.json"
CLOSED_SCENARIO_PATH = OUTPUT_DIR / "mmrm_closed_scenarios.csv"
SIGNAL_HISTORY_DISPLAY_COLUMNS = [
    "1차신호일",
    "2차신호일",
    "3차판정일",
    "결과",
    "3개월후 수익률",
    "6개월후 수익률",
    "9개월후 수익률",
    "12개월후 수익률",
]
SCAN_FAILURE_COLUMNS = ["순위", "티커", "회사명", "시가총액", "오류"]
SCAN_EVENT_DISPLAY_COLUMNS = [
    "티커",
    "회사명",
    "섹터",
    "단계",
    "신호일",
    "종목 3개월 승률",
    "섹터 3개월 승률",
    "결과",
    "신호구분",
    "데이터기준일",
]
ACTIVE_SCENARIO_DISPLAY_COLUMNS = [
    "티커",
    "회사명",
    "섹터",
    "현재상태",
    "1차신호일",
    "2차신호일",
    "종목 3개월 승률",
    "섹터 3개월 승률",
    "데이터기준일",
    "데이터상태",
]
CLOSED_RESULT_DISPLAY_COLUMNS = [
    "티커",
    "회사명",
    "섹터",
    "결과",
    "1차신호일",
    "2차신호일",
    "3차판정일",
    "종료일",
]
CLOSED_SCENARIO_DISPLAY_COLUMNS = [
    "현재 시총순위",
    "티커",
    "회사명",
    "섹터",
    *SIGNAL_HISTORY_DISPLAY_COLUMNS,
]
FIELD_DISPLAY_COLUMNS = [
    "분야",
    "종목 수",
    "종료 사이클",
    "매수 건수",
    "매수 도달률",
    "분석 표본",
    "승률",
    "평균 손익률",
    "중앙값",
]
RANKING_DISPLAY_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "매수 건수",
    "승률",
    "평균 손익률",
    "중앙값",
    "최고",
    "최저",
    "매수 도달률",
    "종합점수",
]


class BuyPointApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.ui_font_family = configure_ui_fonts(self)
        self.title("MMRM 3단계 시나리오 추적 스캐너")
        self.minsize(1800, 680)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.theme_mode = load_theme(UI_SETTINGS_PATH)
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self._apply_theme()

        self.ticker_var = tk.StringVar()
        self.status_var = tk.StringVar(value="티커를 입력해 주세요.")
        self.top100_status_var = tk.StringVar(value="목록을 불러오려면 버튼을 눌러 주세요.")
        self.scan_status_var = tk.StringVar(value="3단계 통합 스캔을 실행하려면 버튼을 눌러 주세요.")
        self.ticker_profile_var = tk.StringVar(value="분야: 미조회")
        self.ticker_cycle_summary_var = tk.StringVar(value="종료 사이클과 매수 도달률을 계산하려면 종목을 검색해 주세요.")
        self.ticker_return_summary_var = tk.StringVar(value="3·6·9·12개월 성과가 여기에 표시됩니다.")
        self.field_level_var = tk.StringVar(value="섹터")
        self.field_horizon_var = tk.StringVar(value="3개월")
        self.ranking_sort_var = tk.StringVar(value="종합점수")
        self.field_status_var = tk.StringVar(value="통합 스캔 후 분야별 성과를 확인할 수 있습니다.")
        self.top100_companies: list[MarketCapCompany] = []
        self.latest_scan_events = pd.DataFrame(columns=SCAN_EVENT_COLUMNS)
        self.latest_active_scenarios = prioritize_active_scenarios(
            load_active_scenarios()
        )
        self.latest_closed_results = pd.DataFrame(columns=CLOSED_RESULT_COLUMNS)
        self.latest_closed_scenarios = load_closed_scenarios()
        self.latest_scan_failures = pd.DataFrame(columns=SCAN_FAILURE_COLUMNS)
        self.latest_scan_date: pd.Timestamp | None = None
        self.latest_classifications = pd.DataFrame()
        self.latest_cycles_by_ticker: dict[str, pd.DataFrame] = {}
        self.latest_analysis_companies: list[MarketCapCompany] = []
        self.latest_sector_performance = pd.DataFrame()
        self.latest_industry_performance = pd.DataFrame()
        self.latest_field_rankings = pd.DataFrame()
        self.selected_field: str | None = None
        self.current_ticker: str | None = None
        self.current_company = ""
        self.current_chart_data = pd.DataFrame()
        self.current_signal_cycles = pd.DataFrame()
        self.chart_window: ChartPreviewWindow | None = None
        self._syncing_chart_history_selection = False
        self.open_chart_after_search = False
        self.pending_chart_first_signal_date: pd.Timestamp | None = None

        self._build_layout()
        active_display = scanner_table_for_display(
            self.latest_active_scenarios,
            ACTIVE_SCENARIO_DISPLAY_COLUMNS,
        )
        populate_table(self.active_tree, active_display)
        self._apply_active_scenario_tags(active_display)
        populate_table(self.closed_scenario_tree, self.latest_closed_scenarios)
        self._apply_history_tags(
            self.latest_closed_scenarios,
            tree=self.closed_scenario_tree,
        )

    def _apply_theme(self) -> None:
        palette = theme_palette(self.theme_mode)
        self.configure(background=palette["window"])

        self.style.configure("TFrame", background=palette["window"])
        self.style.configure(
            "TLabel",
            background=palette["window"],
            foreground=palette["text"],
        )
        self.style.configure(
            "ScanStatus.TLabel",
            background=palette["window"],
            foreground=palette["text"],
        )
        self.style.configure(
            "ScanAlert.TLabel",
            background=palette["signal_third_bg"],
            foreground=palette["signal_third_text"],
            font=(self.ui_font_family, 9, "bold"),
            padding=(6, 5),
        )
        self.style.configure(
            "TLabelframe",
            background=palette["window"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
        )
        self.style.configure(
            "TLabelframe.Label",
            background=palette["window"],
            foreground=palette["text"],
        )
        self.style.configure(
            "TButton",
            background=palette["button"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            focuscolor=palette["selected"],
            padding=(7, 4),
        )
        self.style.map(
            "TButton",
            background=[
                ("pressed", palette["selected"]),
                ("active", palette["button_active"]),
            ],
            foreground=[("pressed", palette["selected_text"])],
        )
        self.style.configure(
            "TEntry",
            fieldbackground=palette["field"],
            foreground=palette["text"],
            insertcolor=palette["text"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=palette["field"],
            background=palette["button"],
            foreground=palette["text"],
            arrowcolor=palette["text"],
            bordercolor=palette["border"],
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["field"])],
            foreground=[("readonly", palette["text"])],
        )
        self.style.configure(
            "Treeview",
            background=palette["field"],
            fieldbackground=palette["field"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            lightcolor=palette["border"],
            darkcolor=palette["border"],
        )
        self.style.map(
            "Treeview",
            background=[("selected", palette["selected"])],
            foreground=[("selected", palette["selected_text"])],
        )
        self.style.configure(
            "Treeview.Heading",
            background=palette["panel"],
            foreground=palette["text"],
            bordercolor=palette["border"],
            relief="flat",
        )
        self.style.map(
            "Treeview.Heading",
            background=[("active", palette["button_active"])],
        )
        self.style.configure(
            "TNotebook",
            background=palette["window"],
            bordercolor=palette["border"],
        )
        self.style.configure(
            "TNotebook.Tab",
            background=palette["button"],
            foreground=palette["text"],
            padding=(9, 4),
        )
        self.style.map(
            "TNotebook.Tab",
            background=[
                ("selected", palette["field"]),
                ("active", palette["button_active"]),
            ],
        )
        for scrollbar_style in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            self.style.configure(
                scrollbar_style,
                background=palette["scroll_thumb"],
                troughcolor=palette["field"],
                bordercolor=palette["field"],
                lightcolor=palette["scroll_thumb"],
                darkcolor=palette["scroll_thumb"],
                arrowcolor=palette["scroll_arrow"],
                gripcount=0,
                arrowsize=8,
                borderwidth=0,
                relief="flat",
            )
            self.style.map(
                scrollbar_style,
                background=[
                    ("pressed", palette["scroll_thumb_active"]),
                    ("active", palette["scroll_thumb_active"]),
                ],
                lightcolor=[
                    ("pressed", palette["scroll_thumb_active"]),
                    ("active", palette["scroll_thumb_active"]),
                ],
                darkcolor=[
                    ("pressed", palette["scroll_thumb_active"]),
                    ("active", palette["scroll_thumb_active"]),
                ],
            )

        self.option_add("*TCombobox*Listbox.background", palette["field"])
        self.option_add("*TCombobox*Listbox.foreground", palette["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", palette["selected"])
        self.option_add("*TCombobox*Listbox.selectForeground", palette["selected_text"])

        if hasattr(self, "theme_button"):
            self.theme_button.configure(text=self._theme_button_text())
        for tree_name in ("scan_tree", "active_tree"):
            tree = getattr(self, tree_name, None)
            if tree is not None:
                self._configure_signal_tree_tags(tree)
        for tree_name in ("buy_tree", "closed_scenario_tree"):
            tree = getattr(self, tree_name, None)
            if tree is not None:
                self._configure_history_tree_tags(tree)
        chart_window = getattr(self, "chart_window", None)
        if chart_window is not None and chart_window.winfo_exists():
            chart_window.set_theme(self.theme_mode)

    def _theme_button_text(self) -> str:
        return "라이트 모드" if self.theme_mode == "dark" else "다크 모드"

    def toggle_theme(self) -> None:
        self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        self._apply_theme()
        save_theme(UI_SETTINGS_PATH, self.theme_mode)

    def _build_layout(self) -> None:
        left_panel_width = 490
        history_panel_width = _table_required_width(SIGNAL_HISTORY_DISPLAY_COLUMNS)
        scanner_panel_width = max(
            _table_required_width(columns)
            for columns in (
                SCAN_EVENT_DISPLAY_COLUMNS,
                ACTIVE_SCENARIO_DISPLAY_COLUMNS,
                CLOSED_RESULT_DISPLAY_COLUMNS,
                CLOSED_SCENARIO_DISPLAY_COLUMNS,
                FIELD_DISPLAY_COLUMNS,
                RANKING_DISPLAY_COLUMNS,
                SCAN_FAILURE_COLUMNS,
            )
        )
        initial_width = (
            left_panel_width
            + history_panel_width
            + scanner_panel_width
            + 48
        )
        self.geometry(f"{initial_width}x820")

        main_frame = ttk.Frame(self, padding=14)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, minsize=left_panel_width)
        main_frame.columnconfigure(1, weight=2, minsize=history_panel_width)
        main_frame.columnconfigure(2, weight=3, minsize=scanner_panel_width)
        main_frame.rowconfigure(0, weight=1)

        left_panel = ttk.LabelFrame(main_frame, text="미국 시총 Top 100", padding=6)
        left_panel.configure(width=left_panel_width)
        left_panel.grid_propagate(False)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)

        self.top100_button = ttk.Button(
            left_panel,
            text="Top 100 불러오기",
            command=self.load_top100,
        )
        self.top100_button.grid(row=0, column=0, sticky="ew")

        top100_status = ttk.Label(
            left_panel,
            textvariable=self.top100_status_var,
            wraplength=330,
            padding=(0, 6, 0, 6),
        )
        top100_status.grid(row=1, column=0, sticky="ew")

        self.top100_tree = self._create_top100_table(left_panel)
        self.top100_tree.bind("<<TreeviewSelect>>", self._on_top100_select)
        self.top100_tree.bind(
            "<Double-1>",
            lambda event: self._on_ticker_double_click(event, self.top100_tree, "ticker"),
        )

        self.center_panel = ttk.Frame(main_frame)
        self.center_panel.configure(width=history_panel_width)
        self.center_panel.grid_propagate(False)
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        center_panel = self.center_panel
        center_panel.rowconfigure(3, weight=1)
        center_panel.columnconfigure(0, weight=1)

        search_frame = ttk.Frame(center_panel)
        search_frame.grid(row=0, column=0, sticky="ew")
        search_frame.columnconfigure(0, weight=1)

        self.search_entry = ttk.Entry(
            search_frame,
            textvariable=self.ticker_var,
            font=(self.ui_font_family, 16),
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", ipady=6)
        self.search_entry.bind("<Return>", lambda _event: self.run_search())
        self.search_entry.focus_set()

        self.search_button = ttk.Button(
            search_frame,
            text="검색",
            command=self.run_search,
        )
        self.search_button.grid(row=0, column=1, padx=(8, 0), ipady=4)

        self.theme_button = ttk.Button(
            search_frame,
            text=self._theme_button_text(),
            command=self.toggle_theme,
        )
        self.theme_button.grid(row=0, column=2, padx=(8, 0), ipady=4)

        status_label = ttk.Label(
            center_panel,
            textvariable=self.status_var,
            wraplength=history_panel_width - 30,
            padding=(0, 8, 0, 8),
        )
        status_label.grid(row=1, column=0, sticky="ew")

        summary_frame = ttk.LabelFrame(center_panel, text="선택 종목 시나리오 성과", padding=5)
        summary_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(
            summary_frame,
            textvariable=self.ticker_profile_var,
            wraplength=history_panel_width - 40,
        ).pack(anchor="w")
        ttk.Label(
            summary_frame,
            textvariable=self.ticker_cycle_summary_var,
            wraplength=history_panel_width - 40,
        ).pack(anchor="w")
        ttk.Label(
            summary_frame,
            textvariable=self.ticker_return_summary_var,
            wraplength=history_panel_width - 40,
        ).pack(anchor="w")

        table_frame = ttk.LabelFrame(
            center_panel,
            text="3단계 신호 과거 기록 (행 더블클릭: 차트 미리보기)",
            padding=4,
        )
        table_frame.grid(row=3, column=0, sticky="nsew")
        self.buy_tree = self._create_table(table_frame)
        self._configure_history_tree_tags()
        populate_table(self.buy_tree, pd.DataFrame(columns=SIGNAL_HISTORY_DISPLAY_COLUMNS))
        self.buy_tree.bind("<<TreeviewSelect>>", self._on_history_select)
        self.buy_tree.bind("<Double-1>", self._on_history_double_click)

        self.scanner_panel = ttk.LabelFrame(
            main_frame,
            text="MMRM 시나리오 추적 스캐너",
            padding=6,
        )
        self.scanner_panel.configure(width=scanner_panel_width)
        self.scanner_panel.grid_propagate(False)
        self.scanner_panel.grid(row=0, column=2, sticky="nsew")
        scanner_panel = self.scanner_panel
        scanner_panel.rowconfigure(2, weight=1)
        scanner_panel.columnconfigure(0, weight=1)

        scan_button_frame = ttk.Frame(scanner_panel)
        scan_button_frame.grid(row=0, column=0, sticky="ew")
        scan_button_frame.columnconfigure(0, weight=1)
        scan_button_frame.columnconfigure(1, weight=1)

        self.scan_button = ttk.Button(
            scan_button_frame,
            text="3단계 통합 스캔",
            command=self.run_top100_scan,
        )
        self.scan_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.scan_save_button = ttk.Button(
            scan_button_frame,
            text="스캔 저장하기",
            command=self.save_latest_scan,
            state="disabled",
        )
        self.scan_save_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        self.scan_status_label = ttk.Label(
            scanner_panel,
            textvariable=self.scan_status_var,
            wraplength=scanner_panel_width - 40,
            padding=(0, 6, 0, 6),
            style="ScanStatus.TLabel",
        )
        self.scan_status_label.grid(row=1, column=0, sticky="ew")

        self.scan_notebook = ttk.Notebook(scanner_panel)
        self.scan_notebook.grid(row=2, column=0, sticky="nsew")

        event_tab = ttk.Frame(self.scan_notebook)
        active_tab = ttk.Frame(self.scan_notebook)
        closed_tab = ttk.Frame(self.scan_notebook)
        closed_scenario_tab = ttk.Frame(self.scan_notebook)
        field_tab = ttk.Frame(self.scan_notebook)
        failure_tab = ttk.Frame(self.scan_notebook)
        self.scan_notebook.add(event_tab, text="이번 스캔 신호")
        self.scan_notebook.add(active_tab, text="활성 시나리오")
        self.scan_notebook.add(closed_tab, text="이번 스캔 종료")
        self.scan_notebook.add(closed_scenario_tab, text="종료 시나리오")
        self.scan_notebook.add(field_tab, text="분야별 성과")
        self.scan_notebook.add(failure_tab, text="오류")

        self.scan_tree = self._create_table(event_tab)
        self.active_tree = self._create_table(active_tab)
        self._configure_signal_tree_tags(self.scan_tree)
        self._configure_signal_tree_tags(self.active_tree)
        self.closed_tree = self._create_table(closed_tab)
        self.closed_scenario_tree = self._create_table(closed_scenario_tab)
        self._configure_history_tree_tags(self.closed_scenario_tree)
        self._build_field_performance_tab(field_tab)
        self.failure_tree = self._create_table(failure_tab)

        populate_table(self.scan_tree, pd.DataFrame(columns=SCAN_EVENT_DISPLAY_COLUMNS))
        populate_table(self.active_tree, pd.DataFrame(columns=ACTIVE_SCENARIO_DISPLAY_COLUMNS))
        populate_table(self.closed_tree, pd.DataFrame(columns=CLOSED_RESULT_DISPLAY_COLUMNS))
        populate_table(
            self.closed_scenario_tree,
            pd.DataFrame(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS),
        )
        populate_table(self.field_tree, pd.DataFrame(columns=FIELD_DISPLAY_COLUMNS))
        populate_table(self.ranking_tree, pd.DataFrame(columns=RANKING_DISPLAY_COLUMNS))
        populate_table(self.failure_tree, pd.DataFrame(columns=SCAN_FAILURE_COLUMNS))

        self.scan_tree.bind(
            "<<TreeviewSelect>>",
            lambda event: self._on_scan_row_select(event, self.scan_tree),
        )
        self.active_tree.bind(
            "<<TreeviewSelect>>",
            lambda event: self._on_scan_row_select(event, self.active_tree),
        )
        self.closed_tree.bind(
            "<<TreeviewSelect>>",
            lambda event: self._on_scan_row_select(event, self.closed_tree),
        )
        self.closed_scenario_tree.bind(
            "<<TreeviewSelect>>",
            self._on_closed_scenario_select,
        )
        self.closed_scenario_tree.bind(
            "<Double-1>",
            self._on_closed_scenario_double_click,
        )
        self.ranking_tree.bind(
            "<<TreeviewSelect>>",
            lambda event: self._on_scan_row_select(event, self.ranking_tree),
        )
        for tree in (self.scan_tree, self.active_tree, self.closed_tree, self.ranking_tree):
            tree.bind(
                "<Double-1>",
                lambda event, source=tree: self._on_ticker_double_click(
                    event,
                    source,
                    "티커",
                ),
            )

    def _configure_signal_tree_tags(self, tree: ttk.Treeview) -> None:
        palette = theme_palette(self.theme_mode)
        tree.tag_configure(
            "signal_first",
            background=palette["signal_first_bg"],
            foreground=palette["signal_first_text"],
        )
        tree.tag_configure(
            "signal_second",
            background=palette["signal_second_bg"],
            foreground=palette["signal_second_text"],
        )
        tree.tag_configure(
            "signal_third",
            background=palette["signal_third_bg"],
            foreground=palette["signal_third_text"],
            font=(self.ui_font_family, 9, "bold"),
        )

    def _apply_scan_event_tags(self, data: pd.DataFrame) -> None:
        for item, (_, row) in zip(self.scan_tree.get_children(), data.iterrows()):
            tag = scan_event_tag(row.get("단계"), row.get("결과"))
            self.scan_tree.item(item, tags=(tag,) if tag else ())

    def _apply_active_scenario_tags(self, data: pd.DataFrame) -> None:
        for item, (_, row) in zip(self.active_tree.get_children(), data.iterrows()):
            tag = active_scenario_tag(row.get("현재상태"))
            self.active_tree.item(item, tags=(tag,) if tag else ())

    def _configure_history_tree_tags(self, tree: ttk.Treeview | None = None) -> None:
        tree = tree or self.buy_tree
        palette = theme_palette(self.theme_mode)
        tag_colors = {
            "history_discard": "history_discard_bg",
            "history_failure": "history_failure_bg",
            "history_success_pending": "history_success_pending_bg",
            "history_success_low": "history_success_low_bg",
            "history_success_medium": "history_success_medium_bg",
            "history_success_high": "history_success_high_bg",
            "history_loss": "history_loss_bg",
            "history_flat": "history_flat_bg",
        }
        for tag, color_key in tag_colors.items():
            options: dict[str, object] = {"background": palette[color_key]}
            if tag == "history_success_high":
                options["font"] = (self.ui_font_family, 9, "bold")
            tree.tag_configure(tag, **options)

    def _apply_history_tags(
        self,
        data: pd.DataFrame,
        tree: ttk.Treeview | None = None,
    ) -> None:
        tree = tree or self.buy_tree
        for item, (_, row) in zip(tree.get_children(), data.iterrows()):
            tag = history_cycle_tag(
                row.get("결과"),
                row.get("3개월후 수익률"),
            )
            tree.item(item, tags=(tag,) if tag else ())

    def _create_table(self, parent: tk.Widget) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, show="headings")
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _build_field_performance_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        controls = ttk.Frame(parent, padding=(4, 4, 4, 0))
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="구분").pack(side="left")
        level_combo = ttk.Combobox(
            controls,
            textvariable=self.field_level_var,
            values=("섹터", "산업"),
            width=8,
            state="readonly",
        )
        level_combo.pack(side="left", padx=(4, 14))
        ttk.Label(controls, text="분석 기간").pack(side="left")
        horizon_combo = ttk.Combobox(
            controls,
            textvariable=self.field_horizon_var,
            values=("3개월", "6개월", "9개월", "12개월"),
            width=8,
            state="readonly",
        )
        horizon_combo.pack(side="left", padx=(4, 14))
        ttk.Label(controls, text="종목 정렬").pack(side="left")
        sort_combo = ttk.Combobox(
            controls,
            textvariable=self.ranking_sort_var,
            values=("종합점수", "승률", "평균 손익률", "매수 도달률"),
            width=14,
            state="readonly",
        )
        sort_combo.pack(side="left", padx=(4, 0))
        for combo in (level_combo, horizon_combo, sort_combo):
            combo.bind("<<ComboboxSelected>>", self._on_field_control_change)

        ttk.Label(
            parent,
            textvariable=self.field_status_var,
            padding=(4, 5, 4, 5),
        ).grid(row=1, column=0, sticky="ew")

        panes = ttk.Panedwindow(parent, orient="vertical")
        panes.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 4))
        field_frame = ttk.LabelFrame(panes, text="분야별 종합 성과", padding=4)
        ranking_frame = ttk.LabelFrame(panes, text="선택 분야 종목 순위", padding=4)
        panes.add(field_frame, weight=1)
        panes.add(ranking_frame, weight=1)
        self.field_tree = self._create_table(field_frame)
        self.ranking_tree = self._create_table(ranking_frame)
        self.field_tree.bind("<<TreeviewSelect>>", self._on_field_select)

    def _create_top100_table(self, parent: tk.Widget) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.grid(row=2, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = ["rank", "ticker", "company", "market_cap"]
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("rank", text="순위")
        tree.heading("ticker", text="티커")
        tree.heading("company", text="회사명")
        tree.heading("market_cap", text="시가총액")
        tree.column("rank", width=52, minwidth=45, anchor="center", stretch=False)
        tree.column("ticker", width=76, minwidth=60, anchor="center", stretch=False)
        tree.column("company", width=230, minwidth=180, stretch=False)
        tree.column("market_cap", width=100, minwidth=95, anchor="e", stretch=False)

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        return tree

    def load_top100(self) -> None:
        self.top100_button.configure(state="disabled")
        self.top100_status_var.set("미국 시총 Top 100 목록을 불러오는 중입니다...")
        self.top100_tree.delete(*self.top100_tree.get_children())

        worker = threading.Thread(target=self._top100_worker, daemon=True)
        worker.start()

    def _top100_worker(self) -> None:
        try:
            companies = fetch_us_top_market_cap(limit=100)
        except Exception as exc:
            self.after(0, self._show_top100_error, exc)
            return

        self.after(0, self._show_top100_result, companies)

    def _show_top100_result(self, companies: list[MarketCapCompany]) -> None:
        self._populate_top100_table(companies)
        self.top100_status_var.set(f"{len(companies)}개 종목을 불러왔습니다. 행을 클릭하면 바로 검색합니다.")
        self.top100_button.configure(state="normal")

    def _populate_top100_table(self, companies: list[MarketCapCompany]) -> None:
        self.top100_companies = list(companies)
        self.top100_tree.delete(*self.top100_tree.get_children())
        for company in companies:
            self.top100_tree.insert(
                "",
                "end",
                values=(company.rank, company.ticker, company.company, company.market_cap),
            )

    def _show_top100_error(self, exc: Exception) -> None:
        if isinstance(exc, MarketCapLoadError):
            message = str(exc)
        else:
            message = f"미국 시총 Top 100 목록을 불러오지 못했습니다: {exc}"

        self.top100_tree.delete(*self.top100_tree.get_children())
        self.top100_companies = []
        self.top100_status_var.set("목록을 불러오지 못했습니다.")
        self.top100_button.configure(state="normal")
        messagebox.showerror("Top 100 조회 실패", message)

    def _on_top100_select(self, _event) -> None:
        selected = self.top100_tree.selection()
        if not selected:
            return
        ticker = self.top100_tree.set(selected[0], "ticker")
        if not ticker:
            return
        self.pending_chart_first_signal_date = None
        self.ticker_var.set(ticker)
        self.run_search()

    def run_top100_scan(self) -> None:
        if str(self.scan_button.cget("state")) == "disabled":
            return

        self.scan_button.configure(state="disabled")
        self.scan_save_button.configure(state="disabled")
        self.top100_button.configure(state="disabled")
        self.scan_status_label.configure(style="ScanStatus.TLabel")
        self.scan_status_var.set("Top 100과 활성 시나리오를 통합 스캔하는 중입니다...")
        self.scan_tree.delete(*self.scan_tree.get_children())
        self.closed_tree.delete(*self.closed_tree.get_children())
        self.field_tree.delete(*self.field_tree.get_children())
        self.ranking_tree.delete(*self.ranking_tree.get_children())
        self.failure_tree.delete(*self.failure_tree.get_children())
        self.latest_scan_events = pd.DataFrame(columns=SCAN_EVENT_COLUMNS)
        self.latest_closed_results = pd.DataFrame(columns=CLOSED_RESULT_COLUMNS)
        self.latest_scan_failures = pd.DataFrame(columns=SCAN_FAILURE_COLUMNS)
        self.latest_scan_date = None
        self.latest_classifications = pd.DataFrame()
        self.latest_cycles_by_ticker = {}
        self.latest_analysis_companies = []
        self.selected_field = None
        self.field_status_var.set("가격 데이터와 시나리오 성과를 계산하는 중입니다...")

        companies = list(self.top100_companies)
        active_scenarios = self.latest_active_scenarios.copy()
        worker = threading.Thread(
            target=self._top100_scan_worker,
            args=(companies, active_scenarios),
            daemon=True,
        )
        worker.start()

    def _top100_scan_worker(
        self,
        companies: list[MarketCapCompany],
        previous_active: pd.DataFrame,
    ) -> None:
        try:
            if not companies:
                self.after(0, self.scan_status_var.set, "Top 100 목록을 먼저 불러오는 중입니다...")
                companies = fetch_us_top_market_cap(limit=100)
                self.after(0, self._show_top100_loaded_by_scan, companies)

            scan_date = pd.Timestamp.today().normalize()
            scan_universe = merge_scan_universe(companies, previous_active)
            events, active_rows, closed_results, failures, cycles_by_ticker = self._scan_companies(
                scan_universe,
                scan_date,
                previous_active,
                progress_label="스캔 중",
            )

            if failures:
                time.sleep(2)
                retry_companies = [failure["company"] for failure in failures]
                (
                    retry_events,
                    retry_active,
                    retry_closed,
                    retry_failures,
                    retry_cycles,
                ) = self._scan_companies(
                    retry_companies,
                    scan_date,
                    previous_active,
                    progress_label="실패 종목 재시도 중",
                )
                events.extend(retry_events)
                active_rows.extend(retry_active)
                closed_results.extend(retry_closed)
                cycles_by_ticker.update(retry_cycles)
                failures = retry_failures

            failed_tickers = {
                failure["company"].ticker
                for failure in failures
            }
            active_rows.extend(
                preserve_failed_active_rows(previous_active, failed_tickers)
            )

            events_df = _sorted_frame(
                events,
                SCAN_EVENT_COLUMNS,
                by=["신호일", "순위"],
                ascending=[False, True],
            )
            events_df = prioritize_scan_events(events_df)
            active_df = _sorted_frame(
                active_rows,
                ACTIVE_SCENARIO_COLUMNS,
                by=["순위", "티커"],
                ascending=[True, True],
            ).drop_duplicates(subset=["티커"], keep="last")
            active_df = prioritize_active_scenarios(active_df)
            closed_df = _sorted_frame(
                closed_results,
                CLOSED_RESULT_COLUMNS,
                by=["종료일", "순위"],
                ascending=[False, True],
            )
            failures_df = pd.DataFrame(
                [_failure_row(failure["company"], failure["error"]) for failure in failures],
                columns=SCAN_FAILURE_COLUMNS,
            )

            self.after(0, self.scan_status_var.set, "종목별 섹터·산업 정보를 확인하는 중입니다...")
            classifications = load_sector_classifications(
                (company.ticker for company in scan_universe),
                progress_callback=lambda index, total, ticker: self.after(
                    0,
                    self.scan_status_var.set,
                    f"분야 정보 확인 중... {index}/{total} {ticker}",
                ),
            )
            events_df = add_sector_column(events_df, classifications)
            active_df = add_sector_column(active_df, classifications)
            closed_df = add_sector_column(closed_df, classifications)
            closed_scenarios_df = build_closed_scenario_history(
                companies,
                cycles_by_ticker,
                classifications,
                previous=self.latest_closed_scenarios,
                failed_tickers=failed_tickers,
            )

            sector_output, industry_output, ranking_output = build_all_field_outputs(
                companies,
                cycles_by_ticker,
                classifications,
            )
            events_df = add_scan_performance_columns(
                events_df,
                companies,
                cycles_by_ticker,
                classifications,
                sector_output,
            )
            active_df = add_scan_performance_columns(
                active_df,
                companies,
                cycles_by_ticker,
                classifications,
                sector_output,
            )
            save_analytics_outputs(sector_output, industry_output, ranking_output)
            save_active_scenarios(active_df)
            save_closed_scenarios(closed_scenarios_df)
        except Exception as exc:
            self.after(0, self._show_scan_error, exc)
            return

        self.after(
            0,
            self._show_scan_result,
            events_df,
            active_df,
            closed_df,
            closed_scenarios_df,
            failures_df,
            scan_date,
            companies,
            cycles_by_ticker,
            classifications,
            sector_output,
            industry_output,
            ranking_output,
        )

    def _show_top100_loaded_by_scan(self, companies: list[MarketCapCompany]) -> None:
        self._populate_top100_table(companies)
        self.top100_status_var.set(f"{len(companies)}개 종목을 불러왔습니다. 이 목록을 기준으로 스캔합니다.")

    def _scan_companies(
        self,
        companies: list[MarketCapCompany],
        scan_date: pd.Timestamp,
        previous_active: pd.DataFrame,
        progress_label: str,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        dict[str, pd.DataFrame],
    ]:
        events: list[dict[str, object]] = []
        active_rows: list[dict[str, object]] = []
        closed_results: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        cycles_by_ticker: dict[str, pd.DataFrame] = {}
        total = len(companies)
        previous_by_ticker = {
            str(row["티커"]).upper(): row
            for _, row in previous_active.iterrows()
        }

        for index, company in enumerate(companies, start=1):
            self.after(
                0,
                self.scan_status_var.set,
                f"{progress_label}... {index}/{total} {company.ticker}",
            )
            try:
                raw_data = load_weekly_data(
                    company.ticker,
                    include_current_week=True,
                    force_refresh=True,
                )
                calculated = calculate_indicators(raw_data)
                cycles, full_table = scan_signal_cycles(calculated)
                cycles_by_ticker[company.ticker.upper()] = cycles.copy()
                ticker_events, active_row, ticker_closed = summarize_ticker_cycles(
                    company,
                    cycles,
                    full_table,
                    scan_date,
                    previous_active=previous_by_ticker.get(company.ticker.upper()),
                )
                events.extend(ticker_events)
                closed_results.extend(ticker_closed)
                if active_row is not None:
                    active_rows.append(active_row)
            except Exception as exc:
                failures.append({"company": company, "error": str(exc)})

        return events, active_rows, closed_results, failures, cycles_by_ticker

    def _show_scan_result(
        self,
        events: pd.DataFrame,
        active_scenarios: pd.DataFrame,
        closed_results: pd.DataFrame,
        closed_scenarios: pd.DataFrame,
        failures: pd.DataFrame,
        scan_date: pd.Timestamp,
        analysis_companies: list[MarketCapCompany],
        cycles_by_ticker: dict[str, pd.DataFrame],
        classifications: pd.DataFrame,
        sector_output: pd.DataFrame,
        industry_output: pd.DataFrame,
        ranking_output: pd.DataFrame,
    ) -> None:
        self.latest_scan_events = events.copy()
        self.latest_active_scenarios = active_scenarios.copy()
        self.latest_closed_results = closed_results.copy()
        self.latest_closed_scenarios = closed_scenarios.copy()
        self.latest_scan_failures = failures.copy()
        self.latest_scan_date = scan_date
        self.latest_analysis_companies = list(analysis_companies)
        self.latest_cycles_by_ticker = {
            ticker: cycles.copy()
            for ticker, cycles in cycles_by_ticker.items()
        }
        self.latest_classifications = classifications.copy()
        self.latest_sector_performance = sector_output.copy()
        self.latest_industry_performance = industry_output.copy()
        self.latest_field_rankings = ranking_output.copy()
        scan_display = scanner_table_for_display(events, SCAN_EVENT_DISPLAY_COLUMNS)
        populate_table(self.scan_tree, scan_display)
        self._apply_scan_event_tags(scan_display)
        active_display = scanner_table_for_display(
            active_scenarios,
            ACTIVE_SCENARIO_DISPLAY_COLUMNS,
        )
        populate_table(self.active_tree, active_display)
        self._apply_active_scenario_tags(active_display)
        populate_table(
            self.closed_tree,
            scanner_table_for_display(closed_results, CLOSED_RESULT_DISPLAY_COLUMNS),
        )
        populate_table(self.closed_scenario_tree, closed_scenarios)
        self._apply_history_tags(closed_scenarios, tree=self.closed_scenario_tree)
        populate_table(self.failure_tree, failures)
        self._refresh_field_analytics(reset_selection=True)

        failed_tickers = ", ".join(failures["티커"].tolist()[:8]) if not failures.empty else ""
        failed_suffix = f": {failed_tickers}" if failed_tickers else ""
        if len(failures) > 8:
            failed_suffix += "..."

        first_count = int((events["단계"] == "1차 신호").sum()) if not events.empty else 0
        second_count = int((events["단계"] == "2차 신호").sum()) if not events.empty else 0
        second_rejection_count = int(
            (events["단계"] == "2차 폐기").sum()
        ) if not events.empty else 0
        third_count = int(
            ((events["단계"] == "3차 신호") & (events["결과"] == "매수 성공")).sum()
        ) if not events.empty else 0
        failed_signal_count = int(
            ((events["단계"] == "3차 신호") & (events["결과"] == "실패")).sum()
        ) if not events.empty else 0
        self.scan_status_var.set(
            f"스캔 완료 | 즉시 확인: 3차 신호 {third_count}개 / "
            f"출발 준비: 2차 신호 {second_count}개 / 관심 편입: 1차 신호 {first_count}개 / "
            f"2차 폐기 {second_rejection_count}개 / "
            f"신호 실패 {failed_signal_count}개 / "
            f"계속 관찰 {len(active_scenarios)}개 / 데이터 오류 {len(failures)}개{failed_suffix}. "
            f"필요하면 스캔 저장하기를 눌러 CSV로 저장하세요."
        )
        self.scan_status_label.configure(
            style="ScanAlert.TLabel" if third_count > 0 else "ScanStatus.TLabel"
        )
        if third_count > 0:
            self.scan_notebook.select(0)
            third_items = [
                item
                for item in self.scan_tree.get_children()
                if self.scan_tree.set(item, "단계") == "3차 신호"
                and self.scan_tree.set(item, "결과") == "매수 성공"
            ]
            if third_items:
                self.scan_tree.focus(third_items[0])
                self.scan_tree.see(third_items[0])
        self.scan_button.configure(state="normal")
        self.scan_save_button.configure(state="normal")
        self.top100_button.configure(state="normal")

    def _show_scan_error(self, exc: Exception) -> None:
        if isinstance(exc, MarketCapLoadError):
            message = str(exc)
        else:
            message = f"Top 100 스캔 중 오류가 발생했습니다: {exc}"

        self.scan_status_var.set(message)
        self.scan_status_label.configure(style="ScanStatus.TLabel")
        self.scan_button.configure(state="normal")
        self.scan_save_button.configure(state="disabled")
        self.top100_button.configure(state="normal")
        messagebox.showerror("Top 100 스캔 실패", message)

    def save_latest_scan(self) -> None:
        if self.latest_scan_date is None:
            messagebox.showinfo("저장할 스캔 없음", "먼저 3단계 통합 스캔을 실행해 주세요.")
            return

        saved_paths = save_tracker_scan_outputs(
            self.latest_scan_events,
            self.latest_active_scenarios,
            self.latest_closed_results,
            self.latest_scan_failures,
            self.latest_scan_date,
        )
        saved_paths += save_analytics_outputs(
            self.latest_sector_performance,
            self.latest_industry_performance,
            self.latest_field_rankings,
            output_dir=DOWNLOADS_DIR,
            date_suffix=self.latest_scan_date.strftime("%Y-%m-%d"),
        )
        saved_paths += (
            save_closed_scenarios(
                self.latest_closed_scenarios,
                DOWNLOADS_DIR
                / f"MMRM_closed_scenarios_{self.latest_scan_date:%Y-%m-%d}.csv",
            ),
        )
        self.scan_status_var.set(
            "스캔 결과 저장 완료: " + " / ".join(str(path) for path in saved_paths)
        )

    def _on_scan_row_select(self, _event, tree: ttk.Treeview) -> None:
        selected = tree.selection()
        if not selected:
            return
        ticker = tree.set(selected[0], "티커")
        if not ticker:
            return
        self.pending_chart_first_signal_date = None
        self.ticker_var.set(ticker)
        self.run_search()

    def _on_closed_scenario_select(self, _event=None) -> None:
        selected = self.closed_scenario_tree.selection()
        if not selected:
            return
        item = selected[0]
        ticker = self.closed_scenario_tree.set(item, "티커")
        first_signal_date = self.closed_scenario_tree.set(item, "1차신호일")
        if not ticker or not first_signal_date:
            return
        self.pending_chart_first_signal_date = pd.Timestamp(first_signal_date)
        self.ticker_var.set(ticker)
        if self.current_ticker == ticker.upper() and not self.current_chart_data.empty:
            if self._chart_is_open():
                self._show_chart(
                    self._cycle_for_first_signal_date(
                        self.pending_chart_first_signal_date
                    )
                )
            return
        self.run_search()

    def _on_closed_scenario_double_click(self, event) -> None:
        item = self.closed_scenario_tree.identify_row(event.y)
        if not item:
            return
        self.closed_scenario_tree.selection_set(item)
        ticker = self.closed_scenario_tree.set(item, "티커")
        first_signal_date = self.closed_scenario_tree.set(item, "1차신호일")
        if not ticker or not first_signal_date:
            return

        self.pending_chart_first_signal_date = pd.Timestamp(first_signal_date)
        self.open_chart_after_search = True
        if self.current_ticker == ticker.upper() and not self.current_chart_data.empty:
            cycle = self._cycle_for_first_signal_date(
                self.pending_chart_first_signal_date
            )
            self._show_chart(cycle)
            self.open_chart_after_search = False
            self.pending_chart_first_signal_date = None

    def _on_ticker_double_click(
        self,
        event,
        tree: ttk.Treeview,
        ticker_column: str,
    ) -> None:
        item = tree.identify_row(event.y)
        if not item:
            return
        tree.selection_set(item)
        ticker = tree.set(item, ticker_column)
        if not ticker:
            return

        self.pending_chart_first_signal_date = None
        self.open_chart_after_search = True
        if (
            self.current_ticker == ticker.upper()
            and not self.current_chart_data.empty
        ):
            self._show_chart(self._latest_cycle())
            self.open_chart_after_search = False
            return

        self.ticker_var.set(ticker)
        self.run_search()

    def _on_history_double_click(self, event) -> None:
        item = self.buy_tree.identify_row(event.y)
        if not item:
            return
        self.buy_tree.selection_set(item)
        self._show_selected_history_cycle(open_window=True)

    def _on_history_select(self, _event=None) -> None:
        if self._syncing_chart_history_selection:
            return
        if self._chart_is_open():
            selected = self.buy_tree.selection()
            if selected:
                selected_position = self.buy_tree.index(selected[0])
                chart_position = signal_cycle_position(
                    self.current_signal_cycles,
                    self.chart_window.cycle,
                )
                if selected_position == chart_position:
                    return
            self._show_selected_history_cycle(open_window=False)

    def _show_selected_history_cycle(self, open_window: bool) -> None:
        selected = self.buy_tree.selection()
        if not selected or self.current_signal_cycles.empty:
            return
        position = self.buy_tree.index(selected[0])
        if position >= len(self.current_signal_cycles):
            return
        if open_window or self._chart_is_open():
            self._show_chart(self.current_signal_cycles.iloc[position])

    def _latest_cycle(self) -> pd.Series | None:
        if self.current_signal_cycles.empty:
            return None
        return self.current_signal_cycles.iloc[-1]

    def _cycle_for_first_signal_date(
        self,
        first_signal_date: pd.Timestamp | None,
    ) -> pd.Series | None:
        if first_signal_date is None or self.current_signal_cycles.empty:
            return None
        dates = pd.to_datetime(
            self.current_signal_cycles["FirstSignalDate"],
            errors="coerce",
        ).dt.normalize()
        matches = self.current_signal_cycles.loc[
            dates.eq(pd.Timestamp(first_signal_date).normalize())
        ]
        if matches.empty:
            return None
        return matches.iloc[-1]

    def _show_chart(self, cycle: pd.Series | None) -> None:
        if not self.current_ticker or self.current_chart_data.empty:
            return
        position = signal_cycle_position(self.current_signal_cycles, cycle)
        if not self._chart_is_open():
            self.chart_window = ChartPreviewWindow(
                self,
                on_close=self._on_chart_closed,
                on_navigate=self._navigate_chart_history,
                theme_mode=self.theme_mode,
            )
        try:
            self.chart_window.show_cycle(
                self.current_ticker,
                self.current_chart_data,
                cycle,
                company=self.current_company,
                navigation_index=position,
                navigation_total=len(self.current_signal_cycles),
            )
            if position is not None:
                self._select_history_position(position)
        except ValueError as exc:
            messagebox.showerror("차트 미리보기 오류", str(exc))

    def _navigate_chart_history(self, direction: int) -> None:
        if self.current_signal_cycles.empty or not self._chart_is_open():
            return
        current = signal_cycle_position(
            self.current_signal_cycles,
            self.chart_window.cycle,
        )
        if current is None:
            current = len(self.current_signal_cycles) - 1
        target = min(
            len(self.current_signal_cycles) - 1,
            max(0, current + (-1 if direction < 0 else 1)),
        )
        if target == current:
            return
        self._show_chart(self.current_signal_cycles.iloc[target])

    def _select_history_position(self, position: int) -> None:
        children = self.buy_tree.get_children()
        if position < 0 or position >= len(children):
            return
        item = children[position]
        self._syncing_chart_history_selection = True
        try:
            self.buy_tree.selection_set(item)
            self.buy_tree.focus(item)
            self.buy_tree.see(item)
        finally:
            self._syncing_chart_history_selection = False

    def _chart_is_open(self) -> bool:
        if self.chart_window is None:
            return False
        try:
            return bool(self.chart_window.winfo_exists())
        except tk.TclError:
            return False

    def _on_chart_closed(self) -> None:
        self.chart_window = None

    def _on_field_control_change(self, _event) -> None:
        self._refresh_field_analytics(reset_selection=False)

    def _on_field_select(self, _event) -> None:
        selected = self.field_tree.selection()
        if not selected:
            return
        field = self.field_tree.set(selected[0], "분야")
        if not field:
            return
        self.selected_field = field
        self._refresh_field_ranking()

    def _refresh_field_analytics(self, reset_selection: bool = False) -> None:
        if not self.latest_analysis_companies:
            populate_table(self.field_tree, pd.DataFrame(columns=FIELD_DISPLAY_COLUMNS))
            populate_table(self.ranking_tree, pd.DataFrame(columns=RANKING_DISPLAY_COLUMNS))
            return

        level = self.field_level_var.get()
        horizon = _horizon_months(self.field_horizon_var.get())
        fields = build_field_performance(
            self.latest_analysis_companies,
            self.latest_cycles_by_ticker,
            self.latest_classifications,
            level,
            horizon,
        )
        display = field_performance_for_display(fields)
        populate_table(self.field_tree, display)

        available_fields = display["분야"].tolist() if not display.empty else []
        if reset_selection or self.selected_field not in available_fields:
            self.selected_field = available_fields[0] if available_fields else None
        if self.selected_field is not None:
            for item in self.field_tree.get_children():
                if self.field_tree.set(item, "분야") == self.selected_field:
                    self.field_tree.selection_set(item)
                    self.field_tree.focus(item)
                    break
        self._refresh_field_ranking()

        analyzed = sum(
            ticker in self.latest_cycles_by_ticker
            for ticker in (company.ticker.upper() for company in self.latest_analysis_companies)
        )
        total = len(self.latest_analysis_companies)
        excluded = total - analyzed
        date_text = (
            self.latest_scan_date.strftime("%Y-%m-%d")
            if self.latest_scan_date is not None
            else "미정"
        )
        self.field_status_var.set(
            f"분석 기준일 {date_text} / 현재 Top 100 구성 종목 기준 / "
            f"정상 분석 {analyzed}종목 / 데이터 오류 제외 {excluded}종목 / "
            f"{horizon}개월 성과"
        )

    def _refresh_field_ranking(self) -> None:
        if not self.selected_field or not self.latest_analysis_companies:
            populate_table(self.ranking_tree, pd.DataFrame(columns=RANKING_DISPLAY_COLUMNS))
            return
        ranking = build_stock_ranking(
            self.latest_analysis_companies,
            self.latest_cycles_by_ticker,
            self.latest_classifications,
            self.field_level_var.get(),
            self.selected_field,
            _horizon_months(self.field_horizon_var.get()),
            self.ranking_sort_var.get(),
        )
        populate_table(self.ranking_tree, ranking_for_display(ranking))

    def run_search(self) -> None:
        if str(self.search_button.cget("state")) == "disabled":
            return

        try:
            ticker = normalize_ticker(self.ticker_var.get())
        except ValueError as exc:
            messagebox.showinfo("입력 필요", str(exc))
            return

        self.search_button.configure(state="disabled")
        self.status_var.set(f"{ticker} 주봉 데이터를 불러오는 중입니다...")
        self.ticker_profile_var.set("분야 정보를 확인하는 중입니다...")
        self.ticker_cycle_summary_var.set("시나리오 성과를 계산하는 중입니다...")
        self.ticker_return_summary_var.set("")

        company = self._company_for_ticker(ticker)
        worker = threading.Thread(
            target=self._search_worker,
            args=(ticker, company),
            daemon=True,
        )
        worker.start()

    def _company_for_ticker(self, ticker: str) -> MarketCapCompany:
        for company in self.top100_companies:
            if company.ticker.upper() == ticker.upper():
                return company
        return MarketCapCompany(
            rank=9999,
            ticker=ticker,
            company=ticker,
            market_cap="",
        )

    def _search_worker(self, ticker: str, company: MarketCapCompany) -> None:
        try:
            raw_data = load_weekly_data(
                ticker,
                include_current_week=True,
                force_refresh=True,
            )
            calculated = calculate_indicators(raw_data)
            signal_cycles, full_table = scan_signal_cycles(calculated)
            signal_path, full_path = save_signal_outputs(ticker, signal_cycles, full_table)
            classifications = load_sector_classifications([ticker])
            cycles_by_ticker = {ticker.upper(): signal_cycles}
            performance_by_horizon = {
                horizon: build_ticker_performance(
                    [company],
                    cycles_by_ticker,
                    classifications,
                    horizon,
                ).iloc[0]
                for horizon in (3, 6, 9, 12)
            }
        except Exception as exc:
            self.after(0, self._show_error, ticker, exc)
            return

        self.after(
            0,
            self._show_result,
            ticker,
            signal_cycles,
            full_table,
            signal_path,
            company,
            classifications,
            performance_by_horizon,
        )

    def _show_result(
        self,
        ticker: str,
        signal_cycles: pd.DataFrame,
        full_table: pd.DataFrame,
        signal_path: Path,
        company: MarketCapCompany,
        classifications: pd.DataFrame,
        performance_by_horizon: dict[int, pd.Series],
    ) -> None:
        self.current_ticker = ticker.upper()
        self.current_company = company.company
        self.current_chart_data = full_table.copy()
        self.current_signal_cycles = signal_cycles.reset_index(drop=True).copy()
        history_display = signal_cycles_for_display(signal_cycles)
        populate_table(self.buy_tree, history_display)
        self._apply_history_tags(history_display)

        count = len(signal_cycles)
        self.status_var.set(
            f"{ticker}: 3단계 신호 사이클 {count}개를 찾았습니다. "
            f"저장: {signal_path}"
        )
        classification = (
            classifications.iloc[0]
            if not classifications.empty
            else pd.Series({"섹터": "미분류", "산업": "미분류"})
        )
        self.ticker_profile_var.set(
            f"{ticker}  |  섹터: {classification.get('섹터', '미분류')}  |  "
            f"산업: {classification.get('산업', '미분류')}"
        )
        base = performance_by_horizon[3]
        self.ticker_cycle_summary_var.set(
            f"종료 사이클 {int(base['종료 사이클'])}건  |  "
            f"3차 매수 도달 {int(base['매수 건수'])}건  |  "
            f"매수 도달률 {format_reach_rate(base['매수 도달률'], base['매수 건수'], base['종료 사이클'])}"
        )
        horizon_text = []
        for horizon in (3, 6, 9, 12):
            row = performance_by_horizon[horizon]
            horizon_text.append(
                f"{horizon}개월 승률 "
                f"{format_rate(row['승률'], row['승리'], row['분석 표본'])}"
            )
        self.ticker_return_summary_var.set("  |  ".join(horizon_text))
        self.search_button.configure(state="normal")

        target_cycle = self._cycle_for_first_signal_date(
            self.pending_chart_first_signal_date
        )
        if target_cycle is None:
            target_cycle = self._latest_cycle()

        if self._chart_is_open() or self.open_chart_after_search:
            children = self.buy_tree.get_children()
            target_position = (
                int(target_cycle.name)
                if target_cycle is not None and isinstance(target_cycle.name, int)
                else len(children) - 1
            )
            if children and 0 <= target_position < len(children):
                target_item = children[target_position]
                self.buy_tree.selection_set(target_item)
                self.buy_tree.focus(target_item)
                self.buy_tree.see(target_item)
            self._show_chart(target_cycle)
        self.open_chart_after_search = False
        self.pending_chart_first_signal_date = None

    def _show_error(self, ticker: str, exc: Exception) -> None:
        if isinstance(exc, DataLoadError):
            message = str(exc)
        else:
            message = f"{ticker} 처리 중 오류가 발생했습니다: {exc}"

        self.status_var.set(message)
        self.ticker_profile_var.set("분야: 조회 실패")
        self.ticker_cycle_summary_var.set("성과를 계산하지 못했습니다.")
        self.ticker_return_summary_var.set("")
        self.search_button.configure(state="normal")
        self.open_chart_after_search = False
        self.pending_chart_first_signal_date = None
        messagebox.showerror("오류", message)


def save_outputs(
    ticker: str,
    buy_points: pd.DataFrame,
    full_table: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buy_path = output_dir / f"{ticker}_buy_points.csv"
    full_path = output_dir / f"{ticker}_full_table.csv"

    buy_points.to_csv(buy_path, index_label="매수포인트날짜", encoding="utf-8-sig")
    full_table.to_csv(full_path, index_label="Date", encoding="utf-8-sig")
    return buy_path, full_path


def save_signal_outputs(
    ticker: str,
    signal_cycles: pd.DataFrame,
    full_table: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_path = output_dir / f"{ticker}_signal_cycles.csv"
    full_path = output_dir / f"{ticker}_full_table.csv"

    signal_cycles_for_display(signal_cycles).to_csv(
        signal_path,
        index=False,
        encoding="utf-8-sig",
    )
    full_table.to_csv(full_path, index_label="Date", encoding="utf-8-sig")
    return signal_path, full_path


def load_closed_scenarios(
    path: Path | str = CLOSED_SCENARIO_PATH,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS)

    data = pd.read_csv(path)
    if "현재 시총순위" not in data.columns and "순위" in data.columns:
        data = data.rename(columns={"순위": "현재 시총순위"})
    data = data.reindex(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS)
    for column in ("1차신호일", "2차신호일", "3차판정일"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    return data


def save_closed_scenarios(
    data: pd.DataFrame,
    path: Path | str = CLOSED_SCENARIO_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = data.reindex(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    normalized.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)
    return path


def save_tracker_scan_outputs(
    events: pd.DataFrame,
    active_scenarios: pd.DataFrame,
    closed_results: pd.DataFrame,
    failures: pd.DataFrame,
    scan_date: pd.Timestamp,
    output_dir: Path | str = DOWNLOADS_DIR,
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_text = scan_date.strftime("%Y-%m-%d")
    event_path = output_dir / f"MMRM_signal_events_{date_text}.csv"
    active_path = output_dir / f"MMRM_active_scenarios_{date_text}.csv"
    closed_path = output_dir / f"MMRM_closed_results_{date_text}.csv"
    failure_path = output_dir / f"MMRM_Top100_scan_failures_{date_text}.csv"

    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    active_scenarios.to_csv(active_path, index=False, encoding="utf-8-sig")
    closed_results.to_csv(closed_path, index=False, encoding="utf-8-sig")
    failures.to_csv(failure_path, index=False, encoding="utf-8-sig")
    return event_path, active_path, closed_path, failure_path


def save_analytics_outputs(
    sector_performance: pd.DataFrame,
    industry_performance: pd.DataFrame,
    field_rankings: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    date_suffix: str | None = None,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{date_suffix}" if date_suffix else ""
    prefix = "MMRM" if date_suffix else "mmrm"
    sector_path = output_dir / f"{prefix}_sector_performance{suffix}.csv"
    industry_path = output_dir / f"{prefix}_industry_performance{suffix}.csv"
    ranking_path = output_dir / f"{prefix}_field_stock_rankings{suffix}.csv"
    sector_performance.to_csv(sector_path, index=False, encoding="utf-8-sig")
    industry_performance.to_csv(industry_path, index=False, encoding="utf-8-sig")
    field_rankings.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    return sector_path, industry_path, ranking_path


def _failure_row(company: MarketCapCompany, error: str) -> dict[str, object]:
    return {
        "순위": company.rank,
        "티커": company.ticker,
        "회사명": company.company,
        "시가총액": company.market_cap,
        "오류": error,
    }


def table_for_display(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.reset_index().rename(columns={"Date": "매수포인트날짜"})
    display = data.reset_index()
    if "Date" not in display.columns:
        display = display.rename(columns={display.columns[0]: "Date"})
    display = display.rename(columns={"Date": "매수포인트날짜"})
    return display


def scanner_table_for_display(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return data.reindex(columns=columns).copy()


def prioritize_scan_events(data: pd.DataFrame) -> pd.DataFrame:
    """Put actionable signals first while preserving recency and market-cap order."""
    display = data.copy()
    if display.empty:
        return display
    display["_신호우선순위"] = display.apply(
        lambda row: scan_event_priority(row.get("단계"), row.get("결과")),
        axis=1,
    )
    return (
        display.sort_values(
            by=["_신호우선순위", "신호일", "순위"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop(columns=["_신호우선순위"])
        .reset_index(drop=True)
    )


def prioritize_active_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    display = data.copy()
    if display.empty:
        return display
    display["_상태우선순위"] = display["현재상태"].map(
        {"3차 신호 대기": 0, "2차 신호 대기": 1}
    ).fillna(2)
    first_dates = pd.to_datetime(display.get("1차신호일"), errors="coerce")
    second_dates = pd.to_datetime(display.get("2차신호일"), errors="coerce")
    display["_단계진입일"] = first_dates
    waiting_for_third = display["현재상태"].eq("3차 신호 대기")
    display.loc[waiting_for_third, "_단계진입일"] = second_dates[waiting_for_third]
    return (
        display.sort_values(
            by=["_상태우선순위", "_단계진입일", "순위", "티커"],
            ascending=[True, False, True, True],
            na_position="last",
        )
        .drop(columns=["_상태우선순위", "_단계진입일"])
        .reset_index(drop=True)
    )


def active_scenario_tag(state: object) -> str:
    if state == "3차 신호 대기":
        return "signal_second"
    if state == "2차 신호 대기":
        return "signal_first"
    return ""


def history_cycle_tag(result: object, three_month_return: object) -> str:
    result_text = str(result)
    if "폐기" in result_text:
        return "history_discard"
    if result_text == "실패":
        return "history_failure"
    if result_text != "매수 성공":
        return ""

    numeric_return = pd.to_numeric(
        pd.Series([three_month_return]), errors="coerce"
    ).iloc[0]
    if pd.isna(numeric_return):
        return "history_success_pending"
    if numeric_return < 0:
        return "history_loss"
    if numeric_return == 0:
        return "history_flat"
    if numeric_return < 10:
        return "history_success_low"
    if numeric_return < 25:
        return "history_success_medium"
    return "history_success_high"


def scan_event_priority(stage: object, result: object) -> int:
    if stage == "3차 신호" and result == "매수 성공":
        return 0
    if stage == "2차 신호":
        return 1
    if stage == "1차 신호":
        return 2
    return 3


def scan_event_tag(stage: object, result: object) -> str:
    priority = scan_event_priority(stage, result)
    return {
        0: "signal_third",
        1: "signal_second",
        2: "signal_first",
    }.get(priority, "")


def add_scan_performance_columns(
    data: pd.DataFrame,
    companies: list[MarketCapCompany],
    cycles_by_ticker: dict[str, pd.DataFrame],
    classifications: pd.DataFrame,
    sector_output: pd.DataFrame,
    horizon_months: int = 3,
) -> pd.DataFrame:
    display = data.copy()
    ticker_column = f"종목 {horizon_months}개월 승률"
    sector_column = f"섹터 {horizon_months}개월 승률"
    display[ticker_column] = "미산출 (0건)"
    display[sector_column] = "미산출 (0건)"
    if display.empty:
        return display

    ticker_performance = build_ticker_performance(
        companies,
        cycles_by_ticker,
        classifications,
        horizon_months,
    )
    ticker_rates = {
        str(row["티커"]).upper(): format_rate(
            row["승률"], row["승리"], row["분석 표본"]
        )
        for _, row in ticker_performance.iterrows()
    }

    horizon_label = f"{horizon_months}개월"
    if "분석 기간" in sector_output.columns:
        sector_rows = sector_output[sector_output["분석 기간"].eq(horizon_label)]
    else:
        sector_rows = pd.DataFrame()
    sector_rates = {
        str(row["분야"]): format_rate(
            row["승률"], row["승리"], row["분석 표본"]
        )
        for _, row in sector_rows.iterrows()
    }

    display[ticker_column] = (
        display["티커"].astype(str).str.upper().map(ticker_rates).fillna("미산출 (0건)")
    )
    if "섹터" in display.columns:
        display[sector_column] = display["섹터"].map(sector_rates).fillna("미산출 (0건)")
    return display


def add_sector_column(data: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    display = data.copy()
    if "섹터" in display.columns:
        display = display.drop(columns=["섹터"])
    if "티커" not in display.columns:
        return display

    if classifications.empty or "티커" not in classifications.columns:
        sector_by_ticker = {}
    else:
        sector_by_ticker = {
            str(row.get("티커", "")).upper(): str(row.get("섹터", "미분류"))
            for _, row in classifications.iterrows()
        }
    position = list(display.columns).index("티커") + 1
    display.insert(
        position,
        "섹터",
        display["티커"].astype(str).str.upper().map(sector_by_ticker).fillna("미분류"),
    )
    return display


def build_closed_scenario_history(
    companies: list[MarketCapCompany],
    cycles_by_ticker: dict[str, pd.DataFrame],
    classifications: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    failed_tickers: set[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    current_tickers = {company.ticker.upper() for company in companies}

    for company in companies:
        cycles = cycles_by_ticker.get(company.ticker.upper())
        if cycles is None or cycles.empty or "Outcome" not in cycles.columns:
            continue
        closed = cycles.loc[
            ~cycles["Outcome"].isin({"2차 신호 대기", "3차 신호 대기"})
        ]
        if closed.empty:
            continue
        display = signal_cycles_for_display(closed)
        display.insert(0, "회사명", company.company)
        display.insert(0, "티커", company.ticker.upper())
        display.insert(0, "현재 시총순위", company.rank)
        frames.append(display)

    if frames:
        current = add_sector_column(
            pd.concat(frames, ignore_index=True),
            classifications,
        )
    else:
        current = pd.DataFrame(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS)

    failed = {ticker.upper() for ticker in (failed_tickers or set())}
    if previous is not None and not previous.empty and failed:
        if "현재 시총순위" not in previous.columns and "순위" in previous.columns:
            previous = previous.rename(columns={"순위": "현재 시총순위"})
        previous_tickers = previous["티커"].astype(str).str.upper()
        preserved = previous.loc[
            previous_tickers.isin(failed & current_tickers)
        ].copy()
        current = pd.concat([current, preserved], ignore_index=True)

    if current.empty:
        return pd.DataFrame(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS)

    current = current.reindex(columns=CLOSED_SCENARIO_DISPLAY_COLUMNS)
    for column in ("1차신호일", "2차신호일", "3차판정일"):
        current[column] = pd.to_datetime(current[column], errors="coerce")
    return (
        current.drop_duplicates(subset=["티커", "1차신호일"], keep="first")
        .sort_values(
            by=["1차신호일", "현재 시총순위", "티커"],
            ascending=[False, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def field_performance_for_display(data: pd.DataFrame) -> pd.DataFrame:
    display = data.copy()
    if display.empty:
        return pd.DataFrame(columns=FIELD_DISPLAY_COLUMNS)
    display["매수 도달률"] = display.apply(
        lambda row: format_reach_rate(
            row["매수 도달률"], row["매수 건수"], row["종료 사이클"]
        ),
        axis=1,
    )
    display["승률"] = display.apply(
        lambda row: format_rate(row["승률"], row["승리"], row["분석 표본"]),
        axis=1,
    )
    return display.reindex(columns=FIELD_DISPLAY_COLUMNS)


def ranking_for_display(data: pd.DataFrame) -> pd.DataFrame:
    display = data.copy()
    if display.empty:
        return pd.DataFrame(columns=RANKING_DISPLAY_COLUMNS)
    display["매수 도달률"] = display.apply(
        lambda row: format_reach_rate(
            row["매수 도달률"], row["매수 건수"], row["종료 사이클"]
        ),
        axis=1,
    )
    display["승률"] = display.apply(
        lambda row: format_rate(row["승률"], row["승리"], row["분석 표본"]),
        axis=1,
    )
    return display.reindex(columns=RANKING_DISPLAY_COLUMNS)


def _horizon_months(value: str) -> int:
    try:
        return int(str(value).replace("개월", "").strip())
    except ValueError:
        return 3


def signal_cycles_for_display(data: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "FirstSignalDate": "1차신호일",
        "SecondSignalDate": "2차신호일",
        "ThirdDecisionDate": "3차판정일",
        "Outcome": "결과",
        "Return3M": "3개월후 수익률",
        "Return6M": "6개월후 수익률",
        "Return9M": "9개월후 수익률",
        "Return12M": "12개월후 수익률",
    }
    display = data.rename(columns=rename_map).copy()
    status_columns = {
        "3개월후 수익률": "Return3MStatus",
        "6개월후 수익률": "Return6MStatus",
        "9개월후 수익률": "Return9MStatus",
        "12개월후 수익률": "Return12MStatus",
    }
    for display_column, status_column in status_columns.items():
        if display_column not in display.columns or status_column not in display.columns:
            continue
        display[display_column] = display[display_column].astype(object)
        missing = display[display_column].isna()
        display.loc[missing, display_column] = display.loc[missing, status_column]
    return display.reindex(columns=SIGNAL_HISTORY_DISPLAY_COLUMNS)


def signal_cycle_position(
    cycles: pd.DataFrame,
    cycle: pd.Series | None,
) -> int | None:
    """Return the table row position corresponding to a chart cycle."""
    if cycles.empty or cycle is None:
        return None

    matches = pd.Series(True, index=cycles.index)
    compared = False
    for column in ("FirstSignalDate", "SecondSignalDate", "ThirdDecisionDate"):
        if column not in cycles.columns or column not in cycle.index:
            continue
        target = pd.to_datetime(cycle.get(column), errors="coerce")
        values = pd.to_datetime(cycles[column], errors="coerce")
        if pd.isna(target):
            matches &= values.isna()
        else:
            matches &= values.dt.normalize().eq(pd.Timestamp(target).normalize())
        compared = True

    if "Outcome" in cycles.columns and "Outcome" in cycle.index:
        matches &= cycles["Outcome"].astype(str).eq(str(cycle.get("Outcome")))
        compared = True
    if not compared:
        return None

    positions = [
        position
        for position, matched in enumerate(matches.to_numpy(dtype=bool))
        if matched
    ]
    return positions[-1] if positions else None


def _sorted_frame(
    rows: list[dict[str, object]],
    columns: list[str],
    by: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    data = pd.DataFrame(rows, columns=columns)
    if data.empty:
        return data
    return data.sort_values(by=by, ascending=ascending, na_position="last").reset_index(drop=True)


def populate_table(tree: ttk.Treeview, data: pd.DataFrame) -> None:
    tree.delete(*tree.get_children())
    columns = list(data.columns)
    tree["columns"] = columns

    formatted_rows = [
        [_format_value(row[column], column) for column in columns]
        for _, row in data.iterrows()
    ]
    default_font = tkfont.nametofont("TkDefaultFont", root=tree)

    for column_index, column in enumerate(columns):
        measured_width = default_font.measure(str(column)) + 28
        for values in formatted_rows:
            measured_width = max(
                measured_width,
                default_font.measure(values[column_index]) + 24,
            )
        maximum_width = 700 if column in {"오류", "ConditionSummary"} else 360
        display_width = min(
            max(_column_width(column), measured_width),
            maximum_width,
        )
        tree.heading(column, text=column)
        tree.column(column, width=display_width, minwidth=60, stretch=False)

    for values in formatted_rows:
        tree.insert("", "end", values=values)


def _table_required_width(columns: list[str]) -> int:
    """Return the panel width needed to show every configured column at once."""
    return sum(_column_width(column) for column in columns) + 42


def _column_width(column: str) -> int:
    if column in {
        "Date",
        "매수포인트날짜",
        "1차신호일",
        "2차신호일",
        "3차판정일",
        "종료일",
        "observation_start_date",
        "주봉시작일",
        "스캔일",
        "신호일",
        "마지막확인일",
        "데이터기준일",
    }:
        return 110
    if column == "회사명":
        return 220
    if column == "오류":
        return 420
    if column in {"섹터", "산업", "분야"}:
        return 175
    if column == "ConditionSummary":
        return 600
    if column in {"macd_area", "macd_flow"}:
        return 140
    if column in {"순위", "현재 시총순위", "티커"}:
        return 70
    if column == "시가총액":
        return 95
    if column in {
        "Close",
        "MA_20",
        "MA_50",
        "MA_150",
        "MA_200",
        "MACD",
        "Signal",
        "RSI",
        "MFI",
    }:
        return 75
    if column == "Momentum":
        return 85
    if column == "결과":
        return 210
    if column in {"승률", "매수 도달률"} or column.endswith("개월 승률"):
        return 145
    if column in {"종합점수", "분석 표본", "매수 건수", "종목 수", "종료 사이클"}:
        return 95
    if column == "MA20_50이격률":
        return 120
    if column in {"단계", "신호구분"}:
        return 95
    if column in {"현재상태", "데이터상태"}:
        return 125
    if column.endswith("수익률") or column.endswith("손익률"):
        return 125
    return 110


def _format_value(value: object, column: str = "") -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if (
        column.endswith("수익률")
        or column.endswith("손익률")
        or column.endswith("이격률")
        or column in {"승률", "매수 도달률"}
    ) and isinstance(
        value, (int, float)
    ):
        return f"{value:+.2f}%"
    if column == "종합점수" and isinstance(value, (int, float)):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


if __name__ == "__main__":
    app = BuyPointApp()
    app.mainloop()
