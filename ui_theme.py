from __future__ import annotations

import json
from pathlib import Path
import tkinter.font as tkfont


DEFAULT_THEME = "light"
UI_FONT_CANDIDATES = ("Noto Sans KR", "Segoe UI", "Malgun Gothic")
TK_UI_FONT_NAMES = (
    "TkDefaultFont",
    "TkTextFont",
    "TkMenuFont",
    "TkHeadingFont",
    "TkCaptionFont",
    "TkSmallCaptionFont",
    "TkIconFont",
    "TkTooltipFont",
)

THEMES = {
    "light": {
        "window": "#f3f4f6",
        "panel": "#f8fafc",
        "field": "#ffffff",
        "text": "#111827",
        "muted": "#5f6b78",
        "border": "#c7cdd4",
        "button": "#e5e7eb",
        "button_active": "#d1d5db",
        "scroll_thumb": "#c7cdd4",
        "scroll_thumb_active": "#9ca7b3",
        "scroll_arrow": "#5f6b78",
        "selected": "#2563eb",
        "selected_text": "#ffffff",
        "chart_background": "#ffffff",
        "chart_panel": "#ffffff",
        "chart_grid": "#edf0f2",
        "chart_border": "#c7cdd4",
        "chart_text": "#5f6b78",
        "crosshair": "#4b5563",
        "date_label_background": "#374151",
        "date_label_text": "#ffffff",
        "zero_line": "#9ca3af",
        "overbought_fill": "#ffd6d6",
        "oversold_fill": "#dbeafe",
    },
    "dark": {
        "window": "#181818",
        "panel": "#202020",
        "field": "#151515",
        "text": "#c9c9c9",
        "muted": "#8f8f8f",
        "border": "#343434",
        "button": "#292929",
        "button_active": "#383838",
        "scroll_thumb": "#383838",
        "scroll_thumb_active": "#555555",
        "scroll_arrow": "#767676",
        "selected": "#3a3a3a",
        "selected_text": "#dedede",
        "chart_background": "#171717",
        "chart_panel": "#1b1b1b",
        "chart_grid": "#303030",
        "chart_border": "#3a3a3a",
        "chart_text": "#969696",
        "crosshair": "#7f7f7f",
        "date_label_background": "#d4d4d4",
        "date_label_text": "#171717",
        "zero_line": "#6f6f6f",
        "overbought_fill": "#47272a",
        "oversold_fill": "#25334a",
    },
}


def configure_ui_fonts(root) -> str:
    available = set(tkfont.families(root))
    fallback = str(tkfont.nametofont("TkDefaultFont", root=root).actual("family"))
    family = next(
        (candidate for candidate in UI_FONT_CANDIDATES if candidate in available),
        fallback,
    )

    for name in TK_UI_FONT_NAMES:
        tkfont.nametofont(name, root=root).configure(family=family, size=9)
    return family


def normalize_theme(value: object) -> str:
    mode = str(value).strip().lower()
    return mode if mode in THEMES else DEFAULT_THEME


def theme_palette(mode: object) -> dict[str, str]:
    return THEMES[normalize_theme(mode)]


def load_theme(path: Path | str) -> str:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return DEFAULT_THEME
    return normalize_theme(payload.get("theme"))


def save_theme(path: Path | str, mode: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"theme": normalize_theme(mode)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
