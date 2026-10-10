"""Answer commands sent to the Telegram bot by starting a scan on GitHub.

A scheduled GitHub Actions workflow runs this every few minutes.  It reads
the messages sent to the bot, and when the owner asked for a scan it starts
the weekly-scan workflow, whose summary then arrives through the usual
Telegram notification.  Only the standard library is used so the runner
needs no package installation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from telegram_api import (
    TELEGRAM_CHAT_ID_ENV,
    TELEGRAM_TOKEN_ENV,
    NotificationError,
    get_updates,
    send_telegram_message,
)


SCAN_WORKFLOW_FILE = "weekly-scan.yml"
SCAN_COMMANDS = {"스캔", "/스캔", "scan", "/scan", "결과", "/결과"}
HELP_COMMANDS = {"/start", "/help", "도움말", "help", "?"}
# A request older than this was sent while polling was not running; a scan
# arriving that late would only be confusing.
MAX_REQUEST_AGE_SECONDS = 60 * 60

SCAN_STARTED_TEXT = (
    "스캔을 시작했습니다. 1~2분 뒤 결과를 보내 드립니다.\n"
    "금요일 장 마감 전이면 잠정 결과입니다."
)
HELP_TEXT = (
    "'스캔'이라고 보내면 지금 기준으로 스캔해서 결과를 보내 드립니다.\n"
    "메시지 확인은 약 5분마다 하므로 답이 오기까지 몇 분 걸릴 수 있습니다.\n"
    "정기 알림은 매주 금요일 미국 장 마감 뒤에 자동으로 옵니다."
)
SCAN_FAILED_TEXT = "스캔을 시작하지 못했습니다. 잠시 뒤 다시 '스캔'이라고 보내 주세요."


def command_for(text: object) -> str:
    """Return "scan", "help" or "" for one message text."""
    normalized = str(text or "").strip().lower()
    if normalized in SCAN_COMMANDS:
        return "scan"
    if normalized in HELP_COMMANDS:
        return "help"
    return ""


def requested_actions(
    updates: list[dict],
    chat_id: str,
    now: float,
) -> tuple[set[str], int | None]:
    """Decide what the owner asked for and which updates to acknowledge.

    Messages from any other chat are ignored: the bot's address is public,
    but only the configured chat may start scans.  Returns the actions
    ("scan", "help", "unknown") and the offset that acknowledges every
    update seen, or ``None`` when there was nothing new.
    """
    actions: set[str] = set()
    last_update_id: int | None = None
    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            last_update_id = update_id if last_update_id is None else max(last_update_id, update_id)
        message = update.get("message") or {}
        if str((message.get("chat") or {}).get("id", "")) != str(chat_id):
            continue
        if now - float(message.get("date", 0) or 0) > MAX_REQUEST_AGE_SECONDS:
            continue
        actions.add(command_for(message.get("text")) or "unknown")
    offset = None if last_update_id is None else last_update_id + 1
    return actions, offset


def start_scan_workflow(repository: str, github_token: str, ref: str = "main") -> None:
    """Start the weekly-scan workflow through the GitHub API."""
    request = Request(
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{SCAN_WORKFLOW_FILE}/dispatches",
        data=json.dumps({"ref": ref}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "mmrm-telegram-commands",
        },
        method="POST",
    )
    with urlopen(request, timeout=20):
        pass


def main() -> int:
    token = os.environ.get(TELEGRAM_TOKEN_ENV, "").strip()
    chat_id = os.environ.get(TELEGRAM_CHAT_ID_ENV, "").strip()
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token or not chat_id:
        print("텔레그램 설정이 없어 명령 확인을 건너뜁니다.")
        return 0

    try:
        updates = get_updates(token)
        actions, offset = requested_actions(updates, chat_id, time.time())
        if offset is not None:
            # Acknowledge first, so a crash below cannot start the same scan twice.
            get_updates(token, offset=offset)
    except NotificationError as exc:
        print(f"경고: {exc}", file=sys.stderr)
        return 0

    print(f"새 메시지 {len(updates)}건, 요청: {sorted(actions) or '없음'}")
    try:
        if "scan" in actions:
            try:
                start_scan_workflow(repository, github_token)
            except (HTTPError, OSError, ValueError) as exc:
                print(f"스캔 시작 실패: {type(exc).__name__}", file=sys.stderr)
                send_telegram_message(SCAN_FAILED_TEXT, token, chat_id)
                return 1
            send_telegram_message(SCAN_STARTED_TEXT, token, chat_id)
        elif actions:
            send_telegram_message(HELP_TEXT, token, chat_id)
    except NotificationError as exc:
        print(f"경고: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
