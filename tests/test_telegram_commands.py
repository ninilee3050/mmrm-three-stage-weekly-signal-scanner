from __future__ import annotations

import json

import pytest

import telegram_api
import telegram_commands
from telegram_commands import (
    MAX_REQUEST_AGE_SECONDS,
    command_for,
    requested_actions,
)

NOW = 1_800_000_000.0
OWNER = "8912890653"


def update(update_id: int, text: str, chat_id: object = OWNER, age: float = 10) -> dict:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "date": NOW - age, "text": text},
    }


def test_scan_and_help_commands_are_recognised() -> None:
    assert command_for(" 스캔 ") == "scan"
    assert command_for("/SCAN") == "scan"
    assert command_for("결과") == "scan"
    assert command_for("/start") == "help"
    assert command_for("안녕") == ""
    assert command_for(None) == ""


def test_only_the_owner_chat_can_request_a_scan() -> None:
    updates = [
        update(10, "스캔", chat_id=111),  # a stranger who found the bot
        update(11, "스캔", chat_id=int(OWNER)),
        update(12, "스캔", chat_id=OWNER),  # sent twice: still one scan
    ]

    actions, offset = requested_actions(updates, OWNER, NOW)

    assert actions == {"scan"}
    assert offset == 13


def test_strangers_alone_start_nothing_but_are_acknowledged() -> None:
    actions, offset = requested_actions([update(20, "스캔", chat_id=111)], OWNER, NOW)

    assert actions == set()
    assert offset == 21


def test_old_requests_are_skipped_and_unknown_text_gets_help() -> None:
    stale = update(30, "스캔", age=MAX_REQUEST_AGE_SECONDS + 5)
    chatter = update(31, "뭐해")

    assert requested_actions([stale], OWNER, NOW) == (set(), 31)
    assert requested_actions([stale, chatter], OWNER, NOW) == ({"unknown"}, 32)
    assert requested_actions([], OWNER, NOW) == (set(), None)


@pytest.fixture
def fake_environment(monkeypatch):
    calls = {"updates": [], "sent": [], "dispatched": []}
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", OWNER)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-token")
    monkeypatch.setattr(telegram_commands.time, "time", lambda: NOW)
    monkeypatch.setattr(
        telegram_commands,
        "send_telegram_message",
        lambda text, token, chat_id: calls["sent"].append((text, chat_id)),
    )
    monkeypatch.setattr(
        telegram_commands,
        "start_scan_workflow",
        lambda repository, token: calls["dispatched"].append(repository),
    )
    return calls


def test_scan_request_acknowledges_then_starts_the_workflow(monkeypatch, fake_environment) -> None:
    calls = fake_environment

    def fake_get_updates(token, offset=None):
        calls["updates"].append(offset)
        return [] if offset is not None else [update(40, "스캔")]

    monkeypatch.setattr(telegram_commands, "get_updates", fake_get_updates)

    assert telegram_commands.main() == 0
    assert calls["updates"] == [None, 41]  # read, then acknowledge
    assert calls["dispatched"] == ["owner/repo"]
    assert calls["sent"] == [(telegram_commands.SCAN_STARTED_TEXT, OWNER)]


def test_no_messages_means_no_dispatch_and_no_reply(monkeypatch, fake_environment) -> None:
    calls = fake_environment
    monkeypatch.setattr(telegram_commands, "get_updates", lambda token, offset=None: [])

    assert telegram_commands.main() == 0
    assert calls["dispatched"] == [] and calls["sent"] == []


def test_failed_dispatch_tells_the_owner(monkeypatch, fake_environment) -> None:
    calls = fake_environment
    monkeypatch.setattr(
        telegram_commands,
        "get_updates",
        lambda token, offset=None: [] if offset is not None else [update(50, "스캔")],
    )

    def failing_dispatch(repository, token):
        raise OSError("network down")

    monkeypatch.setattr(telegram_commands, "start_scan_workflow", failing_dispatch)

    assert telegram_commands.main() == 1
    assert calls["sent"] == [(telegram_commands.SCAN_FAILED_TEXT, OWNER)]


def test_get_updates_sends_the_offset_and_returns_the_result(monkeypatch) -> None:
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"ok": True, "result": [{"update_id": 7}]}).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr(telegram_api, "urlopen", fake_urlopen)

    assert telegram_api.get_updates("TOKEN", offset=8) == [{"update_id": 7}]
    assert seen["url"] == "https://api.telegram.org/botTOKEN/getUpdates"
    assert seen["body"]["offset"] == 8


def test_command_poller_needs_only_the_standard_library() -> None:
    import ast
    import pathlib
    import sys

    allowed = set(sys.stdlib_module_names) | {"telegram_api"}
    for name in ("telegram_commands.py", "telegram_api.py"):
        tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
        imported = {
            (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported <= allowed, (name, imported - allowed)
