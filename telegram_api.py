"""Minimal Telegram Bot API client using only the standard library.

Kept free of third-party imports so the command poller can run on a bare
GitHub Actions runner without installing anything.
"""

from __future__ import annotations

import http.client
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/{method}"
TELEGRAM_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


class NotificationError(RuntimeError):
    """Raised when a Telegram request could not be completed."""


def call_telegram(method: str, token: str, payload: dict[str, object]) -> object:
    """Call one Bot API method and return its ``result`` value."""
    request = Request(
        TELEGRAM_API_URL.format(token=token, method=method),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Error text never includes the request URL, because the URL holds the token.
    try:
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise NotificationError(
            f"텔레그램 전송 실패: HTTP {exc.code} ({_telegram_error_hint(exc.code)})"
        ) from None
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # URLError, timeouts and dropped connections all land here.
        raise NotificationError(
            f"텔레그램 전송 실패: {type(exc).__name__}"
        ) from None
    if not isinstance(body, dict) or not body.get("ok"):
        raise NotificationError("텔레그램 전송 실패: 응답이 올바르지 않습니다.")
    return body.get("result")


def send_telegram_message(text: str, token: str, chat_id: str) -> None:
    call_telegram("sendMessage", token, {"chat_id": chat_id, "text": text})


def get_updates(token: str, offset: int | None = None) -> list[dict]:
    """Messages sent to the bot that have not been acknowledged yet.

    Passing ``offset`` (last update id + 1) acknowledges everything before it,
    so those updates are not returned again.
    """
    payload: dict[str, object] = {"timeout": 0, "allowed_updates": ["message"]}
    if offset is not None:
        payload["offset"] = offset
    result = call_telegram("getUpdates", token, payload)
    return result if isinstance(result, list) else []


def _telegram_error_hint(status: int) -> str:
    return {
        400: "채팅 ID가 틀렸거나 봇에게 먼저 말을 걸지 않았습니다",
        401: "봇 토큰이 틀렸습니다",
        403: "봇을 차단했거나 대화를 시작하지 않았습니다",
        404: "봇 토큰이 틀렸습니다",
        409: "다른 프로그램이 같은 봇의 메시지를 받고 있습니다",
    }.get(status, "잠시 후 다시 시도해 주세요")
