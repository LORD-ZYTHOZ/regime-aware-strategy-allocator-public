"""
live/telegram.py — Telegram sender for the allocator bridge.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_PERSONAL_CHAT_ID from the shared
bridge/.env (same file Singularity uses) so no new credentials are needed.

Two destinations:
  personal  → TELEGRAM_PERSONAL_CHAT_ID  (strategy alerts, budget decisions)
  alert     → TELEGRAM_ALERT_CHAT_ID     (errors, watchdog, solver failures)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger("bridge.telegram")

# Same bridge/.env that Singularity reads
_ENV_PATH = Path(os.getenv("BRIDGE_ENV_PATH", str(Path.home() / "strategies" / "bridge" / ".env")))
_SEND_URL  = "https://api.telegram.org/bot{token}/sendMessage"


def _load_cfg() -> dict:
    cfg: dict = {}
    try:
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return cfg


def _post(token: str, chat_id: str, text: str) -> None:
    try:
        import urllib.request, urllib.parse, json as _json
        body = _json.dumps({"chat_id": chat_id, "text": text}).encode()
        req  = urllib.request.Request(
            _SEND_URL.format(token=token),
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        log.warning("telegram send failed: %s", exc)


def send_personal(text: str) -> None:
    """DM the user (TELEGRAM_PERSONAL_CHAT_ID)."""
    cfg = _load_cfg()
    token = cfg.get("TELEGRAM_BOT_TOKEN", "")
    chat  = cfg.get("TELEGRAM_PERSONAL_CHAT_ID", "")
    if token and chat:
        _post(token, chat, text)


def send_alert(text: str) -> None:
    """Post to the alert channel (TELEGRAM_ALERT_CHAT_ID)."""
    cfg = _load_cfg()
    token = cfg.get("TELEGRAM_BOT_TOKEN", "")
    chat  = cfg.get("TELEGRAM_ALERT_CHAT_ID", "")
    if token and chat:
        _post(token, chat, text)
