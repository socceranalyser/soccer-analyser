"""Private backup of personal data (coupons, coupon draft, favourites) in the owner's Telegram.

The online dashboard (Streamlit Community Cloud) has a temporary disk: every redeploy (each
push to GitHub, the daily data commit) wipes the SQLite file. Coupons must not go to the
public repository, so the bot keeps one pinned JSON document in the owner's private chat:
    push()     - after every change: edit the pinned document (or send + pin a new one)
    restore()  - on start-up, when the database has no coupons: read it back
Only active on the cloud (or with SA_BACKUP=1); the local database is permanent anyway.
"""
from __future__ import annotations

import json
import os
import threading

import requests

from .config import ROOT

NAME = "soccer_analyser_backup.json"
CAPTION = "💾 Резервная копия ваших купонов и избранного (бот обновляет её сам — не удаляйте)"


def enabled() -> bool:
    return str(ROOT).replace("\\", "/").startswith("/mount/src") or os.environ.get("SA_BACKUP") == "1"


def _cfg():
    from .notify import chat_id, load_config
    token, chat = load_config().get("TELEGRAM_BOT_TOKEN"), chat_id()
    return (token, chat) if token and chat else (None, None)


def _api(token, method, **kw):
    r = requests.post(f"https://api.telegram.org/bot{token}/{method}", timeout=30, **kw)
    return r.json() if r.ok else {"ok": False}


def _pinned(token, chat):
    msg = (_api(token, "getChat", data={"chat_id": chat}).get("result") or {}).get("pinned_message")
    if msg and (msg.get("document") or {}).get("file_name") == NAME:
        return msg
    return None


def snapshot() -> dict:
    from . import storage
    return {"coupons": storage.load_coupons(), "draft": storage.load_draft(),
            "favorites": sorted(storage.load_favorites())}


def _push(data: dict):
    token, chat = _cfg()
    if not token:
        return
    blob = (NAME, json.dumps(data, ensure_ascii=False, default=str).encode(), "application/json")
    files = {"f": blob}
    msg = _pinned(token, chat)
    if msg:
        media = json.dumps({"type": "document", "media": "attach://f", "caption": CAPTION})
        res = _api(token, "editMessageMedia", data={"chat_id": chat, "message_id": msg["message_id"],
                                                    "media": media}, files=files)
        if res.get("ok") or "not modified" in str(res.get("description", "")):
            return
    res = _api(token, "sendDocument", data={"chat_id": chat, "caption": CAPTION,
                                            "disable_notification": "true"},
               files={"document": blob})
    if res.get("ok"):
        _api(token, "pinChatMessage", data={"chat_id": chat, "disable_notification": "true",
                                            "message_id": res["result"]["message_id"]})


def push():
    """Back up in the background (never slows down or breaks the page)."""
    if not enabled():
        return
    try:
        data = snapshot()
    except Exception:
        return
    threading.Thread(target=lambda: _safe(_push, data), daemon=True).start()


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:
        pass


def restore() -> int:
    """Fill an empty database from the pinned backup; returns the number of coupons restored."""
    if not enabled():
        return 0
    from . import storage
    if storage.load_coupons() or storage.load_draft() or storage.load_favorites():
        return 0
    token, chat = _cfg()
    if not token:
        return 0
    msg = _pinned(token, chat)
    if not msg:
        return 0
    f = _api(token, "getFile", data={"file_id": msg["document"]["file_id"]}).get("result") or {}
    if not f.get("file_path"):
        return 0
    raw = requests.get(f"https://api.telegram.org/file/bot{token}/{f['file_path']}", timeout=30)
    data = raw.json()
    for c in reversed(data.get("coupons", [])):  # oldest first, keep the original ids
        storage.save_coupon(c["picks"], c["stake"], c["total_odds"], c["model_prob"],
                            created_at=c.get("created_at"), coupon_id=c.get("id"), backup=False)
    if data.get("draft"):
        storage.save_draft(data["draft"], backup=False)
    for key in data.get("favorites", []):
        storage.set_favorite(*key, on=True, backup=False)
    return len(data.get("coupons", []))
