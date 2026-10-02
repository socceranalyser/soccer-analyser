"""Telegram notifications: a daily digest (yesterday's results, today's calls, coupons).

Configuration (never committed): .env in the project root or environment variables
    TELEGRAM_BOT_TOKEN=...   (from @BotFather)
    TELEGRAM_CHAT_ID=...     (filled in by scripts/telegram_setup.py)
"""
from __future__ import annotations

import html
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from .config import LEAGUES, ROOT
from .fixtures import COMP_ORDER

ENV_PATH = ROOT / ".env"
TOP_LEAGUES = ["E0", "SP1", "I1", "D1", "F1", "P1", "N1", "T1", "SC0", "E1", "BRA", "ARG", "USA"]
MAX_TODAY = 25


def load_config() -> dict:
    cfg = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def save_env_value(key: str, value: str):
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    lines = [l for l in lines if not l.startswith(f"{key}=")] + [f"{key}={value}"]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


CHAT_FILE = ROOT / "data" / "state" / "telegram_chat_id.txt"


def chat_id() -> str | None:
    """Chat id from config, from the saved file, or discovered from the bot's messages
    (the owner pressed Start) and then saved, so only the token has to be configured."""
    c = load_config()
    if c.get("TELEGRAM_CHAT_ID"):
        return c["TELEGRAM_CHAT_ID"]
    if CHAT_FILE.exists():
        v = CHAT_FILE.read_text(encoding="utf-8").strip()
        if v:
            return v
    token = c.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return None
    try:
        upd = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30).json()
    except requests.RequestException:
        return None
    chats = [u["message"]["chat"]["id"] for u in upd.get("result", []) if "message" in u]
    if not chats:
        return None
    CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHAT_FILE.write_text(str(chats[-1]), encoding="utf-8")
    return str(chats[-1])


def configured() -> bool:
    return bool(load_config().get("TELEGRAM_BOT_TOKEN") and chat_id())


def send(text: str) -> bool:
    """Send an HTML message (split into Telegram-sized chunks)."""
    token, chat = load_config().get("TELEGRAM_BOT_TOKEN"), chat_id()
    if not token or not chat:
        return False
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    chunks.append(cur)
    ok = True
    for ch in chunks:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": ch, "parse_mode": "HTML",
                                "disable_web_page_preview": "true"}, timeout=30)
        ok &= r.ok
    return ok


# ------------------------------------------------------------------- digest
def _priority(code: str) -> int:
    if code in COMP_ORDER:
        return COMP_ORDER.index(code)
    if code in TOP_LEAGUES:
        return 10 + TOP_LEAGUES.index(code)
    return 100 + (LEAGUES.get(code, {}).get("tier", 5))


def _day(engine, date) -> pd.DataFrame:
    """A day's fixtures with calls; finished matches use the forecast stored before kick-off."""
    from . import storage
    from .verdicts import calls_columns
    day = engine.day(date)
    if day.empty or "p_home" not in day:
        return day
    pre = storage.prematch_forecasts([date - timedelta(days=1), date, date + timedelta(days=1)])
    if len(pre):
        pre = pre.drop_duplicates(["league", "home", "away"], keep="last")
        m = day[["competition", "home", "away"]].reset_index().merge(
            pre.rename(columns={"league": "competition"}), on=["competition", "home", "away"]
        ).set_index("index")
        m = m[day.loc[m.index, "played"]]
        for c in ("p_home", "p_draw", "p_away", "p_over25", "p_btts", "xg_home", "xg_away"):
            ok = m[c].notna()
            day.loc[m.index[ok], c] = m.loc[ok, c].astype(float)
    now = pd.Timestamp.now(tz=day["kickoff"].dt.tz) if day["kickoff"].notna().any() else None
    started = day["kickoff"].notna() & (day["kickoff"] <= now) if now is not None else         pd.Series(False, index=day.index)
    for idx in day.index[~day["played"] & started]:
        res = storage.find_result(day.at[idx, "home"], day.at[idx, "away"], day.at[idx, "date"])
        if res is not None:
            day.loc[idx, ["hg", "ag", "played"]] = [res[0], res[1], True]
    day = day.assign(**calls_columns(day))
    day["prio"] = day["competition"].map(_priority)
    return day.sort_values(["prio", "kickoff"])


def _e(s) -> str:
    return html.escape(str(s))


def daily_digest(engine, coupons: bool = True) -> str:
    today = pd.Timestamp.now().normalize()
    yesterday = today - timedelta(days=1)
    lines = [f"⚽ <b>Soccer Analyser — {today:%d.%m.%Y}</b>"]

    y = _day(engine, yesterday)
    if len(y):
        done = y[y["played"] & y["p_home"].notna() & y["hg"].notna()]
        if len(done):
            s = lambda k: f"{int(done[k].sum())}/{done[k].notna().sum()}"
            lines += ["", f"📊 <b>Вчера ({yesterday:%d.%m})</b>: исход {s('ok_outcome')} · "
                          f"тотал {s('ok_total')} · обе забьют {s('ok_btts')}"]
            try:
                from .analysis import misses, scored_live
                why = {m.match: m.reasons for m in misses(scored_live(), n=200).itertuples()}
            except Exception:
                why = {}
            for _, r in done.head(15).iterrows():
                lines.append(f"{r['checks']}  {_e(r['home'])} <b>{int(r['hg'])}:{int(r['ag'])}"
                             f"</b> {_e(r['away'])} — {_e(r['call_outcome'])}")
                reason = why.get(f"{r['home']} — {r['away']}")
                if reason and not r["ok_outcome"]:
                    lines.append(f"      ↳ <i>{_e(reason)}</i>")

    t = _day(engine, today)
    t = t[t["p_home"].notna()] if len(t) and "p_home" in t else t
    if len(t):
        lines += ["", f"📅 <b>Сегодня — {len(t)} матч(ей) с прогнозом</b>"
                      + (f" (показаны {MAX_TODAY} главных)" if len(t) > MAX_TODAY else "")]
        for comp, g in t.head(MAX_TODAY).groupby("comp_name", sort=False):
            lines.append(f"\n<b>{_e(comp)}</b>")
            for _, r in g.iterrows():
                tm = r["kickoff"].strftime("%H:%M") if pd.notna(r["kickoff"]) else "--:--"
                goals = f"{r['exp_goals']:.1f}" if pd.notna(r["exp_goals"]) else "—"
                lines.append(f"{tm} {_e(r['home'])} — {_e(r['away'])}\n"
                             f"   👉 {_e(r['call_outcome'])} | ⚽ ~{goals} | "
                             f"{_e(r['call_total'])} | обе: {_e(r['call_btts'])}")
    else:
        lines += ["", "📅 Сегодня матчей с прогнозом нет."]

    if coupons:
        try:
            from .coupons import suggest
            from .misli import events_with_model
            ev = events_with_model(engine)
            ev = ev[(ev["kickoff"] > pd.Timestamp.now(tz="UTC"))
                    & (ev["kickoff"] < pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=36))]
            cps = suggest(ev)
            if cps:
                lines += ["", "🎯 <b>Купоны (коэффициенты misli.az)</b>"]
                for cp in cps:
                    lines.append(f"\n<b>{_e(cp['title'])}</b> — коэф. <b>{cp['total_odds']:.2f}"
                                 f"</b>, шанс {cp['prob']:.0%}")
                    for _, pk in cp["picks"].iterrows():
                        lines.append(f" • {_e(pk['match'])}: {_e(pk['label'])} @ {pk['odds']:.2f}")
                lines.append("\n<i>Шанс — по коэффициентам без маржи; это не гарантия.</i>")
        except Exception as exc:  # coupons are optional
            lines.append(f"\n(купоны недоступны: {_e(exc)})")
    return "\n".join(lines)


def diagnose() -> str:
    """Human-readable Telegram status (never prints the token)."""
    token = load_config().get("TELEGRAM_BOT_TOKEN")
    if not token:
        return "telegram: no token"
    try:
        me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=30).json()
    except requests.RequestException as exc:
        return f"telegram: network error {exc}"
    if not me.get("ok"):
        return f"telegram: token rejected by Telegram ({me.get('description')}) - check the secret"
    name = me["result"].get("username")
    return (f"telegram: bot @{name} OK, chat " + ("found" if chat_id() else
            "NOT found - open t.me/" + str(name) + " and press Start (or send /start)"))


def send_daily_digest(engine) -> str:
    if not configured():
        return diagnose()
    return "telegram: sent" if send(daily_digest(engine)) else "telegram: FAILED"
