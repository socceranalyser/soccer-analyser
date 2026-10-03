"""Automatic error analysis of live forecasts — the feedback loop that drives improvement.

* miss_reasons(): why a single call failed (upset, only the model wrong, red card, goals).
* findings(): systematic problems over all scored live forecasts, each with a significance
  test so that noise is not mistaken for a pattern; every finding says what to do.
* fit_live_calibration(): self-correction — refits a calibration map per competition type on
  the model's own live mistakes and keeps it only if it beats the raw forecasts on the most
  recent matches (time split, never on the data it was fitted on).
* weekly_report(): Markdown for data/state/analysis.md and the Telegram weekly message.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import storage
from .calibration import VectorScaling
from .config import LEAGUES, ROOT
from .metrics import log_loss
from .verdicts import outcome_call, result_index, score_calls

STATE = ROOT / "data" / "state"
REPORT = STATE / "analysis.md"
LIVE_CAL = STATE / "live_calibration.json"
CUPS = {"UCL", "UEL", "UECL"}
C = ["p_home", "p_draw", "p_away"]
MIN_PATTERN = 30      # matches before a pattern is even looked at
MIN_SELF_CAL = 200    # matches before self-calibration is attempted


def category(code: str) -> str:
    if code in LEAGUES:
        return "Лиги"
    return "Еврокубки" if code in CUPS else "Сборные"


# ------------------------------------------------------------------- data
def scored_live() -> pd.DataFrame:
    """Live 'final' forecasts with results, plus the bookmaker's view where it was stored."""
    p = storage.load_predictions("live")
    p = p[p["result"].notna()]
    fin = p[p["model"] == "final"].copy()
    if fin.empty:
        return fin
    mk = p[p["model"] == "market"][["league", "date", "home", "away"] + C].rename(
        columns={c: c.replace("p_", "q_") for c in C})
    fin = fin.merge(mk.drop_duplicates(["league", "date", "home", "away"]),
                    on=["league", "date", "home", "away"], how="left")
    fin["cat"] = fin["league"].map(category)
    fin["y"] = [result_index(int(h), int(a)) for h, a in zip(fin["hg"], fin["ag"])]
    fin["goals"] = fin["hg"] + fin["ag"]
    return fin.sort_values("date").reset_index(drop=True)


def red_cards(fin: pd.DataFrame) -> pd.Series:
    """Red cards per match from the livescore archive (0 when unknown)."""
    path = ROOT / "data" / "livescore_results.csv"
    out = pd.Series(0, index=fin.index)
    if not path.exists() or fin.empty:
        return out
    arc = pd.read_csv(path)
    if "red_h" not in arc:
        return out
    from .euro import _norm
    from .names import similarity
    arc["day"] = pd.to_datetime(arc["kickoff"], utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()
    arc["h"], arc["a"] = arc["home_raw"].map(_norm), arc["away_raw"].map(_norm)
    for i, r in fin.iterrows():
        cand = arc[(arc["day"] - r["date"]).abs() <= pd.Timedelta(days=1)]
        if cand.empty:
            continue
        s = cand.apply(lambda c: min(similarity(_norm(r["home"]), c["h"]),
                                     similarity(_norm(r["away"]), c["a"])), axis=1)
        if s.max() >= 0.75:
            m = cand.loc[s.idxmax()]
            out.at[i] = int((m.get("red_h") or 0) + (m.get("red_a") or 0))
    return out


# ------------------------------------------------------------ single misses
def miss_reasons(r, reds: int = 0) -> list[str]:
    """Plain-language reasons for a failed outcome call."""
    p = np.array([r["p_home"], r["p_draw"], r["p_away"]], float)
    y = int(r["y"])
    reasons = []
    has_q = pd.notna(r.get("q_home"))
    q = np.array([r["q_home"], r["q_draw"], r["q_away"]], float) if has_q else None
    if p[y] < 0.22 and (q is None or q[y] < 0.25):
        reasons.append(f"🎲 Сенсация: такой исход ждали редко (модель {p[y]:.0%}"
                       + (f", букмекер {q[y]:.0%})" if q is not None else ")"))
    if q is not None:
        _, cov_q, _ = outcome_call(q, "", "")
        if y in cov_q:
            reasons.append("📉 Ошиблась только модель — букмекер был прав (модель чего-то не знала: "
                           "состав, травмы, мотивация)")
        else:
            reasons.append("🤝 Букмекер ошибся так же")
    if reds:
        reasons.append(f"🟥 Красная карточка в матче ({reds}) — игра изменилась по ходу")
    xg = (r.get("xg_home") or np.nan) + (r.get("xg_away") or np.nan)
    if pd.notna(xg) and abs(r["goals"] - xg) >= 2.5:
        reasons.append(f"⚽ Голов {int(r['goals'])} при ожидаемых {xg:.1f}")
    if not reasons:
        reasons.append("Исход был вполне возможен — обычная вариативность футбола")
    return reasons


def misses(fin: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    if fin.empty:
        return fin
    reds = red_cards(fin)
    rows = []
    for i, r in fin.iterrows():
        sc = score_calls(r[C].to_numpy(float), r.get("p_over25"), r.get("p_btts"),
                         int(r["hg"]), int(r["ag"]))
        if sc["outcome"]:
            continue
        lab, _, pr = outcome_call(r[C].to_numpy(float), r["home"], r["away"])
        rows.append({"date": r["date"], "league": r["league"], "match": f"{r['home']} — {r['away']}",
                     "score": f"{int(r['hg'])}:{int(r['ag'])}", "call": f"{lab} · {pr:.0%}",
                     "reasons": " · ".join(miss_reasons(r, int(reds.at[i]))),
                     "only_model": "Ошиблась только модель" in " ".join(miss_reasons(r, 0)),
                     "upset": "Сенсация" in " ".join(miss_reasons(r, 0))})
    return pd.DataFrame(rows).sort_values("date", ascending=False).head(n) if rows else pd.DataFrame()


# --------------------------------------------------------------- patterns
def _z(observed: float, expected: float, n: int) -> float:
    se = np.sqrt(max(expected * (1 - expected), 1e-6) / n)
    return (observed - expected) / se


def findings(fin: pd.DataFrame) -> list[dict]:
    """Systematic issues. severity: 'ok' | 'watch' | 'act'."""
    out = []
    n = len(fin)
    if n < MIN_PATTERN:
        return [{"severity": "watch", "title": "Мало данных",
                 "text": f"Сыграно {n} матчей с живыми прогнозами. Закономерности ищутся от "
                         f"{MIN_PATTERN} матчей в группе; надёжные выводы — от 200–300.",
                 "todo": None}]
    for cat, g in fin.groupby("cat"):
        if len(g) < MIN_PATTERN:
            continue
        calls = [outcome_call(r, "", "") for r in g[C].to_numpy(float)]
        hit = np.array([y in c[1] for c, y in zip(calls, g["y"])])
        promised = float(np.mean([c[2] for c in calls]))
        z = _z(hit.mean(), promised, len(g))
        sev = "act" if abs(z) >= 2.5 else "watch" if abs(z) >= 1.8 else "ok"
        out.append({"severity": sev, "title": f"{cat}: точность «кто выиграет»",
                    "text": f"обещано {promised:.0%}, сбылось {hit.mean():.0%} ({len(g)} матчей, z={z:+.1f})",
                    "todo": ("Модель переуверена в этой группе → самокалибровка / пересмотр модели"
                             if z <= -2.5 else None)})
        dz = _z((g["y"] == 1).mean(), g["p_draw"].mean(), len(g))
        if abs(dz) >= 1.8:
            out.append({"severity": "act" if abs(dz) >= 2.5 else "watch",
                        "title": f"{cat}: ничьи",
                        "text": f"модель ждала {g['p_draw'].mean():.0%} ничьих, было {(g['y'] == 1).mean():.0%} "
                                f"(z={dz:+.1f})",
                        "todo": "Поправить вероятность ничьих (калибровка / параметр rho)" if abs(dz) >= 2.5 else None})
        o = g.dropna(subset=["p_over25"])
        if len(o) >= MIN_PATTERN:
            oz = _z((o["goals"] > 2.5).mean(), o["p_over25"].mean(), len(o))
            xg = (o["xg_home"] + o["xg_away"]).mean()
            if abs(oz) >= 1.8:
                out.append({"severity": "act" if abs(oz) >= 2.5 else "watch",
                            "title": f"{cat}: тотал 2.5",
                            "text": f"ТБ2.5 ждали в {o['p_over25'].mean():.0%}, было {(o['goals'] > 2.5).mean():.0%}; "
                                    f"голов в среднем {o['goals'].mean():.2f} при ожидаемых {xg:.2f} (z={oz:+.1f})",
                            "todo": "Перенастроить модель голов этой группы" if abs(oz) >= 2.5 else None})
        b = g.dropna(subset=["q_home"])
        if len(b) >= MIN_PATTERN:
            ll_m = log_loss(b[C].to_numpy(float), b["y"].to_numpy())
            ll_q = log_loss(b[["q_home", "q_draw", "q_away"]].to_numpy(float), b["y"].to_numpy())
            out.append({"severity": "watch" if ll_m > ll_q else "ok",
                        "title": f"{cat}: модель против букмекера",
                        "text": f"ошибка (log loss) модель {ll_m:.3f} vs букмекер {ll_q:.3f} на {len(b)} матчах — "
                                + ("букмекер точнее" if ll_m > ll_q else "модель не хуже букмекера"),
                        "todo": None})
    for lg, g in fin.groupby("league"):
        if len(g) < MIN_PATTERN or category(lg) != "Лиги":
            continue
        calls = [outcome_call(r, "", "") for r in g[C].to_numpy(float)]
        hit = np.array([y in c[1] for c, y in zip(calls, g["y"])])
        promised = float(np.mean([c[2] for c in calls]))
        z = _z(hit.mean(), promised, len(g))
        if z <= -2.5:
            out.append({"severity": "act", "title": f"Лига {LEAGUES[lg]['name']}",
                        "text": f"обещано {promised:.0%}, сбылось {hit.mean():.0%} ({len(g)} матчей)",
                        "todo": f"Разобрать {lg}: состав лиги, новички, параметры DC"})
    ms = misses(fin, n=10_000)
    if len(ms):
        share_upset = ms["upset"].mean()
        share_model = ms["only_model"].mean()
        out.append({"severity": "act" if share_model > 0.5 and len(ms) >= 40 else "ok",
                    "title": "Причины промахов",
                    "text": f"{len(ms)} промахов: сенсации {share_upset:.0%}, ошиблась только модель "
                            f"{share_model:.0%}",
                    "todo": ("Больше половины промахов — там, где букмекер был прав: модели не хватает "
                             "информации (составы, важность игроков, мотивация)"
                             if share_model > 0.5 and len(ms) >= 40 else None)})
    return out


# ------------------------------------------------------- self-correction
def fit_live_calibration() -> dict:
    """Per category: fit VectorScaling on older live matches, keep it only if it lowers the
    log loss on the newest 30% (time split). Saved to data/state/live_calibration.json."""
    fin = scored_live()
    result = {}
    for cat, g in fin.groupby("cat") if len(fin) else []:
        if len(g) < MIN_SELF_CAL:
            result[cat] = {"status": f"ждём данных ({len(g)}/{MIN_SELF_CAL})"}
            continue
        cut = int(len(g) * 0.7)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        cal = VectorScaling(l2=5.0).fit(tr[C].to_numpy(float), tr["y"].to_numpy())
        raw = log_loss(te[C].to_numpy(float), te["y"].to_numpy())
        new = log_loss(cal.transform(te[C].to_numpy(float)), te["y"].to_numpy())
        if new < raw - 0.002:
            final = VectorScaling(l2=5.0).fit(g[C].to_numpy(float), g["y"].to_numpy())
            result[cat] = {"status": "включена", "params": final.params.tolist(),
                           "check": f"{raw:.4f} -> {new:.4f}", "n": len(g)}
        else:
            result[cat] = {"status": "не нужна", "check": f"{raw:.4f} -> {new:.4f}", "n": len(g)}
    STATE.mkdir(parents=True, exist_ok=True)
    LIVE_CAL.write_text(json.dumps({"fitted": datetime.now().isoformat(timespec="seconds"),
                                    "categories": result}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return result


def live_calibrator(cat: str):
    try:
        d = json.loads(LIVE_CAL.read_text(encoding="utf-8"))["categories"].get(cat, {})
    except (OSError, ValueError, KeyError):
        return None
    if d.get("status") != "включена":
        return None
    cal = VectorScaling(l2=5.0)
    cal.params = np.array(d["params"])
    return cal


# -------------------------------------------- daily self-correction of goals
LIVE_GOALS = STATE / "live_goals.json"
GOALS_SINCE = "2026-10-04"   # base goals calibration introduced; older forecasts are raw
GOALS_PRIOR_N = 300          # an offset needs ~300 matches of evidence to move fully


def _goals_state() -> dict:
    try:
        return json.loads(LIVE_GOALS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"history": []}


def live_goals_offset(cat: str, at: str | None = None) -> float:
    """Extra logit shift of over-2.5 learnt from live mistakes, in force at time `at` (ISO)."""
    hist = [h for h in _goals_state().get("history", []) if at is None or h["at"] <= at]
    return float(hist[-1]["offset"].get(cat, 0.0)) if hist else 0.0


def fit_live_goals() -> dict:
    """Daily: per category, the logit offset that best explains finished live totals
    (MAP, Gaussian prior worth GOALS_PRIOR_N matches). Each forecast is first "un-shifted"
    by the offset that was in force when it was made, so the loop never double-counts."""
    from scipy.optimize import minimize_scalar
    fin = scored_live()
    st = _goals_state()
    now = datetime.now().isoformat(timespec="seconds")
    out = {}
    if len(fin):
        fin = fin[fin["p_over25"].notna() & (fin["created_at"].str[:10] >= GOALS_SINCE)]
    for cat, g in fin.groupby("cat") if len(fin) else []:
        p = g["p_over25"].astype(float).clip(1e-4, 1 - 1e-4).to_numpy()
        used = np.array([live_goals_offset(cat, d) for d in g["created_at"]])
        base = np.log(p / (1 - p)) - used
        y = (g["goals"] > 2.5).to_numpy(float)
        var = 1 / (GOALS_PRIOR_N * 0.25)

        def obj(o):
            q = 1 / (1 + np.exp(-(base + o)))
            return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)) + 0.5 * o * o / var
        out[cat] = round(float(minimize_scalar(obj, bounds=(-1, 1), method="bounded").x), 4)
    hist = [h for h in st.get("history", []) if "at" in h]
    hist.append({"at": now, "offset": out,
                 "n": {c: int((fin["cat"] == c).sum()) for c in out} if len(fin) else {}})
    STATE.mkdir(parents=True, exist_ok=True)
    LIVE_GOALS.write_text(json.dumps({"history": hist[-400:]}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return out


def daily_learning() -> dict:
    """Everything that learns from yesterday's results, run every day before forecasting."""
    return {"calibration": {k: v["status"] for k, v in fit_live_calibration().items()},
            "goals_offset": fit_live_goals()}


# ------------------------------------------------------------------ report
def weekly_report() -> str:
    fin = scored_live()
    fs = findings(fin)
    cal = fit_live_calibration()
    icon = {"act": "🔴", "watch": "🟡", "ok": "🟢"}
    lines = [f"# Автоанализ прогнозов — {datetime.now():%d.%m.%Y}", "",
             f"Матчей с живыми прогнозами и результатом: **{len(fin)}**", "", "## Находки"]
    for f in fs:
        lines.append(f"- {icon[f['severity']]} **{f['title']}** — {f['text']}")
    todos = [f["todo"] for f in fs if f.get("todo")]
    lines += ["", "## Задачи на улучшение (берёт Claude в начале сессии)"]
    lines += [f"- [ ] {t}" for t in todos] or ["- нет срочных задач"]
    lines += ["", "## Самокалибровка"]
    lines += [f"- {k}: {v['status']}" + (f" ({v['check']})" if "check" in v else "")
              for k, v in cal.items()] or ["- ждём данных"]
    ms = misses(fin, n=15)
    if len(ms):
        lines += ["", "## Последние промахи"]
        lines += [f"- {r.date:%d.%m} {r.match} {r.score} — {r.call}: {r.reasons}"
                  for r in ms.itertuples()]
    text = "\n".join(lines)
    STATE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    return text
