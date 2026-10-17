"""Plain-language calls from probabilities: who wins, over/under 2.5, both teams to score.

Each call names the outcome(s) it covers, so it can be scored against the final result.
When nobody is a clear favourite the outcome call falls back to a double chance
("X will not lose"), which is what a careful tipster would say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# "Win X" only for a clear favourite, otherwise the double chance "X won't lose".
# bt_all 2023-25 (17 292 calls): a 50-55% "win" came true 51.1% (a coin flip; the double chance
# 77.6%), 55-60% 57.7% (82.0%), 60-65% 64.3%. Raising 0.50 -> 0.60: calls that come true
# 67.0% -> 72.8% on history, 69.2% -> 74.1% on live games (2026-10-17, after a 15:30 block
# where 52-53% "wins" like Millwall and Luton failed).
CLEAR_FAVOURITE = 0.60
TOSS_UP = 0.58          # totals / BTTS below this are flagged "почти 50/50" (weak lean only)


def _toss(p: float) -> str:
    return " ⚖️" if p < TOSS_UP else ""  # almost 50/50 (explained under the table)


def outcome_call(p, home: str, away: str) -> tuple[str, frozenset, float]:
    """(label, covered results {0=home,1=draw,2=away}, probability of the call)."""
    p = np.asarray(p, float)
    k = int(np.argmax(p))
    if p[k] >= CLEAR_FAVOURITE and k != 1:
        return f"Победа {home if k == 0 else away}", frozenset({k}), float(p[k])
    hx, xa = p[0] + p[1], p[1] + p[2]
    if hx >= xa:
        return f"{home} не проиграет", frozenset({0, 1}), float(hx)
    return f"{away} не проиграет", frozenset({1, 2}), float(xa)


# "Almost sure": the safest 1X2 call that BOTH our model and the bookmaker rate >= 85%.
# 2023-25 league history (bt_all, 3 088 calls, ~3 per match day): came true 92.4%
# (>= 88%: 94.6%) - but such odds are ~1.03-1.10, so they are for confidence, not for profit.
SURE = 0.85


def sure_call(r, home: str, away: str):
    """(label, model p, market p, covered results) of the safest call, or None."""
    if pd.isna(r.get("q_home")) or pd.isna(r.get("p_home")):
        return None
    p = np.array([r["p_home"], r["p_draw"], r["p_away"]], float)
    q = np.array([r["q_home"], r["q_draw"], r["q_away"]], float)
    opts = [(f"Победа {home}", {0}), (f"Победа {away}", {2}),
            (f"{home} не проиграет", {0, 1}), (f"{away} не проиграет", {1, 2})]
    best = max(opts, key=lambda o: min(p[list(o[1])].sum(), q[list(o[1])].sum()))
    pm, pq = p[list(best[1])].sum(), q[list(best[1])].sum()
    return (best[0], float(pm), float(pq), frozenset(best[1])) if min(pm, pq) >= SURE else None


def total_call(p_over: float, line: float = 2.5) -> tuple[str, bool, float]:
    """(label, True if the call is 'over', probability)."""
    return (f"Больше {line:g}", True, p_over) if p_over >= 0.5 else \
        (f"Меньше {line:g}", False, 1 - p_over)


def btts_call(p_btts: float) -> tuple[str, bool, float]:
    return ("Да", True, p_btts) if p_btts >= 0.5 else ("Нет", False, 1 - p_btts)


def result_index(hg: int, ag: int) -> int:
    return 0 if hg > ag else 1 if hg == ag else 2


def score_calls(p, p_over, p_btts, hg, ag, line: float = 2.5) -> dict:
    """Which of the three calls came true for a final score."""
    _, covered, _ = outcome_call(p, "", "")
    out = {"outcome": result_index(hg, ag) in covered}
    if p_over is not None and not np.isnan(p_over):
        out["total"] = (hg + ag > line) == total_call(p_over, line)[1]
    if p_btts is not None and not np.isnan(p_btts):
        out["btts"] = (hg > 0 and ag > 0) == btts_call(p_btts)[1]
    return out


# model and bookmaker 10+ points apart on the called outcome: in live games the bookmaker was
# the more accurate one in such disagreements -> the verdict carries a warning
DISAGREE = 0.10


def calls_columns(view: pd.DataFrame) -> dict:
    """Plain-language calls for each match plus ✅/❌ once it is finished."""
    cols = {k: [] for k in ("call_outcome", "call_total", "call_btts", "exp_goals", "checks",
                            "ok_outcome", "ok_total", "ok_btts")}
    for _, r in view.iterrows():
        if pd.isna(r.get("p_home")):
            for k in cols:
                cols[k].append("—" if k.startswith("call") or k == "checks" else None)
            cols["checks"][-1] = ""
            continue
        p = [r["p_home"], r["p_draw"], r["p_away"]]
        lab, cov, pr = outcome_call(p, r["home"], r["away"])
        warn = ""
        if pd.notna(r.get("q_home")):
            q = [r["q_home"], r["q_draw"], r["q_away"]]
            qr = sum(q[i] for i in cov)
            if abs(qr - pr) >= DISAGREE:
                warn = f" ⚠️ букмекер {qr:.0%}"
        lu = " 📋" if "учтены" in str(r.get("lineup_note") or "") else ""  # line-ups counted
        cols["call_outcome"].append(f"{'🏆' if len(cov) == 1 else '🛡️'} {lab} · {pr:.0%}{warn}{lu}")
        po, pb = r.get("p_over25"), r.get("p_btts")
        if pd.isna(po):
            cols["call_total"].append("—")
        else:
            tl, over, tp = total_call(po)
            cols["call_total"].append(f"{'⬆️' if over else '⬇️'} {tl} · {tp:.0%}{_toss(tp)}")
        if pd.isna(pb):
            cols["call_btts"].append("—")
        else:
            bl, yes, bp = btts_call(pb)
            cols["call_btts"].append(f"{'⚽' if yes else '🚫'} {bl} · {bp:.0%}{_toss(bp)}")
        xg = (r.get("xg_home") or np.nan) + (r.get("xg_away") or np.nan)
        cols["exp_goals"].append(None if pd.isna(xg) else round(float(xg), 1))
        if r["played"] and pd.notna(r["hg"]):
            sc = score_calls(p, po, pb, int(r["hg"]), int(r["ag"]))
            mark = lambda k: "✅" if sc.get(k) else ("❌" if k in sc else "·")
            cols["checks"].append(f"{mark('outcome')} {mark('total')} {mark('btts')}")
            cols["ok_outcome"].append(sc["outcome"])
            cols["ok_total"].append(sc.get("total"))
            cols["ok_btts"].append(sc.get("btts"))
        else:
            cols["checks"].append("")
            for k in ("ok_outcome", "ok_total", "ok_btts"):
                cols[k].append(None)
    return cols
