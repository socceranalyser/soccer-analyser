"""Plain-language calls from probabilities: who wins, over/under 2.5, both teams to score.

Each call names the outcome(s) it covers, so it can be scored against the final result.
When nobody is a clear favourite the outcome call falls back to a double chance
("X will not lose"), which is what a careful tipster would say.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLEAR_FAVOURITE = 0.50  # below this a single outcome is a coin-flip -> double chance


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
        lab, _, pr = outcome_call(p, r["home"], r["away"])
        cols["call_outcome"].append(f"{lab} · {pr:.0%}")
        po, pb = r.get("p_over25"), r.get("p_btts")
        cols["call_total"].append("—" if pd.isna(po) else "{} · {:.0%}".format(*total_call(po)[::2]))
        cols["call_btts"].append("—" if pd.isna(pb) else "{} · {:.0%}".format(*btts_call(pb)[::2]))
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
