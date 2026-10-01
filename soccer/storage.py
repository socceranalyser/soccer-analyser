"""SQLite store for every prediction we make (backtest runs and live forecasts)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- 'backtest' | 'live'
    created_at TEXT NOT NULL,
    description TEXT,
    params TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL,
    model TEXT NOT NULL,
    league TEXT NOT NULL,
    season INTEGER NOT NULL,
    date TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    p_home REAL, p_draw REAL, p_away REAL,
    p_over25 REAL, xg_home REAL, xg_away REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, model, league, date, home, away)
);
CREATE TABLE IF NOT EXISTS results (
    league TEXT NOT NULL, season INTEGER NOT NULL,
    home TEXT NOT NULL, away TEXT NOT NULL,
    date TEXT NOT NULL, hg INTEGER, ag INTEGER, result TEXT,
    PRIMARY KEY (league, date, home, away)
);
"""
SCHEMA += """
CREATE TABLE IF NOT EXISTS coupons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    stake REAL, total_odds REAL, model_prob REAL,
    picks TEXT NOT NULL              -- JSON list of selections
);
"""
MAX_RESCHEDULE_DAYS = 60  # a forecast matches the result of the nearest same pairing

PRED_FIELDS = ["model", "league", "season", "date", "home", "away", "p_home", "p_draw",
               "p_away", "p_over25", "xg_home", "xg_away"]


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def save_run(run_id: str, kind: str, preds: pd.DataFrame, description: str = "",
             params: dict | None = None, replace_run: bool = True, path=DB_PATH) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    df = preds[PRED_FIELDS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["run_id"] = run_id
    df["created_at"] = now
    with closing(connect(path)) as con, con:
        con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                    (run_id, kind, now, description, json.dumps(params or {}, default=str)))
        if replace_run:
            con.execute("DELETE FROM predictions WHERE run_id = ?", (run_id,))
        cols = ["run_id"] + PRED_FIELDS + ["created_at"]
        con.executemany(
            f"INSERT OR REPLACE INTO predictions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            df[cols].astype(object).where(df[cols].notna(), None).itertuples(index=False))
    return len(df)


def save_results(matches: pd.DataFrame, path=DB_PATH) -> int:
    df = matches[["league", "season", "home", "away", "date", "hg", "ag", "result"]].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    with closing(connect(path)) as con, con:
        con.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?)",
                        df.astype(object).itertuples(index=False))
    return len(df)


def save_coupon(picks: list[dict], stake: float, total_odds: float, model_prob: float,
                path=DB_PATH) -> int:
    with closing(connect(path)) as con, con:
        cur = con.execute(
            "INSERT INTO coupons (created_at, stake, total_odds, model_prob, picks) "
            "VALUES (?,?,?,?,?)", (datetime.now().isoformat(timespec="seconds"), stake,
                                   total_odds, model_prob, json.dumps(picks, default=str)))
        return int(cur.lastrowid)


def load_coupons(path=DB_PATH) -> list[dict]:
    with closing(connect(path)) as con:
        rows = con.execute("SELECT id, created_at, stake, total_odds, model_prob, picks "
                           "FROM coupons ORDER BY id DESC").fetchall()
    return [{"id": r[0], "created_at": r[1], "stake": r[2], "total_odds": r[3],
             "model_prob": r[4], "picks": json.loads(r[5])} for r in rows]


def delete_coupon(coupon_id: int, path=DB_PATH) -> None:
    with closing(connect(path)) as con, con:
        con.execute("DELETE FROM coupons WHERE id = ?", (coupon_id,))


def prematch_forecasts(dates, path=DB_PATH) -> pd.DataFrame:
    """Headline ('final') forecasts stored before kick-off for the given dates."""
    days = sorted({pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates})
    with closing(connect(path)) as con:
        df = pd.read_sql(
            f"SELECT league, date, home, away, p_home, p_draw, p_away FROM predictions "
            f"WHERE run_id = 'live' AND model = 'final' AND date IN ({','.join('?' * len(days))})",
            con, params=days)
    return df


def find_result(home: str, away: str, date, days: int = 2, path=DB_PATH):
    """(hg, ag) of a finished match between home and away around `date`, else None."""
    d = pd.Timestamp(date)
    lo, hi = (d - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), \
        (d + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    with closing(connect(path)) as con:
        r = con.execute("SELECT hg, ag FROM results WHERE home = ? AND away = ? AND date "
                        "BETWEEN ? AND ? LIMIT 1", (home, away, lo, hi)).fetchone()
    return None if r is None or r[0] is None else (int(r[0]), int(r[1]))


def load_runs(path=DB_PATH) -> pd.DataFrame:
    with closing(connect(path)) as con:
        return pd.read_sql("SELECT * FROM runs ORDER BY created_at DESC", con)


def load_predictions(run_id: str | None = None, kind: str | None = None,
                     path=DB_PATH) -> pd.DataFrame:
    """Predictions joined with actual results (NaN result = not played yet).

    A forecast is matched to the same pairing's result on the nearest date (within
    MAX_RESCHEDULE_DAYS), so postponed matches still get scored.
    """
    q = "SELECT p.*, runs.kind FROM predictions p JOIN runs USING (run_id) WHERE 1=1"
    args = []
    if run_id:
        q += " AND p.run_id = ?"
        args.append(run_id)
    if kind:
        q += " AND runs.kind = ?"
        args.append(kind)
    with closing(connect(path)) as con:
        preds = pd.read_sql(q, con, params=args)
        res = pd.read_sql("SELECT league, home, away, date AS r_date, hg, ag, result "
                          "FROM results", con)
    preds["date"] = pd.to_datetime(preds["date"])
    res["r_date"] = pd.to_datetime(res["r_date"])
    preds = preds.reset_index(names="_row")
    m = preds.merge(res, on=["league", "home", "away"], how="inner")
    m["_gap"] = (m["r_date"] - m["date"]).abs().dt.days
    m = m[m["_gap"] <= MAX_RESCHEDULE_DAYS].sort_values("_gap").drop_duplicates("_row")
    out = preds.merge(m[["_row", "hg", "ag", "result"]], on="_row", how="left")
    return out.drop(columns="_row")
