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
    p_over25 REAL, xg_home REAL, xg_away REAL, p_btts REAL,
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
               "p_away", "p_over25", "xg_home", "xg_away", "p_btts"]


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
    if "p_btts" not in cols:  # migration for databases created before BTTS was stored
        con.execute("ALTER TABLE predictions ADD COLUMN p_btts REAL")
    return con


def save_run(run_id: str, kind: str, preds: pd.DataFrame, description: str = "",
             params: dict | None = None, replace_run: bool = True, path=DB_PATH) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    df = preds.reindex(columns=PRED_FIELDS).copy()
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
            f"SELECT league, date, home, away, p_home, p_draw, p_away, p_over25, p_btts, "
            f"xg_home, xg_away FROM predictions "
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


# --------------------------------------------------------- cloud state sync
# In the cloud the SQLite file lives on a throw-away disk, so the small, valuable part of
# it (live forecasts, coupons) is kept as text files in the repository (data/state/) and
# the large, static part (backtests) as one compressed CSV.
STATE_DIR = DB_PATH.parent / "state"


def export_state(state_dir=STATE_DIR, path=DB_PATH, include_backtests: bool = False) -> list:
    state_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with closing(connect(path)) as con:
        runs = pd.read_sql("SELECT * FROM runs", con)
        live = pd.read_sql("SELECT * FROM predictions WHERE run_id = 'live'", con)
        coupons = pd.read_sql("SELECT * FROM coupons", con)
        runs.to_csv(state_dir / "runs.csv", index=False)
        live.to_csv(state_dir / "live_predictions.csv", index=False)
        coupons.to_csv(state_dir / "coupons.csv", index=False)
        written += ["runs.csv", "live_predictions.csv", "coupons.csv"]
        bt = state_dir / "backtests.csv.gz"
        if include_backtests or not bt.exists():
            pd.read_sql("SELECT * FROM predictions WHERE run_id != 'live'", con).to_csv(
                bt, index=False, compression="gzip")
            written.append(bt.name)
    return written


def import_state(state_dir=STATE_DIR, path=DB_PATH) -> dict:
    """Load data/state/* into the SQLite file (missing rows only; safe to call often)."""
    counts = {}
    if not state_dir.exists():
        return counts
    with closing(connect(path)) as con, con:
        have_bt = con.execute("SELECT COUNT(*) FROM predictions WHERE run_id != 'live'").fetchone()[0]
        for name, table, cond in (("runs.csv", "runs", None),
                                  ("live_predictions.csv", "predictions", None),
                                  ("backtests.csv.gz", "predictions", "bt"),
                                  ("coupons.csv", "coupons", None)):
            f = state_dir / name
            if not f.exists() or (cond == "bt" and have_bt):
                continue
            df = pd.read_csv(f)
            if df.empty:
                continue
            cols = [c for c in df.columns
                    if c in {r[1] for r in con.execute(f"PRAGMA table_info({table})")}]
            con.executemany(
                f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                df[cols].astype(object).where(df[cols].notna(), None).itertuples(index=False))
            counts[name] = len(df)
    return counts


# ------------------------------------------------------------ coupon draft
SCHEMA_DRAFT = """CREATE TABLE IF NOT EXISTS coupon_draft (
    id INTEGER PRIMARY KEY CHECK (id = 1), updated_at TEXT NOT NULL, picks TEXT NOT NULL)"""


def save_draft(picks: list[dict], path=DB_PATH) -> None:
    """The coupon being built, kept across page reloads and restarts."""
    with closing(connect(path)) as con, con:
        con.execute(SCHEMA_DRAFT)
        con.execute("INSERT OR REPLACE INTO coupon_draft VALUES (1, ?, ?)",
                    (datetime.now().isoformat(timespec="seconds"),
                     json.dumps(picks, default=str)))


def load_draft(path=DB_PATH) -> list[dict]:
    with closing(connect(path)) as con:
        con.execute(SCHEMA_DRAFT)
        r = con.execute("SELECT picks FROM coupon_draft WHERE id = 1").fetchone()
    return json.loads(r[0]) if r else []
