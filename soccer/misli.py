"""Pre-match football odds from misli.az (public JSON used by the site itself).

Each event is linked to our models when possible:
  * national teams (Azerbaijani names translated to English),
  * European cups (club names matched strictly against all known clubs),
  * domestic matches (both clubs found in the same league of that country this season;
    clubs from different leagues only in cup competitions).
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, LEAGUES
from .names import CLUB_ALIASES, best_match

API = "https://apivx.misli.az/api/web/v1/sportsbook"
HEADERS = {"User-Agent": "Mozilla/5.0 (soccer-analyser personal use)", "Accept": "application/json",
           "Origin": "https://www.misli.az", "Referer": "https://www.misli.az/"}
CACHE_SECONDS = 600  # be polite: one download per 10 minutes at most

# misli country ids -> our country names
COUNTRY = {
    "ENG": "Англия", "SCO": "Шотландия", "DE": "Германия", "IT": "Италия", "ES": "Испания",
    "FR": "Франция", "NL": "Нидерланды", "BE": "Бельгия", "PT": "Португалия", "TR": "Турция",
    "GR": "Греция", "AR": "Аргентина", "AT": "Австрия", "BR": "Бразилия", "CN": "Китай",
    "DK": "Дания", "FI": "Финляндия", "IE": "Ирландия", "JP": "Япония", "MX": "Мексика",
    "NO": "Норвегия", "PL": "Польша", "RO": "Румыния", "RU": "Россия", "SE": "Швеция",
    "CH": "Швейцария", "US": "США",
}
# competitions that are not senior men's football
EXCLUDE = re.compile(r"qad[ıi]n|u\d{2}|rezerv|\(q\)|olimp|asiya oyunlar|futzal|amator|həvəskar"
                     r"|\bII\b|\bB$|\s2$", re.IGNORECASE)  # women, youth, reserve/B teams
CUP_WORDS = re.compile(r"kubo|kuboku|cup|pokal|coppa|copa|taça|coupe", re.IGNORECASE)
EURO_CUPS = {"çempionlar liqası": "UCL", "avropa liqası": "UEL", "konfrans liqası": "UECL"}

# Azerbaijani -> English (martj42 spelling) national-team names
NATIONS_AZ = {
    "Almaniya": "Germany", "Serbiya": "Serbia", "İspaniya": "Spain", "Xorvatiya": "Croatia",
    "İngiltərə": "England", "Çexiya": "Czech Republic", "İtaliya": "Italy", "Türkiyə": "Turkey",
    "Rumıniya": "Romania", "İsveç": "Sweden", "Fransa": "France", "Belçika": "Belgium",
    "İslandiya": "Iceland", "Bolqarıstan": "Bulgaria", "Portuqaliya": "Portugal",
    "Norveç": "Norway", "Belarus": "Belarus", "San Marino": "San Marino",
    "Niderland": "Netherlands", "Hollandiya": "Netherlands", "Bosniya və Herseqovina":
    "Bosnia and Herzegovina", "Polşa": "Poland", "İsveçrə": "Switzerland",
    "Sloveniya": "Slovenia", "Estoniya": "Estonia", "Lüksemburq": "Luxembourg",
    "Şimali İrlandiya": "Northern Ireland", "Gürcüstan": "Georgia", "Ukrayna": "Ukraine",
    "Macarıstan": "Hungary", "Finlandiya": "Finland", "Albaniya": "Albania", "Kosovo": "Kosovo",
    "Avstriya": "Austria", "Yunanıstan": "Greece", "İrlandiya": "Republic of Ireland",
    "İsrail": "Israel", "Şimali Makedoniya": "North Macedonia", "Şotlandiya": "Scotland",
    "Uels": "Wales", "Danimarka": "Denmark", "Farer Adaları": "Faroe Islands",
    "Slovakiya": "Slovakia", "Malta": "Malta", "Cəbəllüttariq": "Gibraltar",
    "Azərbaycan": "Azerbaijan", "Lixtenşteyn": "Liechtenstein", "Litva": "Lithuania",
    "Latviya": "Latvia", "Moldova": "Moldova", "Moldova Respublikası": "Moldova",
    "Ermənistan": "Armenia", "Qazaxıstan": "Kazakhstan", "Qazaxstan": "Kazakhstan",
    "Bosniya-Herseqovina": "Bosnia and Herzegovina", "Çexiya Respublikası": "Czech Republic",
    "Şimali Makedoniya Respublikası": "North Macedonia", "Rusiya Federasiyası": "Russia",
    "Andorra": "Andorra", "Monteneqro": "Montenegro", "Kipr": "Cyprus", "Rusiya": "Russia",
    "Argentina": "Argentina", "Braziliya": "Brazil", "Uruqvay": "Uruguay", "Kolumbiya": "Colombia",
    "Çili": "Chile", "Peru": "Peru", "Ekvador": "Ecuador", "Paraqvay": "Paraguay",
    "Boliviya": "Bolivia", "Venesuela": "Venezuela", "ABŞ": "United States", "Meksika": "Mexico",
    "Kanada": "Canada", "Kosta-Rika": "Costa Rica", "Panama": "Panama", "Honduras": "Honduras",
    "Yamayka": "Jamaica", "Haiti": "Haiti", "Kuba": "Cuba", "Qvatemala": "Guatemala",
    "Nikaraqua": "Nicaragua", "El Salvador": "El Salvador", "Kürasao": "Curaçao",
    "Trinidad və Tobaqo": "Trinidad and Tobago", "Dominikan Respublikası": "Dominican Republic",
    "Sent-Kits və Nevis": "Saint Kitts and Nevis", "Qrenada": "Grenada", "Qayana": "Guyana",
    "Dominika": "Dominica", "Kayman Adaları": "Cayman Islands", "Puerto Riko": "Puerto Rico",
    "Boneyr": "Bonaire", "Surinam": "Suriname", "Bermud Adaları": "Bermuda",
    "Barbados": "Barbados", "Martinika": "Martinique", "Montserrat": "Montserrat",
    "Qvadelupa": "Guadeloupe", "Saint-Martin": "Saint Martin", "Sent-Lüsiya": "Saint Lucia",
    "Vircin Adaları": "United States Virgin Islands",
    "Vircin Adaları, Britaniya": "British Virgin Islands", "Salvador": "El Salvador",
    "Belize": "Belize", "Aruba": "Aruba", "Antiqua və Barbuda": "Antigua and Barbuda",
    "Baham Adaları": "Bahamas", "Sent-Vinsent və Qrenadinlər": "Saint Vincent and the Grenadines",
    "Yaponiya": "Japan", "Cənubi Koreya": "South Korea", "Koreya Respublikası": "South Korea",
    "Şimali Koreya": "North Korea", "Çin": "China", "Avstraliya": "Australia", "İran": "Iran",
    "Səudiyyə Ərəbistanı": "Saudi Arabia", "Qətər": "Qatar", "BƏƏ": "United Arab Emirates",
    "İraq": "Iraq", "İordaniya": "Jordan", "Özbəkistan": "Uzbekistan", "Hindistan": "India",
    "Tayland": "Thailand", "Vyetnam": "Vietnam", "İndoneziya": "Indonesia", "Malayziya": "Malaysia",
    "Filippin": "Philippines", "Pakistan": "Pakistan", "Oman": "Oman", "Bəhreyn": "Bahrain",
    "Küveyt": "Kuwait", "Suriya": "Syria", "Livan": "Lebanon", "Fələstin": "Palestine",
    "Qırğızıstan": "Kyrgyzstan", "Tacikistan": "Tajikistan", "Türkmənistan": "Turkmenistan",
    "Yeni Zelandiya": "New Zealand", "Mərakeş": "Morocco", "Əlcəzair": "Algeria", "Tunis": "Tunisia",
    "Misir": "Egypt", "Nigeriya": "Nigeria", "Seneqal": "Senegal", "Kamerun": "Cameroon",
    "Qana": "Ghana", "Kot-d'İvuar": "Ivory Coast", "Fil Dişi Sahili": "Ivory Coast",
    "Mali": "Mali", "Burkina-Faso": "Burkina Faso", "Cənubi Afrika": "South Africa",
    "CAR": "South Africa", "Konqo DR": "DR Congo", "Kabo-Verde": "Cape Verde", "Qvineya": "Guinea",
    "Zambiya": "Zambia", "Uqanda": "Uganda", "Keniya": "Kenya", "Tanzaniya": "Tanzania",
    "Benin": "Benin", "Qabon": "Gabon", "Ekvatorial Qvineya": "Equatorial Guinea",
    "Anqola": "Angola", "Mozambik": "Mozambique", "Zimbabve": "Zimbabwe", "Liviya": "Libya",
    "Sudan": "Sudan", "Efiopiya": "Ethiopia", "Madaqaskar": "Madagascar", "Namibiya": "Namibia",
}


def _get(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def _competitions() -> dict:
    menu = _get(f"{API}/left-menu?type=PRE_EVENT")
    soccer = next(s for s in menu if s.get("st") == "SOCCER")
    out = {}
    for country in soccer["c"]:
        for comp in country["c"]:
            out[comp["i"]] = (country["n"], comp["n"].strip())
    return out


def _odds(market: dict) -> dict:
    return {o["on"]: o.get("od") for o in market.get("o", []) if market.get("st") == "OPEN"}


_cache: dict = {}


def fetch_events(force: bool = False) -> pd.DataFrame:
    """All pre-match football events with the four main markets."""
    if not force and _cache.get("t", 0) > time.time() - CACHE_SECONDS:
        return _cache["df"].copy()
    comps = _competitions()
    events = _get(f"{API}/event/0?sportType=SOCCER&betType=PRE_EVENT")["e"]
    rows = []
    for e in events:
        if len(e.get("p", [])) != 2:
            continue
        country, comp = comps.get(e["cp"], ("", ""))
        row = {"event_id": e["i"], "kickoff": pd.Timestamp(e["d"], unit="ms", tz="UTC"),
               "home_raw": e["p"][0]["n"].strip(), "away_raw": e["p"][1]["n"].strip(),
               "ct": e.get("ct", ""), "country_az": country, "comp_az": comp,
               "mbs": int(e.get("mbs", 1))}
        for m in e.get("m", []):
            o = _odds(m)
            if m["t"] == 1 and m["s"] == 1:
                row.update(o1=o.get(1), ox=o.get(2), o2=o.get(3))
            elif m["t"] == 2 and m["s"] == 92:
                row.update(o1x=o.get(1), o12=o.get(2), ox2=o.get(3))
            elif m["t"] == 2 and m["s"] == 785:
                row.update(ou_line=float(str(m.get("ov", "2.5")).replace(",", ".")),
                           o_under=o.get(1), o_over=o.get(2))
            elif m["t"] == 2 and m["s"] == 89:
                row.update(o_btts_yes=o.get(1), o_btts_no=o.get(2))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["competition_az"] = (df["country_az"] + " · " + df["comp_az"]).str.strip(" ·")
    _cache.update(t=time.time(), df=df)
    return df.copy()


# --------------------------------------------------------------- live results
LIVE_STATUS = {"FIRST_HALF": "1-й тайм", "SECOND_HALF": "2-й тайм", "BREAK": "перерыв",
               "EXTRA_TIME": "доп. время", "PENALTIES": "пенальти", "HALF_TIME": "перерыв"}
ENDED = re.compile(r"ENDED|FINISHED")


def fetch_results(dates) -> pd.DataFrame:
    """Scores (finished and in-play) for the given calendar dates, from misli.az."""
    rows = []
    for date in dates:
        day = pd.Timestamp(date).strftime("%Y-%m-%d")
        try:
            data = _get(f"https://apivx.misli.az/api/web/v1/statistics/sport/SOCCER/matches"
                        f"?date={day}")
        except requests.RequestException:
            continue
        for m in data or []:
            h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
            hs, as_ = h.get("scores") or {}, a.get("scores") or {}
            status = m.get("status") or ""
            ended = bool(ENDED.search(status))
            # 90-minute score for finished matches, the running score otherwise
            hg = hs.get("REGULAR") if ended and hs.get("REGULAR") is not None else hs.get("CURRENT")
            ag = as_.get("REGULAR") if ended and as_.get("REGULAR") is not None else as_.get("CURRENT")
            rows.append({"home_raw": (h.get("teamName") or "").strip(),
                         "away_raw": (a.get("teamName") or "").strip(),
                         "kickoff": pd.Timestamp(m["date"], unit="ms", tz="UTC"),
                         "status": status, "minute": m.get("minute"), "ended": ended,
                         "live": status in LIVE_STATUS, "hg": hg, "ag": ag})
    df = pd.DataFrame(rows)
    # misli only serves a rolling window of recent matches -> keep every finished score we
    # have ever seen in a local archive and answer from archive + fresh data
    archive = _archive_results(df[df["ended"]] if len(df) else df)
    lo = pd.Timestamp(min(dates)).tz_localize("UTC") - pd.Timedelta(days=1)
    hi = pd.Timestamp(max(dates)).tz_localize("UTC") + pd.Timedelta(days=2)
    old = archive[(archive["kickoff"] >= lo) & (archive["kickoff"] < hi)]
    df = pd.concat([df, old], ignore_index=True) if len(df) else old
    if df.empty:
        return df
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df["ended"] = df["ended"].astype(bool)
    df["live"] = df["live"].fillna(False).astype(bool)
    # a fresh row wins over the archive for the same match (sort: live/ended fresh first)
    return df.drop_duplicates(["home_raw", "away_raw", "kickoff"]).reset_index(drop=True)


ARCHIVE = DATA_DIR / "misli_results.csv"


def _archive_results(new: pd.DataFrame) -> pd.DataFrame:
    cols = ["home_raw", "away_raw", "kickoff", "status", "minute", "ended", "live", "hg", "ag"]
    try:
        old = pd.read_csv(ARCHIVE, parse_dates=["kickoff"]) if ARCHIVE.exists() else             pd.DataFrame(columns=cols)
    except Exception:
        old = pd.DataFrame(columns=cols)
    if len(old):
        old["kickoff"] = pd.to_datetime(old["kickoff"], utc=True)
    allr = pd.concat([old, new[cols]] if len(new) else [old], ignore_index=True)
    allr = allr.dropna(subset=["hg", "ag"]).drop_duplicates(
        ["home_raw", "away_raw", "kickoff"], keep="last")
    if len(new):
        try:
            ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
            allr.to_csv(ARCHIVE, index=False)
        except OSError:
            pass
    allr["ended"] = allr["ended"].astype(bool)
    allr["live"] = False
    return allr


def attach_results(day: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Match misli scores onto our fixtures of a day (same pairing, kick-off ±3 h)."""
    out = day.copy()
    for c in ("live_status", "live_minute"):
        out[c] = None
    if res.empty or day.empty:
        return out
    from .names import similarity
    from .euro import _norm
    res = res.assign(h_en=res["home_raw"].map(lambda n: NATIONS_AZ.get(n, n)).map(_norm),
                     a_en=res["away_raw"].map(lambda n: NATIONS_AZ.get(n, n)).map(_norm))
    for idx, r in day.iterrows():
        k = r.get("kickoff")
        cand = res if pd.isna(k) else res[(res["kickoff"] - k).abs() <= pd.Timedelta(hours=3)]
        if cand.empty:
            continue
        names_h = {_norm(str(x)) for x in (r.get("home"), r.get("home_src")) if pd.notna(x)}
        names_a = {_norm(str(x)) for x in (r.get("away"), r.get("away_src")) if pd.notna(x)}
        sh = cand["h_en"].map(lambda c: max(similarity(n, c) for n in names_h))
        sa = cand["a_en"].map(lambda c: max(similarity(n, c) for n in names_a))
        lo, hi = np.minimum(sh, sa), np.maximum(sh, sa)
        same_minute = ((cand["kickoff"] - k).abs() <= pd.Timedelta(minutes=15)) if pd.notna(k)             else pd.Series(False, index=cand.index)
        # both names close, or same kick-off with one name certain and the other plausible
        # (spellings like "Qazaxstan" / "Kazakhstan" differ a lot)
        ok = (lo >= 0.75) | (same_minute & (hi >= 0.85) & (lo >= 0.55))
        if not ok.any():
            continue
        m = cand.loc[((lo + hi) / 2).where(ok).idxmax()]
        if m["hg"] is None or pd.isna(m["hg"]):
            continue
        if m["ended"]:
            out.loc[idx, ["hg", "ag", "played"]] = [int(m["hg"]), int(m["ag"]), True]
        elif m["live"]:
            out.loc[idx, ["live_status", "live_minute"]] = [LIVE_STATUS.get(m["status"], m["status"]), m["minute"]]
            out.loc[idx, ["hg", "ag"]] = [int(m["hg"]), int(m["ag"])]
    return out


# -------------------------------------------------------------------- linking
def link_events(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Add kind / competition / home / away (+ keys) for events our models can forecast."""
    seasons = engine.matches.groupby("league")["season"].max()
    cur = engine.matches[engine.matches["season"] == engine.matches["league"].map(seasons)]
    by_country: dict[str, dict[str, str]] = {}  # country -> team -> league (this season)
    for lg, g in cur.groupby("league"):
        for t in set(g["home"]) | set(g["away"]):
            by_country.setdefault(LEAGUES[lg]["country"], {})[t] = lg
    nations = set(engine.nat_elo.ratings)
    club_keys = [k for k in engine.elo.ratings if not k.startswith("?")]

    out = []
    for r in df.itertuples():
        link = {"kind": None, "competition": None, "home": None, "away": None,
                "home_key": None, "away_key": None}
        comp = r.comp_az.replace("İ", "i").replace("I", "ı").lower()  # Azerbaijani casing
        if EXCLUDE.search(comp) or EXCLUDE.search(r.home_raw) or EXCLUDE.search(r.away_raw):
            out.append(link)
            continue
        if r.ct == "INT":
            euro = next((code for word, code in EURO_CUPS.items() if word in comp), None)
            if euro:
                hk, _ = best_match(r.home_raw, club_keys, CLUB_ALIASES, 0.82, both_ways=False)
                ak, _ = best_match(r.away_raw, club_keys, CLUB_ALIASES, 0.82, both_ways=False)
                if hk and ak:
                    link.update(kind="cup", competition=euro, home=hk.split("|", 1)[1],
                                away=ak.split("|", 1)[1], home_key=hk, away_key=ak)
            else:
                h, a = NATIONS_AZ.get(r.home_raw), NATIONS_AZ.get(r.away_raw)
                if h in nations and a in nations:
                    link.update(kind="national", competition="UNL" if "millətlər" in comp
                                else "INT", home=h, away=a)
        elif r.ct in COUNTRY and COUNTRY[r.ct] in by_country:
            teams = by_country[COUNTRY[r.ct]]
            h, sh = best_match(r.home_raw, teams, CLUB_ALIASES, 0.8)
            a, sa = best_match(r.away_raw, teams, CLUB_ALIASES, 0.8)
            if h and a and h != a:
                lh, la = teams[h], teams[a]
                if lh == la:
                    link.update(kind="league", competition=lh, home=h, away=a)
                elif CUP_WORDS.search(comp):  # domestic cup between different tiers
                    c = COUNTRY[r.ct]
                    link.update(kind="cup", competition=lh, home=h, away=a,
                                home_key=f"{c}|{h}", away_key=f"{c}|{a}")
        out.append(link)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def model_probs(f: dict, line: float | None) -> dict:
    """Model probabilities for every market misli offers, from an Engine forecast."""
    p = f["probs"]
    m = f["matrix"]
    n = m.shape[-1]
    i, j = np.indices((n, n))
    line = 2.5 if line is None or pd.isna(line) else float(line)
    over = float((m * (i + j > line)).sum())
    btts = f["markets"]["btts"]
    return {"o1": p[0], "ox": p[1], "o2": p[2], "o1x": p[0] + p[1], "o12": p[0] + p[2],
            "ox2": p[1] + p[2], "o_over": over, "o_under": 1 - over,
            "o_btts_yes": btts, "o_btts_no": 1 - btts}


MARKET_LABELS = {"o1": "П1", "ox": "Х", "o2": "П2", "o1x": "1X", "o12": "12", "ox2": "X2",
                 "o_over": "ТБ {line}", "o_under": "ТМ {line}", "o_btts_yes": "Обе забьют — да",
                 "o_btts_no": "Обе забьют — нет"}


def outcome_won(market: str, hg: int, ag: int, line: float | None) -> bool:
    line = 2.5 if line is None or pd.isna(line) else line
    return {"o1": hg > ag, "ox": hg == ag, "o2": hg < ag, "o1x": hg >= ag, "o12": hg != ag,
            "ox2": hg <= ag, "o_over": hg + ag > line, "o_under": hg + ag < line,
            "o_btts_yes": hg > 0 and ag > 0, "o_btts_no": hg == 0 or ag == 0}[market]


# ------------------------------------------------------------------- live now
def fetch_live() -> pd.DataFrame:
    """Matches being played right now (minute, score, competition) from misli.az."""
    data = _get("https://apivx.misli.az/api/web/v1/statistics/sport/SOCCER/matches/live")
    rows = []
    for m in (data or {}).get("data", []):
        status = m.get("status") or ""
        if status not in LIVE_STATUS:
            continue
        h, a = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        hs, as_ = h.get("scores") or {}, a.get("scores") or {}
        t, c = m.get("tournament") or {}, m.get("country") or {}
        rows.append({
            "id": m.get("id"), "home_raw": (h.get("teamName") or "").strip(),
            "away_raw": (a.get("teamName") or "").strip(),
            "hg": hs.get("CURRENT", 0) or 0, "ag": as_.get("CURRENT", 0) or 0,
            "minute": m.get("minute"), "status": LIVE_STATUS[status],
            "competition": f"{c.get('misliName') or c.get('name') or ''} · "
                           f"{t.get('misliName') or t.get('name') or ''}".strip(" ·"),
            "top": bool(t.get("isTopCompetition")),
            "kickoff": pd.Timestamp(m["date"], unit="ms", tz="UTC") if m.get("date") else pd.NaT,
            "red_h": h.get("redCards") or 0, "red_a": a.get("redCards") or 0,
        })
    return pd.DataFrame(rows)


def link_live(live: pd.DataFrame, day: pd.DataFrame) -> pd.DataFrame:
    """Attach our fixture (names + pre-match call) to live matches when they are ours."""
    from .euro import _norm
    from .names import similarity
    live = live.copy()
    live["our_idx"] = None
    if live.empty or day is None or day.empty:
        return live
    for i, r in live.iterrows():
        h = _norm(NATIONS_AZ.get(r["home_raw"], r["home_raw"]))
        a = _norm(NATIONS_AZ.get(r["away_raw"], r["away_raw"]))
        best, score = None, 0.0
        for j, d in day.iterrows():
            nh = {_norm(str(x)) for x in (d.get("home"), d.get("home_src")) if pd.notna(x)}
            na = {_norm(str(x)) for x in (d.get("away"), d.get("away_src")) if pd.notna(x)}
            s = min(max(similarity(n, h) for n in nh), max(similarity(n, a) for n in na))
            if s > score:
                best, score = j, s
        if score >= 0.75:
            live.at[i, "our_idx"] = best
    return live


def events_with_model(engine, force: bool = False) -> pd.DataFrame:
    """misli.az events linked to our models, with model probabilities per market."""
    df = link_events(fetch_events(force=force), engine)
    probs = []
    for r in df.itertuples():
        if r.kind is None or pd.isna(r.kind):
            probs.append({})
            continue
        try:
            f = engine.forecast(r.kind, r.competition, r.home, r.away, False,
                                r.home_key, r.away_key)
            probs.append({f"p_{k}": v for k, v in model_probs(f, r.ou_line).items()}
                         | {"p_top_score": f["markets"]["top_scores"][0][0]})
        except Exception:
            probs.append({})
    return pd.concat([df, pd.DataFrame(probs)], axis=1)


def save_market_snapshot(engine, events: pd.DataFrame | None = None) -> int:
    """Store misli.az 1X2 (margin-free) next to our live forecasts as model 'market', for
    matches that have not started, so the Accuracy page always compares with a bookmaker."""
    from . import storage
    from .coupons import market_probs
    from .fixtures import local_tz
    ev = events if events is not None else events_with_model(engine)
    if ev.empty or "kind" not in ev:
        return 0
    ev = ev[ev["kind"].notna() & (ev["kickoff"] > pd.Timestamp.now(tz="UTC"))]
    rows = []
    for _, r in ev.iterrows():
        mp = market_probs(r)
        if "o1" not in mp:
            continue
        kick = r["kickoff"].tz_convert(local_tz())
        rows.append({"model": "market", "league": r["competition"], "season": kick.year,
                     "date": kick.tz_localize(None).normalize(), "home": r["home"],
                     "away": r["away"], "p_home": mp["o1"], "p_draw": mp["ox"],
                     "p_away": mp["o2"], "p_over25": mp.get("o_over"),
                     "p_btts": mp.get("o_btts_yes")})
    if not rows:
        return 0
    return storage.save_run("live", "live", pd.DataFrame(rows), description="ежедневные прогнозы",
                            replace_run=False)
