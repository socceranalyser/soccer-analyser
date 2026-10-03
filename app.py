"""Soccer Analyser dashboard:  streamlit run app.py"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from soccer import storage
from soccer.backtest import common_matches
from soccer.config import LEAGUES, season_label
from soccer.engine import Engine
from soccer.fixtures import CUPS, NATIONAL, competition_name, local_tz
from soccer.metrics import OUTCOME_INDEX, reliability, summary_table
from soccer.national import TOURNAMENT_RU
from soccer.pipeline import current_season, current_teams
from soccer.simulation import add_probabilities, league_table, simulate_season

st.set_page_config(page_title="Soccer Analyser", page_icon="⚽", layout="wide")


def _inject_css():
    from pathlib import Path
    css = Path(__file__).with_name("assets") / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Gradient banner instead of a plain st.title."""
    import html as _html
    icon, _, text = title.partition(" ")  # emoji stays outside the gradient text
    st.markdown(f'<div class="sa-hero"><h1><span class="sa-emoji">{icon}</span> '
                f'<span class="sa-title">{_html.escape(text)}</span></h1>'
                + (f"<p>{_html.escape(subtitle)}</p>" if subtitle else "") + "</div>",
                unsafe_allow_html=True)


_inject_css()

COLORS = {"П1": "#2E7D32", "Х": "#9E9E9E", "П2": "#1565C0"}
MODEL_LABELS = {"final": "Итоговый прогноз", "dixon_coles": "Dixon-Coles", "elo": "Elo",
                "market": "Букмекеры", "news": "Модель + новости"}
KIND_RU = {"league": "Лига", "cup": "Еврокубок", "national": "Сборные"}


# ============================================================================ data
@st.cache_resource(show_spinner="Восстанавливаю сохранённые прогнозы…")
def _restore_state() -> dict:
    """Cloud: the database disk is temporary -> reload forecasts saved in data/state/."""
    try:
        return storage.import_state()
    except Exception:
        return {}


def get_engine() -> Engine:
    _restore_state()
    return _engine(st.session_state.get("refresh_token", 0))


@st.cache_resource(ttl=3 * 3600, show_spinner="Загружаю данные и обучаю модели (≈1 мин)…")
def _engine(refresh_token: int) -> Engine:
    import threading
    eng = Engine(refresh=refresh_token > 0)

    def store():  # results + forecasts for the next days, without blocking the page
        try:
            eng.save_results()
            today = pd.Timestamp.now().normalize()
            eng.record_days([today + timedelta(days=i) for i in range(4)])
        except Exception:
            pass  # storage problems must never break the dashboard

    threading.Thread(target=store, daemon=True).start()
    return eng


@st.cache_data(ttl=3 * 3600, show_spinner="Считаю прогнозы на день…")
def get_day(date: pd.Timestamp, built_at: pd.Timestamp, tz: str | None) -> pd.DataFrame:
    return get_engine().day(date, tz)


@st.cache_data(ttl=120, show_spinner=False)
def get_scores(date: pd.Timestamp) -> pd.DataFrame:
    """Finished and live scores from livescore.com + misli.az (refreshed every 2 minutes)."""
    from soccer.scores import fetch_results
    try:
        return fetch_results([date - timedelta(days=1), date, date + timedelta(days=1)])
    except Exception:
        return pd.DataFrame()


def with_scores(day: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Overlay scores on a day's fixtures, show pre-match forecasts, store results."""
    from soccer.misli import attach_results
    _store_upcoming(day)
    day = attach_results(day, get_scores(date))
    now = now_in_tz()
    started = day["played"] | (day["kickoff"].notna() & (day["kickoff"] <= now))
    # scores misli no longer serves (rolling window) -> our own results table
    for idx in day.index[started & ~day["played"]]:
        r = day.loc[idx]
        res = storage.find_result(r["home"], r["away"], r["sched_date"])
        if res is not None:
            day.loc[idx, ["hg", "ag", "played"]] = [res[0], res[1], True]
    # started/finished matches: show the forecast stored BEFORE kick-off (the model has
    # meanwhile learnt the result, so a fresh forecast would flatter it)
    fields = ["p_home", "p_draw", "p_away", "p_over25", "p_btts", "xg_home", "xg_away"]
    try:
        pre = storage.prematch_forecasts([date - timedelta(days=1), date, date + timedelta(days=1)])
    except Exception:
        pre = pd.DataFrame()
    if len(pre) and "p_home" in day:
        pre = pre.drop_duplicates(["league", "home", "away"], keep="last")
        m = day[["competition", "home", "away"]].reset_index().merge(
            pre.rename(columns={"league": "competition"}), on=["competition", "home", "away"],
            how="inner").set_index("index")
        m = m[started.reindex(m.index).fillna(False)]
        for c in fields:
            ok = m[c].notna()
            day.loc[m.index[ok], c] = m.loc[ok, c].astype(float)
    fresh = day[day["played"] & day["hg"].notna()]
    if len(fresh):
        try:
            storage.save_results(fresh.assign(
                league=fresh["competition"], date=fresh["sched_date"],
                season=fresh["season"].fillna(date.year).astype(int),
                hg=fresh["hg"].astype(int), ag=fresh["ag"].astype(int),
                result=np.select([fresh.hg > fresh.ag, fresh.hg == fresh.ag], ["H", "D"], "A")))
        except Exception:
            pass
    return day


def _store_upcoming(day: pd.DataFrame):
    """Save headline forecasts of matches that have not kicked off (latest pre-match wins)."""
    if "p_home" not in day:
        return
    now = now_in_tz()
    up = day[day["p_home"].notna() & ~day["played"]
             & (day["kickoff"].isna() | (day["kickoff"] > now))]
    if up.empty:
        return
    rows = up.assign(model="final", league=up["competition"], date=up["sched_date"],
                     season=up["season"].fillna(up["sched_date"].dt.year).astype(int))
    try:
        storage.save_run("live", "live", rows, description="ежедневные прогнозы",
                         replace_run=False)
    except Exception:
        pass


def status_label(r, now) -> str:
    if r.get("played"):
        return "✅ завершён"
    if r.get("live_status"):
        minute = f" {int(r['live_minute'])}'" if pd.notna(r.get("live_minute")) else ""
        return f"🔴 {r['live_status']}{minute}"
    k = r.get("kickoff")
    if pd.notna(k) and k <= now:
        return "⏱ идёт / ждём счёт"
    return ""


TIMEZONES = {"Время компьютера": None, "Баку (UTC+4)": "Asia/Baku",
             "Москва (UTC+3)": "Europe/Moscow", "Стамбул (UTC+3)": "Europe/Istanbul",
             "Киев (UTC+2/+3)": "Europe/Kyiv", "Берлин / Мадрид": "Europe/Berlin",
             "Лондон": "Europe/London", "UTC": "UTC", "Нью-Йорк": "America/New_York"}


def user_tz() -> str | None:
    """IANA zone chosen in the sidebar; None = this computer's zone (schedule default)."""
    return TIMEZONES.get(st.session_state.get("tz_label", "Время компьютера"))


def now_in_tz() -> pd.Timestamp:
    return pd.Timestamp.now(tz=TIMEZONES.get(st.session_state.get("tz_label")) or local_tz())


@st.cache_data(show_spinner="Монте-Карло симуляция сезона…")
def get_simulation(league: str, built_at, n_sims: int, uncertainty: bool):
    eng = get_engine()
    season = current_season(eng.matches, league)
    played = eng.matches[(eng.matches["league"] == league) & (eng.matches["season"] == season)]
    teams = current_teams(eng.matches, league)
    res = simulate_season(eng.dc(league), played, teams, n_sims=n_sims,
                          param_uncertainty=uncertainty)
    cfg = LEAGUES[league]
    return add_probabilities(res, cfg["top"], cfg["relegated"]), res["position_probs"], \
        res["n_remaining"]


def pct(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{100 * x:.0f}%"


def pick_label(p) -> str:
    """Most likely outcome, flagged when no outcome is clearly ahead."""
    k = int(np.argmax(p))
    label = f"{['П1', 'Х', 'П2'][k]} · {pct(p[k])}"
    return f"≈ равные шансы ({label})" if p[k] < 0.40 else label


# ===================================================================== match card
def team_form(eng: Engine, kind: str, team: str, league: str | None, n: int = 6) -> pd.DataFrame:
    if kind == "national":
        src = eng.intl
    else:
        country = LEAGUES.get(league, {}).get("country")
        if country:
            src = eng.matches[eng.matches["league"].map(
                lambda l: LEAGUES[l]["country"] == country)]
        else:
            src = eng.matches
    tm = src[(src["home"] == team) | (src["away"] == team)].sort_values("date").tail(n)
    if tm.empty:
        return tm
    tm = tm.iloc[::-1]
    is_home = tm["home"] == team
    gf = np.where(is_home, tm["hg"], tm["ag"])
    ga = np.where(is_home, tm["ag"], tm["hg"])
    return pd.DataFrame({
        "Дата": tm["date"].dt.strftime("%d.%m.%y"),
        "": np.where(is_home, "дома", "в гостях"),
        "Соперник": np.where(is_home, tm["away"], tm["home"]),
        "Счёт": [f"{a}:{b}" for a, b in zip(gf, ga)],
        "Итог": np.where(gf > ga, "✅ В", np.where(gf == ga, "➖ Н", "❌ П")),
    })


def explain(f: dict, home: str, away: str, kind: str) -> str:
    p = f["probs"]
    k = int(np.argmax(p))
    who = [f"победа {home}", "ничья", f"победа {away}"][k]
    conf = "уверенный" if p[k] >= 0.6 else "умеренный" if p[k] >= 0.45 else "осторожный"
    diff = (f["elo_h"] or 0) - (f["elo_a"] or 0)
    lines = [f"**Самый вероятный исход — {who} ({pct(p[k])})**, прогноз {conf}."]
    if f["elo_h"] and f["elo_a"]:
        stronger = home if diff > 0 else away
        lines.append(f"По рейтингу Elo сильнее **{stronger}** ({f['elo_h']:.0f} против "
                     f"{f['elo_a']:.0f}, разница {abs(diff):.0f}).")
    mk = f["markets"]
    lines.append(f"Ожидаемые голы: {mk['xg_home']:.2f} — {mk['xg_away']:.2f}; "
                 f"тотал больше 2.5 — {pct(mk['totals'][2.5])}, обе забьют — {pct(mk['btts'])}.")
    if not f["known"]:
        lines.append("⚠️ Об одной из команд мало данных — прогноз менее надёжен.")
    return "  \n".join(lines)


def match_card(eng: Engine, kind: str, competition: str, home: str, away: str,
               neutral=False, home_key=None, away_key=None, odds=None, result=None, date=None):
    if date is None:
        date = now_in_tz().tz_localize(None).normalize()
    f = eng.forecast(kind, competition, home, away, neutral, home_key, away_key, odds, date)
    mk, p = f["markets"], f["probs"]
    st.markdown(f"### {home} — {away}")
    st.caption(f"{competition_name(competition)} · {KIND_RU[kind]}"
               + (" · нейтральное поле" if neutral else ""))
    if result is not None:
        st.info(f"Матч сыгран: **{result}**")
    from soccer.verdicts import btts_call, outcome_call, total_call
    lab, _, pr = outcome_call(p, home, away)
    tl, _, tp = total_call(mk["totals"][2.5])
    bl, _, bp = btts_call(mk["btts"])
    c = st.columns(4)
    c[0].metric("Кто выиграет", lab, pct(pr), delta_color="off")
    c[1].metric("Голов ожидается", f"{mk['xg_home'] + mk['xg_away']:.1f}",
                f"{mk['xg_home']:.1f} : {mk['xg_away']:.1f}", delta_color="off")
    c[2].metric("Тотал 2.5", tl, pct(tp), delta_color="off")
    c[3].metric("Обе забьют", bl, pct(bp), delta_color="off")
    st.caption(f"Вероятности исходов: П1 {pct(p[0])} · Х {pct(p[1])} · П2 {pct(p[2])}. "
               f"Самый вероятный точный счёт — {mk['top_scores'][0][0]} "
               f"(всего {pct(mk['top_scores'][0][1])}: точный счёт угадать трудно даже в лучшем случае).")
    st.markdown(explain(f, home, away, kind))
    absent = eng.absences(competition, home, away, date) if kind == "league" else None
    if absent is not None:
        lines = []
        for side, team in (("home", home), ("away", away)):
            out, doubt = absent[side], absent[f"{side}_doubt"]
            txt = ", ".join(out) if out else "все в строю"
            lines.append(f"**{team}** — не сыграют ({len(out)}): {txt}"
                         + (f"; под вопросом: {', '.join(doubt)}" if doubt else ""))
        st.markdown("🚑 **Травмы и дисквалификации** (API-Football, учтены в прогнозе):  \n"
                    + "  \n".join(lines))

    left, right = st.columns([3, 2])
    with left:
        rows = {"Итоговый прогноз": p, **f["models"]}
        df = pd.DataFrame(rows, index=["П1", "Х", "П2"]).T
        fig = go.Figure()
        for o in ["П1", "Х", "П2"]:
            fig.add_bar(y=df.index, x=df[o], name=o, orientation="h", marker_color=COLORS[o],
                        text=[pct(v) for v in df[o]], textposition="inside")
        fig.update_layout(barmode="stack", height=70 + 45 * len(rows), xaxis_tickformat=".0%",
                          xaxis_range=[0, 1], margin=dict(l=0, r=0, t=30, b=0),
                          legend=dict(orientation="h", y=1.18, x=0, traceorder="normal"),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Итоговый прогноз — то, что модель считает лучшей оценкой. Ниже — мнения "
                   "отдельных моделей и букмекеров (если есть коэффициенты).")
        g = 6
        heat = px.imshow(f["matrix"][:g, :g], text_auto=".0%", color_continuous_scale="Greens",
                         labels=dict(x=f"Голы {away}", y=f"Голы {home}", color="P"),
                         aspect="auto")
        heat.update_traces(texttemplate="%{z:.0%}")
        heat.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0),
                           title="Вероятность каждого точного счёта", coloraxis_showscale=False)
        st.plotly_chart(heat, width="stretch")
    with right:
        st.markdown("**Самые вероятные счета**")
        st.dataframe(pd.DataFrame(mk["top_scores"][:8], columns=["Счёт", "Вероятность"])
                     .assign(Вероятность=lambda d: d["Вероятность"].map(pct)),
                     hide_index=True, width="stretch")
        st.markdown("**Тоталы и другие рынки**")
        t = [{"Рынок": f"Тотал больше {k}", "Вероятность": pct(v)}
             for k, v in mk["totals"].items()]
        t += [{"Рынок": f"Тотал меньше {k}", "Вероятность": pct(1 - v)}
              for k, v in mk["totals"].items() if k in (1.5, 2.5, 3.5)]
        t += [{"Рынок": "Обе забьют — да", "Вероятность": pct(mk["btts"])},
              {"Рынок": f"{home} не пропустит", "Вероятность": pct(mk["home_clean_sheet"])},
              {"Рынок": f"{away} не пропустит", "Вероятность": pct(mk["away_clean_sheet"])}]
        st.dataframe(pd.DataFrame(t), hide_index=True, width="stretch")

    st.markdown("**Последние матчи**")
    fc = st.columns(2)
    lg_home = competition if competition in LEAGUES else _league_of(eng, home_key, home)
    lg_away = competition if competition in LEAGUES else _league_of(eng, away_key, away)
    for col, team, lg in zip(fc, (home, away), (lg_home, lg_away)):
        form = team_form(eng, kind, team, lg)
        col.markdown(f"**{team}**")
        if form.empty:
            col.caption("нет данных о последних матчах")
        else:
            col.dataframe(form, hide_index=True, width="stretch")


def _league_of(eng: Engine, key, team) -> str | None:
    """Domestic league of a club given its 'country|team' key (for form tables)."""
    if not isinstance(key, str) or not key or key.startswith("?"):  # national teams: NaN key
        return None
    country, name = key.split("|", 1)
    recent = eng.matches[eng.matches["season"] >= eng.matches["season"].max() - 1]
    for lg in recent.loc[(recent["home"] == name), "league"].unique():
        if LEAGUES[lg]["country"] == country:
            return lg
    return None


# ======================================================================== pages
def page_today():
    eng = get_engine()
    page_header("📅 Матчи дня и прогнозы", "Кто выиграет, сколько голов, тотал и «обе забьют» — для каждого матча, плюс live-счёт")
    today = now_in_tz().tz_localize(None).normalize()
    today_day = get_day(today, eng.built_at, user_tz())
    if not today_day.empty and "p_home" in today_day:
        today_day = today_day.assign(**_calls_columns(today_day.assign(
            played=False, hg=np.nan, ag=np.nan)))
    with st.container(border=True):
        live_panel(today_day)
    favorites_panel(today, eng.built_at)
    c = st.columns([2, 3, 3])
    choice = c[0].segmented_control("📅 День", ["Вчера", "Сегодня", "Завтра", "Дата…"],
                                    default="Сегодня")
    if choice == "Дата…":
        date = pd.Timestamp(c[0].date_input("📆 Дата", today, format="DD.MM.YYYY"))
    else:
        date = today + timedelta(days={"Вчера": -1, "Сегодня": 0, "Завтра": 1}.get(choice, 0))
    query = c[1].text_input("🔎 Поиск команды", placeholder="например: Arsenal, Spain…")
    day = get_day(date, eng.built_at, user_tz())
    if date == today:
        # late matches from yesterday (kick-off after 21:00) still running or just finished
        prev = get_day(date - timedelta(days=1), eng.built_at, user_tz())
        if not prev.empty and prev["kickoff"].notna().any():
            cut = pd.Timestamp(date).tz_localize(prev["kickoff"].dt.tz) - pd.Timedelta(hours=3)
            late = prev[prev["kickoff"] >= cut].assign(from_yesterday=True)
            day = pd.concat([late, day], ignore_index=True) if len(late) else day
    if not day.empty:
        day = with_scores(day, date)
    if day.empty:
        st.info(f"На {date:%d.%m.%Y} матчей в расписании нет. Попробуйте другой день.")
        _upcoming_hint(eng, date)
        return
    comps = list(dict.fromkeys(day["comp_name"]))
    sel = c[2].multiselect("🏆 Турниры", comps, placeholder="все турниры")
    view = day.copy()
    if sel:
        view = view[view["comp_name"].isin(sel)]
    if query:
        q = query.strip().lower()
        view = view[view["home"].str.lower().str.contains(q, regex=False)
                    | view["away"].str.lower().str.contains(q, regex=False)
                    | view["home_src"].fillna("").str.lower().str.contains(q, regex=False)
                    | view["away_src"].fillna("").str.lower().str.contains(q, regex=False)]

    has_p = view["p_home"].notna() if "p_home" in view else pd.Series(False, index=view.index)
    view = view.assign(**_calls_columns(view))
    played = view[view["played"] & has_p & view["hg"].notna()]
    m = st.columns(5)
    m[0].metric("Матчей", len(view))
    m[1].metric("Уже сыграно", int(view["played"].sum()))
    for col, key, label in ((m[2], "ok_outcome", "Исход угадан"), (m[3], "ok_total", "Тотал угадан"),
                            (m[4], "ok_btts", "«Обе забьют» угадано")):
        s_ = played[key].dropna() if len(played) else pd.Series(dtype=float)
        col.metric(label, f"{int(s_.sum())} из {len(s_)}" if len(s_) else "—")
    st.caption(f"Часовой пояс: {st.session_state.get('tz_label', 'Время компьютера')} "
               "(меняется в меню слева). Нажмите на строку — откроется подробный разбор. "
               "Для начавшихся и сыгранных матчей показан прогноз, сделанный до начала.")

    if view.empty:
        st.warning("Ничего не найдено.")
        return
    now = now_in_tz()
    show_probs = st.toggle("📊 Показать вероятности П1 / Х / П2", value=False)
    table = pd.DataFrame({
        "Время": [("вчера " if y is True else "") + (k.strftime("%H:%M") if pd.notna(k) else "")
                  for k, y in zip(view["kickoff"], view.get("from_yesterday",
                                                            pd.Series(False, index=view.index)))],
        "Турнир": view["comp_name"],
        "Матч": view["home"] + " — " + view["away"],
        "Кто выиграет (прогноз)": view["call_outcome"],
        "Голов ожидается": view["exp_goals"],
        "Тотал 2.5 (прогноз)": view["call_total"],
        "Обе забьют (прогноз)": view["call_btts"],
        "Статус": [status_label(r, now) for _, r in view.iterrows()],
        "Реальный счёт": [f"{int(h)}:{int(a)}" if pd.notna(h) else "" for h, a in
                          zip(view["hg"], view["ag"])],
        "Сбылось (исход · тотал · обе)": view["checks"],
    })
    cfg = {"Голов ожидается": st.column_config.NumberColumn(format="%.1f")}
    if show_probs:
        pct_col = lambda label: st.column_config.ProgressColumn(label, format="percent",
                                                                min_value=0, max_value=1)
        for k, c in (("П1", "p_home"), ("Х", "p_draw"), ("П2", "p_away")):
            table.insert(table.columns.get_loc("Кто выиграет (прогноз)"), k, view.get(c).to_numpy())
            cfg[k] = pct_col(k)
    favs = storage.load_favorites()
    keys = [(r["competition"], pd.Timestamp(r.get("sched_date", r.get("date"))).strftime("%Y-%m-%d"),
             r["home"], r["away"]) for _, r in view.iterrows()]
    table.insert(0, "⭐", [k in favs for k in keys])
    cfg["⭐"] = st.column_config.CheckboxColumn("⭐", help="Отметьте — матч появится в «⭐ Мои матчи» "
                                                "наверху страницы с live-счётом", width="small")
    edited = st.data_editor(table, hide_index=True, width="stretch", key=f"tbl_{date:%Y%m%d}",
                            disabled=[c for c in table.columns if c != "⭐"],
                            height=min(38 * (len(table) + 1), 700), column_config=cfg)
    changed = [i for i, (old, new_) in enumerate(zip(table["⭐"], edited["⭐"])) if old != new_]
    for i in changed:
        storage.set_favorite(*keys[i], on=bool(edited["⭐"].iloc[i]))
    if changed:
        st.rerun()
    labels = ["— выберите матч —"] + [f"{t} · {m}" for t, m in zip(table["Время"], table["Матч"])]
    pick = st.selectbox("🔍 Подробный разбор матча", range(len(labels)),
                        format_func=lambda i: labels[i], key=f"card_{date:%Y%m%d}")
    rows = [pick - 1] if pick else []
    if rows:
        r = view.iloc[rows[0]]
        st.divider()
        odds = None
        if pd.notna(r.get("odds_h")):
            odds = np.array([r["odds_h"], r["odds_d"], r["odds_a"]], float)
        res = f"{int(r['hg'])}:{int(r['ag'])}" if r["played"] else None
        match_card(eng, r["kind"], r["competition"], r["home"], r["away"],
                   bool(r.get("neutral", False)), r.get("home_key"), r.get("away_key"),
                   odds, res, r.get("sched_date", r.get("date")))


# ====================================================================== favourites
@st.fragment(run_every=30)
def favorites_panel(today: pd.Timestamp, built_at):
    """Starred matches with score, minute and the pre-match calls; refreshes every 30 s."""
    import html as _h
    favs = storage.load_favorites()
    if not favs:
        return
    rows = []
    for d in (today - timedelta(days=1), today, today + timedelta(days=1)):
        day = get_day(d, built_at, user_tz())
        if day.empty:
            continue
        key = [(c, pd.Timestamp(sd).strftime("%Y-%m-%d"), h, a) for c, sd, h, a in
               zip(day["competition"], day.get("sched_date", day["date"]), day["home"], day["away"])]
        sub_ = day[[k in favs for k in key]]
        if len(sub_):
            sub_ = with_scores(sub_.copy(), d)
            rows.append(sub_.assign(**_calls_columns(sub_)))
    if not rows:
        return
    fav = pd.concat(rows, ignore_index=True).sort_values("kickoff")
    now = now_in_tz()
    live = get_live()
    if not live.empty:
        from soccer.misli import link_live
        live = link_live(live, fav)
        for _, lv in live[live["our_idx"].notna()].iterrows():
            i = int(lv["our_idx"])
            fav.loc[i, ["hg", "ag", "live_status", "live_minute"]] = [
                lv["hg"], lv["ag"], lv["status"], lv["minute"]]
    html = []
    for _, r in fav.iterrows():
        when = r["kickoff"].strftime("%d.%m %H:%M") if pd.notna(r["kickoff"]) else ""
        st_ = status_label(r, now) or when
        score = f"{int(r['hg'])}:{int(r['ag'])}" if pd.notna(r.get("hg")) else "–:–"
        checks = f" {r['checks']}" if r.get("played") and r.get("checks") else ""
        html.append(f"<div class='sa-lrow'><span class='sa-lmin'>{_h.escape(st_)}</span>"
                    f"<span class='sa-lteams'>{_h.escape(r['home'])} <b>{score}</b> "
                    f"{_h.escape(r['away'])}{checks}</span>"
                    f"<span class='sa-ltip'>{_h.escape(str(r['call_outcome']))} · "
                    f"{_h.escape(str(r['call_total']).split(' · ⚖️')[0])}</span></div>")
    with st.container(border=True):
        st.markdown(f"**⭐ Мои матчи: {len(fav)}** · обновляется каждые 30 с · "
                    "убрать — снимите звёздочку в таблице")
        st.markdown("".join(html), unsafe_allow_html=True)


# ====================================================================== live now
@st.cache_data(ttl=25, show_spinner=False)
def get_live() -> pd.DataFrame:
    from soccer.scores import fetch_live
    try:
        return fetch_live()
    except Exception:
        return pd.DataFrame()


@st.fragment(run_every=30)
def live_panel(day: pd.DataFrame):
    """Matches in play right now; refreshes itself every 30 s, flags new goals."""
    import time as _time
    from soccer.misli import link_live
    live = get_live()
    if live.empty:
        st.caption("🔴 Сейчас live-матчей нет (обновляется каждые 30 с).")
        return
    live = link_live(live, day)
    ours = live[live["our_idx"].notna()]
    # goal detection against the previous refresh
    prev = st.session_state.setdefault("live_scores", {})
    flash = st.session_state.setdefault("live_flash", {})
    now = _time.time()
    for _, r in live.iterrows():
        key, score = r["id"], (int(r["hg"]), int(r["ag"]))
        if key in prev and prev[key] != score:
            flash[key] = now
            scorer = r["home_raw"] if score[0] > prev[key][0] else r["away_raw"]
            st.toast(f"⚽ ГОЛ! {scorer} — {r['home_raw']} {score[0]}:{score[1]} {r['away_raw']}",
                     icon="⚽")
        prev[key] = score
    import html as _h
    head = st.columns([3, 2])
    head[0].markdown(f"**🔴 Сейчас идут: {len(live)}**"
                     + (f" · с нашим прогнозом: {len(ours)}" if len(ours) else ""))
    show_all = head[1].toggle("Все live-матчи", value=len(ours) == 0, key="live_all")
    show = live if show_all else ours
    show = show.sort_values(["top", "minute"], ascending=[False, False])
    if show.empty:
        st.caption("Матчей из вашего расписания сейчас нет — включите «Все live-матчи».")
        return
    rows = []
    for _, r in show.iterrows():
        d = day.loc[r["our_idx"]] if pd.notna(r["our_idx"]) else None
        home = d["home"] if d is not None else r["home_raw"]
        away = d["away"] if d is not None else r["away_raw"]
        goal = now - flash.get(r["id"], 0) < 120
        minute = f"{int(r['minute'])}'" if pd.notna(r["minute"]) else _h.escape(str(r["status"]))
        reds = (" " + "🟥" * int(r["red_h"]) + "|" + "🟥" * int(r["red_a"])
                if (r["red_h"] or r["red_a"]) else "")
        tip = (f"<span class='sa-ltip'>прогноз: {_h.escape(d['call_outcome'])}</span>"
               if d is not None and isinstance(d.get("call_outcome"), str) else "")
        rows.append(f"<div class='sa-lrow{' goal' if goal else ''}'><span class='sa-lmin'>{minute}</span>"
                    f"<span class='sa-lteams'>{_h.escape(str(home))} <b>{int(r['hg'])}:{int(r['ag'])}</b> "
                    f"{_h.escape(str(away))}{' ⚽' if goal else ''}{reds}</span>{tip}</div>")
    with st.container(height=min(42 + 30 * len(rows), 200), border=True):
        st.markdown("".join(rows), unsafe_allow_html=True)


def _calls_columns(view: pd.DataFrame) -> dict:
    from soccer.verdicts import calls_columns
    return calls_columns(view)


def _upcoming_hint(eng: Engine, date):
    nxt = eng.schedule[eng.schedule["date"] > date]["date"]
    if len(nxt):
        st.caption(f"Ближайший игровой день в расписании: {nxt.min():%d.%m.%Y}")


def page_match():
    eng = get_engine()
    page_header("🔮 Прогноз любого матча", "Любые две команды: одна лига, разные лиги или сборные")
    kind = st.radio("Тип матча", ["Клубы одной лиги", "Клубы разных лиг (еврокубок)",
                                  "Сборные"], horizontal=True)
    if kind == "Клубы одной лиги":
        c = st.columns(2)
        countries = list(dict.fromkeys(v["country"] for v in LEAGUES.values()))
        country = c[0].selectbox("Страна", countries)
        league = c[1].selectbox("Лига", [k for k, v in LEAGUES.items()
                                         if v["country"] == country],
                                format_func=lambda k: LEAGUES[k]["name"].split(" — ", 1)[-1])
        teams = current_teams(eng.matches, league)
        c = st.columns(2)
        home = c[0].selectbox("Хозяева", teams)
        away = c[1].selectbox("Гости", [t for t in teams if t != home])
        match_card(eng, "league", league, home, away)
    elif kind == "Клубы разных лиг (еврокубок)":
        keys = sorted(k for k in eng.elo.ratings if "|" in k and not k.startswith("?"))
        recent = eng.matches[eng.matches["season"] >= eng.matches["season"].max() - 1]
        active = {f"{LEAGUES[l]['country']}|{t}" for l, t in
                  zip(recent["league"], recent["home"]) if LEAGUES[l]["tier"] <= 2}
        cup_keys = set(eng.euro["home_key"]) | set(eng.euro["away_key"])
        keys = [k for k in keys if k in active or k in cup_keys]
        fmt = lambda k: f"{k.split('|', 1)[1]}  ({k.split('|', 1)[0]})"
        c = st.columns(3)
        hk = c[0].selectbox("Хозяева", keys, index=_idx(keys, "Англия|Arsenal"), format_func=fmt)
        ak = c[1].selectbox("Гости", keys, index=_idx(keys, "Испания|Real Madrid"),
                            format_func=fmt)
        neutral = c[2].toggle("Нейтральное поле (финал)")
        match_card(eng, "cup", "UCL", hk.split("|", 1)[1], ak.split("|", 1)[1], neutral, hk, ak)
    else:
        recent = eng.intl[eng.intl["date"] >= eng.intl["date"].max() - pd.Timedelta(days=4 * 365)]
        nations = sorted(set(recent["home"]) | set(recent["away"]))
        c = st.columns(3)
        home = c[0].selectbox("Хозяева", nations, index=_idx(nations, "Spain"))
        away = c[1].selectbox("Гости", nations, index=_idx(nations, "Germany"))
        neutral = c[2].toggle("Нейтральное поле")
        match_card(eng, "national", "UNL", home, away, neutral)


def _idx(items, value) -> int:
    return items.index(value) if value in items else 0


def page_leagues():
    eng = get_engine()
    page_header("🏆 Лиги и симуляция сезона", "Таблица, шансы на чемпионство и вылет, сила атаки и обороны")
    c = st.columns(2)
    countries = list(dict.fromkeys(v["country"] for v in LEAGUES.values()))
    country = c[0].selectbox("Страна", countries)
    league = c[1].selectbox("Лига", [k for k, v in LEAGUES.items() if v["country"] == country],
                            format_func=lambda k: LEAGUES[k]["name"].split(" — ", 1)[-1])
    season = current_season(eng.matches, league)
    teams = current_teams(eng.matches, league)
    played = eng.matches[(eng.matches["league"] == league) & (eng.matches["season"] == season)]
    st.subheader(f"{LEAGUES[league]['name']} · сезон {season_label(league, season)}")
    t1, t2, t3 = st.tabs(["Таблица и симуляция", "Сила команд", "Последние результаты"])
    with t1:
        lc, rc = st.columns([2, 3])
        lc.markdown("**Текущая таблица**")
        lc.dataframe(league_table(played, teams).rename(columns={
            "team": "Команда", "P": "И", "W": "В", "D": "Н", "L": "П", "GF": "ЗМ", "GA": "ПМ",
            "GD": "РМ", "Pts": "О"}), hide_index=True, width="stretch",
            height=35 * (len(teams) + 1) + 3)
        if not LEAGUES[league]["sim"]:
            rc.info("Симуляция сезона недоступна: в этой лиге разделение таблицы, плей-офф, "
                    "конференции или больше двух встреч пар. Прогнозы матчей работают.")
        else:
            with rc:
                st.markdown("**Чем закончится сезон (Монте-Карло)**")
                sc = st.columns(2)
                n_sims = sc[0].select_slider("Симуляций", [2000, 10000, 20000], value=10000)
                unc = sc[1].toggle("Учитывать неопределённость силы команд", value=True)
                summ, posp, n_rem = get_simulation(league, eng.built_at, n_sims, unc)
                cfg = LEAGUES[league]
                st.caption(f"Осталось матчей: {n_rem}. Каждый сезон разыгран {n_sims:,} раз.")
                show = summ.rename(columns={
                    "team": "Команда", "pts_now": "Очки", "exp_pts": "Ожид. очки",
                    "pts_p10": "мин (10%)", "pts_p90": "макс (90%)", "exp_pos": "Ожид. место",
                    "p_title": "Чемпион", f"p_top{cfg['top']}": f"Топ-{cfg['top']}",
                    "p_relegation": "Вылет"})
                pc = lambda l: st.column_config.ProgressColumn(l, format="percent",
                                                               min_value=0, max_value=1)
                st.dataframe(show, hide_index=True, width="stretch",
                             height=35 * (len(teams) + 1) + 3,
                             column_config={"Чемпион": pc("Чемпион"),
                                            f"Топ-{cfg['top']}": pc(f"Топ-{cfg['top']}"),
                                            "Вылет": pc("Вылет"),
                                            "Ожид. очки": st.column_config.NumberColumn(format="%.1f"),
                                            "Ожид. место": st.column_config.NumberColumn(format="%.1f")})
            order = summ["team"].tolist()
            heat = px.imshow(posp.loc[order], color_continuous_scale="Blues", aspect="auto",
                             labels=dict(x="Итоговое место", y="", color="P"))
            heat.update_traces(texttemplate="%{z:.0%}")
            heat.update_layout(height=40 + 26 * len(order), margin=dict(l=0, r=0, t=10, b=0),
                               coloraxis_showscale=False)
            st.markdown("**Вероятности итоговых мест**")
            st.plotly_chart(heat, width="stretch")
    with t2:
        dc = eng.dc(league)
        r = dc.ratings()
        r = r[r["team"].isin(teams)]
        r["elo"] = r["team"].map(lambda t: eng.elo.rating(t, league))
        lc, rc = st.columns([3, 2])
        fig = px.scatter(r, x="attack", y="defence", text="team",
                         labels={"attack": "Атака → (больше забивает)",
                                 "defence": "Оборона → (меньше пропускает)"})
        fig.update_traces(textposition="top center", marker_size=9)
        fig.add_hline(y=0, line_dash="dot", opacity=0.4)
        fig.add_vline(x=0, line_dash="dot", opacity=0.4)
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
        lc.plotly_chart(fig, width="stretch")
        lc.caption(f"Ноль — средний уровень лиги. Домашнее преимущество: "
                   f"×{np.exp(dc.theta[-2]):.2f} к голам хозяев.")
        rc.dataframe(r[["team", "attack", "defence", "elo"]].rename(columns={
            "team": "Команда", "attack": "Атака", "defence": "Оборона", "elo": "Elo"}),
            hide_index=True, width="stretch", height=600,
            column_config={"Атака": st.column_config.NumberColumn(format="%+.2f"),
                           "Оборона": st.column_config.NumberColumn(format="%+.2f"),
                           "Elo": st.column_config.NumberColumn(format="%.0f")})
    with t3:
        last = played.sort_values("date", ascending=False).head(40)
        st.dataframe(pd.DataFrame({"Дата": last["date"].dt.strftime("%d.%m.%Y"),
                                   "Хозяева": last["home"], "Счёт": last["hg"].astype(str) + ":"
                                   + last["ag"].astype(str), "Гости": last["away"]}),
                     hide_index=True, width="stretch")
        if eng.n_fresh:
            st.caption("Часть свежих результатов взята из расписания fixturedownload.com "
                       "(football-data.co.uk публикует их с задержкой).")


def page_ratings():
    eng = get_engine()
    page_header("📊 Рейтинги силы", "Все клубы на единой шкале и сборные мира")
    t1, t2 = st.tabs(["Клубы (все лиги, единая шкала)", "Сборные"])
    with t1:
        recent = eng.matches[eng.matches["season"] >= eng.matches["season"].max() - 1]
        last_league = recent.sort_values("date").groupby("home")["league"].last()
        rows = []
        for team, lg in last_league.items():
            rows.append({"Команда": team, "Страна": LEAGUES[lg]["country"],
                         "Лига": LEAGUES[lg]["name"].split(" — ", 1)[-1],
                         "Elo": eng.elo.rating(team, lg)})
        df = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
        df.insert(0, "#", range(1, len(df) + 1))
        c = st.columns(2)
        ctry = c[0].multiselect("Страны", sorted(df["Страна"].unique()), placeholder="все")
        q = c[1].text_input("🔎 Поиск", key="club_search")
        v = df[df["Страна"].isin(ctry)] if ctry else df
        if q:
            v = v[v["Команда"].str.lower().str.contains(q.lower(), regex=False)]
        st.dataframe(v.head(300), hide_index=True, width="stretch", height=600,
                     column_config={"Elo": st.column_config.NumberColumn(format="%.0f")})
        st.caption("Рейтинги разных стран сравнимы благодаря еврокубкам: результаты клубов в ЛЧ, "
                   "ЛЕ и ЛК частично переносятся на всю лигу их страны.")
    with t2:
        last = eng.intl[eng.intl["date"] >= eng.intl["date"].max() - pd.Timedelta(days=3 * 365)]
        active = set(last["home"]) | set(last["away"])
        df = pd.DataFrame([(t, r) for t, r in eng.nat_elo.ratings.items() if t in active],
                          columns=["Сборная", "Elo"]).sort_values("Elo", ascending=False)
        df.insert(0, "#", range(1, len(df) + 1))
        q = st.text_input("🔎 Поиск", key="nat_search")
        if q:
            df = df[df["Сборная"].str.lower().str.contains(q.lower(), regex=False)]
        st.dataframe(df, hide_index=True, width="stretch", height=600,
                     column_config={"Elo": st.column_config.NumberColumn(format="%.0f")})


def render_auto_analysis():
    """Automatic error analysis of live forecasts: findings + reasons for each miss."""
    from soccer import analysis
    try:
        fin = analysis.scored_live()
        fs = analysis.findings(fin)
        ms = analysis.misses(fin, n=40)
    except Exception as exc:
        st.caption(f"Автоанализ недоступен: {exc}")
        return
    icon = {"act": "🔴", "watch": "🟡", "ok": "🟢"}
    with st.expander(f"🔍 Автоанализ живых прогнозов · {len(fin)} сыгранных матчей", expanded=True):
        st.caption("Система сама ищет систематические ошибки (с проверкой на случайность), "
                   "раз в неделю присылает отчёт в Telegram и переучивает калибровку, когда "
                   "данных достаточно. Задачи, требующие новой идеи, Claude берёт в работу "
                   "в начале каждой сессии.")
        for f in fs:
            st.markdown(f"{icon[f['severity']]} **{f['title']}** — {f['text']}"
                        + (f"  \n&nbsp;&nbsp;&nbsp;➜ *{f['todo']}*" if f.get("todo") else ""))
        if len(ms):
            st.markdown("**Разбор промахов** — почему не сбылось:")
            st.dataframe(pd.DataFrame({
                "Дата": ms["date"].dt.strftime("%d.%m"), "Матч": ms["match"],
                "Счёт": ms["score"], "Прогноз": ms["call"], "Вероятная причина": ms["reasons"]}),
                hide_index=True, width="stretch")


def page_accuracy():
    page_header("🎯 Точность прогнозов", "Как часто сбываются прогнозы — живые и на истории, в сравнении с букмекером")
    render_auto_analysis()
    runs = storage.load_runs()
    if runs.empty:
        st.info("Прогнозов пока нет.")
        return
    labels = {r.run_id: ("Ежедневные прогнозы (реальные, до матча)" if r.kind == "live"
                         else f"Бэктест · {r.description or r.run_id}") for r in runs.itertuples()}
    live_done = "live" in labels and storage.load_predictions("live")["result"].notna().any()
    # real forecasts first once some have been scored; otherwise the full backtest
    order = sorted(labels, key=lambda k: (0 if (k == "live" and live_done) else
                                          1 if k == "bt_all" else 2, k))
    run_id = st.selectbox("Набор прогнозов", order, format_func=labels.get)
    preds = storage.load_predictions(run_id)
    done = preds.dropna(subset=["result"])
    pending = preds[preds["result"].isna()]
    n_models = max(done["model"].nunique(), 1)
    st.caption(f"Сыграно матчей с прогнозом: {len(done) // n_models:,} · ждут результата: "
               f"{len(pending) // max(pending['model'].nunique(), 1):,}")
    with st.expander("Как читать метрики"):
        st.markdown(
            "- **Угадан исход** — доля матчей, где самый вероятный исход совпал с результатом.\n"
            "- **Log loss** и **RPS** — штраф за ошибку с учётом уверенности: чем ниже, тем "
            "лучше. Уверенная ошибка штрафуется сильно.\n"
            "- **Калибровка** — если модель говорит 60%, событие должно случаться ~60% раз.\n"
            "- **Букмекеры** — ориентир: обогнать рынок очень сложно.")
    if done.empty:
        st.info("Ещё нет сыгранных матчей с сохранёнными прогнозами — загляните после игр.")
        return
    cat = done["league"].map(lambda l: "Лиги" if l in LEAGUES else "Еврокубки" if l in CUPS
                             else "Сборные")
    cats = [c for c in ("Лиги", "Еврокубки", "Сборные") if (cat == c).any()]
    for tab, c in zip(st.tabs(cats), cats):
        with tab:
            _accuracy_block(done[cat == c], run_id, key=c)


def _accuracy_block(done: pd.DataFrame, run_id: str, key: str):
    from soccer.verdicts import score_calls
    head = done[done["model"] == ("final" if (done["model"] == "final").any()
                                  else sorted(done["model"].unique())[0])]
    if len(head):
        checks = [score_calls([r.p_home, r.p_draw, r.p_away], r.p_over25,
                              getattr(r, "p_btts", np.nan), int(r.hg), int(r.ag))
                  for r in head.itertuples()]
        c = st.columns(3)
        for col, k, label in ((c[0], "outcome", "«Кто выиграет» сбылось"),
                              (c[1], "total", "Тотал 2.5 сбылся"),
                              (c[2], "btts", "«Обе забьют» сбылось")):
            vals = [ch[k] for ch in checks if k in ch]
            col.metric(label, f"{np.mean(vals):.1%}" if vals else "—",
                       help=f"по {len(vals):,} матчам" if vals else "модель не давала этот прогноз")
    news_done = done
    done = done[done["model"] != "news"]  # news covers ~12 matches/day: compared separately
    models = sorted(done["model"].unique())
    fair = common_matches(done, models) if len(models) > 1 else done
    if fair.empty:
        fair = done
    if (news_done["model"] == "news").any():
        nm = [m for m in ("final", "news", "market") if (news_done["model"] == m).any()]
        nf = common_matches(news_done, nm)
        if len(nf):
            st.markdown("**📰 Помогают ли новости?** Те же матчи: модель без новостей, с новостями "
                        "и букмекер (меньше log loss — лучше).")
            st.dataframe(summary_table(nf).assign(model=lambda d: d["model"].map(
                lambda m: MODEL_LABELS.get(m, m)))[["model", "n", "accuracy", "log_loss"]]
                .rename(columns={"model": "Модель", "n": "Матчей", "accuracy": "Угадан исход",
                                 "log_loss": "Log loss"}), hide_index=True, width="stretch",
                column_config={"Угадан исход": st.column_config.NumberColumn(format="percent"),
                               "Log loss": st.column_config.NumberColumn(format="%.4f")})
    tbl = summary_table(fair).assign(model=lambda d: d["model"].map(
        lambda m: MODEL_LABELS.get(m, m)))
    st.dataframe(tbl.rename(columns={
        "model": "Модель", "n": "Матчей", "accuracy": "Угадан исход", "log_loss": "Log loss",
        "rps": "RPS", "brier": "Brier", "ece": "Ошибка калибровки",
        "ou25_log_loss": "Log loss ТБ2.5"}), hide_index=True, width="stretch",
        column_config={"Угадан исход": st.column_config.NumberColumn(format="percent"),
                       "Log loss": st.column_config.NumberColumn(format="%.4f"),
                       "RPS": st.column_config.NumberColumn(format="%.4f"),
                       "Brier": st.column_config.NumberColumn(format="%.4f"),
                       "Ошибка калибровки": st.column_config.NumberColumn(format="%.4f"),
                       "Log loss ТБ2.5": st.column_config.NumberColumn(format="%.4f")})
    by_league = summary_table(fair, by=("league", "model"))
    by_league["Турнир"] = by_league["league"].map(competition_name)
    by_league["model"] = by_league["model"].map(lambda m: MODEL_LABELS.get(m, m))
    with st.expander("По турнирам"):
        st.dataframe(by_league.pivot_table(index="Турнир", columns="model", values="accuracy")
                     .style.format("{:.1%}"), width="stretch")
    lc, rc = st.columns(2)
    with lc:
        outcome = st.radio("Калибровка исхода", ["П1", "Х", "П2"], horizontal=True,
                           key=f"cal_{key}")
        lab = {"П1": "H", "Х": "D", "П2": "A"}[outcome]
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Идеал",
                        line=dict(dash="dot", color="gray"))
        for mdl, g in fair.groupby("model"):
            rel = reliability(g[["p_home", "p_draw", "p_away"]].to_numpy(),
                              g["result"].map(OUTCOME_INDEX).to_numpy(), bins=10)
            rel = rel[(rel["outcome"] == lab) & (rel["n"] >= 10)]
            fig.add_scatter(x=rel["predicted"], y=rel["observed"], mode="lines+markers",
                            name=MODEL_LABELS.get(mdl, mdl))
        fig.update_layout(height=380, xaxis_title="Прогноз модели", yaxis_title="Как вышло",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch", key=f"acc_1_{key}")
    with rc:
        daily = fair.copy()
        daily["hit"] = daily[["p_home", "p_draw", "p_away"]].to_numpy().argmax(1) == \
            daily["result"].map(OUTCOME_INDEX).to_numpy()
        period = "D" if run_id == "live" else "Q"
        agg = daily.groupby([daily["date"].dt.to_period(period).dt.start_time, "model"])["hit"] \
            .agg(["mean", "size"]).reset_index()
        min_n = 1 if run_id == "live" else 30  # tiny periods only add noise
        agg = agg[agg["size"] >= min_n].rename(columns={"mean": "hit"})
        agg["model"] = agg["model"].map(lambda m: MODEL_LABELS.get(m, m))
        st.markdown("**Доля угаданных исходов по " + ("дням" if run_id == "live" else
                                                       "кварталам") + "**")
        fig = px.line(agg, x="date", y="hit", color="model", markers=True,
                      labels={"date": "", "hit": "Угадан исход", "model": ""})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch", key=f"acc_2_{key}")
    st.markdown("**Последние прогнозы и результаты**")
    from soccer.verdicts import calls_columns
    head_model = "final" if "final" in models else models[0]
    last = done[done["model"] == head_model].sort_values("date", ascending=False).head(50)
    if len(last):
        last = last.assign(played=True).reset_index(drop=True)
        calls = pd.DataFrame(calls_columns(last))
        mk = done[done["model"] == "market"].set_index(["league", "date", "home", "away"])
        key = pd.MultiIndex.from_frame(last[["league", "date", "home", "away"]])
        book = []
        for k, r in zip(key, last.itertuples()):
            if k in mk.index:
                q = mk.loc[k, ["p_home", "p_draw", "p_away"]]
                q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
                from soccer.verdicts import outcome_call, result_index
                lab, cov, pr = outcome_call(q.to_numpy(float), r.home, r.away)
                ok = result_index(int(r.hg), int(r.ag)) in cov
                book.append(f"{'✅' if ok else '❌'} {lab} · {pr:.0%}")
            else:
                book.append("—")
        st.dataframe(pd.DataFrame({
            "Дата": last["date"].dt.strftime("%d.%m.%Y"),
            "Турнир": last["league"].map(competition_name),
            "Матч": last["home"] + " — " + last["away"],
            "Счёт": last["hg"].astype("Int64").astype(str) + ":" + last["ag"].astype("Int64").astype(str),
            "Прогноз модели": calls["call_outcome"],
            "Сбылось (исход · тотал · обе)": calls["checks"],
            "Букмекер сказал бы": book}), hide_index=True, width="stretch")
        st.caption("«Кто выиграет» — как в «Матчах дня»: «Победа X» при шансе от 50%, иначе "
                   "«X не проиграет». Колонка букмекера — тот же вердикт по коэффициентам misli.az.")

# =================================================================== coupon page
@st.cache_data(ttl=600, show_spinner="Загружаю коэффициенты misli.az…")
def get_misli(built_at, refresh_token: int) -> pd.DataFrame:
    """misli.az events linked to our models, with model probabilities per market."""
    from soccer.misli import events_with_model, save_market_snapshot
    ev = events_with_model(get_engine(), force=refresh_token > 0)
    try:
        save_market_snapshot(get_engine(), ev)
    except Exception:
        pass
    return ev


ODDS_COLS = ["o1", "ox", "o2", "o1x", "o12", "ox2", "o_over", "o_under", "o_btts_yes",
             "o_btts_no"]


def _best_value(r) -> tuple[str | None, float]:
    best, val = None, -1.0
    for c in ODDS_COLS:
        o, p = r.get(c), r.get(f"p_{c}")
        if pd.notna(o) and pd.notna(p) and o > 1:
            v = p * o - 1
            if v > val:
                best, val = c, v
    return best, val


def _market_label(col: str, line) -> str:
    from soccer.misli import MARKET_LABELS
    line = 2.5 if line is None or pd.isna(line) else line
    return MARKET_LABELS[col].format(line=f"{line:g}")


def _coupon() -> list[dict]:
    if "coupon" not in st.session_state:  # restore the coupon being built (survives reloads)
        try:
            st.session_state["coupon"] = storage.load_draft()
        except Exception:
            _set_coupon([])
    return st.session_state["coupon"]


def _set_coupon(picks: list[dict]):
    st.session_state["coupon"] = picks
    try:
        storage.save_draft(picks)
    except Exception:
        pass


def _market_p(r, col: str):
    from soccer.coupons import market_probs
    v = market_probs(r).get(col)
    return None if v is None else float(v)


def _in_ten(p: float) -> str:
    """0.82 -> 'примерно 8 из 10'."""
    n = int(round(p * 10))
    return f"примерно {n} из 10" if 0 < n < 10 else ("почти всегда" if n >= 10 else "редко")


def _add_pick(r, col: str):
    picks = [p for p in _coupon() if p["event_id"] != int(r["event_id"])]  # one pick per match
    picks.append({
        "event_id": int(r["event_id"]), "kickoff": str(r["kickoff"]),
        "match": f"{r['home_raw']} — {r['away_raw']}", "comp": r["competition_az"],
        "market": col, "label": _market_label(col, r.get("ou_line")), "odds": float(r[col]),
        "p_model": None if pd.isna(r.get(f"p_{col}")) else float(r[f"p_{col}"]),
        "p_market": _market_p(r, col),
        "home": r["home"] if pd.notna(r.get("home")) else None,
        "away": r["away"] if pd.notna(r.get("away")) else None,
        "line": None if pd.isna(r.get("ou_line")) else float(r["ou_line"]),
        "mbs": int(r["mbs"])})
    _set_coupon(picks)


def render_coupon_sidebar():
    picks = _coupon()
    with st.sidebar:
        st.markdown(f"### 🎟️ Мой купон ({len(picks)})")
        if not picks:
            st.caption("Пусто. Выберите матч и нажмите «➕» у нужного исхода "
                       "или загрузите готовый купон.")
            return
        for p in picks:
            c = st.columns([5, 1])
            pm, pk = p.get("p_model"), p.get("p_market")
            use_model = st.session_state.get("chance_source", "Мой анализ (рекомендую)") == "Только модель"
            main = (pm if pm is not None else pk) if use_model else (pk if pk is not None else pm)
            parts = []
            if pm is not None:
                parts.append(f"модель {pct(pm)}")
            if pk is not None:
                parts.append(f"букмекер {pct(pk)}")
            c[0].markdown(f"**{p['match']}**  \n{p['label']} @ **{p['odds']:.2f}**  \n"
                          +(f"<small>Сыграет {_in_ten(main)} · {' · '.join(parts)}</small>"
                             if main is not None else ""), unsafe_allow_html=True)
            if c[1].button("✖", key=f"rm_{p['event_id']}", help="убрать"):
                _set_coupon([q for q in picks if q is not p])
                st.rerun()
        total = float(np.prod([p["odds"] for p in picks]))
        st.metric("Общий коэффициент", f"{total:.2f}")
        stake = st.number_input("Сумма ставки, ₼", min_value=0.0, value=1.0, step=1.0)
        st.markdown(f"Если купон зайдёт, получите **{stake * total:.2f} ₼**")
        pk_all = [p.get("p_market") for p in picks]
        pm_all = [p.get("p_model") for p in picks]
        prob_k = float(np.prod(pk_all)) if all(v is not None for v in pk_all) else None
        prob_m = float(np.prod(pm_all)) if all(v is not None for v in pm_all) else None
        use_model = st.session_state.get("chance_source", "Мой анализ (рекомендую)") == "Только модель"
        prob = (prob_m if prob_m is not None else prob_k) if use_model else             (prob_k if prob_k is not None else prob_m)
        if prob is not None:
            src = " · ".join(x for x in (
                f"модель {pct(prob_m)}" if prob_m is not None else "",
                f"букмекер {pct(prob_k)}" if prob_k is not None else "") if x)
            st.markdown(f"**Купон зайдёт {_in_ten(prob)} раз** ({src})")
            back = prob * total * 100
            verdict = ("🟢 на дистанции в плюсе" if back > 100 else
                       "🔴 на дистанции в минусе (маржа букмекера)")
            st.markdown(f"Если ставить такой купон 100 раз по 1 ₼: потратите 100 ₼, "
                        f"вернётся ≈ **{back:.0f} ₼** — {verdict}")
            st.caption("Главный шанс — " + ("наша модель" if use_model else
                                             "итоговая оценка (по истории ≈ букмекер без маржи)")
                       + "; переключается на странице «Купон».")
        else:
            st.caption("Для части матчей нет оценки шанса — шанс купона не посчитан.")
        need = max(p["mbs"] for p in picks)
        if len(picks) < need:
            st.warning(f"По правилам misli для выбранных матчей нужно минимум {need} события "
                       f"в купоне (сейчас {len(picks)}).")
        c = st.columns(2)
        if c[0].button("💾 Сохранить", width="stretch"):
            storage.save_coupon(picks, stake, total, prob if prob is not None else float("nan"))
            st.toast("Купон сохранён — результат появится внизу страницы после матчей.")
        if c[1].button("🗑️ Очистить", width="stretch"):
            _set_coupon([])
            st.rerun()


def render_news():
    """What Claude found in today's news and how the maths weighed it."""
    from soccer import news
    items = sorted(news.notable(min_abs=0.0), key=lambda n: n["kickoff"])
    if not items:
        if not news.available():
            st.caption("📰 Анализ новостей включится, когда будет добавлен ключ ANTHROPIC_API_KEY.")
        return
    W, G, n = news.weights()
    with st.expander(f"📰 Новости дня — проанализировано {len(items)} матч(ей)", expanded=False):
        st.caption(f"Claude ищет в интернете травмы, ротацию, мотивацию, смену тренера и т.п.; "
                   f"формула переводит найденное в поправку (вес новостей W = {W:.3f}"
                   + (f", подобран по {n} сыгранным матчам)" if n >= news.MIN_FIT
                      else f", пока осторожный стартовый — подберётся после {news.MIN_FIT} матчей; "
                           f"сейчас сыграно {n})"))
        for it in items:
            (ph, px, pa), _, _ = news.adjust((it["p_home"], it["p_draw"], it["p_away"]),
                                             it["s"], w=(W, G, n))
            arrow = "⬆️" if it["s"] > 0.3 else "⬇️" if it["s"] < -0.3 else "➖"
            st.markdown(f"**{it['home']} — {it['away']}** {arrow} "
                        f"П1 {pct(it['p_home'])}→**{pct(ph)}** · Х {pct(it['p_draw'])}→**{pct(px)}** · "
                        f"П2 {pct(it['p_away'])}→**{pct(pa)}**"
                        + (" · ⛔ **исключён из купонов**" if it["avoid"] else "")
                        + "  \n" + it["summary"])
            for f in it.get("factors", []):
                side = {"home": it["home"], "away": it["away"]}.get(f["team"], "обе")
                st.caption(f"{'+' if f['impact'] > 0 else ''}{f['impact']} · {side} · "
                           f"{f['description']} (уверенность {f['certainty']:.0%})")


def render_suggestions(df: pd.DataFrame):
    """Ready-made coupons (1, 2 and 3 matches) from upcoming matches."""
    from soccer.coupons import suggest
    upcoming = df[df["kickoff"] > pd.Timestamp.now(tz="UTC")]
    st.markdown("### 🎯 Готовые купоны")
    c0 = st.columns([3, 2])
    c0[0].radio("Как отбирать", ["Мой анализ (рекомендую)", "Только модель", "Только букмекер"],
                horizontal=True, key="chance_source",
                help="Мой анализ: в купон попадают только исходы, которые уверенно подтверждают и "
                     "моя модель (сила команд, xG, травмы, новости), и рынок. Шанс указан честный: "
                     "на истории 2023–25 обещанные проценты совпали с реальностью.")
    c0[1].toggle("Показывать мнение букмекера для сравнения", value=True, key="show_book")
    source = {"Мой анализ (рекомендую)": "combined", "Только модель": "model",
              "Только букмекер": "market"}[st.session_state.get("chance_source",
                                                                 "Мой анализ (рекомендую)")]
    coupons = suggest(upcoming, source)
    if not coupons:
        st.info("Для выбранного периода не хватает матчей с прогнозом модели, чтобы собрать "
                "купоны. Попробуйте «Все» вместо «Сегодня».")
        return
    stake = st.number_input("Пример ставки для расчёта, ₼", min_value=1.0, value=10.0,
                            step=1.0, key="sugg_stake")
    singles = [cp for cp in coupons if cp["style"] == "single"]
    others = [cp for cp in coupons if cp["style"] != "single"]
    slots = ([st.container()] if singles else []) + (list(st.columns(len(others))) if others else [])
    for col, cp in zip(slots, singles + others):
        with col.container(border=True):
            st.markdown(f"**{cp['title']}**")
            for _, pk in cp["picks"].iterrows():
                t = pd.Timestamp(pk["kickoff"]).tz_convert(user_tz() or local_tz())
                st.markdown(f"{t:%d.%m %H:%M} · {pk['match']}  \n"
                            f"**{pk['label']}** @ **{pk['odds']:.2f}** · шанс {pct(pk['p'])}"
                            + (f" · букмекер {pct(pk['p_market'])}"
                               if st.session_state.get("show_book", True) else "")
                            + f"  \n<small>💡 {pk['why']}</small>", unsafe_allow_html=True)
            st.divider()
            c = st.columns(2)
            c[0].metric("Общий коэф.", f"{cp['total_odds']:.2f}")
            c[1].metric("Шанс купона", pct(cp["prob"]), _in_ten(cp["prob"]) + " раз",
                        delta_color="off")
            from soccer.coupons import HISTORY
            h = HISTORY.get(cp["style"])
            if h:
                st.caption(f"📜 Такие купоны в 2023–25: сбылись в {h['hit']:.0%} дней, самая длинная "
                           f"серия проигрышей — {h['streak']} дн. подряд, итог {h['ret']:+.0%} на ставку.")
            st.caption(f"Ставка {stake:.0f} ₼ → если зайдёт, {stake * cp['total_odds']:.2f} ₼. "
                       f"Если ставить так 100 раз по 1 ₼, вернётся ≈ "
                       f"{(cp['ev'] + 1) * 100:.0f} ₼ (маржа букмекера).")
            if st.button("➕ Загрузить в мой купон", key=f"load_{cp['style']}",
                         width="stretch"):
                _set_coupon([
                    {"event_id": int(pk["event_id"]), "kickoff": str(pk["kickoff"]),
                     "match": pk["match"], "comp": pk["comp"], "market": pk["market"],
                     "label": pk["label"], "odds": float(pk["odds"]),
                     "p_model": float(pk["p_model"]) if pd.notna(pk.get("p_model")) else None,
                     "p_market": float(pk["p_market"]),
                     "home": pk["home"], "away": pk["away"], "line": pk["line"],
                     "mbs": int(pk["mbs"])} for _, pk in cp["picks"].iterrows()])
                st.rerun()
    st.caption("Шанс каждого исхода — по выбранному источнику (по умолчанию наша модель); "
               "коэффициент misli.az — это только цена ставки. Шанс купона — произведение "
               "шансов. Это не гарантия выигрыша — ставьте только то, что готовы потерять.")


def page_coupon():
    from soccer.misli import outcome_won
    eng = get_engine()
    page_header("🎟️ Купоны", "Коэффициенты misli.az, оценка модели и готовые купоны дня")
    c = st.columns([2, 3, 2, 2])
    day = c[0].segmented_control("🗓️ Когда", ["Сегодня", "Завтра", "Все"], default="Все")
    query = c[1].text_input("🔎 Поиск команды или турнира", key="coupon_q")
    only_model = c[2].toggle("🧠 Только с прогнозом модели", value=True)
    sort = c[3].selectbox("↕️ Сортировка", ["По времени", "По преимуществу модели"])
    if c[3].button("🔄 Обновить коэффициенты"):
        st.session_state["misli_token"] = st.session_state.get("misli_token", 0) + 1
    try:
        df = get_misli(eng.built_at, st.session_state.get("misli_token", 0))
    except Exception as exc:
        st.error(f"Не удалось загрузить misli.az: {exc}")
        return
    tz = user_tz() or local_tz()
    df["kick_local"] = df["kickoff"].dt.tz_convert(tz)
    today = now_in_tz().tz_localize(None).normalize()
    dates = df["kick_local"].dt.tz_localize(None).dt.normalize()
    if day == "Сегодня":
        df = df[dates == today]
    elif day == "Завтра":
        df = df[dates == today + timedelta(days=1)]
    if only_model:
        df = df[df["p_o1"].notna()] if "p_o1" in df else df.iloc[:0]
    if query:
        q = query.lower()
        df = df[df["home_raw"].str.lower().str.contains(q, regex=False)
                | df["away_raw"].str.lower().str.contains(q, regex=False)
                | df["competition_az"].str.lower().str.contains(q, regex=False)
                | df["home"].fillna("").str.lower().str.contains(q, regex=False)
                | df["away"].fillna("").str.lower().str.contains(q, regex=False)]
    best = df.apply(_best_value, axis=1, result_type="expand") if len(df) else None
    if best is not None:
        df = df.assign(best_col=best[0], best_val=best[1])
        if sort == "По преимуществу модели":
            df = df.sort_values("best_val", ascending=False)
        else:
            df = df.sort_values("kickoff")
    margin = (1 / df[["o1", "ox", "o2"]]).sum(axis=1) - 1 if len(df) else pd.Series(dtype=float)
    st.caption(f"Матчей: {len(df)} · средняя маржа букмекера на 1X2: "
               f"{margin.mean():.1%}" if len(df) else "Матчей не найдено.")
    with st.expander("ℹ️ Как пользоваться и как понимать «преимущество»", expanded=False):
        st.markdown(
            "- Выберите матч в таблице → внизу появятся все исходы с коэффициентами misli и "
            "вероятностями модели → нажмите **➕**, чтобы добавить в купон (он слева).\n"
            "- **Справедливый коэффициент** = 1 / вероятность модели. Если коэффициент misli "
            "выше справедливого, модель считает ставку выгодной (**преимущество > 0**).\n"
            "- **Важно:** проверка на 47 тыс. матчей показала, что закрывающие коэффициенты "
            "букмекеров точнее нашей модели, и добавлять к ним модель почти бесполезно. Поэтому "
            "«преимущество» модели — это место, где модель и букмекер расходятся, а не "
            "гарантированная выгода. Маржа misli ~6–10% на матч, в экспрессе она перемножается.\n"
            "- **MBS** — минимальное число событий в купоне по правилам misli для этого матча.")
    if df.empty:
        render_coupon_sidebar()
        return
    render_news()
    render_suggestions(df)
    view = pd.DataFrame({
        "Время": df["kick_local"].dt.strftime("%d.%m %H:%M"),
        "Турнир": df["competition_az"],
        "Матч": df["home_raw"] + " — " + df["away_raw"],
        "1": df["o1"], "X": df["ox"], "2": df["o2"],
        "Модель П1": df.get("p_o1"), "Модель Х": df.get("p_ox"), "Модель П2": df.get("p_o2"),
        "Лучший вариант по модели": [
            (f"{_market_label(b, l)} @ {r[b]:.2f} ({v:+.0%})" if isinstance(b, str) else "—")
            for b, v, l, (_, r) in zip(df["best_col"], df["best_val"], df["ou_line"],
                                       df.iterrows())],
        "MBS": df["mbs"],
    })
    pc = lambda l: st.column_config.ProgressColumn(l, format="percent", min_value=0, max_value=1)
    event = st.dataframe(view, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", height=min(38 * (len(view) + 1), 560),
                         column_config={"Модель П1": pc("Модель П1"), "Модель Х": pc("Модель Х"),
                                        "Модель П2": pc("Модель П2"),
                                        "1": st.column_config.NumberColumn(format="%.2f"),
                                        "X": st.column_config.NumberColumn(format="%.2f"),
                                        "2": st.column_config.NumberColumn(format="%.2f")})
    rows = event.selection.rows if event and event.selection else []
    if rows:
        r = df.iloc[rows[0]]
        st.divider()
        st.markdown(f"### {r['home_raw']} — {r['away_raw']}")
        st.caption(f"{r['competition_az']} · {r['kick_local']:%d.%m %H:%M} · MBS {r['mbs']}"
                   + (f" · модель: {r['home']} — {r['away']}" if pd.notna(r.get("home")) else
                      " · модель этот матч не знает"))
        head = st.columns([3, 2, 2, 2, 2, 1])
        for h, t in zip(head, ["Исход", "Коэф. misli", "Модель", "Справедл. коэф.",
                               "Преимущество", ""]):
            h.markdown(f"**{t}**")
        for col in ODDS_COLS:
            o = r.get(col)
            if pd.isna(o):
                continue
            p = r.get(f"p_{col}")
            cells = st.columns([3, 2, 2, 2, 2, 1])
            cells[0].write(_market_label(col, r.get("ou_line")))
            cells[1].write(f"{o:.2f}")
            cells[2].write(pct(p) if pd.notna(p) else "—")
            cells[3].write(f"{1 / p:.2f}" if pd.notna(p) and p > 0 else "—")
            if pd.notna(p):
                v = p * o - 1
                cells[4].write(f"{'🟢' if v > 0 else '🔴'} {v:+.0%}")
            else:
                cells[4].write("—")
            if cells[5].button("➕", key=f"add_{r['event_id']}_{col}", help="в купон"):
                _add_pick(r, col)
                st.rerun()
        if pd.notna(r.get("home")):
            with st.expander("Подробный разбор матча моделью"):
                match_card(eng, r["kind"], r["competition"], r["home"], r["away"], False,
                           r.get("home_key"), r.get("away_key"),
                           np.array([r["o1"], r["ox"], r["o2"]], float), None,
                           r["kickoff"].tz_convert(local_tz()).tz_localize(None).normalize())
    render_coupon_sidebar()

    st.caption("Сохранённые купоны и их результаты — на странице «🧾 Мои купоны».")

# ================================================================ my coupons
@st.cache_data(ttl=120, show_spinner="Проверяю результаты…")
def _scores_for(dates: tuple) -> pd.DataFrame:
    from soccer.scores import fetch_results
    try:
        return fetch_results(list(dates))
    except Exception:
        return pd.DataFrame()


def _status_lines(picks: list[dict]) -> tuple[str, list[str]]:
    from soccer.coupons import coupon_status
    days = set()
    for p in picks:
        d = pd.Timestamp(p["kickoff"])
        d = (d.tz_convert("UTC").tz_localize(None) if d.tzinfo else d).normalize()
        days |= {d - timedelta(days=1), d, d + timedelta(days=1)}
    stt = coupon_status(picks, _scores_for(tuple(sorted(days))))
    lines = []
    for r in stt["picks"]:
        if r["result"] is None:
            mark = "⏳ ещё не сыгран" if pd.Timestamp(r["kickoff"]) > pd.Timestamp.now(tz="UTC")                 else "⏳ ждём результат"
        else:
            mark = f"{'✅' if r['won'] else '❌'} {r['result'][0]}:{r['result'][1]}"
        t = pd.Timestamp(r["kickoff"]).tz_convert(user_tz() or local_tz())             if pd.Timestamp(r["kickoff"]).tzinfo else pd.Timestamp(r["kickoff"])
        lines.append(f"- {t:%d.%m %H:%M} · **{r['match']}** — {r['label']} @ {r['odds']:.2f} → {mark}")
    return stt["state"], lines


def page_my_coupons():
    page_header("🧾 Мои купоны", "Ваши купоны и их результаты — обновляются сами")
    st.caption("Результаты подтягиваются сами (livescore.com + misli.az), обновление раз в 2 минуты.")
    draft = _coupon()
    if draft:
        st.markdown("### Текущий купон (ещё не сохранён)")
        state, lines = _status_lines(draft)
        st.markdown("\n".join(lines))
        total = float(np.prod([p["odds"] for p in draft]))
        c = st.columns([2, 2, 3])
        c[0].metric("Общий коэф.", f"{total:.2f}")
        stake = c[1].number_input("Ставка, ₼", min_value=0.0, value=1.0, step=1.0, key="my_stake")
        if c[2].button("💾 Сохранить этот купон", type="primary"):
            known = all(p.get("p_market") is not None for p in draft)
            prob = float(np.prod([p["p_market"] for p in draft])) if known else float("nan")
            storage.save_coupon(draft, stake, total, prob)
            _set_coupon([])
            st.toast("Купон сохранён")
            st.rerun()
        st.divider()
    saved = storage.load_coupons()
    if not saved:
        st.info("Сохранённых купонов пока нет. Соберите купон на странице «Купон (misli.az)» "
                "или загрузите готовый и нажмите «Сохранить».")
        return
    results = []
    blocks = []
    for cpn in saved:
        state, lines = _status_lines(cpn["picks"])
        results.append((state, cpn))
        blocks.append((state, cpn, lines))
    won = [c for s_, c in results if s_ == "won"]
    lost = [c for s_, c in results if s_ == "lost"]
    staked = sum(c["stake"] for c in won + lost)
    returned = sum(c["stake"] * c["total_odds"] for c in won)
    m = st.columns(4)
    m[0].metric("Купонов", len(saved))
    m[1].metric("Выиграно", len(won))
    m[2].metric("Проиграно", len(lost))
    m[3].metric("Баланс по сыгранным", f"{returned - staked:+.2f} ₼")
    head = {"won": "✅ выигран", "lost": "❌ проигран", "pending": "⏳ в игре"}
    for state, cpn, lines in blocks:
        title = (f"#{cpn['id']} · {cpn['created_at'][:16].replace('T', ' ')} · коэф. "
                 f"{cpn['total_odds']:.2f} · ставка {cpn['stake']:.2f} ₼ · {head[state]}"
                 + (f" (+{cpn['stake'] * cpn['total_odds']:.2f} ₼)" if state == "won" else ""))
        with st.expander(title, expanded=state == "pending"):
            st.markdown("\n".join(lines))
            if st.button("Удалить", key=f"del_{cpn['id']}"):
                storage.delete_coupon(cpn["id"])
                st.rerun()


def page_about():
    page_header("ℹ️ Как это работает", "Модели, данные и проверка качества — простыми словами")
    st.markdown("""
**Что прогнозирует анализатор.** Вероятности исходов (П1 / Х / П2), точные счета, тоталы,
«обе забьют» — для матчей 38 национальных лиг, Лиги чемпионов, Лиги Европы, Лиги конференций,
Лиги наций и других матчей сборных.

**Модели**
- **Dixon-Coles** — модель голов: у каждой команды сила атаки и обороны, плюс домашнее
  преимущество. Свежие матчи важнее старых. Новички лиги по умолчанию считаются похожими
  на вылетевшие команды.
- **Elo** — рейтинг, который растёт за победы и падает за поражения (сильнее — за крупный счёт).
  Еврокубки связывают рейтинги разных стран, поэтому клубы разных лиг сравнимы.
- **Итоговый прогноз** для лиг — среднее Dixon-Coles и Elo (так точнее, чем каждая модель по
  отдельности); для еврокубков и сборных — Elo, который на истории оказался точнее.
- **Симуляция сезона** — оставшиеся матчи разыгрываются 10 000 раз с учётом того, что сила
  команд известна неточно; отсюда шансы на чемпионство, еврокубки и вылет.

**Проверка качества.** Каждая модель и каждое улучшение проверяются на прошлых сезонах так,
будто мы делали прогноз до матча (без подглядывания в будущее). Все реальные прогнозы
сохраняются и сверяются с результатами в разделе «Точность».

**Источники данных.** football-data.co.uk (результаты и коэффициенты), fixturedownload.com
(расписание), openfootball (история еврокубков), martj42/international_results (сборные).

**Важно.** Футбол очень случаен: даже хорошая модель угадывает исход примерно в 50–55% матчей
лиг. Вероятности показывают шансы, а не гарантии.
""")


# ========================================================================= login
def _app_password() -> str | None:
    """Password from Streamlit secrets / environment; none set (local use) -> no login."""
    import os
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
    except Exception:  # no secrets file at all
        pass
    return os.environ.get("APP_PASSWORD") or None


def _send_login_code() -> bool:
    """Second factor: a fresh 6-digit code to the owner's Telegram (valid 5 minutes)."""
    import hashlib
    import secrets
    import time
    from soccer.notify import send
    code = f"{secrets.randbelow(1_000_000):06d}"
    st.session_state["2fa"] = {"hash": hashlib.sha256(code.encode()).hexdigest(),
                               "expires": time.time() + 300, "tries": 0, "sent": time.time()}
    when = now_in_tz().strftime("%d.%m %H:%M")
    return send(f"🔐 Код для входа на сайт Soccer Analyser: <b>{code}</b>\n"
                f"Действует 5 минут ({when}).\n"
                "Если это не вы — кто-то знает ваш пароль: смените APP_PASSWORD в Streamlit.")


def require_login():
    """Password, then (when the Telegram bot is configured) a one-time code sent to Telegram."""
    import hashlib
    import hmac
    import time
    from soccer.notify import configured as telegram_ready
    password = _app_password()
    if not password or st.session_state.get("authenticated"):
        return
    _inject_css()
    page_header("⚽ Soccer Analyser", "Вход только для владельца")
    if not st.session_state.get("password_ok"):
        with st.form("login"):
            entered = st.text_input("🔑 Пароль", type="password")
            ok = st.form_submit_button("Войти", type="primary")
        if ok:
            if hmac.compare_digest(entered.encode(), password.encode()):
                if telegram_ready():
                    st.session_state["password_ok"] = True
                    if not _send_login_code():
                        st.session_state.pop("password_ok")
                        st.error("Не удалось отправить код в Telegram — попробуйте ещё раз.")
                        st.stop()
                    st.rerun()
                st.session_state["authenticated"] = True  # no bot configured: password only
                st.rerun()
            time.sleep(1.5)  # slows down password guessing
            st.error("Неверный пароль")
        st.stop()
    # step 2: code from Telegram
    tf = st.session_state.get("2fa") or {}
    st.info("📲 Мы отправили 6-значный код в ваш Telegram (бот Soccer Analyser).")
    with st.form("code"):
        code = st.text_input("Код из Telegram", max_chars=6)
        ok = st.form_submit_button("Подтвердить", type="primary")
    c = st.columns(2)
    if c[0].button("Отправить код ещё раз"):
        if time.time() - tf.get("sent", 0) < 60:
            st.warning("Повторно можно через минуту.")
        else:
            _send_login_code()
            st.success("Новый код отправлен.")
    if c[1].button("Назад к паролю"):
        for k in ("password_ok", "2fa"):
            st.session_state.pop(k, None)
        st.rerun()
    if ok:
        tf["tries"] = tf.get("tries", 0) + 1
        if time.time() > tf.get("expires", 0):
            st.error("Код истёк — нажмите «Отправить код ещё раз».")
        elif tf["tries"] > 5:
            for k in ("password_ok", "2fa"):
                st.session_state.pop(k, None)
            st.error("Слишком много попыток. Введите пароль заново.")
        elif hmac.compare_digest(hashlib.sha256(code.strip().encode()).hexdigest(),
                                 tf.get("hash", "")):
            st.session_state["authenticated"] = True
            st.session_state.pop("2fa", None)
            st.rerun()
        else:
            time.sleep(1)
            st.error(f"Неверный код (попытка {tf['tries']} из 5).")
    st.stop()

require_login()

# ======================================================================== layout
with st.sidebar:
    st.markdown('<div class="sa-brand"><div style="font-size:1.8rem">⚽</div><div>'
                '<b>Soccer Analyser</b><span>прогнозы · купоны · live</span></div></div>',
                unsafe_allow_html=True)
pages = st.navigation([
    st.Page(page_today, title="Матчи дня", icon="📅", default=True, url_path="today"),
    st.Page(page_coupon, title="Купон (misli.az)", icon="🎟️", url_path="coupon"),
    st.Page(page_my_coupons, title="Мои купоны", icon="🧾", url_path="my-coupons"),
    st.Page(page_match, title="Прогноз любого матча", icon="🔮", url_path="match"),
    st.Page(page_leagues, title="Лиги и симуляция", icon="🏆", url_path="leagues"),
    st.Page(page_ratings, title="Рейтинги", icon="📊", url_path="ratings"),
    st.Page(page_accuracy, title="Точность", icon="🎯", url_path="accuracy"),
    st.Page(page_about, title="Как это работает", icon="ℹ️", url_path="about"),
])
with st.sidebar:
    st.selectbox("🕒 Часовой пояс", list(TIMEZONES), key="tz_label",
                 help=f"Часы компьютера: {local_tz()}")
    eng = get_engine()
    st.caption(f"Модели обновлены: {eng.built_at:%d.%m %H:%M}  \n"
               f"Матчей в базе: {len(eng.matches) + len(eng.intl) + len(eng.euro):,}")
    if st.button("🔄 Обновить данные", help="скачать свежие результаты и расписание"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state["refresh_token"] = st.session_state.get("refresh_token", 0) + 1
        st.rerun()
pages.run()
