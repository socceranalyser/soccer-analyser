"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from soccer import storage
from soccer.backtest import common_matches
from soccer.config import LEAGUES, season_label
from soccer.data import implied_probs, load_fixtures, load_matches
from soccer.metrics import OUTCOME_INDEX, reliability, summary_table
from soccer.pipeline import current_season, current_teams, fit_dc, fit_elo, record_live
from soccer.probability import markets
from soccer.simulation import add_probabilities, league_table, simulate_season

st.set_page_config(page_title="Soccer Analyser", page_icon="⚽", layout="wide")

MODEL_LABELS = {"dixon_coles": "Dixon-Coles", "elo": "Elo", "market": "Букмекеры"}
OUTCOME_COLORS = {"П1": "#2E7D32", "Х": "#9E9E9E", "П2": "#1565C0"}


# ----------------------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner="Загрузка матчей…")
def get_matches() -> pd.DataFrame:
    return load_matches()


@st.cache_data(ttl=3 * 3600, show_spinner="Загрузка расписания…")
def get_fixtures() -> pd.DataFrame:
    fx = load_fixtures()
    if fx.empty:
        return fx
    return fx[fx["date"] >= pd.Timestamp.today().normalize()]


def data_key(df: pd.DataFrame) -> str:
    return f"{len(df)}_{df['date'].max():%Y%m%d}"


@st.cache_resource(show_spinner="Обучение Dixon-Coles…")
def get_dc(league: str, key: str):
    return fit_dc(get_matches(), league, get_fixtures())


@st.cache_resource(show_spinner="Расчёт Elo…")
def get_elo(key: str):
    return fit_elo(get_matches())


@st.cache_data(ttl=3 * 3600, show_spinner="Запись прогнозов на ближайшие матчи…")
def auto_record() -> dict:
    return record_live(refresh=False)


@st.cache_data(show_spinner="Монте-Карло симуляция сезона…")
def get_simulation(league: str, key: str, n_sims: int, uncertainty: bool):
    matches = get_matches()
    season = current_season(matches, league)
    played = matches[(matches["league"] == league) & (matches["season"] == season)]
    teams = current_teams(matches, league)
    res = simulate_season(get_dc(league, key), played, teams, n_sims=n_sims,
                          param_uncertainty=uncertainty)
    cfg = LEAGUES[league]
    return add_probabilities(res, cfg["top"], cfg["relegated"]), res["position_probs"], \
        res["n_remaining"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_simulation(container):
    with container:
        st.markdown("**Монте-Карло симуляция оставшихся матчей**")
        sc = st.columns(2)
        n_sims = sc[0].select_slider("Симуляций", [1000, 5000, 10000, 20000], value=10000)
        unc = sc[1].toggle("Учитывать неопределённость силы команд (Байес)", value=True)
        summ, posp, n_rem = get_simulation(league, key, n_sims, unc)
        st.caption(f"Осталось матчей: {n_rem}. Тай-брейк: разница, затем забитые.")
        cfg = LEAGUES[league]
        show = summ.rename(columns={
            "team": "Команда", "pts_now": "Очки", "exp_pts": "Ожид. очки",
            "pts_p10": "P10", "pts_p90": "P90", "exp_pos": "Ожид. место",
            "p_title": "Чемпион", f"p_top{cfg['top']}": f"Топ-{cfg['top']}",
            "p_relegation": "Вылет"})
        st.dataframe(
            show.style.format({"Ожид. очки": "{:.1f}", "P10": "{:.0f}", "P90": "{:.0f}",
                               "Ожид. место": "{:.1f}", "Чемпион": "{:.1%}",
                               f"Топ-{cfg['top']}": "{:.1%}", "Вылет": "{:.1%}"}),
            hide_index=True, use_container_width=True, height=35 * (len(teams) + 1) + 3)
    order = summ["team"].tolist()
    heat = px.imshow(posp.loc[order], color_continuous_scale="Blues", aspect="auto",
                     text_auto=".0%", labels=dict(x="Место", y="", color="P"))
    heat.update_traces(texttemplate="%{z:.0%}")
    heat.update_layout(height=40 + 26 * len(order), margin=dict(l=0, r=0, t=10, b=0),
                       coloraxis_showscale=False)
    st.markdown("**Распределение итоговых мест**")
    st.plotly_chart(heat, use_container_width=True)


# ------------------------------------------------------------------------- sidebar
matches = get_matches()
key = data_key(matches)
try:
    rec = auto_record()
except Exception as exc:  # network hiccups must not break the dashboard
    rec = {"error": str(exc)}

with st.sidebar:
    st.title("⚽ Soccer Analyser")
    countries = list(dict.fromkeys(v["country"] for v in LEAGUES.values()))
    country = st.selectbox("Страна", countries)
    league = st.selectbox("Лига", [c for c, v in LEAGUES.items() if v["country"] == country],
                          format_func=lambda c: LEAGUES[c]["name"].split(" — ", 1)[-1])
    st.caption(f"Данные: {len(matches):,} матчей, последний — {matches['date'].max():%d.%m.%Y}")
    if "saved" in rec:
        st.caption(f"Записано прогнозов на ближайшие матчи: {rec['saved']}")
    if st.button("Обновить данные и прогнозы"):
        st.cache_data.clear()
        st.cache_resource.clear()
        with st.spinner("Обновление…"):
            r = record_live(refresh=True)
        st.success(f"Матчей: {r['matches']:,}, записано прогнозов: {r['saved']}")
        st.rerun()

elo = get_elo(key)
dc = get_dc(league, key)
fixtures = get_fixtures()
league_fx = fixtures[fixtures["league"] == league] if not fixtures.empty else fixtures
teams = current_teams(matches, league, fixtures)
same_country = [c for c, v in LEAGUES.items() if v["country"] == LEAGUES[league]["country"]]
country_matches = matches[matches["league"].isin(same_country)]

tab_match, tab_table, tab_ratings, tab_acc = st.tabs(
    ["Матч", "Таблица и симуляция", "Рейтинги", "Точность"])

# --------------------------------------------------------------------------- match
with tab_match:
    mode_opts = ["Ближайшие матчи", "Любая пара"] if not league_fx.empty else ["Любая пара"]
    mode = st.radio("Выбор матча", mode_opts, horizontal=True)
    fx_row = None
    if mode == "Ближайшие матчи":
        labels = [f"{r.date:%d.%m %a} {r.time if isinstance(r.time, str) else ''} — "
                  f"{r.home} vs {r.away}" for r in league_fx.itertuples()]
        i = st.selectbox("Матч", range(len(labels)), format_func=lambda k: labels[k])
        fx_row = league_fx.iloc[i]
        home, away = fx_row["home"], fx_row["away"]
    else:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Хозяева", teams, index=0)
        away = c2.selectbox("Гости", [t for t in teams if t != home], index=0)

    m = dc.score_matrices([home], [away])[0]
    mk = markets(m)
    elo_p = elo.predict([home], [away], league)
    rows = {"Dixon-Coles": [mk["p_home"], mk["p_draw"], mk["p_away"]],
            "Elo": list(elo_p["probs"][0])}
    if fx_row is not None and pd.notna(fx_row["odds_h"]):
        rows["Букмекеры"] = list(implied_probs([fx_row["odds_h"]], [fx_row["odds_d"]],
                                               [fx_row["odds_a"]])[0])

    st.subheader(f"{home} — {away}")
    c = st.columns(5)
    c[0].metric("П1", pct(mk["p_home"]))
    c[1].metric("Х", pct(mk["p_draw"]))
    c[2].metric("П2", pct(mk["p_away"]))
    c[3].metric("Ожид. голы", f"{mk['xg_home']:.2f} : {mk['xg_away']:.2f}")
    c[4].metric("Тотал > 2.5", pct(mk["totals"][2.5]))

    left, right = st.columns([3, 2])
    with left:
        probs_df = pd.DataFrame(rows, index=["П1", "Х", "П2"]).T
        fig = go.Figure()
        for outcome in ["П1", "Х", "П2"]:
            fig.add_bar(y=probs_df.index, x=probs_df[outcome], name=outcome, orientation="h",
                        marker_color=OUTCOME_COLORS[outcome],
                        text=[pct(v) for v in probs_df[outcome]], textposition="inside")
        fig.update_layout(barmode="stack", height=80 + 50 * len(rows), xaxis_tickformat=".0%",
                          margin=dict(l=0, r=0, t=30, b=0), xaxis_range=[0, 1],
                          legend=dict(orientation="h", y=1.15, x=0, traceorder="normal"))
        st.plotly_chart(fig, use_container_width=True)

        g = 7
        z = m[:g, :g]
        heat = px.imshow(z, text_auto=".1%", color_continuous_scale="Greens", aspect="auto",
                         labels=dict(x=f"Голы {away}", y=f"Голы {home}", color="P"))
        heat.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                           title="Вероятности точного счёта", coloraxis_showscale=False)
        st.plotly_chart(heat, use_container_width=True)
    with right:
        st.markdown("**Вероятные счета**")
        st.dataframe(pd.DataFrame(mk["top_scores"], columns=["Счёт", "P"])
                     .assign(P=lambda d: d["P"].map(pct)), hide_index=True,
                     use_container_width=True)
        st.markdown("**Тоталы и другие рынки**")
        tot = [{"Рынок": f"Тотал больше {k}", "P": pct(v)} for k, v in mk["totals"].items()]
        tot += [{"Рынок": f"Тотал меньше {k}", "P": pct(1 - v)} for k, v in mk["totals"].items()
                if k in (1.5, 2.5, 3.5)]
        tot += [{"Рынок": "Обе забьют", "P": pct(mk["btts"])},
                {"Рынок": f"{home} на ноль", "P": pct(mk["home_clean_sheet"])},
                {"Рынок": f"{away} на ноль", "P": pct(mk["away_clean_sheet"])}]
        st.dataframe(pd.DataFrame(tot), hide_index=True, use_container_width=True)

    st.markdown("### Форма")
    fc = st.columns(2)
    for col, team in zip(fc, (home, away)):
        tm = country_matches[(country_matches["home"] == team)
                             | (country_matches["away"] == team)].tail(8).iloc[::-1]
        if tm.empty:
            col.info(f"Нет истории для {team}")
            continue
        is_home = tm["home"] == team
        gf = np.where(is_home, tm["hg"], tm["ag"])
        ga = np.where(is_home, tm["ag"], tm["hg"])
        res = np.where(gf > ga, "В", np.where(gf == ga, "Н", "П"))
        form = pd.DataFrame({
            "Дата": tm["date"].dt.strftime("%d.%m.%y"),
            "": np.where(is_home, "Д", "Г"),
            "Соперник": np.where(is_home, tm["away"], tm["home"]),
            "Счёт": [f"{a}:{b}" for a, b in zip(gf, ga)],
            "Итог": res,
        })
        if tm["hxg"].notna().any():
            xgf = np.where(is_home, tm["hxg"], tm["axg"])
            xga = np.where(is_home, tm["axg"], tm["hxg"])
            form["xG"] = [f"{a:.2f}:{b:.2f}" if pd.notna(a) else "" for a, b in zip(xgf, xga)]
        pts = (res == "В").sum() * 3 + (res == "Н").sum()
        col.markdown(f"**{team}** — {''.join(res[:5][::-1])} · {pts / len(res):.2f} очка/матч, "
                     f"голы {gf.mean():.2f}:{ga.mean():.2f}")
        col.dataframe(form, hide_index=True, use_container_width=True)

    st.markdown("### Elo")
    ec = st.columns(3)
    ec[0].metric(f"Elo {home}", f"{elo_p['elo_h'][0]:.0f}")
    ec[1].metric(f"Elo {away}", f"{elo_p['elo_a'][0]:.0f}")
    ec[2].metric("Разница (с учётом поля)",
                 f"{elo_p['elo_h'][0] + elo.home_adv - elo_p['elo_a'][0]:+.0f}")
    hist = elo.history[elo.history["league"].isin(same_country)]
    since = matches["date"].max() - pd.Timedelta(days=800)
    series = []
    for team in (home, away):
        h = hist[(hist["home"] == team) & (hist["date"] >= since)][["date", "elo_h"]]
        a = hist[(hist["away"] == team) & (hist["date"] >= since)][["date", "elo_a"]]
        s = pd.concat([h.rename(columns={"elo_h": "elo"}), a.rename(columns={"elo_a": "elo"})])
        series.append(s.assign(team=team))
    eh = pd.concat(series).sort_values("date")
    if not eh.empty:
        fig = px.line(eh, x="date", y="elo", color="team", labels={"date": "", "elo": "Elo"})
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- table
with tab_table:
    season = current_season(matches, league, fixtures)
    played = matches[(matches["league"] == league) & (matches["season"] == season)]
    st.subheader(f"Сезон {season_label(league, season)}")
    lc, rc = st.columns([2, 3])
    with lc:
        st.markdown("**Текущая таблица**")
        st.dataframe(league_table(played, teams), hide_index=True, use_container_width=True,
                     height=35 * (len(teams) + 1) + 3)
    if LEAGUES[league]["sim"]:
        render_simulation(rc)
    else:
        rc.info("Симуляция сезона недоступна: в этой лиге есть разделение таблицы, плей-офф, "
                "конференции или больше двух встреч пар за сезон. Прогнозы матчей работают.")

# ------------------------------------------------------------------------- ratings
with tab_ratings:
    r = dc.ratings()
    r = r[r["team"].isin(teams)]
    r["elo"] = r["team"].map(lambda t: elo.rating(t, league))
    lc, rc = st.columns([3, 2])
    with lc:
        fig = px.scatter(r, x="attack", y="defence", text="team",
                         error_x="attack_sd", error_y="defence_sd",
                         labels={"attack": "Атака (выше — больше забивает)",
                                 "defence": "Оборона (выше — меньше пропускает)"})
        fig.update_traces(textposition="top center", marker_size=9,
                          error_x_thickness=1, error_y_thickness=1, error_x_width=0,
                          error_y_width=0, error_x_color="rgba(120,120,120,0.35)",
                          error_y_color="rgba(120,120,120,0.35)")
        fig.add_hline(y=0, line_dash="dot", opacity=0.4)
        fig.add_vline(x=0, line_dash="dot", opacity=0.4)
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Параметры Dixon-Coles (лог-шкала, ±1σ апостериорной неопределённости). "
                   f"Домашнее преимущество: {np.exp(dc.theta[-2]):.2f}× к голам хозяев, "
                   f"ρ = {dc.rho:.3f}.")
    with rc:
        st.dataframe(r[["team", "attack", "defence", "strength", "elo"]]
                     .rename(columns={"team": "Команда", "attack": "Атака",
                                      "defence": "Оборона", "strength": "Сила", "elo": "Elo"})
                     .style.format({"Атака": "{:+.2f}", "Оборона": "{:+.2f}",
                                    "Сила": "{:+.2f}", "Elo": "{:.0f}"}),
                     hide_index=True, use_container_width=True, height=600)

# ------------------------------------------------------------------------ accuracy
with tab_acc:
    runs = storage.load_runs()
    if runs.empty:
        st.info("Прогнозов пока нет. Запустите `python scripts/run_backtest.py`.")
    else:
        labels = {r.run_id: f"{'Живые прогнозы' if r.kind == 'live' else 'Бэктест'} · "
                            f"{r.run_id} · {r.description}" for r in runs.itertuples()}
        run_id = st.selectbox("Набор прогнозов", list(labels), format_func=labels.get)
        preds = storage.load_predictions(run_id)
        done = preds.dropna(subset=["result"])
        pending = preds[preds["result"].isna()]
        scope = st.radio("Лиги", ["Все", "Текущая"], horizontal=True)
        if scope == "Текущая":
            done = done[done["league"] == league]
            pending = pending[pending["league"] == league]
        st.caption(f"Сыграно: {done['league'].size // max(done['model'].nunique(), 1):,} матчей "
                   f"· ожидают результата: {pending['league'].size // max(pending['model'].nunique(), 1)}")
        if done.empty:
            st.info("Ещё нет сыгранных матчей с сохранёнными прогнозами.")
        else:
            models = sorted(done["model"].unique())
            fair = common_matches(done, models)
            tbl = summary_table(fair).assign(model=lambda d: d["model"].map(MODEL_LABELS))
            st.markdown("**Метрики** (на матчах, где есть прогноз каждой модели; ниже — лучше, "
                        "кроме точности)")
            st.dataframe(tbl.rename(columns={
                "model": "Модель", "n": "Матчей", "log_loss": "Log loss", "rps": "RPS",
                "brier": "Brier", "accuracy": "Точность исхода", "ece": "ECE",
                "ou25_log_loss": "Log loss ТБ2.5"}).style.format(
                {"Log loss": "{:.4f}", "RPS": "{:.4f}", "Brier": "{:.4f}",
                 "Точность исхода": "{:.1%}", "ECE": "{:.4f}", "Log loss ТБ2.5": "{:.4f}"}),
                hide_index=True, use_container_width=True)
            if done["season"].nunique() > 1 or scope == "Все":
                by = ("season", "model") if done["season"].nunique() > 1 else ("league", "model")
                t2 = summary_table(fair, by=by)
                fig = px.bar(t2.assign(model=t2["model"].map(MODEL_LABELS)), x=by[0], y="rps",
                             color="model", barmode="group",
                             labels={"rps": "RPS", by[0]: "", "model": ""})
                fig.update_yaxes(range=[t2["rps"].min() * 0.97, t2["rps"].max() * 1.01])
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

            lc, rc = st.columns(2)
            with lc:
                outcome = st.radio("Калибровка исхода", ["П1", "Х", "П2"], horizontal=True)
                lab = {"П1": "H", "Х": "D", "П2": "A"}[outcome]
                fig = go.Figure()
                fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Идеал",
                                line=dict(dash="dot", color="gray"))
                for mdl, g in fair.groupby("model"):
                    p = g[["p_home", "p_draw", "p_away"]].to_numpy()
                    rel = reliability(p, g["result"].map(OUTCOME_INDEX).to_numpy(), bins=10)
                    rel = rel[(rel["outcome"] == lab) & (rel["n"] >= 15)]
                    fig.add_scatter(x=rel["predicted"], y=rel["observed"], mode="lines+markers",
                                    name=MODEL_LABELS.get(mdl, mdl),
                                    marker_size=np.sqrt(rel["n"]) / 1.5)
                fig.update_layout(height=380, xaxis_title="Прогноз", yaxis_title="Факт",
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
            with rc:
                if "market" in models:
                    st.markdown("**Накопленная разница log loss против букмекеров** "
                                "(ниже нуля — модель лучше рынка)")
                    piv = fair.copy()
                    y = piv["result"].map(OUTCOME_INDEX).to_numpy()
                    p = piv[["p_home", "p_draw", "p_away"]].to_numpy()
                    piv["ll"] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None))
                    w = piv.pivot_table(index=["date", "home"], columns="model", values="ll")
                    w = w.sort_index()
                    diff = w.drop(columns="market").sub(w["market"], axis=0).cumsum()
                    diff = diff.reset_index().melt(id_vars=["date", "home"], var_name="model",
                                                   value_name="cum")
                    fig = px.line(diff.assign(model=diff["model"].map(MODEL_LABELS)), x="date",
                                  y="cum", color="model", labels={"cum": "", "date": "", "model": ""})
                    fig.add_hline(y=0, line_dash="dot")
                    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Последние прогнозы**")
            last = done[done["model"] == "dixon_coles"].sort_values("date", ascending=False).head(30)
            st.dataframe(last.assign(
                Матч=last["home"] + " — " + last["away"], Счёт=last["hg"].astype("Int64").astype(str)
                + ":" + last["ag"].astype("Int64").astype(str),
                П1=last["p_home"].map(pct), Х=last["p_draw"].map(pct), П2=last["p_away"].map(pct))
                [["date", "league", "Матч", "П1", "Х", "П2", "Счёт"]]
                .rename(columns={"date": "Дата", "league": "Лига"}),
                hide_index=True, use_container_width=True)
        if not pending.empty:
            st.markdown("**Ожидают результата**")
            pv = pending.pivot_table(index=["date", "league", "home", "away"], columns="model",
                                     values="p_home").reset_index()
            st.dataframe(pv.rename(columns={**MODEL_LABELS, "date": "Дата", "league": "Лига",
                                            "home": "Хозяева", "away": "Гости"}),
                         hide_index=True, use_container_width=True)
