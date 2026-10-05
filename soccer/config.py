"""Global configuration: leagues, seasons, paths."""
from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "predictions.db"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
EXTRA_URL = "https://www.football-data.co.uk/new/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
EXTRA_FIXTURES_URL = "https://www.football-data.co.uk/new_league_fixtures.csv"


def _lg(name, country, tier=1, source="main", sim=True, top=4, relegated=3, extra_name=None):
    return {"name": name, "country": country, "tier": tier, "source": source, "sim": sim,
            "top": top, "relegated": relegated, "extra_name": extra_name}


# sim=False: league has splits/playoffs/conferences or >2 meetings per pair, so the
# double-round-robin Monte Carlo would be wrong.
LEAGUES = {
    "E0": _lg("Англия — Премьер-лига", "Англия", 1, top=4, relegated=3),
    "E1": _lg("Англия — Чемпионшип", "Англия", 2, top=2, relegated=3),
    "E2": _lg("Англия — Лига 1", "Англия", 3, top=2, relegated=4),
    "E3": _lg("Англия — Лига 2", "Англия", 4, top=3, relegated=2),
    "EC": _lg("Англия — Национальная лига", "Англия", 5, top=1, relegated=4),
    "SC0": _lg("Шотландия — Премьершип", "Шотландия", 1, sim=False),
    "SC1": _lg("Шотландия — Чемпионшип", "Шотландия", 2, sim=False),
    "SC2": _lg("Шотландия — Лига 1", "Шотландия", 3, sim=False),
    "SC3": _lg("Шотландия — Лига 2", "Шотландия", 4, sim=False),
    "D1": _lg("Германия — Бундеслига", "Германия", 1, top=4, relegated=2),
    "D2": _lg("Германия — 2. Бундеслига", "Германия", 2, top=2, relegated=2),
    "I1": _lg("Италия — Серия A", "Италия", 1, top=4, relegated=3),
    "I2": _lg("Италия — Серия B", "Италия", 2, top=2, relegated=3),
    "SP1": _lg("Испания — Ла Лига", "Испания", 1, top=4, relegated=3),
    "SP2": _lg("Испания — Сегунда", "Испания", 2, top=2, relegated=4),
    "F1": _lg("Франция — Лига 1", "Франция", 1, top=3, relegated=2),
    "F2": _lg("Франция — Лига 2", "Франция", 2, top=2, relegated=2),
    "N1": _lg("Нидерланды — Эредивизи", "Нидерланды", 1, top=2, relegated=2),
    "B1": _lg("Бельгия — Про-лига", "Бельгия", 1, sim=False),
    "P1": _lg("Португалия — Примейра", "Португалия", 1, top=2, relegated=2),
    "T1": _lg("Турция — Суперлига", "Турция", 1, top=2, relegated=3),
    "G1": _lg("Греция — Суперлига", "Греция", 1, sim=False),
    # one-file-per-country leagues (football-data.co.uk/new/), data from 2012
    "ARG": _lg("Аргентина — Лига Професьональ", "Аргентина", source="extra", sim=False),
    "AUT": _lg("Австрия — Бундеслига", "Австрия", source="extra", sim=False),
    "BRA": _lg("Бразилия — Серия A", "Бразилия", source="extra", top=4, relegated=4),
    "CHN": _lg("Китай — Суперлига", "Китай", source="extra", top=3, relegated=2),
    "DNK": _lg("Дания — Суперлига", "Дания", source="extra", sim=False),
    "FIN": _lg("Финляндия — Вейккауслига", "Финляндия", source="extra", sim=False),
    "IRL": _lg("Ирландия — Премьер-дивизион", "Ирландия", source="extra", sim=False),
    "JPN": _lg("Япония — Джей-лига", "Япония", source="extra", top=3, relegated=3),
    "MEX": _lg("Мексика — Лига MX", "Мексика", source="extra", sim=False),
    "NOR": _lg("Норвегия — Элитсериен", "Норвегия", source="extra", top=3, relegated=2),
    "POL": _lg("Польша — Экстракласа", "Польша", source="extra", top=3, relegated=3),
    "ROU": _lg("Румыния — Суперлига", "Румыния", source="extra", sim=False),
    # "RUS" (Russia) removed 2026-10-05 at the user's request (source also stopped updating in Aug 2026)
    "SWE": _lg("Швеция — Аллсвенскан", "Швеция", source="extra", top=3, relegated=2),
    "SWZ": _lg("Швейцария — Суперлига", "Швейцария", source="extra", sim=False),
    "USA": _lg("США — MLS", "США", source="extra", sim=False),
}
CALENDAR_YEAR = {"ARG", "BRA", "CHN", "FIN", "IRL", "NOR", "SWE", "USA"}  # season = year

# country name used inside the extra-league files / new_league_fixtures.csv
EXTRA_COUNTRY = {
    "ARG": "Argentina", "AUT": "Austria", "BRA": "Brazil", "CHN": "China", "DNK": "Denmark",
    "FIN": "Finland", "IRL": "Ireland", "JPN": "Japan", "MEX": "Mexico", "NOR": "Norway",
    "POL": "Poland", "ROU": "Romania", "RUS": "Russia", "SWE": "Sweden", "SWZ": "Switzerland",
    "USA": "USA",
}
TOP5 = ["E0", "SP1", "I1", "D1", "F1"]

FIRST_SEASON = 2005  # 2005/06 — earliest season we download


def season_code(start_year: int) -> str:
    """2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """European (Aug-May) seasons: the start year of the season in progress."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def all_seasons(today: date | None = None) -> list[int]:
    return list(range(FIRST_SEASON, current_season_start(today) + 1))


def season_label(league: str, season: int) -> str:
    if league in CALENDAR_YEAR:
        return str(season)
    return f"{season}/{(season + 1) % 100:02d}"
