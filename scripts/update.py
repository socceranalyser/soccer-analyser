"""Refresh data, store finished results and forecast upcoming top-5 fixtures.

Run daily (e.g. Windows Task Scheduler):  .venv\\Scripts\\python.exe scripts\\update.py
"""
from soccer.pipeline import record_live

if __name__ == "__main__":
    r = record_live(refresh=True)
    print(f"matches: {r['matches']:,} | upcoming fixtures: {r['fixtures']} | "
          f"forecast rows saved: {r['saved']}")
