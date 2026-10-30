"""Load a real race with FastF1 into the same `Race` container the simulator emits.

Optional dependency:  pip install "pitwall-ai[f1]"

    race = load_race(2023, "Bahrain")

Fuel load is not published, so it is estimated from a linear burn profile.
"""
from __future__ import annotations

import numpy as np

from ..physics import C_IDX, TrackConfig
from .schema import Race


def load_race(year: int, gp: str, cache_dir: str = ".fastf1-cache") -> Race:
    try:
        import fastf1
    except ImportError as e:  # pragma: no cover
        raise ImportError('Install FastF1 support with: pip install "pitwall-ai[f1]"') from e

    import os
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    ses = fastf1.get_session(year, gp, "R")
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = ses.laps
    n_laps = int(laps["LapNumber"].max())
    drivers = sorted(laps["Driver"].unique())
    med = float(laps["LapTime"].dt.total_seconds().median())
    pit_loss = float(np.nanmedian(
        (laps.loc[laps["PitInTime"].notna(), "LapTime"].dt.total_seconds() - med).clip(lower=0))) or 21.0
    track = TrackConfig(name=f"{year} {gp}", laps=n_laps, base_lap=med, pit_loss=pit_loss)

    n = len(drivers)
    shape = (n, n_laps)
    lap_time = np.full(shape, np.nan)
    comp = np.full(shape, -1, dtype=int)
    age = np.zeros(shape, dtype=int)
    fuel = np.zeros(shape)
    pit_in = np.zeros(shape, dtype=bool)
    position = np.zeros(shape, dtype=int)
    cum = np.full(shape, np.nan)
    sc = np.zeros(n_laps, dtype=bool)

    for i, d in enumerate(drivers):
        dl = laps.pick_drivers(d)
        for _, r in dl.iterrows():
            j = int(r["LapNumber"]) - 1
            if r["LapTime"] == r["LapTime"]:
                lap_time[i, j] = r["LapTime"].total_seconds()
            comp[i, j] = C_IDX.get(str(r["Compound"]).upper(), -1)
            age[i, j] = int(r["TyreLife"]) if r["TyreLife"] == r["TyreLife"] else 0
            fuel[i, j] = track.fuel_at(j + 1)
            pit_in[i, j] = r["PitInTime"] == r["PitInTime"]
            if r["Position"] == r["Position"]:
                position[i, j] = int(r["Position"])
            if r["Time"] == r["Time"]:
                cum[i, j] = r["Time"].total_seconds()
            status = str(r.get("TrackStatus", ""))
            if "4" in status or "6" in status or "7" in status:  # SC / VSC / VSC ending
                sc[j] = True
    return Race(track, drivers, lap_time, comp, age, fuel, pit_in, sc, position, cum)
