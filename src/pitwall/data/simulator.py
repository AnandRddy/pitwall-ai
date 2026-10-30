"""Synthetic but physically structured F1 race and practice generator.

Used for offline development, CI and the closed-loop backtest. The same
`Race` / `LapTable` containers are produced by `fastf1_loader` from real data.
"""
from __future__ import annotations

import numpy as np

from ..physics import COMPOUNDS, C_IDX, TRUE_TYRES, TrackConfig, TruthCar, sample_sc_laps
from .schema import LapTable, Race

Plan = tuple[str, tuple[tuple[int, str], ...]]  # (start compound, ((pit_lap, next_compound), ...))


def random_plan(track: TrackConfig, rng: np.random.Generator) -> Plan:
    start = str(rng.choice(COMPOUNDS, p=[0.35, 0.45, 0.20]))
    if rng.random() < 0.6:
        nxt = [c for c in COMPOUNDS if c != start]
        return start, ((int(rng.integers(18, 33)), str(rng.choice(nxt))),)
    laps = sorted(int(x) for x in (rng.integers(12, 23), rng.integers(30, 43)))
    c2 = str(rng.choice(COMPOUNDS))
    c3 = str(rng.choice(COMPOUNDS))
    if len({start, c2, c3}) < 2:
        c3 = str(rng.choice([c for c in COMPOUNDS if c != start]))
    return start, ((laps[0], c2), (laps[1], c3))


def simulate_race(track: TrackConfig | None = None, n_cars: int = 20, seed: int = 0, rain: float = 0.0,
                  plans: dict[str, Plan] | None = None, sc: np.ndarray | None = None):
    """Simulate a full grid. Returns (Race, true_base_pace dict)."""
    track = track or TrackConfig()
    rng = np.random.default_rng(seed)
    L = track.laps
    drivers = [f"CAR{i + 1:02d}" for i in range(n_cars)]
    offsets = rng.normal(0.0, 0.35, n_cars)
    true_base = {d: track.base_lap + float(o) for d, o in zip(drivers, offsets)}
    sc = sample_sc_laps(track, rng, rain) if sc is None else sc

    lap_time = np.zeros((n_cars, L))
    comp = np.zeros((n_cars, L), dtype=int)
    age = np.zeros((n_cars, L), dtype=int)
    fuel = np.zeros((n_cars, L))
    pit_in = np.zeros((n_cars, L), dtype=bool)

    for i, d in enumerate(drivers):
        start, stops = (plans or {}).get(d) or random_plan(track, rng)
        pit_map = {lap: c for lap, c in stops}
        car = TruthCar(track, true_base[d], start)
        for lap in range(1, L + 1):
            t, c, a, pitted = car.step(
                lap, bool(sc[lap - 1]), pit_map.get(lap) if lap < L else None,
                lap_noise=rng.normal(0.0, track.lap_noise), pit_noise=rng.normal(0.0, 0.5),
                slow_stop=float(rng.random() < 0.03) * rng.uniform(3, 7))
            lap_time[i, lap - 1], comp[i, lap - 1], age[i, lap - 1], pit_in[i, lap - 1] = t, C_IDX[c], a, pitted
            fuel[i, lap - 1] = track.fuel_at(lap)

    # cumulative time with grid offsets and gap compression when the SC period ends
    cum = np.zeros((n_cars, L))
    run = np.arange(n_cars) * 0.35
    position = np.zeros((n_cars, L), dtype=int)
    for j in range(L):
        run = run + lap_time[:, j]
        if sc[j] and (j + 1 == L or not sc[j + 1]):
            order = np.argsort(run)
            ranks = np.empty(n_cars)
            ranks[order] = np.arange(n_cars)
            run = run[order[0]] + ranks * 0.9
        cum[:, j] = run
        position[np.argsort(run), j] = np.arange(1, n_cars + 1)
    race = Race(track, drivers, lap_time, comp, age, fuel, pit_in, sc, position, cum)
    return race, true_base


def simulate_practice(track: TrackConfig, true_base: dict[str, float], seed: int = 0) -> LapTable:
    """Long-run practice laps: every driver runs two compounds, short stints, no cliffs reached."""
    rng = np.random.default_rng(seed + 10_000)
    max_run = {"SOFT": 14, "MEDIUM": 20, "HARD": 22}
    drv, cmp_, ag, fu, lt = [], [], [], [], []
    for d, base in true_base.items():
        for c in rng.choice(COMPOUNDS, size=2, replace=False):
            n = int(rng.integers(8, max_run[c] + 1))
            f0 = rng.uniform(35, 95)
            for a in range(1, n + 1):
                f = max(f0 - track.fuel_per_lap * (a - 1), 5.0)
                t = base + float(TRUE_TYRES[c].deg(a)) + track.fuel_effect * f + rng.normal(0, track.lap_noise)
                drv.append(d); cmp_.append(C_IDX[c]); ag.append(a); fu.append(f); lt.append(t)
    return LapTable(np.array(drv, dtype=object), np.array(cmp_), np.array(ag), np.array(fu), np.array(lt))
