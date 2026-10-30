"""Vectorised Monte Carlo race simulator for the remainder of a race.

Every candidate strategy is evaluated against the *same* random scenarios
(common random numbers), so strategy-to-strategy differences are low variance
and "probability this plan is fastest" is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.safety_car import SafetyCarModel
from ..models.tyre_deg import TyreDegModel
from ..physics import TrackConfig
from .types import RaceState, Stops

MAX_STOPS = 6


@dataclass
class Scenarios:
    sc: np.ndarray         # (n, m) bool, laps under SC
    noise: np.ndarray      # (n, m) lap-time noise
    pit_noise: np.ndarray  # (n, MAX_STOPS)
    slow: np.ndarray       # (n, MAX_STOPS) slow-stop penalty

    @property
    def n(self) -> int:
        return self.sc.shape[0]

    @property
    def m(self) -> int:
        return self.sc.shape[1]

    @classmethod
    def sample(cls, state: RaceState, sc_model: SafetyCarModel, sigma: float, n: int,
               rng: np.random.Generator) -> "Scenarios":
        m = state.total_laps - state.lap
        sc = np.zeros((n, m), dtype=bool)
        rem = np.full(n, state.sc_remaining, dtype=int)
        since = np.full(n, state.laps_since_sc, dtype=float)
        for j in range(m):
            lap = state.lap + 1 + j
            active = rem > 0
            h = sc_model.hazard(lap, state.rain_prob, since)
            start = (~active) & (rng.random(n) < h)
            now = active | start
            sc[:, j] = now
            rem = np.where(start, rng.integers(3, 6, n) - 1, np.where(active, rem - 1, 0))
            since = np.where(now, 0.0, since + 1.0)
        return cls(sc, rng.normal(0.0, sigma, (n, m)), rng.normal(0.0, 0.5, (n, MAX_STOPS)),
                   (rng.random((n, MAX_STOPS)) < 0.03) * rng.uniform(3, 7, (n, MAX_STOPS)))


def simulate_strategy(state: RaceState, stops: Stops, model: TyreDegModel, track: TrackConfig,
                      scen: Scenarios) -> np.ndarray:
    """Remaining race time (n,) for one strategy under every scenario."""
    base = model.base(state.driver)
    pit_at = {lap: c for lap, c in stops}
    comp = state.compound
    age = np.full(scen.n, state.tyre_age, dtype=int)
    total = np.zeros(scen.n)
    sc_lap = track.sc_lap_factor * track.base_lap
    k = 0
    for j in range(scen.m):
        lap = state.lap + 1 + j
        sc = scen.sc[:, j]
        age = age + (~sc)  # tyres barely wear behind the safety car
        green = base + model.curve_at(comp, age) + model.fuel_effect * track.fuel_at(lap)
        t = np.where(sc, sc_lap, green + scen.noise[:, j])
        if lap in pit_at:
            t = t + track.pit_loss * np.where(sc, track.sc_pit_factor, 1.0) + scen.pit_noise[:, k] + scen.slow[:, k]
            comp, age, k = pit_at[lap], np.zeros(scen.n, dtype=int), k + 1
        total += t
    return total
