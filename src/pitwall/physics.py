"""Ground-truth race physics.

The simulator and the closed-loop backtest use this as "reality". The strategy
engine never sees these parameters: it must learn them from noisy practice and
live race laps, which is what makes the evaluation honest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
C_IDX = {c: i for i, c in enumerate(COMPOUNDS)}


@dataclass(frozen=True)
class TyreTruth:
    offset: float      # s/lap slower than SOFT when new
    wear: float        # linear degradation, s per lap of age
    quad: float        # quadratic degradation
    cliff_age: int     # age after which performance falls off a cliff
    cliff_k: float     # cliff severity

    def deg(self, age):
        age = np.asarray(age, dtype=float)
        over = np.maximum(age - self.cliff_age, 0.0)
        return self.offset + self.wear * age + self.quad * age**2 + self.cliff_k * over**1.6


TRUE_TYRES = {
    "SOFT": TyreTruth(0.00, 0.060, 0.0020, 16, 0.12),
    "MEDIUM": TyreTruth(0.55, 0.035, 0.0008, 28, 0.10),
    "HARD": TyreTruth(0.95, 0.020, 0.0004, 40, 0.10),
}


@dataclass(frozen=True)
class TrackConfig:
    name: str = "Synthetic GP"
    laps: int = 57
    base_lap: float = 90.0
    pit_loss: float = 21.5        # green-flag time lost in the pit lane
    sc_pit_factor: float = 0.45   # share of pit loss paid when pitting under SC
    sc_lap_factor: float = 1.45   # SC lap time / base lap time
    fuel_start: float = 105.0     # kg
    fuel_effect: float = 0.032    # s per kg
    lap_noise: float = 0.25       # s, 1-sigma
    sc_base_hazard: float = 0.010  # per green lap

    @property
    def fuel_per_lap(self) -> float:
        return self.fuel_start / self.laps

    def fuel_at(self, lap: int) -> float:
        """Fuel load (kg) at the start of `lap` (1-indexed)."""
        return max(self.fuel_start - self.fuel_per_lap * (lap - 1), 0.0)


def sc_hazard_truth(lap, rain, since, base=0.010):
    """True per-lap probability that a safety car is deployed."""
    lap = np.asarray(lap, dtype=float)
    since = np.asarray(since, dtype=float)
    h = base * (1.0 + 1.5 * rain) + np.where(lap <= 2, 0.04, 0.0)
    h = h * np.where(since < 8, 0.15, 1.0)
    return np.clip(h, 0.0, 0.5)


def sample_sc_laps(track: TrackConfig, rng: np.random.Generator, rain: float = 0.0) -> np.ndarray:
    """Boolean mask (laps,) of laps run under safety car."""
    sc = np.zeros(track.laps, dtype=bool)
    remaining, since = 0, 99
    for i in range(track.laps):
        lap = i + 1
        if remaining > 0:
            sc[i] = True
            remaining -= 1
            since = 0
        elif rng.random() < float(sc_hazard_truth(lap, rain, since, track.sc_base_hazard)):
            sc[i] = True
            remaining = int(rng.integers(3, 6)) - 1
            since = 0
        else:
            since += 1
    return sc


class TruthCar:
    """Steps one car through the race using true tyre physics."""

    def __init__(self, track: TrackConfig, base_pace: float, compound: str):
        self.track = track
        self.base = base_pace
        self.compound = compound
        self.age = 0

    def step(self, lap: int, sc: bool, pit_to: str | None = None,
             lap_noise: float = 0.0, pit_noise: float = 0.0, slow_stop: float = 0.0):
        """Run one lap. Returns (lap_time, compound_used, age_used, pitted)."""
        tr = self.track
        if not sc:  # tyres barely wear behind the safety car
            self.age += 1
        used, age = self.compound, self.age
        if sc:
            t = tr.sc_lap_factor * tr.base_lap
        else:
            t = (self.base + float(TRUE_TYRES[used].deg(age))
                 + tr.fuel_effect * tr.fuel_at(lap) + lap_noise)
        if pit_to is not None:
            loss = tr.pit_loss * (tr.sc_pit_factor if sc else 1.0) + pit_noise + slow_stop
            t += loss
            self.compound, self.age = pit_to, 0
        return t, used, age, pit_to is not None
