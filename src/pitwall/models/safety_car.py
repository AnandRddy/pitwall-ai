"""Safety-car hazard model.

A logistic regression learns P(SC deploys on lap l | race context) from
historical race histories (simulated here; swap in FastF1 `TrackStatus` history
for real circuits). The per-lap hazard feeds the Monte Carlo strategy engine.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..physics import TrackConfig, sc_hazard_truth


def _features(lap, rain, since):
    lap, rain, since = np.broadcast_arrays(np.asarray(lap, float), np.asarray(rain, float), np.asarray(since, float))
    return np.stack([(lap <= 2).astype(float), rain, (since < 8).astype(float), lap / 60.0], axis=-1)


class SafetyCarModel:
    def __init__(self):
        self.coef = np.zeros(4)
        self.intercept = -4.5

    def hazard(self, lap, rain, since) -> np.ndarray:
        z = _features(lap, rain, since) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def prob_at_least_one(self, start_lap: int, horizon: int, total: int, rain: float = 0.0,
                          since: int = 99) -> float:
        """P(an SC is deployed within the next `horizon` laps), ignoring SC cool-down shape."""
        p_none, s = 1.0, since
        for lap in range(start_lap, min(start_lap + horizon, total + 1)):
            p_none *= 1.0 - float(self.hazard(lap, rain, s))
            s += 1
        return 1.0 - p_none

    @classmethod
    def train(cls, track: TrackConfig, n_races: int = 3000, seed: int = 0) -> "SafetyCarModel":
        rng = np.random.default_rng(seed)
        rain = (rng.random(n_races) < 0.2).astype(float)
        remaining = np.zeros(n_races, dtype=int)
        since = np.full(n_races, 99)
        X, y = [], []
        for lap in range(1, track.laps + 1):
            active = remaining > 0
            h = sc_hazard_truth(lap, rain, since, track.sc_base_hazard)
            start = (~active) & (rng.random(n_races) < h)
            X.append(_features(lap, rain, since)[~active])
            y.append(start[~active])
            now = active | start
            remaining = np.where(start, rng.integers(3, 6, n_races) - 1, np.where(active, remaining - 1, 0))
            since = np.where(now, 0, since + 1)
        X, y = np.concatenate(X), np.concatenate(y)
        clf = LogisticRegression(C=50.0, max_iter=500).fit(X, y)
        m = cls()
        m.coef, m.intercept = clf.coef_[0].copy(), float(clf.intercept_[0])
        return m

    @classmethod
    def pretrained(cls, track: TrackConfig) -> "SafetyCarModel":
        return _cached(track)


@lru_cache(maxsize=8)
def _cached(track: TrackConfig) -> SafetyCarModel:
    return SafetyCarModel.train(track)
