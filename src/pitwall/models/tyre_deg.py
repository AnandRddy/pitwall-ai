"""Tyre degradation model learned from lap data.

lap_time = driver_pace + compound_curve(age) + fuel_effect * fuel_kg + noise

* driver fixed effects separate car/driver pace from tyre behaviour,
* per-compound cubic age curves (ridge-regularised) capture non-linear wear,
* curves are forced monotone and extrapolated conservatively (linearly) beyond
  the oldest tyre age ever observed - exactly where real cliffs hide, which is
  why the live pipeline pairs this model with a CUSUM cliff detector.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import LapTable
from ..physics import COMPOUNDS, C_IDX

MAX_AGE = 90


class TyreDegModel:
    def __init__(self, degree: int = 3, ridge: float = 1e-3):
        self.degree = degree
        self.ridge = ridge
        self.fitted = False
        self.driver_base: dict[str, float] = {}
        self.curves = np.zeros((len(COMPOUNDS), MAX_AGE + 1))
        self.max_seen = np.zeros(len(COMPOUNDS), dtype=int)
        self.fuel_effect = 0.03
        self.sigma = 0.3

    # -- fitting ---------------------------------------------------------
    def _design(self, d_idx, comp, age, fuel, n_drv):
        n, deg = len(age), self.degree
        X = np.zeros((n, n_drv + 2 + 3 * deg + 1))
        X[np.arange(n), d_idx] = 1.0
        X[:, n_drv] = comp == 1
        X[:, n_drv + 1] = comp == 2
        a = age / 20.0
        for c in range(3):
            m = comp == c
            for p in range(1, deg + 1):
                X[m, n_drv + 2 + c * deg + p - 1] = a[m] ** p
        X[:, -1] = fuel / 100.0
        return X

    def fit(self, table: LapTable) -> "TyreDegModel":
        ok = np.isfinite(table.lap_time) & (table.compound >= 0)
        t = table.mask(ok)
        drivers = sorted(set(t.driver.tolist()))
        d_map = {d: i for i, d in enumerate(drivers)}
        d_idx = np.array([d_map[d] for d in t.driver])
        X = self._design(d_idx, t.compound, t.age.astype(float), t.fuel, len(drivers))
        n_drv = len(drivers)
        pen = np.full(X.shape[1], self.ridge)
        pen[:n_drv] = 0.0
        beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ t.lap_time)
        resid = t.lap_time - X @ beta
        self.sigma = float(np.sqrt(np.sum(resid**2) / max(len(resid) - X.shape[1], 1)))
        self.fuel_effect = float(beta[-1] / 100.0)

        # driver pace is quoted for SOFT, age 0, zero fuel
        self.driver_base = {d: float(beta[i]) for i, d in enumerate(drivers)}
        offsets = np.array([0.0, beta[n_drv], beta[n_drv + 1]])
        ages = np.arange(MAX_AGE + 1, dtype=float)
        for c in range(3):
            poly = np.zeros_like(ages)
            for p in range(1, self.degree + 1):
                poly += beta[n_drv + 2 + c * self.degree + p - 1] * (ages / 20.0) ** p
            curve = offsets[c] + poly
            seen = t.age[t.compound == c]
            ms = int(seen.max()) if len(seen) else 0
            curve[1:] = np.maximum.accumulate(curve[1:])
            if 3 < ms < MAX_AGE:
                slope = max((curve[ms] - curve[ms - 3]) / 3.0, 0.02)
                d = ages[ms + 1:] - ms
                # risk-averse prior: unseen ages get a convex penalty (cliffs are never in practice data)
                curve[ms + 1:] = curve[ms] + slope * d + 0.01 * d**2
            self.curves[c], self.max_seen[c] = curve, ms
        self.fitted = True
        return self

    # -- inference -------------------------------------------------------
    def base(self, driver: str) -> float:
        if driver in self.driver_base:
            return self.driver_base[driver]
        return float(np.mean(list(self.driver_base.values())))

    def curve_at(self, compound: str | int, age) -> np.ndarray:
        c = C_IDX[compound] if isinstance(compound, str) else compound
        return self.curves[c][np.clip(np.asarray(age, dtype=int), 0, MAX_AGE)]

    def lap_time(self, driver: str, compound: str | int, age, fuel) -> np.ndarray:
        return self.base(driver) + self.curve_at(compound, age) + self.fuel_effect * np.asarray(fuel)

    def mae(self, table: LapTable) -> float:
        pred = np.array([float(self.lap_time(d, int(c), a, f))
                         for d, c, a, f in zip(table.driver, table.compound, table.age, table.fuel)])
        return float(np.mean(np.abs(pred - table.lap_time)))
