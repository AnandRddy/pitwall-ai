"""Tyre degradation model learned from lap data.

lap_time = driver_pace + compound_curve(age) + fuel_effect * fuel_kg + noise

* driver fixed effects separate car/driver pace from tyre behaviour,
* per-compound cubic age curves (ridge-regularised) capture non-linear wear,
* the fuel coefficient is shrunk toward a physical prior (~0.03 s/kg): on real data fuel burn and
  tyre age are almost collinear in the opening laps, so an unconstrained fit drifts badly,
* an optional *prior* table (earlier races, normalised to a common pace scale) is pooled in with a
  lower weight - the team's historical database - so a model early in a race is not extrapolating
  from ten laps per driver,
* a recent-pace correction per driver absorbs track evolution and set-up effects,
* curves are forced monotone and extrapolated conservatively beyond the oldest observed age.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import LapTable
from ..physics import COMPOUNDS, C_IDX

MAX_AGE = 90


class TyreDegModel:
    def __init__(self, degree: int = 3, ridge: float = 1e-3, fuel_prior: float = 0.03,
                 fuel_strength: float = 4.0, extrap_penalty: float = 0.01, bias_laps: int = 5):
        self.degree, self.ridge = degree, ridge
        self.fuel_prior, self.fuel_strength = fuel_prior, fuel_strength
        self.extrap_penalty, self.bias_laps = extrap_penalty, bias_laps
        self.fitted = False
        self.driver_base: dict[str, float] = {}
        self.bias: dict[str, float] = {}
        self.curves = np.zeros((len(COMPOUNDS), MAX_AGE + 1))
        self.max_seen = np.zeros(len(COMPOUNDS), dtype=int)
        self.fuel_effect = fuel_prior
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

    def fit(self, table: LapTable, prior: LapTable | None = None, prior_weight: float = 0.25) -> "TyreDegModel":
        ok = np.isfinite(table.lap_time) & (table.compound >= 0)
        t = table.mask(ok)
        w = np.ones(len(t))
        if prior is not None and len(prior):
            pk = np.isfinite(prior.lap_time) & (prior.compound >= 0)
            t = LapTable.concat([t, prior.mask(pk)])
            w = np.concatenate([w, np.full(int(pk.sum()), prior_weight)])
        drivers = sorted(set(t.driver.tolist()))
        d_map = {d: i for i, d in enumerate(drivers)}
        d_idx = np.array([d_map[d] for d in t.driver])
        n_drv = len(drivers)
        X = self._design(d_idx, t.compound, t.age.astype(float), t.fuel, n_drv)
        y = t.lap_time - self.fuel_prior * t.fuel          # fuel enters as prior + a shrunk correction
        pen = np.full(X.shape[1], self.ridge)
        pen[:n_drv] = 0.0
        pen[-1] = self.fuel_strength
        XtW = X.T * w
        beta = np.linalg.solve(XtW @ X + np.diag(pen), XtW @ y)
        resid = y - X @ beta
        self.sigma = float(np.sqrt(np.sum(w * resid**2) / max(w.sum() - X.shape[1], 1)))
        self.fuel_effect = float(self.fuel_prior + beta[-1] / 100.0)

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
                # risk-averse prior: unseen ages get a convex penalty (cliffs are rarely in the data)
                curve[ms + 1:] = curve[ms] + slope * d + self.extrap_penalty * d**2
            self.curves[c], self.max_seen[c] = curve, ms
        # A compound nobody has run yet would otherwise get a flat "soft, no wear" curve and look
        # unbeatable. Borrow the nearest observed compound's curve, shifted ~0.5 s per step harder.
        seen_c = [c for c in range(3) if self.max_seen[c] > 3]
        for c in range(3):
            if self.max_seen[c] <= 3 and seen_c:
                k = min(seen_c, key=lambda s: abs(s - c))
                self.curves[c] = self.curves[k] + 0.5 * (c - k) * (np.arange(MAX_AGE + 1) > 0)

        # recent-pace correction: mean residual over each driver's latest laps of the primary table
        self.bias = {}
        main = table.mask(ok)
        for d in drivers:
            idx = np.nonzero(main.driver == d)[0][-self.bias_laps:]
            if len(idx) >= 3:
                r = [float(main.lap_time[i] - self._raw(d, int(main.compound[i]), main.age[i], main.fuel[i]))
                     for i in idx]
                self.bias[d] = float(np.mean(r)) * len(idx) / (len(idx) + 2.0)   # shrink toward 0
        self.fitted = True
        return self

    # -- inference -------------------------------------------------------
    def _raw(self, driver, compound, age, fuel):
        base = self.driver_base.get(driver, float(np.mean(list(self.driver_base.values()))))
        return base + self.curve_at(compound, age) + self.fuel_effect * np.asarray(fuel)

    def base(self, driver: str) -> float:
        """Driver pace incl. recent-pace correction."""
        b = self.driver_base.get(driver)
        if b is None:
            b = float(np.mean(list(self.driver_base.values())))
        return b + self.bias.get(driver, 0.0)

    def curve_at(self, compound: str | int, age) -> np.ndarray:
        c = C_IDX[compound] if isinstance(compound, str) else compound
        return self.curves[c][np.clip(np.asarray(age, dtype=int), 0, MAX_AGE)]

    def lap_time(self, driver: str, compound: str | int, age, fuel) -> np.ndarray:
        return self.base(driver) + self.curve_at(compound, age) + self.fuel_effect * np.asarray(fuel)

    def mae(self, table: LapTable) -> float:
        pred = np.array([float(self.lap_time(d, int(c), a, f))
                         for d, c, a, f in zip(table.driver, table.compound, table.age, table.fuel)])
        return float(np.mean(np.abs(pred - table.lap_time)))
