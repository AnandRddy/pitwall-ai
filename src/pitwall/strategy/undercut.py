"""Undercut / overcut analysis against a specific rival.

If we box now and the rival stays out for k more laps (k is uncertain - they may
cover us), do we emerge ahead once they have stopped? Monte Carlo over rival
response lag, stop-time variance and traffic on rejoin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.tyre_deg import TyreDegModel
from ..physics import COMPOUNDS, TrackConfig
from .types import RaceState


@dataclass
class UndercutResult:
    rival: str
    compound: str
    p_success: float
    expected_margin: float   # seconds ahead (+) / behind (-) once rival has stopped
    gap: float
    recommended: bool

    def to_dict(self) -> dict:
        return {"rival": self.rival, "compound": self.compound, "p_success": round(self.p_success, 3),
                "expected_margin": round(self.expected_margin, 2), "gap": round(self.gap, 2),
                "recommended": self.recommended}


def evaluate_undercut(state: RaceState, rival: str, rival_compound: str, rival_age: int, gap: float,
                      model: TyreDegModel, track: TrackConfig, n: int = 4000, seed: int = 0,
                      lag_probs=(0.35, 0.40, 0.25), threshold: float = 0.55) -> UndercutResult:
    """`gap` is how many seconds the rival is AHEAD of us."""
    rng = np.random.default_rng(seed)
    horizon = len(lag_probs)
    b_us, b_riv = model.base(state.driver), model.base(rival)
    sigma = model.sigma
    best = None
    for new in COMPOUNDS:
        if len(set(state.used) | {new}) < 2:
            continue
        margins = []
        for k in range(1, horizon + 1):
            ours = rival_t = 0.0
            for i in range(k + 1):  # laps next .. next+k  (rival pits on lap k, out-lap is k+1)
                lap = state.lap + 1 + i
                fuel = track.fuel_at(lap)
                if i == 0:
                    ours += b_us + float(model.curve_at(state.compound, state.tyre_age + 1)) \
                        + model.fuel_effect * fuel + track.pit_loss
                else:
                    ours += b_us + float(model.curve_at(new, i)) + model.fuel_effect * fuel
                if i < k:
                    rival_t += b_riv + float(model.curve_at(rival_compound, rival_age + 1 + i)) \
                        + model.fuel_effect * fuel + (track.pit_loss if i == k - 1 else 0.0)
                else:
                    rival_t += b_riv + float(model.curve_at(new, 1)) + model.fuel_effect * fuel
            margins.append(rival_t - gap - ours)  # rival is `gap` s ahead; >0 => we end up ahead
        margins = np.array(margins)
        k_idx = rng.choice(horizon, size=n, p=lag_probs)
        noise = rng.normal(0.0, np.sqrt(2 * sigma**2 * (k_idx + 2) + 2 * 0.5**2))
        traffic = rng.exponential(1.2, n) * (rng.random(n) < 0.35)
        m = margins[k_idx] + noise - traffic
        res = UndercutResult(rival, new, float((m > 0).mean()), float(m.mean()), gap, bool((m > 0).mean() >= threshold))
        if best is None or res.p_success > best.p_success:
            best = res
    return best or UndercutResult(rival, state.compound, 0.0, -gap, gap, False)
