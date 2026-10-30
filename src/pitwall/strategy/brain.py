"""StrategyBrain: the stateful core that learns during the race.

It starts from a practice-fitted tyre model, refits as live laps arrive, watches
the car's own residuals with CUSUM, and answers "what do we do now?" queries.
"""
from __future__ import annotations

from ..data.schema import LapTable
from ..models.safety_car import SafetyCarModel
from ..models.tyre_deg import TyreDegModel
from ..physics import COMPOUNDS, TrackConfig
from .anomaly import CusumDetector
from .optimizer import optimize
from .types import RaceState, Recommendation


class StrategyBrain:
    def __init__(self, track: TrackConfig, practice: LapTable, driver: str, prior: LapTable | None = None):
        self.track, self.driver = track, driver
        self.sc_model = SafetyCarModel.pretrained(track)
        self._tables = [practice]
        self.prior = prior
        self.tyre = TyreDegModel().fit(practice, prior=prior)
        self.cusum = CusumDetector()
        self.last_residual = 0.0

    def observe_own_lap(self, compound: int, age: int, fuel: float, lap_time: float, clean: bool) -> float:
        """Score our own lap against the model *before* it learns from it."""
        if not clean:
            return 0.0
        pred = float(self.tyre.lap_time(self.driver, compound, age, fuel))
        self.last_residual = lap_time - pred
        self.cusum.update(self.last_residual)
        return self.last_residual

    def predicted(self, compound: int, age: int, fuel: float) -> float:
        return float(self.tyre.lap_time(self.driver, compound, age, fuel))

    def ingest(self, table: LapTable, refit: bool = True) -> None:
        if len(table):
            self._tables.append(table)
        if refit:
            self.tyre = TyreDegModel().fit(LapTable.concat(self._tables), prior=self.prior)
            if self.cusum.alarm:
                # model has absorbed the new data: re-arm to flag the next surprise
                self.cusum.reset()

    def on_pit(self) -> None:
        self.cusum.reset()

    def recommend(self, state: RaceState, **kw) -> Recommendation:
        return optimize(state, self.tyre, self.sc_model, self.track, **kw)

    def curves(self, max_age: int = 35) -> dict:
        return {c: [round(float(v), 3) for v in self.tyre.curves[i][: max_age + 1]]
                for i, c in enumerate(COMPOUNDS)}
