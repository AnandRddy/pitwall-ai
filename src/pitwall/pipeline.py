"""ReplaySession: runs a race lap by lap through the StrategyBrain in *shadow mode*.

The car follows a preset plan (as the real team did); PitWall advises alongside,
learns from every lap and flags where it would have decided differently.
"""
from __future__ import annotations

from .agent.radio import RadioEvent, parse_radio
from .data.schema import Race
from .data.simulator import simulate_practice, simulate_race
from .physics import COMPOUNDS, TrackConfig
from .strategy.brain import StrategyBrain
from .strategy.types import RaceState
from .strategy.undercut import evaluate_undercut

OUR_PLAN = ("MEDIUM", ((26, "HARD"),))  # conventional 1-stop template the car actually runs


class ReplaySession:
    def __init__(self, seed: int = 7, driver: str = "CAR07", rain: float = 0.0, fast: bool = True):
        self.track = TrackConfig()
        self.driver, self.rain, self.fast = driver, rain, fast
        self.race, self.true_base = simulate_race(self.track, seed=seed, rain=rain, plans={driver: OUR_PLAN})
        self.brain = StrategyBrain(self.track, simulate_practice(self.track, self.true_base, seed), driver)
        self.lap = 0
        self.radio: list[RadioEvent] = []
        self.last_rec = None
        self.last_undercut = None

    @property
    def done(self) -> bool:
        return self.lap >= self.track.laps

    def state(self) -> RaceState:
        r, lap = self.race, self.lap
        comp, age, used = r.tyre_state(self.driver, lap) if lap else (OUR_PLAN[0], 0, (OUR_PLAN[0],))
        sc_rem, since = r.sc_context(lap)
        ga, gb, ahead, behind = r.gaps(self.driver, lap) if lap else (None, None, None, None)
        pos = int(r.position[r.idx(self.driver), lap - 1]) if lap else 1
        rain = max(self.rain, max((e.severity for e in self.radio if e.kind == "rain"), default=0.0))
        return RaceState(lap, self.track.laps, self.driver, comp, age, used, sc_rem, since, rain, pos,
                         ga, gb, ahead, behind)

    def add_radio(self, text: str) -> RadioEvent:
        ev = parse_radio(text)
        self.radio.append(ev)
        if ev.kind in ("tyres", "damage"):
            self.brain.cusum.s += 0.5 * ev.severity * self.brain.cusum.h  # prior nudge toward an alarm
            self.brain.cusum.alarm = self.brain.cusum.s > self.brain.cusum.h
        return ev

    def step(self) -> dict:
        """Advance one lap and return the full pit-wall payload."""
        r, tr = self.race, self.track
        self.lap += 1
        lap, j, i = self.lap, self.lap - 1, r.idx(self.driver)
        clean = bool(not r.sc[j] and not r.pit_in[i, j] and r.tyre_age[i, j] > 1 and lap > 1)
        c, a, f, lt = int(r.compound[i, j]), int(r.tyre_age[i, j]), float(r.fuel[i, j]), float(r.lap_time[i, j])
        pred = self.brain.predicted(c, a, f)
        resid = self.brain.observe_own_lap(c, a, f, lt, clean)
        alarm = self.brain.cusum.alarm
        if r.pit_in[i, j]:
            self.brain.on_pit()
        self.brain.ingest(r.table_for_lap(lap), refit=True)

        state = self.state()
        payload = {"lap": lap, "total": tr.laps, "position": state.position, "lap_time": round(lt, 3),
                   "predicted": round(pred, 3), "residual": round(resid, 3), "cusum": round(self.brain.cusum.s, 2),
                   "cliff_alarm": alarm, "compound": COMPOUNDS[c], "tyre_age": a, "sc": bool(r.sc[j]),
                   "pitted": bool(r.pit_in[i, j]), "gap_ahead": state.gap_ahead, "gap_behind": state.gap_behind,
                   "curves": self.brain.curves(), "ours_next": None}
        if not self.done:
            rec = self.brain.recommend(state, n_scenarios=250, top_k=14, max_stops=2 if lap > 20 else 3,
                                       refine=False, seed=lap)
            self.last_rec = rec
            payload["rec"] = rec.to_dict()
            payload["sc_prob_5"] = round(self.brain.sc_model.prob_at_least_one(
                lap + 1, 5, tr.laps, state.rain_prob, state.laps_since_sc), 3)
            self.last_undercut = None
            if state.car_ahead and state.gap_ahead is not None and state.gap_ahead < 4.0:
                k = r.idx(state.car_ahead)
                rc, ra, _ = r.tyre_state(state.car_ahead, lap)
                self.last_undercut = evaluate_undercut(state, state.car_ahead, rc, ra, state.gap_ahead,
                                                       self.brain.tyre, tr, n=1500, seed=lap)
                payload["undercut"] = self.last_undercut.to_dict()
        else:
            payload["rec"], payload["sc_prob_5"] = None, 0.0
        return payload
