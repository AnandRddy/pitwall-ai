import numpy as np
import pytest

from pitwall.data.simulator import simulate_practice, simulate_race
from pitwall.physics import TrackConfig
from pitwall.strategy.brain import StrategyBrain
from pitwall.strategy.optimizer import enumerate_candidates
from pitwall.strategy.types import RaceState
from pitwall.strategy.undercut import evaluate_undercut

TRACK = TrackConfig()


@pytest.fixture(scope="module")
def brain():
    _, base = simulate_race(TRACK, seed=3)
    return StrategyBrain(TRACK, simulate_practice(TRACK, base, seed=3), "CAR07")


def state(**kw):
    d = dict(lap=0, total_laps=57, driver="CAR07", compound="MEDIUM", tyre_age=0, used=("MEDIUM",))
    d.update(kw)
    return RaceState(**d)


def test_all_candidates_respect_two_compound_rule_and_min_stint():
    st = state()
    n = 0
    for stops in enumerate_candidates(st, max_stops=3, min_stint=6):
        n += 1
        assert len(set(st.used) | {c for _, c in stops}) >= 2
        laps = [l for l, _ in stops]
        assert laps == sorted(laps) and all(b - a >= 6 for a, b in zip(laps, laps[1:]))
    assert n > 1000


def test_optimizer_returns_ranked_plans_fast(brain):
    rec = brain.recommend(state(), n_scenarios=300, seed=1)
    means = [p.mean for p in rec.plans]
    assert means == sorted(means)
    assert rec.best.delta == 0 and abs(sum(p.p_best for p in rec.plans) - 1) < 1e-6
    assert rec.elapsed_ms < 3000
    assert rec.best.stops, "a zero-stop race must be illegal on a single compound"


def test_worn_tyres_force_the_stop_earlier(brain):
    fresh = brain.recommend(state(lap=20, compound="SOFT", tyre_age=3, used=("SOFT",)), n_scenarios=300, seed=2)
    worn = brain.recommend(state(lap=20, compound="SOFT", tyre_age=20, used=("SOFT",)), n_scenarios=300, seed=2)
    assert worn.best.stops[0][0] <= fresh.best.stops[0][0]


def test_safety_car_makes_boxing_cheaper(brain):
    green = state(lap=14, compound="MEDIUM", tyre_age=14, used=("MEDIUM",))
    sc = state(lap=14, compound="MEDIUM", tyre_age=14, used=("MEDIUM",), sc_remaining=3, laps_since_sc=0)
    g = brain.recommend(green, n_scenarios=400, seed=5)
    s = brain.recommend(sc, n_scenarios=400, seed=5)
    assert s.pit_now_cost < g.pit_now_cost
    assert s.pit_now_cost < 1.0


def test_undercut_probability_falls_with_gap(brain):
    st = state(lap=20, compound="MEDIUM", tyre_age=20, used=("MEDIUM",))
    close = evaluate_undercut(st, "CAR08", "MEDIUM", 20, 0.8, brain.tyre, TRACK, seed=1)
    far = evaluate_undercut(st, "CAR08", "MEDIUM", 20, 6.0, brain.tyre, TRACK, seed=1)
    assert close.p_success > far.p_success
    assert far.p_success < 0.2


def test_common_random_numbers_make_runs_reproducible(brain):
    a = brain.recommend(state(), n_scenarios=200, seed=9)
    b = brain.recommend(state(), n_scenarios=200, seed=9)
    assert a.best.stops == b.best.stops and a.best.mean == b.best.mean
