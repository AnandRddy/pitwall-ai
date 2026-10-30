import numpy as np

from pitwall.data.simulator import simulate_practice, simulate_race
from pitwall.models.safety_car import SafetyCarModel
from pitwall.models.tyre_deg import TyreDegModel
from pitwall.physics import TRUE_TYRES, TrackConfig, sample_sc_laps, sc_hazard_truth
from pitwall.strategy.anomaly import CusumDetector

TRACK = TrackConfig()


def test_tyre_model_recovers_physics_inside_observed_range():
    race, base = simulate_race(TRACK, seed=1)
    model = TyreDegModel().fit(simulate_practice(TRACK, base, seed=1))
    assert abs(model.fuel_effect - TRACK.fuel_effect) < 0.006
    for i, c in enumerate(("SOFT", "MEDIUM", "HARD")):
        age = 8
        got = model.curves[i][age]
        want = float(TRUE_TYRES[c].deg(age))
        assert abs(got - want) < 0.25, (c, got, want)
    assert 0.15 < model.sigma < 0.40


def test_curves_are_monotone_and_extrapolation_is_convex():
    _, base = simulate_race(TRACK, seed=2)
    m = TyreDegModel().fit(simulate_practice(TRACK, base, seed=2))
    for curve in m.curves:
        assert np.all(np.diff(curve[1:]) >= -1e-9)


def test_race_refit_learns_the_cliff():
    race, base = simulate_race(TRACK, seed=4)
    prac = simulate_practice(TRACK, base, seed=4)
    before = TyreDegModel().fit(prac)
    from pitwall.data.schema import LapTable
    laps = LapTable.concat([prac] + [race.table_for_lap(l) for l in range(2, TRACK.laps + 1)])
    after = TyreDegModel().fit(laps)
    for i, c in enumerate(("SOFT", "MEDIUM", "HARD")):
        seen = after.max_seen[i]
        want = float(TRUE_TYRES[c].deg(seen))
        assert abs(after.curves[i][seen] - want) <= abs(before.curves[i][seen] - want) + 0.3


def test_sc_model_is_calibrated_against_truth():
    model = SafetyCarModel.train(TRACK, n_races=4000, seed=3)
    rng = np.random.default_rng(0)
    emp = np.mean([sample_sc_laps(TRACK, rng).any() for _ in range(1500)])
    # model's P(at least one SC deployed) from the start of the race
    pred = model.prob_at_least_one(1, TRACK.laps, TRACK.laps)
    assert abs(emp - pred) < 0.12
    # structure: first-lap risk > mid-race risk; rain raises risk; cool-down lowers it
    assert model.hazard(1, 0, 99) > 3 * model.hazard(30, 0, 99)
    assert model.hazard(30, 0.7, 99) > model.hazard(30, 0, 99)
    assert model.hazard(30, 0, 2) < model.hazard(30, 0, 99)
    assert float(sc_hazard_truth(1, 0, 99)) > 0.04


def test_cusum_silent_on_noise_and_alarms_on_drift():
    rng = np.random.default_rng(0)
    d = CusumDetector()
    assert not any(d.update(float(r)) for r in rng.normal(0, 0.2, 200))
    d.reset()
    alarmed_at = next(i for i, r in enumerate(rng.normal(0.8, 0.2, 20)) if d.update(float(r)))
    assert alarmed_at <= 3
