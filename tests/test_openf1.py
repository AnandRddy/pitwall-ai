import numpy as np

from pitwall.data import openf1_loader as of
from pitwall.evaluation import real_data as rd
from pitwall.models.tyre_deg import TyreDegModel


def fake_session(n_laps=50, sc=(20, 22)):
    laps, stints = [], []
    for num, pit in ((1, 25), (44, 30)):
        for lap in range(1, n_laps + 1):
            stint_lap = lap if lap <= pit else lap - pit
            t = 90 + 0.05 * stint_lap + (0.0 if num == 1 else 0.3)
            if sc[0] <= lap <= sc[1]:
                t = 130
            if lap == pit:
                t += 20
            laps.append({"driver_number": num, "lap_number": lap, "lap_duration": t, "is_pit_out_lap": lap == pit + 1})
        stints += [{"driver_number": num, "stint_number": 1, "lap_start": 1, "lap_end": pit,
                    "compound": "MEDIUM", "tyre_age_at_start": 0},
                   {"driver_number": num, "stint_number": 2, "lap_start": pit + 1, "lap_end": n_laps,
                    "compound": "HARD", "tyre_age_at_start": 0}]
    drivers = [{"driver_number": 1, "name_acronym": "AAA"}, {"driver_number": 44, "name_acronym": "BBB"}]
    rc = [{"date": "2024-01-01T10:00:00", "lap_number": sc[0], "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
          {"date": "2024-01-01T10:10:00", "lap_number": sc[1], "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"}]
    return {"laps": laps, "stints": stints, "drivers": drivers, "race_control": rc}


def patch(monkeypatch, payload):
    monkeypatch.setattr(of, "_get", lambda ep, **kw: payload[ep])


SESSION = {"session_key": 1, "year": 2024, "country_name": "Testland", "circuit_short_name": "TST"}


def test_loader_builds_race_with_stints_pits_and_safety_car(monkeypatch):
    patch(monkeypatch, fake_session())
    race, meta = of._build(SESSION)
    assert race.drivers == ["AAA", "BBB"] and race.n_laps == 50
    assert race.pit_in[0, 24] and race.pit_in[1, 29] and race.pit_in.sum() == 2
    assert race.compound[0, 0] == 1 and race.compound[0, 30] == 2     # MEDIUM then HARD
    assert race.tyre_age[0, 0] == 1 and race.tyre_age[0, 25] == 1      # age resets after the stop
    assert meta.sc_deployments == [20] and race.sc[19:22].all() and not race.sc[22]
    assert meta.usable


def test_pit_loss_analysis_reads_real_stop_cost(monkeypatch):
    patch(monkeypatch, fake_session())
    race, meta = of._build(SESSION)
    out = rd.pit_loss([(race, meta)])
    assert out["n_green_stops"] == 2
    assert 15 < out["median_pit_loss_green_s"] < 25


def test_unseen_compound_does_not_look_free():
    from pitwall.data.simulator import simulate_practice, simulate_race
    from pitwall.physics import TrackConfig
    tr = TrackConfig()
    _, base = simulate_race(tr, seed=1)
    prac = simulate_practice(tr, base, seed=1)
    keep = prac.compound != 1  # nobody has run the MEDIUM yet
    m = TyreDegModel().fit(prac.mask(keep))
    assert m.max_seen[1] == 0
    assert m.curves[1][10] > m.curves[0][10] - 0.2  # not magically faster than a soft
