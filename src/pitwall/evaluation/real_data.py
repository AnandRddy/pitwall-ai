"""Validation of PitWall's models against REAL races (OpenF1 data).

Four analyses, each answering a question a team would ask before trusting the tool:

1. tyre_forecast   Can the live-refit tyre model forecast the next 10 laps better than a
                   plain "recent pace + fuel" engineer heuristic?   (out-of-sample, rolling origin)
2. sc_hazard       Does the safety-car hazard model beat a constant base rate on unseen races?
                   (leave-one-race-out cross-validation)
3. pit_loss        What do real pit stops cost, and how much cheaper are they under safety car?
                   (checks the simulator's 21.5 s / 0.45 assumptions)
4. shadow_replay   At lap N, how far is each driver's *actual* plan from PitWall's optimum?
                   (model-estimated, exploratory: NOT a measure of real race-time gain)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..data.openf1_loader import RaceMeta, list_races, load_race
from ..data.schema import LapTable, Race
from ..models.safety_car import _features
from ..models.tyre_deg import TyreDegModel
from ..physics import COMPOUNDS
from ..strategy.brain import StrategyBrain
from ..strategy.types import RaceState


# ----------------------------------------------------------------------------- helpers
def clean_table(race: Race, first: int, last: int) -> LapTable:
    """Clean laps (no SC / pit / out-lap / traffic outliers) for race laps first..last."""
    t = LapTable.concat([race.table_for_lap(l) for l in range(max(first, 2), min(last, race.n_laps) + 1)])
    base = race.track.base_lap
    return t.mask((t.lap_time > 0.93 * base) & (t.lap_time < 1.07 * base))


def load_season(years, verbose=True):
    out = []
    for y in years:
        for s in list_races(y):
            try:
                race, meta = load_race(session=s, year=y)
            except Exception as e:  # pragma: no cover - network / data gaps
                if verbose:
                    print(f"skip {y} {s.get('country_name')}: {e}")
                continue
            out.append((race, meta))
    return out


# ----------------------------------------------------------------------------- 1. tyre forecast
def prior_table(prev, k=4):
    """Pool clean laps of the previous `k` usable races (team history database, normalised pace scale)."""
    return LapTable.concat([clean_table(r, 2, r.n_laps) for r, _ in prev[-k:]])


def tyre_forecast(races, origins=(15, 25, 35), horizon=10, fuel_rule=0.03, min_prior=2):
    """Rolling-origin forecast of the next `horizon` laps. Three predictors on identical laps:
    heuristic baseline, model on this race only, model with pooled prior races."""
    rows = []
    usable = [(r, m) for r, m in races if m.usable]
    for k, (race, meta) in enumerate(usable):
        if k < min_prior:
            continue
        prior = prior_table(usable[:k])
        for o in origins:
            if o + horizon > race.n_laps:
                continue
            train, test = clean_table(race, 2, o), clean_table(race, o + 1, o + horizon)
            if len(train) < 150 or not len(test):
                continue
            m_solo = TyreDegModel().fit(train)
            m_pool = TyreDegModel().fit(train, prior=prior)
            seen_max = {c: int(m_pool.max_seen[c]) for c in range(3)}
            last = {}
            for d in set(train.driver.tolist()):
                idx = np.nonzero(train.driver == d)[0][-5:]
                if len(idx) >= 3:
                    last[d] = (train.lap_time[idx].mean(), train.fuel[idx].mean())
            for d, c, a, f, y in zip(test.driver, test.compound, test.age, test.fuel, test.lap_time):
                if d not in last or d not in m_pool.driver_base or d not in m_solo.driver_base:
                    continue
                pb = last[d][0] + fuel_rule * (f - last[d][1])
                rows.append({"race": meta.name, "origin": o, "err_base": abs(pb - y),
                             "err_solo": abs(float(m_solo.lap_time(d, int(c), a, f)) - y),
                             "err_model": abs(float(m_pool.lap_time(d, int(c), a, f)) - y),
                             "extrap": bool(a > seen_max[int(c)])})
    return _summarise_forecast(rows)


def _summarise_forecast(rows):
    def blk(sel):
        if not sel:
            return None
        m = np.array([r["err_model"] for r in sel])
        so = np.array([r["err_solo"] for r in sel])
        b = np.array([r["err_base"] for r in sel])
        return {"n_laps": len(sel), "mae_baseline_s": float(b.mean()), "mae_model_race_only_s": float(so.mean()),
                "mae_model_with_history_s": float(m.mean()),
                "improvement_vs_baseline_pct": float(100 * (1 - m.mean() / b.mean()))}
    out = {"all": blk(rows), "by_origin": {}, "extrapolating_unseen_ages": blk([r for r in rows if r["extrap"]]),
           "within_seen_ages": blk([r for r in rows if not r["extrap"]]),
           "n_races": len({r["race"] for r in rows})}
    for o in sorted({r["origin"] for r in rows}):
        out["by_origin"][str(o)] = blk([r for r in rows if r["origin"] == o])
    return out


# ----------------------------------------------------------------------------- 2. SC hazard
def _sc_rows(race: Race, meta: RaceMeta):
    deploy = set(meta.sc_deployments)
    X, y, since = [], [], 99
    for lap in range(1, race.n_laps + 1):
        active = bool(race.sc[lap - 1])
        if (not active) or lap in deploy:
            X.append(_features(lap, float(meta.wet), since))
            y.append(lap in deploy)
        since = 0 if active else since + 1
    return np.array(X), np.array(y, dtype=bool)


def sc_hazard(races):
    data = [(m, *_sc_rows(r, m)) for r, m in races if not m.red_flag and m.n_laps >= 40]
    n = len(data)
    ll_m = ll_b = br_m = br_b = cnt = 0.0
    pred_any, obs_any = [], []
    for k in range(n):
        Xtr = np.concatenate([d[1] for i, d in enumerate(data) if i != k])
        ytr = np.concatenate([d[2] for i, d in enumerate(data) if i != k])
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr)
        p_base = float(np.clip(ytr.mean(), 1e-6, 1 - 1e-6))
        Xte, yte = data[k][1], data[k][2]
        p = np.clip(clf.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        ll_m += -np.sum(yte * np.log(p) + (~yte) * np.log(1 - p))
        ll_b += -np.sum(yte * np.log(p_base) + (~yte) * np.log(1 - p_base))
        br_m += np.sum((p - yte) ** 2)
        br_b += np.sum((p_base - yte) ** 2)
        cnt += len(yte)
        pred_any.append(1 - np.prod(1 - p))
        obs_any.append(float(yte.any()))
    pred_any, obs_any = np.array(pred_any), np.array(obs_any)
    return {"n_races": n, "observed_races_with_sc": float(obs_any.mean()),
            "mean_predicted_p_any_sc": float(pred_any.mean()),
            "logloss_model": float(ll_m / cnt), "logloss_constant_rate": float(ll_b / cnt),
            "brier_race_level_model": float(np.mean((pred_any - obs_any) ** 2)),
            "brier_race_level_constant": float(np.mean((obs_any.mean() - obs_any) ** 2)),
            "first_laps_sc_rate": _first_lap_rate(data)}


def _first_lap_rate(data):
    lap12 = [(any(l <= 2 for l in m.sc_deployments)) for m, *_ in data]
    return float(np.mean(lap12)) if lap12 else None


# ----------------------------------------------------------------------------- 3. pit loss
def pit_loss(races):
    """Time lost to the field over a stop's in-lap + out-lap, measured against cars that did not
    stop on the same laps. Comparing against the field (not against green-flag pace) keeps safety-car
    stops on the same footing as green-flag ones."""
    green, scl = [], []
    for race, meta in races:
        if meta.red_flag:
            continue
        lt, n = race.lap_time, race.n_laps
        out_lap = np.zeros_like(race.pit_in)
        out_lap[:, 1:] = race.pit_in[:, :-1]
        free = np.where(race.pit_in | out_lap, np.nan, lt)
        field = np.nanmedian(free, axis=0)
        vsc = {v + d for v in meta.vsc_deployments for d in (-2, -1, 0, 1)}   # 1-indexed laps near a VSC
        for i in range(len(race.drivers)):
            for j in np.nonzero(race.pit_in[i])[0]:
                if j + 1 >= n or not (np.isfinite(lt[i, j]) and np.isfinite(lt[i, j + 1])):
                    continue
                if {j + 1, j + 2} & vsc:   # virtual-SC stops are a different regime: exclude
                    continue
                if not (np.isfinite(field[j]) and np.isfinite(field[j + 1])):
                    continue
                loss = (lt[i, j] - field[j]) + (lt[i, j + 1] - field[j + 1])
                if 5 < loss < 60:
                    (scl if (race.sc[j] or race.sc[j + 1]) else green).append(loss)
    g, s = np.array(green), np.array(scl)
    return {"n_green_stops": len(g), "n_sc_stops": len(s),
            "median_pit_loss_green_s": float(np.median(g)) if len(g) else None,
            "median_pit_loss_under_sc_s": float(np.median(s)) if len(s) else None,
            "sc_discount_ratio": float(np.median(s) / np.median(g)) if len(g) and len(s) else None,
            "simulator_default": {"pit_loss_s": 22.7, "sc_pit_factor": 0.67}}


# ----------------------------------------------------------------------------- 4. shadow replay
def _actual_plan(race: Race, i: int, origin: int):
    stops = []
    for j in np.nonzero(race.pit_in[i, origin:])[0]:
        lap = origin + int(j) + 1
        if lap >= race.n_laps:
            return None
        c = int(race.compound[i, lap]) if lap < race.n_laps else -1  # compound used on the next lap
        if c < 0:
            return None
        stops.append((lap, COMPOUNDS[c]))
    return tuple(stops)


def shadow_replay(races, origin=18, n_scenarios=300, max_races=None):
    rows = []
    usable = [(r, m) for r, m in races if m.usable]
    for k, (race, meta) in enumerate(usable[:max_races] if max_races else usable):
        if k < 2 or origin + 10 >= race.n_laps or race.sc[origin]:
            continue
        practice = clean_table(race, 2, origin)
        if len(practice) < 200:
            continue
        brain = StrategyBrain(race.track, practice, race.drivers[0], prior=prior_table(usable[:k]))
        sc_rem, since = race.sc_context(origin)
        for i, d in enumerate(race.drivers):
            plan = _actual_plan(race, i, origin)
            if plan is None or not plan or race.compound[i, origin - 1] < 0 or d not in brain.tyre.driver_base:
                continue
            comp, age, used = race.tyre_state(d, origin)
            if len({*used, *(c for _, c in plan)}) < 2:
                continue
            st = RaceState(origin, race.n_laps, d, comp, age, used, sc_rem, since, float(meta.wet))
            brain.driver = d
            rec = brain.recommend(st, n_scenarios=n_scenarios, top_k=14, max_stops=2, refine=False, seed=origin,
                                  baseline=plan)
            mine = next((p for p in rec.plans if p.stops == plan), None)
            if mine is None:
                continue
            first = rec.best.stops[0][0] if rec.best.stops else None
            p0 = int(race.position[i, origin - 1])
            p1 = int(race.position[i, -1])
            rows.append({"race": meta.name, "driver": d, "gap_s": mine.delta, "best_stop": first,
                         "actual_stop": plan[0][0], "same_stops": len(plan) == len(rec.best.stops),
                         "pos_change": p0 - p1})
    return _summarise_shadow(rows, origin)


def _summarise_shadow(rows, origin):
    if not rows:
        return {"n_driver_plans": 0}
    gap = np.array([r["gap_s"] for r in rows])
    d_lap = np.array([abs(r["best_stop"] - r["actual_stop"]) for r in rows if r["best_stop"]])
    pos = np.array([r["pos_change"] for r in rows], float)

    def rank(x):
        return np.argsort(np.argsort(x)).astype(float)
    rho = float(np.corrcoef(rank(gap), rank(pos))[0, 1]) if len(rows) > 3 else None
    return {"origin_lap": origin, "n_driver_plans": len(rows), "n_races": len({r["race"] for r in rows}),
            "median_gap_to_optimum_s": float(np.median(gap)), "mean_gap_to_optimum_s": float(gap.mean()),
            "share_within_1s_of_optimum": float((gap < 1.0).mean()),
            "median_abs_first_stop_diff_laps": float(np.median(d_lap)) if len(d_lap) else None,
            "spearman_plan_gap_vs_places_gained": rho,
            "note": "gap is estimated by PitWall's own model; spearman uses real finishing positions "
                    "(negative = drivers with bigger estimated gaps gained fewer places)"}


# ----------------------------------------------------------------------------- CLI
def run_all(years=(2023, 2024), shadow_races=None):
    races = load_season(years)
    usable = [(r, m) for r, m in races if m.usable]
    return {"years": list(years), "races_loaded": len(races), "races_usable": len(usable),
            "excluded": {"red_flag": sum(m.red_flag for _, m in races), "wet": sum(m.wet for _, m in races)},
            "tyre_forecast": tyre_forecast(usable), "sc_hazard": sc_hazard(races),
            "pit_loss": pit_loss(races), "shadow_replay": shadow_replay(usable, max_races=shadow_races)}


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024])
    ap.add_argument("--out", default="reports/real_data.json")
    ap.add_argument("--shadow-races", type=int, default=None)
    a = ap.parse_args()
    res = run_all(tuple(a.years), a.shadow_races)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
