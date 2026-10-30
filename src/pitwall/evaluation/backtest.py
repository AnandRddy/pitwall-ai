"""Closed-loop backtest: does adaptive AI strategy beat a static plan on the same race?

Every policy drives the SAME car through the SAME hidden reality (true tyre physics,
safety-car laps, lap noise, pit-stop variance). The AI policies only ever see
noisy practice laps plus the laps they have driven, so model error is real.

Policies
  template  conventional Medium>Hard one-stopper at lap 24, never revisited
  static    pre-race Monte Carlo optimum from the practice-fitted model, never revisited
  adaptive  same start, but re-optimises on safety cars, CUSUM tyre alarms and every 5 laps,
            refitting its tyre model on its own laps as it goes
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..data.schema import LapTable, sc_context_of
from ..data.simulator import simulate_practice
from ..physics import C_IDX, COMPOUNDS, TrackConfig, TruthCar, sample_sc_laps
from ..strategy.brain import StrategyBrain
from ..strategy.types import RaceState

DRIVER = "CAR07"


def _state(track, lap, car, used, sc, rain):
    rem, since = sc_context_of(sc, lap)
    return RaceState(lap, track.laps, DRIVER, car.compound, car.age, tuple(sorted(used)), rem, since, rain)


def _choose_start(brain, track, rain):
    best = None
    for c in COMPOUNDS:
        st = RaceState(0, track.laps, DRIVER, c, 0, (c,), 0, 99, rain)
        rec = brain.recommend(st, n_scenarios=300, top_k=16, max_stops=3, refine=False, seed=1)
        if best is None or rec.best.mean < best[0]:
            best = (rec.best.mean, c, rec.best.stops)
    return best[1], best[2]


def run_policy(policy, track, base, sc, noise, practice, rain=0.0):
    brain = StrategyBrain(track, practice, DRIVER)
    if policy == "template":
        start, plan = "MEDIUM", ((24, "HARD"),)
    else:
        start, plan = _choose_start(brain, track, rain)
    car = TruthCar(track, base, start)
    used, total, k = {start}, 0.0, 0
    for lap in range(1, track.laps + 1):
        s = bool(sc[lap - 1])
        if policy == "adaptive" and lap > 1:
            st = _state(track, lap - 1, car, used, sc, rain)
            if lap % 5 == 0 or st.sc_remaining or brain.cusum.alarm:
                rec = brain.recommend(st, n_scenarios=200, top_k=12, max_stops=2, refine=False, seed=lap)
                plan = rec.best.stops
        pit_to = plan[0][1] if plan and lap >= plan[0][0] and lap < track.laps else None
        age_before = car.age + 1
        comp_before = car.compound
        fuel = track.fuel_at(lap)
        t, c, a, pitted = car.step(lap, s, pit_to, noise["lap"][lap - 1], noise["pit"][k], noise["slow"][k])
        total += t
        if pitted:
            used.add(pit_to)
            plan = plan[1:]
            k += 1
            brain.on_pit()
        if policy == "adaptive":
            clean = not s and not pitted and a > 1 and lap > 1
            brain.observe_own_lap(C_IDX[c], a, fuel, t, clean)
            if clean:
                brain.ingest(LapTable(np.array([DRIVER], dtype=object), np.array([C_IDX[c]]), np.array([a]),
                                      np.array([fuel]), np.array([t])))
    return total


def run_backtest(n_races: int = 20, seed: int = 0, rain_frac: float = 0.0, verbose: bool = True):
    track = TrackConfig()
    rows = []
    t0 = time.time()
    for r in range(n_races):
        rng = np.random.default_rng(seed * 100_003 + r)
        base_off = {f"CAR{i + 1:02d}": track.base_lap + float(rng.normal(0, 0.35)) for i in range(20)}
        rain = 0.0 if rng.random() >= rain_frac else 0.7
        sc = sample_sc_laps(track, rng, rain)
        noise = {"lap": rng.normal(0, track.lap_noise, track.laps), "pit": rng.normal(0, 0.5, 8),
                 "slow": (rng.random(8) < 0.03) * rng.uniform(3, 7, 8)}
        practice = simulate_practice(track, base_off, seed=seed * 1000 + r)
        res = {p: run_policy(p, track, base_off[DRIVER], sc, noise, practice, rain)
               for p in ("template", "static", "adaptive")}
        rows.append({"sc": bool(sc.any()), **res})
        if verbose:
            print(f"race {r + 1:>3}/{n_races}  SC={'Y' if sc.any() else 'n'}  "
                  f"adaptive vs static {res['static'] - res['adaptive']:+6.1f}s  "
                  f"vs template {res['template'] - res['adaptive']:+6.1f}s", flush=True)
    return summarise(rows, time.time() - t0)


def summarise(rows, elapsed):
    def block(sel):
        if not sel:
            return None
        g_static = np.array([r["static"] - r["adaptive"] for r in sel])
        g_templ = np.array([r["template"] - r["adaptive"] for r in sel])
        s_templ = np.array([r["template"] - r["static"] for r in sel])
        return {"n": len(sel), "adaptive_vs_static_s": float(g_static.mean()),
                "adaptive_vs_template_s": float(g_templ.mean()), "static_vs_template_s": float(s_templ.mean()),
                "adaptive_win_rate_vs_static": float((g_static > 0).mean()),
                "adaptive_win_rate_vs_template": float((g_templ > 0).mean())}
    out = {"all": block(rows), "with_sc": block([r for r in rows if r["sc"]]),
           "no_sc": block([r for r in rows if not r["sc"]]), "seconds": elapsed}
    return out


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rain", type=float, default=0.0)
    a = ap.parse_args()
    out = run_backtest(a.races, a.seed, a.rain)
    print("\n=== Backtest (positive = AI gains seconds) ===")
    for k in ("all", "with_sc", "no_sc"):
        b = out[k]
        if b:
            print(f"{k:>8} n={b['n']:<3} adaptive-vs-static {b['adaptive_vs_static_s']:+6.2f}s "
                  f"(win {b['adaptive_win_rate_vs_static']:.0%}) | adaptive-vs-template "
                  f"{b['adaptive_vs_template_s']:+6.2f}s (win {b['adaptive_win_rate_vs_template']:.0%}) | "
                  f"static-vs-template {b['static_vs_template_s']:+6.2f}s")
    print(f"runtime {out['seconds']:.0f}s")


if __name__ == "__main__":  # pragma: no cover
    main()
