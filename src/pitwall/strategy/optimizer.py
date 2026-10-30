"""Two-stage strategy search.

1. Enumerate every legal strategy (<= 3 stops, two-compound rule, minimum stint)
   and rank it with a closed-form *nominal* cost from cumulative tyre curves -
   thousands of candidates in milliseconds.
2. Evaluate a diverse shortlist with the Monte Carlo engine (safety cars,
   pit-stop variance, lap noise) on shared scenarios, then locally refine the
   leaders by nudging pit laps.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from ..models.safety_car import SafetyCarModel
from ..models.tyre_deg import TyreDegModel
from ..physics import COMPOUNDS, TrackConfig
from .monte_carlo import Scenarios, simulate_strategy
from .types import Plan, RaceState, Recommendation, Stops


def enumerate_candidates(state: RaceState, max_stops: int = 3, min_stint: int = 6):
    L, first = state.total_laps, state.lap + 1
    used = set(state.used)
    if len(used) >= 2:
        yield ()
    steps = {1: 1, 2: 2, 3: 4}
    for k in range(1, max_stops + 1):
        s = steps.get(k, 4)
        grid = range(first, L, s)
        for laps in itertools.combinations(grid, k):
            if any(b - a < min_stint for a, b in zip(laps, laps[1:])) or L - laps[-1] < 2:
                continue
            for comps in itertools.product(COMPOUNDS, repeat=k):
                if len(used | set(comps)) >= 2:
                    yield tuple(zip(laps, comps))


def _nominal_costs(model: TyreDegModel, track: TrackConfig):
    cum = np.concatenate([np.zeros((3, 1)), np.cumsum(model.curves[:, 1:], axis=1)], axis=1)
    return {c: cum[i] for i, c in enumerate(COMPOUNDS)}


def nominal_cost(state: RaceState, stops: Stops, cum, pit_loss: float) -> float:
    comp, a0, prev, total = state.compound, state.tyre_age, state.lap, 0.0
    for lap, c in stops:
        m = lap - prev
        total += cum[comp][a0 + m] - cum[comp][a0]
        comp, a0, prev = c, 0, lap
    m = state.total_laps - prev
    total += cum[comp][a0 + m] - cum[comp][a0]
    return total + pit_loss * len(stops)


def _evaluate(state, cands, model, track, scen):
    return np.stack([simulate_strategy(state, s, model, track, scen) for s in cands])


def _neighbours(stops: Stops, state: RaceState, min_stint: int):
    out = set()
    for i in range(len(stops)):
        for d in (-2, -1, 1, 2):
            lap = stops[i][0] + d
            new = stops[:i] + ((lap, stops[i][1]),) + stops[i + 1:]
            laps = [x[0] for x in new]
            if laps[0] > state.lap and laps[-1] < state.total_laps - 1 and all(
                    b - a >= min_stint for a, b in zip(laps, laps[1:])):
                out.add(new)
    return out


def optimize(state: RaceState, model: TyreDegModel, sc_model: SafetyCarModel, track: TrackConfig,
             n_scenarios: int = 400, top_k: int = 24, max_stops: int = 3, min_stint: int = 6,
             risk_aversion: float = 0.0, seed: int = 0, refine: bool = True,
             baseline: Stops | None = None) -> Recommendation:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    cum = _nominal_costs(model, track)
    cands = list(enumerate_candidates(state, max_stops, min_stint))
    if not cands:
        cands = [()]
    scored = sorted(cands, key=lambda s: nominal_cost(state, s, cum, track.pit_loss))

    # diverse shortlist: <=3 per compound sequence, plus the best "box next lap" options
    short, per_seq = [], {}
    for s in scored:
        key = tuple(c for _, c in s)
        if per_seq.get(key, 0) < 3 and len(short) < top_k:
            short.append(s)
            per_seq[key] = per_seq.get(key, 0) + 1
    first = state.lap + 1
    pit_now = []
    for k in range(1, max_stops + 1):  # best "box next lap" options for each stop count
        pit_now += [s for s in scored if len(s) == k and s[0][0] == first][:2]
    extra = [s for s in pit_now + ([baseline] if baseline is not None else []) if s not in short]
    short += extra

    scen = Scenarios.sample(state, sc_model, model.sigma, n_scenarios, rng)
    seen = {s: i for i, s in enumerate(short)}
    T = _evaluate(state, short, model, track, scen)

    if refine:
        lead = np.argsort(T.mean(axis=1))[:2]
        new = [n for i in lead for n in _neighbours(short[i], state, min_stint) if n not in seen]
        if new:
            T = np.vstack([T, _evaluate(state, new, model, track, scen)])
            short = short + new

    score = T.mean(axis=1) + risk_aversion * T.std(axis=1)
    order = np.argsort(score)
    winners = np.argmin(T, axis=0)
    p_best = np.bincount(winners, minlength=len(short)) / T.shape[1]
    best_mean = T[order[0]].mean()
    plans = [Plan(short[i], float(T[i].mean()), float(T[i].std()), float(np.percentile(T[i], 10)),
                  float(np.percentile(T[i], 90)), float(p_best[i]), float(T[i].mean() - best_mean))
             for i in order]

    best = plans[0]
    next_lap = best.stops[0][0] if best.stops else None
    now_plans = [p for p in plans if p.stops and p.stops[0][0] == first]
    cost = (now_plans[0].mean - best.mean) if now_plans else float("inf")
    notes = []
    if state.sc_remaining:
        notes.append("Safety car out: pit loss is discounted.")
    return Recommendation(plans, next_lap == first, next_lap, best.stops[0][1] if best.stops else None,
                          cost, len(cands), (time.perf_counter() - t0) * 1000.0, notes)
