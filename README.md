# PitWall AI

[![ci](https://github.com/AnandRddy/pitwall-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AnandRddy/pitwall-ai/actions/workflows/ci.yml)

**A real-time race-strategy engine for Formula 1 pit walls**: it learns tyre behaviour live, simulates thousands of race futures per lap, and tells the strategist *box or stay out*, with the probability and the cost of being wrong. A tool-using LLM race engineer turns those numbers into radio calls.

```
 practice laps ─┐                       ┌─► Monte Carlo strategy search ─┐
 live lap data ─┼─► Tyre-deg model ─────┤   (8,000+ strategies / lap)    ├─► BOX / STAY OUT
 timing gaps  ──┤   (ridge, per-compound│                                │   + P(best), cost of
 radio text   ──┤    curves, refit/lap) ├─► Undercut evaluator           │   boxing now
                │                       │                                │
                ├─► CUSUM cliff alarm ──┘   Safety-car hazard model ─────┘
                │                           (logistic, learned)
                └─► Claude race engineer (tool use) ──► radio call ──► live dashboard
```

## Why this is a real pit-wall problem

Strategy calls are made under uncertainty with seconds to decide: tyres degrade non-linearly, a safety car can erase a pit stop's cost, and rivals react. PitWall AI treats it as what it is, a decision problem under uncertainty, not a lap-time regression.

| Capability | How it works |
|---|---|
| **Tyre degradation learning** | Joint ridge regression: driver pace + per-compound cubic age curves + fuel effect. Curves forced monotone; unseen ages get a convex, risk-averse extrapolation. Refit every lap on all cars' clean laps. |
| **Strategy optimisation** | Stage 1 ranks every legal strategy (≤3 stops, two-compound rule, min stint) with a closed-form cost from cumulative tyre curves. Stage 2 runs a diverse shortlist through a vectorised Monte Carlo with shared random scenarios (safety cars, stop-time variance, slow stops, lap noise), then locally refines the leaders. |
| **Safety-car hazard model** | Logistic regression learns P(SC deploys) from race context (opening laps, rain, cool-down). Feeds the Monte Carlo and a "SC risk next 5 laps" gauge. Pit loss is discounted when the SC is out, so the engine says *BOX* when it should. |
| **Undercut / overcut** | Monte Carlo over the rival's reaction lag, stop variance and rejoin traffic: P(we emerge ahead). |
| **Tyre-cliff detection** | CUSUM on model residuals flags sustained pace loss within ~3 laps; also nudged by parsed driver radio ("rear left is gone"). |
| **LLM race engineer** | Claude tool-use loop over 5 tools (state, optimise, undercut, SC outlook, tyre status). Invoked only on events (box call, SC, cliff alarm, driver radio) to bound cost and latency; every number must come from a tool. Falls back to a deterministic engineer offline. |
| **Live dashboard** | FastAPI WebSocket feed: lap-time vs model, learned tyre curves, strategy table with P(best), undercut odds, radio log, driver radio input. |

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                   # 17 tests
python -m pitwall.api.server                # dashboard at http://127.0.0.1:8000
python -m pitwall.evaluation.backtest --races 200 --seed 42
```

Optional: `pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY` for the Claude engineer; `pip install -e ".[f1]"` for real races via FastF1 (`pitwall.data.fastf1_loader.load_race`). Docker: `docker build -t pitwall . && docker run -p 8000:8000 pitwall`.

## Results: closed-loop backtest (200 races, seed 42)

All policies drive the *same* car through the *same* hidden reality (true tyre physics, safety-car laps, lap noise, stop variance). The AI only sees noisy practice data (which never reaches the tyre cliff) plus the laps it has driven.

| Race type | Adaptive AI vs static pre-race optimum | Adaptive AI vs conventional M>H template |
|---|---|---|
| All races (n=200) | **+1.07 s** (wins 61%) | **+2.22 s** (wins 85%) |
| With safety car (n=90) | **+2.29 s** (wins 70%) | **+3.15 s** (wins 74%) |
| No safety car (n=110) | +0.08 s (wins 54%, roughly parity) | +1.46 s (wins 94%) |

Reading it honestly: the adaptive policy's edge comes from exploiting safety cars and correcting the practice-fitted model's blind spot (the cliff); on green-flag races it ties the static optimum. Seconds matter: a typical pit-stop gap between cars is 1-3 s. Reproduce with `backtest_results.txt` / the command above.

## Limitations (read before citing numbers)

- **Reality is synthetic.** The backtest world is a physically structured simulator (`physics.py`), not real timing data. Gains are demonstrated against a hidden-truth simulator with model mismatch, not against real races. A FastF1 loader is included but real-race backtesting is not done yet.
- Single-car optimisation: rivals are only modelled in the undercut module, not in a full multi-agent race simulation. No traffic or dirty-air model in the main search.
- Tyre age does not advance behind the safety car; no inters/wets; no virtual safety car; SC length is assumed 3-5 laps.
- The LLM path has not been exercised against the live API in CI (it needs a key); the tool layer it calls is tested.

## Roadmap

1. Real-data calibration and backtest on FastF1 seasons (per-circuit SC hazard, per-compound priors).
2. Multi-car game-theoretic simulation (rival strategy inference, traffic, DRS trains).
3. Learned tyre model with uncertainty (hierarchical Bayesian / GP) instead of ridge polynomials.
4. Weather nowcasting inputs for the rain/inters decision.

## Layout

```
src/pitwall/
  physics.py            hidden-truth tyre/fuel/SC physics (simulator + backtest only)
  data/                 Race/LapTable schema, simulator, FastF1 loader
  models/               tyre_deg.py (ridge curves), safety_car.py (logistic hazard)
  strategy/             monte_carlo.py, optimizer.py, undercut.py, anomaly.py (CUSUM), brain.py
  agent/                engineer.py (tools + Claude loop + rule fallback), radio.py
  pipeline.py           lap-by-lap shadow-mode replay session
  api/                  FastAPI + WebSocket + dashboard
  evaluation/backtest.py  closed-loop policy comparison
tests/                  models, strategy, system (API, WebSocket, backtest)
```
