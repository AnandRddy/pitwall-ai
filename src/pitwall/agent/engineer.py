"""The race engineer: turns model outputs into pit-wall decisions and radio calls.

* `ToolBox` exposes the quantitative engine as tools (also usable by any LLM).
* `RuleEngineer` answers deterministically offline - the default, zero-cost path.
* `LLMEngineer` runs a Claude tool-use loop and is only invoked on *events*
  (box call, SC, cliff alarm, driver radio) to keep latency and cost sane. Every
  number it quotes must come from a tool result.
"""
from __future__ import annotations

import json
import os

from ..pipeline import ReplaySession

SYSTEM = (
    "You are the lead race strategist on an F1 pit wall. Be terse and decisive, like a radio call. "
    "Use ONLY numbers returned by your tools; never invent gaps, times or probabilities. "
    "Lead with the decision (BOX / STAY OUT / SC call), then one line of reasoning with the key number, "
    "then any risk. Max 60 words. Plain text only: no markdown, no asterisks. "
    "Stop laps are race lap numbers; when box_now is true the stop is on the NEXT lap (state lap + 1)."
)

TOOLS = [
    {"name": "get_race_state", "description": "Current lap, tyres, position, gaps, safety-car status.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "optimize_strategy", "description": "Monte Carlo search for the best remaining strategy; returns top plans, "
     "whether to box next lap and the cost of boxing now.",
     "input_schema": {"type": "object", "properties": {"risk_aversion": {"type": "number", "description": "0=expected time, 1=penalise variance"}}}},
    {"name": "evaluate_undercut", "description": "Probability that pitting now undercuts the car ahead.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "safety_car_outlook", "description": "Probability of a safety car in the next N laps.",
     "input_schema": {"type": "object", "properties": {"horizon": {"type": "integer"}}}},
    {"name": "tyre_status", "description": "Residual vs model, CUSUM cliff alarm and model curve for current tyre.",
     "input_schema": {"type": "object", "properties": {}}},
]


class ToolBox:
    def __init__(self, session: ReplaySession):
        self.s = session

    def get_race_state(self) -> dict:
        st = self.s.state()
        return {k: v for k, v in st.__dict__.items()}

    def optimize_strategy(self, risk_aversion: float = 0.0) -> dict:
        st = self.s.state()
        rec = self.s.brain.recommend(st, n_scenarios=300, top_k=14, max_stops=2, refine=False,
                                     risk_aversion=risk_aversion, seed=st.lap)
        self.s.last_rec = rec
        return rec.to_dict()

    def evaluate_undercut(self) -> dict:
        u = self.s.last_undercut
        return u.to_dict() if u else {"note": "No car within 4s ahead."}

    def safety_car_outlook(self, horizon: int = 5) -> dict:
        st = self.s.state()
        p = self.s.brain.sc_model.prob_at_least_one(st.lap + 1, horizon, st.total_laps, st.rain_prob, st.laps_since_sc)
        return {"horizon": horizon, "p_safety_car": round(p, 3), "sc_active": st.sc_remaining > 0}

    def tyre_status(self) -> dict:
        b = self.s.brain
        return {"residual_s": round(b.last_residual, 3), "cusum": round(b.cusum.s, 2), "cliff_alarm": b.cusum.alarm,
                "model_sigma_s": round(b.tyre.sigma, 3)}

    def call(self, name: str, args: dict) -> dict:
        return getattr(self, name)(**(args or {}))


class RuleEngineer:
    def respond(self, session: ReplaySession, payload: dict, radio: str | None = None) -> str:
        rec = payload.get("rec")
        if rec is None:
            return "Chequered flag. Good job, team."
        parts = []
        if radio:
            ev = session.radio[-1]
            parts.append({"damage": "Copy damage, checking data.", "rain": "Copy rain, monitoring radar.",
                          "tyres": "Copy tyres, we are watching the degradation.",
                          "traffic": "Copy, working on track position."}.get(ev.kind, "Copy."))
        best = rec["plans"][0]
        if payload["sc"] and rec["box_now"]:
            parts.append(f"SAFETY CAR. BOX BOX, {rec['next_compound']}s. Pit loss is discounted.")
        elif rec["box_now"]:
            parts.append(f"BOX THIS LAP, {rec['next_compound']}s. Plan {best['label']}, {best['p_best']:.0%} best.")
        elif best["stops"]:
            parts.append(f"Stay out. Target L{rec['next_stop_lap']} for {rec['next_compound']}s; "
                         f"boxing now costs {rec['pit_now_cost']:.1f}s.")
        else:
            parts.append("Stay out, no further stop needed.")
        if payload["cliff_alarm"]:
            parts.append("Tyre drop-off detected, pace is below model.")
        u = payload.get("undercut")
        if u and u["recommended"]:
            parts.append(f"Undercut on {u['rival']} is {u['p_success']:.0%} on.")
        if payload["sc_prob_5"] > 0.2:
            parts.append(f"SC risk next 5 laps {payload['sc_prob_5']:.0%}.")
        return " ".join(parts)


class LLMEngineer:
    def __init__(self, toolbox: ToolBox, model: str = "claude-sonnet-5-5", client=None, max_turns: int = 6):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.tb, self.client, self.model, self.max_turns = toolbox, client, model, max_turns

    def respond(self, session: ReplaySession, payload: dict, radio: str | None = None) -> str:
        trigger = radio and f'Driver radio: "{radio}"' or (
            "Safety car deployed." if payload["sc"] else "Cliff alarm." if payload["cliff_alarm"]
            else "Box decision check." if payload["rec"] and payload["rec"]["box_now"] else "Routine lap check.")
        msgs = [{"role": "user", "content": f"Lap {payload['lap']}/{payload['total']}. {trigger} What is the call?"}]
        for _ in range(self.max_turns):
            resp = self.client.messages.create(model=self.model, max_tokens=500, system=SYSTEM,
                                               tools=TOOLS, messages=msgs)
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text").replace("**", "").strip()
            msgs.append({"role": "assistant", "content": resp.content})
            results = [{"type": "tool_result", "tool_use_id": b.id,
                        "content": json.dumps(self.tb.call(b.name, b.input), default=str)}
                       for b in resp.content if b.type == "tool_use"]
            msgs.append({"role": "user", "content": results})
        return "Unable to reach a call within the tool budget."


def make_engineer(session: ReplaySession):
    """LLM engineer when ANTHROPIC_API_KEY is set, otherwise deterministic rules."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMEngineer(ToolBox(session)), True
        except ImportError:
            pass
    return RuleEngineer(), False


def is_event(payload: dict, radio: str | None) -> bool:
    rec = payload.get("rec")
    return bool(radio or payload["sc"] or payload["cliff_alarm"] or (rec and rec["box_now"]))
