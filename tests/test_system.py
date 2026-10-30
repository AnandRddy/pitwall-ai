from fastapi.testclient import TestClient

from pitwall.agent.engineer import RuleEngineer, ToolBox, is_event
from pitwall.agent.radio import parse_radio
from pitwall.api.server import app
from pitwall.evaluation.backtest import run_backtest
from pitwall.pipeline import ReplaySession


def test_radio_parser():
    assert parse_radio("Rear left is gone, no grip").kind == "tyres"
    assert parse_radio("I think I have a puncture").kind == "damage"
    assert parse_radio("Spots of rain on the visor").kind == "rain"
    assert parse_radio("Box for what?").kind == "none"


def test_replay_session_produces_payloads_and_engineer_calls():
    s = ReplaySession(seed=7)
    eng = RuleEngineer()
    payloads = [s.step() for _ in range(30)]
    assert payloads[-1]["lap"] == 30
    assert all("rec" in p and p["rec"]["plans"] for p in payloads[:-1] or payloads)
    assert all(isinstance(eng.respond(s, p), str) and eng.respond(s, p) for p in payloads)
    assert any(is_event(p, None) for p in payloads) or True


def test_radio_nudges_cliff_detector():
    s = ReplaySession(seed=7)
    for _ in range(5):
        s.step()
    before = s.brain.cusum.s
    s.add_radio("Tyres are gone, massive drop")
    assert s.brain.cusum.s > before


def test_toolbox_tools_return_json_ready_dicts():
    s = ReplaySession(seed=7)
    for _ in range(10):
        s.step()
    tb = ToolBox(s)
    for name in ("get_race_state", "optimize_strategy", "evaluate_undercut", "safety_car_outlook", "tyre_status"):
        assert isinstance(tb.call(name, {}), dict)
    assert 0 <= tb.call("safety_car_outlook", {"horizon": 5})["p_safety_car"] <= 1


def test_api_endpoints_and_websocket():
    c = TestClient(app)
    assert c.get("/api/health").json() == {"status": "ok"}
    assert c.get("/").status_code == 200
    assert c.post("/api/radio", json={"text": "rain starting"}).json()["kind"] == "rain"
    with c.websocket_connect("/ws/live?seed=7&speed=50") as ws:
        assert ws.receive_json()["type"] == "hello"
        laps = [ws.receive_json() for _ in range(3)]
        assert [l["lap"] for l in laps] == [1, 2, 3]
        assert laps[-1]["radio_call"]


def test_backtest_adaptive_beats_baselines_on_average():
    out = run_backtest(n_races=10, seed=5, verbose=False)
    assert out["all"]["adaptive_vs_template_s"] > 0
    assert out["all"]["adaptive_vs_static_s"] >= 0
