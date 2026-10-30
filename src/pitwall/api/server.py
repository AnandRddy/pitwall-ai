"""FastAPI server: REST for one-shot queries, WebSocket for the live pit-wall feed."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..agent.engineer import ToolBox, is_event, make_engineer, RuleEngineer
from ..agent.radio import parse_radio
from ..pipeline import ReplaySession

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="PitWall AI", version="0.1.0")


class RadioIn(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/radio")
def radio(body: RadioIn):
    return parse_radio(body.text).to_dict()


@app.get("/api/strategy")
def strategy(seed: int = 7, lap: int = 0):
    """One-shot recommendation after `lap` laps of the replay."""
    s = ReplaySession(seed=seed)
    payload = None
    for _ in range(lap):
        payload = s.step()
    if lap == 0:
        return s.brain.recommend(s.state(), n_scenarios=300, seed=1).to_dict()
    return payload


@app.get("/api/races")
def races(year: int = 2024):
    """Real races available from OpenF1 for the dashboard's race picker."""
    from ..data.openf1_loader import list_races
    return [{"name": f"{r['country_name']} ({r['circuit_short_name']})", "key": r["country_name"],
             "date": r["date_start"][:10]} for r in list_races(year)]


@app.websocket("/ws/live")
async def live(ws: WebSocket, seed: int = 7, speed: float = 2.0, rain: float = 0.0,
               year: int | None = None, gp: str | None = None, driver: str | None = None, start: int = 1):
    await ws.accept()
    try:
        session = await asyncio.to_thread(
            lambda: ReplaySession(seed=seed, rain=rain, year=year, gp=gp, driver=driver) if year
            else ReplaySession(seed=seed, rain=rain))
    except Exception as e:
        await ws.send_json({"type": "error", "message": f"Could not load race: {e}"})
        await ws.close()
        return
    if start > 1:
        await asyncio.to_thread(session.skip_to, start - 1)
    engineer, llm = make_engineer(session)
    fallback = RuleEngineer()
    await ws.send_json({"type": "hello", "llm": llm, "track": session.track.name, "laps": session.track.laps,
                        "source": session.source, "driver": session.driver, "drivers": session.race.drivers,
                        "outline": session.track_data.outline if session.track_data else None})
    pending: list[str] = []

    async def listen():
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("radio"):
                    pending.append(str(msg["radio"]))
        except (WebSocketDisconnect, RuntimeError):
            pass

    listener = asyncio.create_task(listen())
    try:
        while not session.done:
            text = pending.pop(0) if pending else None
            if text:
                session.add_radio(text)
            payload = await asyncio.to_thread(session.step)
            if is_event(payload, text):
                try:
                    payload["radio_call"] = await asyncio.to_thread(engineer.respond, session, payload, text)
                except Exception:
                    payload["radio_call"] = fallback.respond(session, payload, text)
            else:
                payload["radio_call"] = fallback.respond(session, payload, text)
            payload["driver_radio"] = text
            await ws.send_json({"type": "lap", **payload})
            await asyncio.sleep(1.0 / max(speed, 0.1))
        await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()


def main():  # pragma: no cover
    import uvicorn
    uvicorn.run("pitwall.api.server:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
