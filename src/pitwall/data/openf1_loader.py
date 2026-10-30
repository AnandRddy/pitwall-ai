"""Load real races from the open OpenF1 API (https://openf1.org) into `Race`.

No account or API key is needed. Responses are cached on disk, so each race is
downloaded once. Produces the same container as the simulator, so every model in
this repo runs unchanged on real data.

    from pitwall.data.openf1_loader import load_race, list_races
    race, meta = load_race(2024, "Bahrain")

Approximations (OpenF1 does not publish them): fuel load is a linear burn
profile; cumulative time / positions are rebuilt from lap durations.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..physics import C_IDX, TrackConfig
from .schema import Race

API = "https://api.openf1.org/v1"
CACHE = Path(".openf1-cache")
_last_call = [0.0]


def _get(endpoint: str, **params):
    q = urllib.parse.urlencode(params)
    key = CACHE / (endpoint + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items())) + ".json")
    if key.exists():
        return json.loads(key.read_text())
    url = f"{API}/{endpoint}?{q}"
    for attempt in range(6):
        wait = 0.45 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pitwall-ai"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = []
                break
            if e.code in (429, 500, 502, 503) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise
    CACHE.mkdir(exist_ok=True)
    key.write_text(json.dumps(data))
    return data


@dataclass
class RaceMeta:
    year: int
    name: str
    session_key: int
    red_flag: bool = False
    wet: bool = False
    sc_deployments: list[int] = field(default_factory=list)   # lap each real safety car was deployed
    vsc_deployments: list[int] = field(default_factory=list)
    n_laps: int = 0
    real_base_lap: float = 90.0   # circuit's real median green lap (s); Race lap times are shifted to 90

    @property
    def usable(self) -> bool:
        return not self.red_flag and not self.wet and self.n_laps >= 40


def list_races(year: int) -> list[dict]:
    rows = _get("sessions", year=year, session_name="Race")
    return sorted(rows, key=lambda r: r["date_start"])


def load_race(year: int, grand_prix: str | None = None, session: dict | None = None):
    """Load by country/location substring (e.g. 'Bahrain', 'Monaco') or a sessions row."""
    if session is None:
        g = grand_prix.lower()
        hits = [r for r in list_races(year) if g in (r["country_name"] + " " + r["location"]
                                                     + " " + r["circuit_short_name"]).lower()]
        if not hits:
            raise ValueError(f"No {year} race matching {grand_prix!r}")
        session = hits[0]
    return _build(session)


def _build(session: dict):
    sk = session["session_key"]
    laps = _get("laps", session_key=sk)
    stints = _get("stints", session_key=sk)
    drivers = _get("drivers", session_key=sk)
    rc = _get("race_control", session_key=sk)
    if not laps or not stints:
        raise ValueError(f"No lap/stint data for session {sk}")

    code = {d["driver_number"]: d.get("name_acronym") or str(d["driver_number"]) for d in drivers}
    nums = sorted({r["driver_number"] for r in laps})
    names = [code.get(n, str(n)) for n in nums]
    n_laps = max(r["lap_number"] for r in laps)
    shape = (len(nums), n_laps)
    lap_time = np.full(shape, np.nan)
    pit_out = np.zeros(shape, dtype=bool)
    for r in laps:
        i, j = nums.index(r["driver_number"]), r["lap_number"] - 1
        if r.get("lap_duration"):
            lap_time[i, j] = r["lap_duration"]
        pit_out[i, j] = bool(r.get("is_pit_out_lap"))

    comp = np.full(shape, -1, dtype=int)
    age = np.zeros(shape, dtype=int)
    pit_in = np.zeros(shape, dtype=bool)
    wet = False
    by_drv: dict[int, list[dict]] = {}
    for s in stints:
        by_drv.setdefault(s["driver_number"], []).append(s)
    for n, sts in by_drv.items():
        if n not in nums:
            continue
        i = nums.index(n)
        sts.sort(key=lambda s: s["stint_number"])
        for k, s in enumerate(sts):
            c = str(s.get("compound") or "").upper()
            if c in ("INTERMEDIATE", "WET"):
                wet = True
            a0 = s.get("tyre_age_at_start") or 0
            end = min(s["lap_end"] or n_laps, n_laps)
            for lap in range(s["lap_start"], end + 1):
                comp[i, lap - 1] = C_IDX.get(c, -1)
                age[i, lap - 1] = a0 + (lap - s["lap_start"]) + 1
            if k < len(sts) - 1 and 1 <= end <= n_laps:
                pit_in[i, end - 1] = True

    sc = np.zeros(n_laps, dtype=bool)
    meta = RaceMeta(session["year"], f"{session['country_name']} ({session['circuit_short_name']})", sk,
                    wet=wet, n_laps=n_laps)
    open_sc = None
    for m in sorted(rc, key=lambda m: m["date"]):
        msg, lap = (m.get("message") or "").upper(), m.get("lap_number")
        if m.get("flag") == "RED":
            meta.red_flag = True
        if m.get("category") != "SafetyCar" or not lap:
            continue
        if "VIRTUAL" in msg:
            if "DEPLOYED" in msg:
                meta.vsc_deployments.append(lap)
                sc[lap - 1:lap + 1] |= True  # VSC laps are slow: exclude from model fits
        elif "DEPLOYED" in msg:
            meta.sc_deployments.append(lap)
            open_sc = lap
        elif "IN THIS LAP" in msg and open_sc:
            sc[open_sc - 1:lap] = True
            open_sc = None
    if open_sc:
        sc[open_sc - 1:] = True

    green = np.where(sc[None, :], np.nan, np.where(pit_in | pit_out, np.nan, lap_time))
    base = float(np.nanmedian(green))
    meta.real_base_lap = base
    # Normalise every race to a common 90 s pace scale so laps from different circuits can be pooled
    # as prior data (relative tyre/fuel behaviour transfers; absolute lap time does not).
    lap_time = lap_time + (90.0 - base)
    track = TrackConfig(name=meta.name, laps=n_laps, base_lap=90.0)

    filled = np.where(np.isfinite(lap_time), lap_time, np.nanmedian(lap_time, axis=1, keepdims=True))
    cum = np.cumsum(filled, axis=1)
    # drivers who retire: push them to the back once their laps run out
    for i in range(len(nums)):
        last = int(np.max(np.nonzero(np.isfinite(lap_time[i]))[0])) if np.isfinite(lap_time[i]).any() else -1
        cum[i, last + 1:] = 1e9
    position = np.argsort(np.argsort(cum, axis=0), axis=0) + 1
    fuel = np.tile([track.fuel_at(l) for l in range(1, n_laps + 1)], (len(nums), 1))
    return Race(track, names, lap_time, comp, age, fuel, pit_in, sc, position, cum), meta
