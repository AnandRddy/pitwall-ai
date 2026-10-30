"""Car GPS positions from OpenF1 -> animated track map frames.

OpenF1 publishes an (x, y) point per car roughly every 0.26 s. We downsample to one point
every `step` seconds per car (cached), and interpolate smooth frames inside any lap window.
"""
from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timezone

import numpy as np

from . import openf1_loader as of


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


class TrackData:
    def __init__(self, outline, cars):
        self.outline = outline                 # [[x, y], ...] one clean lap of the winner
        self.cars = cars                       # code -> (t, x, y) numpy arrays

    def frames(self, t0: float, t1: float, n: int = 20) -> dict:
        """Positions of every car at n evenly spaced instants inside [t0, t1)."""
        ts = t0 + (np.arange(n) + 0.5) / n * (t1 - t0)
        out = {}
        for code, (t, x, y) in self.cars.items():
            if len(t) < 2:
                continue
            out[code] = np.stack([np.interp(ts, t, x), np.interp(ts, t, y)], axis=1).round().astype(int).tolist()
        return out

    def to_json(self) -> dict:
        return {"outline": self.outline,
                "cars": {c: [t.tolist(), x.tolist(), y.tolist()] for c, (t, x, y) in self.cars.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "TrackData":
        return cls(d["outline"], {c: (np.array(t), np.array(x), np.array(y)) for c, (t, x, y) in d["cars"].items()})


def load_track(meta, step: float = 2.0) -> TrackData:
    """Fetch (once) and cache downsampled GPS tracks for every driver of a race."""
    cache = of.CACHE / f"track_{meta.session_key}_{step}.json"
    if cache.exists():
        return TrackData.from_json(json.loads(cache.read_text()))
    lo, hi = meta.lap_starts[0] - 5.0, meta.race_end + 5.0
    ref = meta.winner_code
    outline_win = (meta.lap_starts[2], meta.lap_starts[3]) if len(meta.lap_starts) > 4 else (lo, lo + 100)
    cars, outline = {}, []
    for code, num in meta.driver_numbers.items():
        raw = "date%3E%3D" + urllib.parse.quote(_iso(lo)) + "&date%3C%3D" + urllib.parse.quote(_iso(hi))
        rows = of._get("location", _raw=raw, _cache=False, session_key=meta.session_key, driver_number=num)
        t_, x_, y_, last = [], [], [], -1e18
        for r in rows:
            if r["x"] == 0 and r["y"] == 0:
                continue
            t = _epoch(r["date"])
            if code == ref and outline_win[0] <= t < outline_win[1]:
                outline.append([r["x"], r["y"]])
            if t - last >= step:
                t_.append(t); x_.append(r["x"]); y_.append(r["y"]); last = t
        if len(t_) > 10:
            cars[code] = (np.array(t_), np.array(x_, float), np.array(y_, float))
    td = TrackData(outline[::2], cars)
    of.CACHE.mkdir(exist_ok=True)
    cache.write_text(json.dumps(td.to_json()))
    return td
