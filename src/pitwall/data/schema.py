"""Data containers shared by the simulator, FastF1 loader and models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..physics import COMPOUNDS, TrackConfig


@dataclass
class LapTable:
    """Flat table of clean laps used to fit the tyre model."""
    driver: np.ndarray
    compound: np.ndarray   # int index into COMPOUNDS
    age: np.ndarray        # tyre age in laps (1 = first lap on the tyre)
    fuel: np.ndarray       # kg at start of lap
    lap_time: np.ndarray   # seconds

    def __len__(self) -> int:
        return len(self.lap_time)

    @staticmethod
    def empty() -> "LapTable":
        return LapTable(np.array([], dtype=object), np.array([], dtype=int), np.array([], dtype=int),
                        np.array([], dtype=float), np.array([], dtype=float))

    @staticmethod
    def concat(tables: list["LapTable"]) -> "LapTable":
        tables = [t for t in tables if len(t)]
        if not tables:
            return LapTable.empty()
        return LapTable(*(np.concatenate([getattr(t, f) for t in tables])
                          for f in ("driver", "compound", "age", "fuel", "lap_time")))

    def mask(self, m: np.ndarray) -> "LapTable":
        return LapTable(self.driver[m], self.compound[m], self.age[m], self.fuel[m], self.lap_time[m])


@dataclass
class Race:
    """A full race: per-car, per-lap arrays of shape (n_cars, laps)."""
    track: TrackConfig
    drivers: list[str]
    lap_time: np.ndarray
    compound: np.ndarray    # compound used DURING the lap
    tyre_age: np.ndarray    # age during the lap
    fuel: np.ndarray
    pit_in: np.ndarray      # bool: car pitted at the end of this lap
    sc: np.ndarray          # (laps,) bool
    position: np.ndarray    # position at the end of the lap (1-based)
    cum: np.ndarray         # cumulative race time (gap-adjusted under SC)

    @property
    def n_laps(self) -> int:
        return self.lap_time.shape[1]

    def idx(self, driver: str) -> int:
        return self.drivers.index(driver)

    def table_for_lap(self, lap: int) -> LapTable:
        """Clean, green-flag, non-pit, non-out laps of all cars for 1-indexed `lap`."""
        j = lap - 1
        ok = (~self.pit_in[:, j]) & (self.compound[:, j] >= 0) & np.isfinite(self.lap_time[:, j])
        ok &= (not self.sc[j]) and lap > 1
        ok &= self.tyre_age[:, j] > 1  # drop out-laps
        d = np.array(self.drivers, dtype=object)
        return LapTable(d[ok], self.compound[ok, j], self.tyre_age[ok, j], self.fuel[ok, j], self.lap_time[ok, j])

    def gaps(self, driver: str, lap: int):
        """(gap to car ahead, gap to car behind, car ahead, car behind) at end of `lap`."""
        j, i = lap - 1, self.idx(driver)
        pos = self.position[:, j]
        p = pos[i]
        order = np.argsort(pos)
        ahead = self.drivers[order[p - 2]] if p > 1 else None
        behind = self.drivers[order[p]] if p < len(pos) else None
        ga = float(self.cum[i, j] - self.cum[order[p - 2], j]) if ahead else None
        gb = float(self.cum[order[p], j] - self.cum[i, j]) if behind else None
        return ga, gb, ahead, behind

    def tyre_state(self, driver: str, lap: int):
        """(compound, age, used compounds) of the car after `lap` laps (post any stop)."""
        i = self.idx(driver)
        j = lap - 1
        used = {COMPOUNDS[c] for c in self.compound[i, : lap] if c >= 0}
        if lap == 0:
            return COMPOUNDS[self.compound[i, 0]], 0, (COMPOUNDS[self.compound[i, 0]],)
        if self.pit_in[i, j] and j + 1 < self.n_laps:
            comp, age = COMPOUNDS[self.compound[i, j + 1]], 0
        else:
            comp, age = COMPOUNDS[self.compound[i, j]], int(self.tyre_age[i, j])
        used.add(comp)
        return comp, age, tuple(sorted(used))

    def sc_context(self, lap: int):
        return sc_context_of(self.sc, lap)


def sc_context_of(sc: np.ndarray, lap: int):
    """SC knowledge available before lap `lap+1` starts: (remaining estimate, laps since SC).

    The only look-ahead is the immediate lap's flag - the "safety car deployed" notice a
    real pit wall receives during the lap just completed.
    """
    if lap < len(sc) and sc[lap]:
        elapsed, k = 0, lap - 1
        while k >= 0 and sc[k]:
            elapsed += 1
            k -= 1
        return max(1, 3 - elapsed), 0
    since, k = 0, lap - 1
    while k >= 0 and not sc[k]:
        since += 1
        k -= 1
    return 0, (since if k >= 0 else 99)
