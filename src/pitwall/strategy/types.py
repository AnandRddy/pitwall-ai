from __future__ import annotations

from dataclasses import dataclass, field

Stops = tuple[tuple[int, str], ...]  # ((pit_lap, compound_fitted), ...)


@dataclass(frozen=True)
class RaceState:
    """What the pit wall knows after `lap` laps are complete."""
    lap: int
    total_laps: int
    driver: str
    compound: str
    tyre_age: int
    used: tuple[str, ...]
    sc_remaining: int = 0       # SC laps still to run (>0 means SC is out for the next lap)
    laps_since_sc: int = 99
    rain_prob: float = 0.0
    position: int = 1
    gap_ahead: float | None = None
    gap_behind: float | None = None
    car_ahead: str | None = None
    car_behind: str | None = None


@dataclass
class Plan:
    stops: Stops
    mean: float            # expected remaining race time (s)
    std: float
    p10: float
    p90: float
    p_best: float = 0.0    # probability this plan is the fastest among the finalists
    delta: float = 0.0     # mean - best mean

    @property
    def label(self) -> str:
        if not self.stops:
            return "No further stop"
        return " > ".join(f"L{lap} {c[0]}" for lap, c in self.stops)

    def to_dict(self) -> dict:
        return {"label": self.label, "stops": [[lap, c] for lap, c in self.stops], "mean": round(self.mean, 2),
                "std": round(self.std, 2), "p10": round(self.p10, 2), "p90": round(self.p90, 2),
                "p_best": round(self.p_best, 3), "delta": round(self.delta, 2)}


@dataclass
class Recommendation:
    plans: list[Plan]
    box_now: bool
    next_stop_lap: int | None
    next_compound: str | None
    pit_now_cost: float          # seconds lost vs. best plan if we box on the next lap
    n_evaluated: int
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Plan:
        return self.plans[0]

    def to_dict(self) -> dict:
        return {"plans": [p.to_dict() for p in self.plans[:5]], "box_now": self.box_now,
                "next_stop_lap": self.next_stop_lap, "next_compound": self.next_compound,
                "pit_now_cost": round(self.pit_now_cost, 2) if self.pit_now_cost != float("inf") else None, "n_evaluated": self.n_evaluated,
                "elapsed_ms": round(self.elapsed_ms, 1), "notes": self.notes}
