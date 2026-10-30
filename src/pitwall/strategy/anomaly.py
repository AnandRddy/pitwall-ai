"""CUSUM change detector for tyre-cliff / damage onset.

Residual = actual lap time - model prediction. A tyre falling off a cliff (or a
puncture / damage) shows up as a sustained positive drift that a per-lap
threshold would miss but a cumulative sum catches within a few laps.
"""
from __future__ import annotations


class CusumDetector:
    def __init__(self, k: float = 0.15, h: float = 0.9):
        self.k, self.h = k, h
        self.s = 0.0
        self.alarm = False

    def update(self, residual: float) -> bool:
        self.s = max(0.0, self.s + residual - self.k)
        self.alarm = self.s > self.h
        return self.alarm

    def reset(self) -> None:
        self.s, self.alarm = 0.0, False
