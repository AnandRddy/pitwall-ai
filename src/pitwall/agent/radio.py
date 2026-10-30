"""Driver radio -> structured event (keyword parser; the LLM agent handles free-form text)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_RULES = [
    ("damage", re.compile(r"puncture|damage|front wing|floor|debris|lost (a )?part|vibration", re.I)),
    ("rain", re.compile(r"rain|drops|spots|wet|damp", re.I)),
    ("tyres", re.compile(r"grain|blister|tyres? (are |is )?(gone|dead|going|done)|no grip|rear (left|right)|cliff|sliding", re.I)),
    ("traffic", re.compile(r"traffic|stuck behind|dirty air|can'?t (get|pass)", re.I)),
]


@dataclass
class RadioEvent:
    kind: str          # damage | rain | tyres | traffic | none
    text: str
    severity: float    # 0..1

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "severity": self.severity}


def parse_radio(text: str) -> RadioEvent:
    for kind, rx in _RULES:
        if rx.search(text):
            sev = 0.9 if kind in ("damage", "tyres") and re.search(r"gone|dead|puncture|massive|big", text, re.I) else 0.6
            return RadioEvent(kind, text, sev)
    return RadioEvent("none", text, 0.0)
