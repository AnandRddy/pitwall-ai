"""Export one real race (OpenF1) to a self-contained HTML explorer.

    python scripts/real_data_view.py 2023 Azerbaijan docs/real_race.html
"""
import json
import sys
from pathlib import Path

import numpy as np

from pitwall.data.openf1_loader import load_race
from pitwall.physics import COMPOUNDS

year, gp, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
race, meta = load_race(year, gp)
shift = meta.real_base_lap - 90.0            # undo the 90 s normalisation: show real lap times
order = np.argsort(race.position[:, -1])
drivers = []
for i in order:
    laps = []
    for j in range(race.n_laps):
        t = race.lap_time[i, j]
        laps.append({"lap": j + 1, "t": None if not np.isfinite(t) else round(float(t + shift), 3),
                     "c": None if race.compound[i, j] < 0 else COMPOUNDS[race.compound[i, j]],
                     "age": int(race.tyre_age[i, j]), "pit": bool(race.pit_in[i, j]), "sc": bool(race.sc[j]),
                     "pos": int(race.position[i, j])})
    drivers.append({"code": race.drivers[i], "laps": laps})
data = {"title": f"{meta.name} {meta.year}", "n_laps": race.n_laps, "sc_deployed": meta.sc_deployments,
        "base": round(meta.real_base_lap, 2), "drivers": drivers}

html = (Path(__file__).parent / "real_data_view.template.html").read_text(encoding="utf-8")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html.replace("/*DATA*/null", json.dumps(data)), encoding="utf-8")
print("wrote", out, f"({out.stat().st_size // 1024} KB)")
