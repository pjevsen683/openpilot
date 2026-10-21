#!/usr/bin/env python3
"""Matches the car to our own OSM map file and publishes what the map says.

The map file is built on a PC by build_map.py and copied to MAP_PATH; it is
data, not code, and lives outside the repo. Without it this process idles.

Twice a second the latest GPS fix is matched to a road (see matcher.py) and
the result written as JSON to OUT_PATH in /dev/shm, where tavascan_web picks it
up for the live page and its trace. Every TRACE_PERIOD_S a slim copy is also
appended to TRACE_PATH, so a drive can be reviewed against the map afterwards.

Read-only towards the car: it subscribes to cereal and never publishes, writes
params or sends CAN. Runs onroad only.
"""
import json
import os
import time

from openpilot.cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.tavascan_map.matcher import Map

MAP_PATH = os.getenv("TAVASCAN_MAP_PATH", "/data/tavascan_map/tavascan_map.sqlite")
OUT_PATH = "/dev/shm/tavascan_map.json"
TRACE_PATH = os.getenv("TAVASCAN_MAP_TRACE", "/data/tavascan_map.jsonl")
TRACE_MAX_BYTES = 8 * 1024 * 1024
TRACE_PERIOD_S = 2.0
STALE_FIX_S = 2.0


def _write_atomic(path: str, data: dict) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(data, f, ensure_ascii=False)
  os.replace(tmp, path)


def _trace(rec: dict) -> None:
  try:
    if os.path.exists(TRACE_PATH) and os.path.getsize(TRACE_PATH) > TRACE_MAX_BYTES:
      os.replace(TRACE_PATH, TRACE_PATH + ".1")
    with open(TRACE_PATH, "a") as f:
      f.write(json.dumps(rec, ensure_ascii=False) + "\n")
  except OSError:
    pass


def main() -> None:
  gps_service = get_gps_location_service(Params())
  sm = messaging.SubMaster([gps_service])
  rk = Ratekeeper(2.0)
  m = None
  last_fix = 0.0
  last_trace = 0.0
  while True:
    rk.keep_time()
    if m is None:
      if not os.path.exists(MAP_PATH):
        time.sleep(10)
        continue
      try:
        m = Map(MAP_PATH)
        cloudlog.info("tavascan_map: using %s", MAP_PATH)
      except Exception:
        cloudlog.exception("tavascan_map: cannot open map")
        time.sleep(30)
        continue

    sm.update(0)
    g = sm[gps_service]
    now = time.monotonic()
    out = {"t": round(time.time(), 1), "fix": False, "match": None}
    if sm.updated[gps_service] and g.hasFix:
      last_fix = now
      try:
        st = m.match(g.latitude, g.longitude, g.bearingDeg, g.speed)
      except Exception:
        cloudlog.exception("tavascan_map: match failed")
        st = None
      out.update(fix=True, lat=round(g.latitude, 6), lon=round(g.longitude, 6),
                 bearing=round(g.bearingDeg), speed_kph=round(g.speed * 3.6, 1),
                 accuracy_m=round(g.horizontalAccuracy, 1), match=st)
    elif now - last_fix < STALE_FIX_S:
      continue   # no new fix this tick; keep the last output on disk
    try:
      _write_atomic(OUT_PATH, out)
    except OSError:
      pass
    if now - last_trace >= TRACE_PERIOD_S and out["fix"]:
      _trace(out)
      last_trace = now


if __name__ == "__main__":
  main()
