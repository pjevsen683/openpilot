#!/usr/bin/env python3
"""Matches the car to our own OSM map file and publishes what the map says.

The map file is built on a PC by build_map.py and copied to MAP_PATH; it is
data, not code, and lives outside the repo.

Twice a second the latest GPS fix is matched to a road (see matcher.py). Ten
times a second the lane filter (lanes.py, fed by lane_runner.py) updates which
lane of the carriageway we are in; it runs with or without the map. Both go
into the JSON at OUT_PATH in /dev/shm, written twice a second, where
tavascan_web picks it up for the live page and its trace. Every TRACE_PERIOD_S
a copy without the drawn road is appended to TRACE_PATH, so a drive can be
reviewed afterwards.

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
from openpilot.tavascan_map.lane_runner import LaneRunner
from openpilot.tavascan_map.matcher import Map
from openpilot.tavascan_web import geometry

MAP_PATH = os.getenv("TAVASCAN_MAP_PATH", "/data/tavascan_map/tavascan_map.sqlite")
OUT_PATH = "/dev/shm/tavascan_map.json"
TRACE_PATH = os.getenv("TAVASCAN_MAP_TRACE", "/data/tavascan_map.jsonl")
TRACE_MAX_BYTES = 8 * 1024 * 1024
TRACE_PERIOD_S = 2.0
STALE_FIX_S = 2.0
LOOP_HZ = 10.0
MATCH_PERIOD_S = 0.5
WRITE_PERIOD_S = 0.5
RADAR_STALE_S = 0.5


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


def _model_inputs(md) -> dict:
  return {"lc_state": str(md.meta.laneChangeState), "lc_dir": str(md.meta.laneChangeDirection),
          "lane_ys": [ll.y[0] for ll in md.laneLines if len(ll.y)],
          "lane_probs": list(md.laneLineProbs),
          "edge_ys": [e.y[0] for e in md.roadEdges if len(e.y)],
          "edge_stds": list(md.roadEdgeStds)}


def main() -> None:
  gps_service = get_gps_location_service(Params())
  sm = messaging.SubMaster([gps_service, "modelV2", "carState", "radarTracks"])
  rk = Ratekeeper(LOOP_HZ)
  m = None
  next_open = 0.0
  runner = LaneRunner()
  radar: list = []
  radar_t = -1e9
  last_fix = last_match = last_write = last_trace = -1e9
  out: dict = {"t": round(time.time(), 1), "fix": False, "match": None}
  while True:
    rk.keep_time()
    now = time.monotonic()
    if m is None and now >= next_open:
      if os.path.exists(MAP_PATH):
        try:
          m = Map(MAP_PATH)
          cloudlog.info("tavascan_map: using %s", MAP_PATH)
        except Exception:
          cloudlog.exception("tavascan_map: cannot open map")
      next_open = now + 30.0

    sm.update(0)
    g = sm[gps_service]
    if m is not None and sm.updated[gps_service] and g.hasFix and now - last_match >= MATCH_PERIOD_S:
      last_match = last_fix = now
      try:
        st = m.match(g.latitude, g.longitude, g.bearingDeg, g.speed)
      except Exception:
        cloudlog.exception("tavascan_map: match failed")
        st = None
      out = {"fix": True, "lat": round(g.latitude, 6), "lon": round(g.longitude, 6),
             "bearing": round(g.bearingDeg), "speed_kph": round(g.speed * 3.6, 1),
             "accuracy_m": round(g.horizontalAccuracy, 1), "match": st}
    elif now - last_fix > STALE_FIX_S:
      out = {"fix": False, "match": None}

    # The lane filter runs every tick, map or no map: lane changes take a few
    # seconds and are missed at the 2 Hz the map is matched at.
    cs = sm["carState"]
    if sm.updated["radarTracks"]:
      radar = geometry.radar_points(sm["radarTracks"], cs.vEgo)
      radar_t = now
    elif now - radar_t > RADAR_STALE_S:
      radar = []
    try:
      # a stale model frame (modeld behind) is passed as None so it is not counted twice
      model = _model_inputs(sm["modelV2"]) if sm.updated["modelV2"] else None
      out["lane"] = runner.step(now, cs.vEgo, 1.0 / LOOP_HZ, out.get("match"), model,
                                {"left_blinker": cs.leftBlinker, "right_blinker": cs.rightBlinker}, radar)
    except Exception:
      cloudlog.exception("tavascan_map: lane filter failed")
    out["t"] = round(time.time(), 1)

    if now - last_write >= WRITE_PERIOD_S:
      last_write = now
      try:
        _write_atomic(OUT_PATH, out)
      except OSError:
        pass
    if now - last_trace >= TRACE_PERIOD_S and (out["fix"] or out.get("lane")):
      last_trace = now
      rec = dict(out)
      if rec.get("match"):
        rec["match"] = {k: v for k, v in rec["match"].items() if k != "path"}
      _trace(rec)


if __name__ == "__main__":
  main()
