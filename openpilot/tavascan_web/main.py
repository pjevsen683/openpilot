#!/usr/bin/env python3
"""Serves a small live page showing what the shadow rules WOULD have done.

Read-only by design. This process subscribes to cereal and evaluates rules from
tavascan_web.shadow against the car's radar, which opendbc already parses for
us into radarTracks -- all six per-lane objects, no bit twiddling needed.

It never publishes to cereal, writes params or sends CAN. Nothing here can
influence how the car drives.

Two ways to use it:
  * Live, at http://<device-ip>:8088/ -- intended for a passenger, not the driver.
  * Afterwards, from the JSONL trace in /data/tavascan_shadow.jsonl, which is
    written whenever a rule would have engaged. This is the safer of the two and
    is the reason the trace exists.

Runs onroad only.
"""
import json
import os
import threading
import time

from openpilot.cereal import messaging
from openpilot.common.realtime import Ratekeeper
from opendbc.car.common.conversions import Conversions as CV
from openpilot.tavascan_web import geometry, osm, server, shadow

PORT = int(os.getenv("TAVASCAN_WEB_PORT", "8088"))
TRACE = os.getenv("TAVASCAN_WEB_TRACE", "/data/tavascan_shadow.jsonl")
TRACE_MAX_BYTES = 8 * 1024 * 1024
# A rule engaging is rare, so the trace would be empty on a drive where nothing
# fired -- and then there is nothing to review afterwards. A slow heartbeat means
# every drive leaves a record of what the model and radar actually saw.
HEARTBEAT_S = 10.0
PAGE = os.path.join(os.path.dirname(__file__), "page.html")

_snapshot: dict = {"ready": False}
_lock = threading.Lock()


def collector() -> None:
  sm = messaging.SubMaster(["carState", "modelV2", "radarTracks", "liveMapDataSP"])
  rk = Ratekeeper(10.0)
  points: list = []
  last_radar = 0.0
  was_active = False
  osm_view: dict = {"params": {}, "maps": osm.maps_installed()}
  osm_tick = 0
  last_beat = 0.0

  while True:
    sm.update(50)
    v_ego = sm["carState"].vEgo

    if sm.updated["radarTracks"]:
      points = geometry.radar_points(sm["radarTracks"], v_ego)
      last_radar = time.monotonic()
    radar_age = time.monotonic() - last_radar if last_radar else None
    if radar_age is not None and radar_age > 2.0:
      points = []

    # The params directory is a filesystem read; once a second is plenty.
    osm_tick += 1
    if osm_tick % 10 == 1:
      osm_view = {"params": osm.read_params(), "maps": osm.maps_installed()}
    osm_view["live"] = osm.read_live(sm)
    osm_view["road_ahead"] = osm.road_ahead()

    lanes = shadow.lane_position(sm["modelV2"].laneLineProbs, sm["modelV2"].roadEdges)
    ut = shadow.undertake(points, v_ego, lanes["rightmost"])
    mg = shadow.merge_yield(points, v_ego)

    v_cruise = sm["carState"].cruiseState.speed
    caps = [r["cap"] for r in (ut, mg) if r["cap"] is not None]
    combined = min(caps) if caps else None

    snap = {
      "ready": True,
      "t": round(time.time(), 1),
      "v_ego_kph": round(v_ego * CV.MS_TO_KPH, 1),
      "v_cruise_kph": round(v_cruise * CV.MS_TO_KPH, 1) if v_cruise else None,
      "engaged": bool(sm["carState"].cruiseState.enabled),
      "radar_age_s": round(radar_age, 1) if radar_age is not None else None,
      "points": points,
      "scene": geometry.scene(sm["modelV2"]),
      "lanes": lanes,
      "osm": osm_view,
      "undertake": ut,
      "merge_yield": mg,
      "would_cap_kph": round(combined * CV.MS_TO_KPH, 1) if combined else None,
      # The whole point: how much slower than now would the car be asked to go.
      "delta_kph": round((combined - v_ego) * CV.MS_TO_KPH, 1) if combined else None,
    }

    with _lock:
      _snapshot.clear()
      _snapshot.update(snap)

    active = ut["active"] or mg["active"]
    now = time.monotonic()
    if active or was_active:
      append_trace(snap)
      last_beat = now
    elif now - last_beat >= HEARTBEAT_S:
      append_trace(snap)
      last_beat = now
    was_active = active

    rk.keep_time()


def append_trace(snap: dict) -> None:
  """Appends one line per tick while a rule is engaged. Bounded in size."""
  try:
    if os.path.exists(TRACE) and os.path.getsize(TRACE) > TRACE_MAX_BYTES:
      return
    with open(TRACE, "a") as f:
      f.write(json.dumps(snap) + "\n")
  except OSError:
    pass


def main() -> None:
  threading.Thread(target=collector, daemon=True).start()
  server.serve(PORT, PAGE, _snapshot, _lock)


if __name__ == "__main__":
  main()
