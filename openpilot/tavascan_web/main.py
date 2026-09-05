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
from openpilot.tavascan_web import geometry, osm, psd as psd_mod, server, shadow

PORT = int(os.getenv("TAVASCAN_WEB_PORT", "8088"))
TRACE = os.getenv("TAVASCAN_WEB_TRACE", "/data/tavascan_shadow.jsonl")
TRACE_MAX_BYTES = 8 * 1024 * 1024
# A rule engaging is rare, so the trace would be empty on a drive where nothing
# fired -- and then there is nothing to review afterwards. A slow heartbeat means
# every drive leaves a record of what the model and radar actually saw.
HEARTBEAT_S = 10.0
# While a rule is engaged, 10 Hz is far more detail than reviewing needs and it
# fills the cap in minutes. 2 Hz still shows how an engagement developed.
ACTIVE_PERIOD_S = 0.5
PAGE = os.path.join(os.path.dirname(__file__), "page.html")

_snapshot: dict = {"ready": False}
_lock = threading.Lock()


def collector() -> None:
  sm = messaging.SubMaster(["carState", "modelV2", "radarTracks", "liveMapDataSP"])
  # PSD is not in the cereal schema, so it is read straight off the bus.
  can_sock = messaging.sub_sock("can", timeout=0)
  psd = psd_mod.PSD()
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

    for m in messaging.drain_sock(can_sock):
      for c in m.can:
        if c.src == psd_mod.BUS and c.address in (psd_mod.ADDR_04, psd_mod.ADDR_05, psd_mod.ADDR_06):
          psd.feed(c.address, bytes(c.dat))

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
    left = shadow.left_lane_report(points, v_ego)
    mg = shadow.merge_yield(points, v_ego, lanes["rightmost"])

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
      "psd": psd.snapshot(),
      "undertake": ut,
      "left_lane": left,
      "merge_yield": mg,
      "would_cap_kph": round(combined * CV.MS_TO_KPH, 1) if combined else None,
      # The whole point: how much slower than now would the car be asked to go.
      "delta_kph": round((combined - v_ego) * CV.MS_TO_KPH, 1) if combined else None,
    }

    with _lock:
      _snapshot.clear()
      _snapshot.update(snap)

    active = ut["active"] or mg["active"] or bool(left["slower"])
    now = time.monotonic()
    period = ACTIVE_PERIOD_S if (active or was_active) else HEARTBEAT_S
    if now - last_beat >= period:
      append_trace(snap)
      last_beat = now
    was_active = active

    rk.keep_time()


def trace_record(snap: dict) -> dict:
  """A slimmed copy for the trace.

  The full snapshot is ~4 kB, mostly the mapd param dump and the scene
  polylines, and it filled the 8 MB cap on the first day. What matters
  afterwards is the verdict and the geometry that produced it, so the params
  go and the lane lines keep their probabilities but not their points.
  """
  sc = snap.get("scene") or {}
  osm_live = (snap.get("osm") or {}).get("live") or {}
  return {
    "t": snap.get("t"),
    "v_ego_kph": snap.get("v_ego_kph"),
    "v_cruise_kph": snap.get("v_cruise_kph"),
    "engaged": snap.get("engaged"),
    "delta_kph": snap.get("delta_kph"),
    "undertake": snap.get("undertake"),
    "left_lane": snap.get("left_lane"),
    "merge_yield": snap.get("merge_yield"),
    "points": snap.get("points"),
    "lanes": snap.get("lanes"),
    "lane_probs": [l["prob"] if l else None for l in (sc.get("lane_lines") or [])],
    "road": osm_live.get("road_name"),
    "limit_kph": osm_live.get("speed_limit_kph"),
    "radar_age_s": snap.get("radar_age_s"),
    "psd": {k: (snap.get("psd") or {}).get(k)
            for k in ("guidance", "here", "branches", "age_s")},
  }


def append_trace(snap: dict) -> None:
  """Appends one line per tick while a rule is engaged. Bounded in size."""
  try:
    if os.path.exists(TRACE) and os.path.getsize(TRACE) > TRACE_MAX_BYTES:
      return
    with open(TRACE, "a") as f:
      f.write(json.dumps(trace_record(snap)) + "\n")
  except OSError:
    pass


def main() -> None:
  threading.Thread(target=collector, daemon=True).start()
  server.serve(PORT, PAGE, _snapshot, _lock)


if __name__ == "__main__":
  main()
