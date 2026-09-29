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
from openpilot.common.swaglog import cloudlog
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
  """Never lets an exception kill the thread silently.

  It did: removing one rule took a helper with it, the collector raised on every
  iteration, and the page served {"ready": false} indefinitely while the server
  carried on. Nothing in the logs, because nothing was catching it.
  """
  while True:
    try:
      _collect()
    except Exception:
      cloudlog.exception("tavascan_web: collector failed, retrying")
      time.sleep(2.0)


def _collect() -> None:
  sm = messaging.SubMaster(["carState", "modelV2", "radarTracks", "liveMapDataSP"])
  # PSD is not in the cereal schema, so it is read straight off the bus.
  can_sock = messaging.sub_sock("can", timeout=0)
  psd = psd_mod.PSD()
  # Raw side-assist frames, shown as hex so they can be stared at. Nothing
  # decodes them yet: the front radar's layout produced -461 km/h and 32 m
  # lateral offsets when applied here, and a byte that tracked one overtake at
  # r=0.98 gave slopes spanning a factor of ten across four more.
  side_raw = {"0x24C": None, "0x24D": None}
  side_seen: dict = {}
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
        elif c.src == 0 and c.address in (0x24C, 0x24D):
          raw = bytes(c.dat)
          key = "0x%X" % c.address
          side_raw[key] = raw.hex()
          # remember which byte positions have ever moved, so the page can mark
          # the live ones instead of a wall of identical numbers
          was = side_seen.get(key)
          if was is None:
            side_seen[key] = [set([v]) for v in raw]
          else:
            for i, v in enumerate(raw[:len(was)]):
              was[i].add(v)

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
    # Cheap enough at 10 Hz -- one shm read and a json parse, the same the line
    # above already does -- and it is the one map signal that could be early
    # enough to brake for a roundabout.
    osm_view["horizon"] = osm.map_horizon()

    lanes = shadow.lane_position(sm["modelV2"].laneLineProbs, sm["modelV2"].roadEdges)
    ut = shadow.undertake(points, v_ego, lanes["rightmost"])
    left = shadow.left_lane_report(points, v_ego)
    psd_snap = psd.snapshot()

    # carState.cruiseState.speed is the CAR's own cruise setting and reads zero
    # throughout when openpilot does longitudinal. vCruise is openpilot's own set
    # speed, already in km/h. Checked against a whole drive: cruiseState.speed was
    # 0 for all 102 segments while vCruise tracked every change.
    v_cruise_kph = sm["carState"].vCruise
    caps = [r["cap"] for r in (ut,) if r["cap"] is not None]
    combined = min(caps) if caps else None

    snap = {
      "ready": True,
      "t": round(time.time(), 1),
      "v_ego_kph": round(v_ego * CV.MS_TO_KPH, 1),
      "v_cruise_kph": round(v_cruise_kph, 1) if 0 < v_cruise_kph < 250 else None,
      "engaged": bool(sm["carState"].cruiseState.enabled),
      "radar_age_s": round(radar_age, 1) if radar_age is not None else None,
      "points": points,
      # Side and rear radar. The car gives presence only -- no distance, no
      # position -- so this is a yes or no per side, drawn alongside us rather
      # than anywhere in particular. opendbc decodes it from
      # MEB_Side_Assist_01 into carState, so no raw CAN needed here.
      "side_raw": dict(side_raw),
      "side_live": {k: [len(x) > 1 for x in v] for k, v in side_seen.items()},
      "blindspot": {"left": bool(sm["carState"].leftBlindspot),
                    "right": bool(sm["carState"].rightBlindspot)},
      "scene": geometry.scene(sm["modelV2"]),
      "lanes": lanes,
      "osm": osm_view,
      "psd": psd_snap,
      "undertake": ut,
      "left_lane": left,
      "would_cap_kph": round(combined * CV.MS_TO_KPH, 1) if combined else None,
      # The whole point: how much slower than now would the car be asked to go.
      "delta_kph": round((combined - v_ego) * CV.MS_TO_KPH, 1) if combined else None,
    }

    with _lock:
      _snapshot.clear()
      _snapshot.update(snap)

    active = ut["active"] or bool(left["slower"])
    # A 10 s heartbeat cannot describe an approach to a bend: 162 m at 80 km/h
    # is seven seconds, so the whole event can fall between two records. When
    # the map says the road ahead wants us much slower than we are going, write
    # at the active rate so the approach is there to look at afterwards.
    hz = osm_view.get("horizon") or {}
    curve_ahead = bool(hz.get("available") and
                       hz.get("slowest_kph", 999) < v_ego * CV.MS_TO_KPH - 15.0)
    now = time.monotonic()
    period = ACTIVE_PERIOD_S if (active or was_active or curve_ahead) else HEARTBEAT_S
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
    "blindspot": snap.get("blindspot"),
    "points": snap.get("points"),
    "lanes": snap.get("lanes"),
    "lane_probs": [l["prob"] if l else None for l in (sc.get("lane_lines") or [])],
    "road": osm_live.get("road_name"),
    "limit_kph": osm_live.get("speed_limit_kph"),
    # Kept in the trace because it is the whole question for curve braking: how
    # far ahead does the map name the slow point, and is that further than the
    # distance it takes to get down to that speed.
    "map_horizon": (snap.get("osm") or {}).get("horizon"),
    "radar_age_s": snap.get("radar_age_s"),
    "psd": {k: (snap.get("psd") or {}).get(k)
            for k in ("guidance", "here", "branches", "age_s")},
  }


def append_trace(snap: dict) -> None:
  """Appends one line per tick while a rule is engaged. Bounded in size.

  Rotates rather than stopping. The first version simply returned once the file
  passed the cap, and it passed the cap on 11 September: every drive since then
  wrote nothing at all, in silence, including the ones this was built to record.
  Rolling to a single .1 file bounds the disk at twice the cap and never stops.
  """
  try:
    if os.path.exists(TRACE) and os.path.getsize(TRACE) > TRACE_MAX_BYTES:
      os.replace(TRACE, TRACE + ".1")
    with open(TRACE, "a") as f:
      f.write(json.dumps(trace_record(snap)) + "\n")
  except OSError:
    pass


def main() -> None:
  threading.Thread(target=collector, daemon=True).start()
  server.serve(PORT, PAGE, _snapshot, _lock)


if __name__ == "__main__":
  main()
