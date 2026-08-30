#!/usr/bin/env python3
"""Serves a small live page showing what the shadow rules WOULD have done.

Read-only by design. This process subscribes to cereal, decodes the car's radar
and evaluates rules from tavascan_web.shadow. It never publishes to cereal, never
writes params, never sends CAN. Nothing here can influence how the car drives.

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import Ratekeeper
from opendbc.car.common.conversions import Conversions as CV
from openpilot.tavascan_web import osm, radar, shadow

PORT = int(os.getenv("TAVASCAN_WEB_PORT", "8088"))
TRACE = os.getenv("TAVASCAN_WEB_TRACE", "/data/tavascan_shadow.jsonl")
TRACE_MAX_BYTES = 8 * 1024 * 1024
PAGE = os.path.join(os.path.dirname(__file__), "page.html")

_snapshot: dict = {"ready": False}
_lock = threading.Lock()


def collector() -> None:
  can_sock = messaging.sub_sock("can", timeout=20)
  sm = messaging.SubMaster(["carState", "modelV2", "longitudinalPlan", "liveMapDataSP"])
  rk = Ratekeeper(10.0)
  objects: dict = {}
  last_radar = 0.0
  was_active = False
  osm_view: dict = {"params": {}, "maps": osm.maps_installed()}
  osm_tick = 0

  while True:
    sm.update(0)
    v_ego = sm["carState"].vEgo if sm.updated["carState"] else 0.0

    for msg in messaging.drain_sock(can_sock):
      for c in msg.can:
        if c.address == radar.RADAR_ADDR and c.src == radar.RADAR_BUS:
          decoded = radar.decode(bytes(c.dat), v_ego)
          if decoded is not None:
            objects = decoded
            last_radar = time.monotonic()

    radar_age = time.monotonic() - last_radar if last_radar else None
    if radar_age is not None and radar_age > 2.0:
      objects = {}

    lanes = shadow.lane_position(sm["modelV2"].laneLineProbs, sm["modelV2"].roadEdges)
    ut = shadow.undertake(objects, v_ego, lanes["rightmost"])
    mg = shadow.merge_yield(objects, v_ego)

    # The params directory is a filesystem read; once a second is plenty.
    osm_tick += 1
    if osm_tick % 10 == 1:
      osm_view = {"params": osm.read_params(), "maps": osm.maps_installed()}
    osm_view["live"] = osm.read_live(sm)

    v_cruise = sm["carState"].cruiseState.speed if sm.updated["carState"] else 0.0
    caps = [r["cap"] for r in (ut, mg) if r["cap"] is not None]
    combined = min(caps) if caps else None

    snap = {
      "ready": True,
      "t": round(time.time(), 1),
      "v_ego_kph": round(v_ego * CV.MS_TO_KPH, 1),
      "v_cruise_kph": round(v_cruise * CV.MS_TO_KPH, 1) if v_cruise else None,
      "engaged": bool(sm["carState"].cruiseState.enabled) if sm.updated["carState"] else False,
      "radar_age_s": round(radar_age, 1) if radar_age is not None else None,
      "objects": objects,
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
    if active or was_active:
      append_trace(snap)
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


class Handler(BaseHTTPRequestHandler):
  def log_message(self, fmt, *args):
    pass  # do not spam the openpilot logs with request lines

  def _send(self, code: int, body: bytes, ctype: str) -> None:
    self.send_response(code)
    self.send_header("Content-Type", ctype)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self):
    if self.path.startswith("/data.json"):
      with _lock:
        body = json.dumps(_snapshot).encode()
      self._send(200, body, "application/json")
    elif self.path in ("/", "/index.html"):
      try:
        with open(PAGE, "rb") as f:
          self._send(200, f.read(), "text/html; charset=utf-8")
      except OSError:
        self._send(500, b"page missing", "text/plain")
    else:
      self._send(404, b"not found", "text/plain")


def main() -> None:
  threading.Thread(target=collector, daemon=True).start()
  cloudlog.info("tavascan_web: serving on :%d", PORT)
  ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
  main()
