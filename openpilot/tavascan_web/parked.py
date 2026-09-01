#!/usr/bin/env python3
"""The offroad half: what the car looks like while it is parked.

Serves the same port as the onroad shadow page. The two never overlap -- this
one is only_offroad, the other only_onroad -- so opening the same address gives
you whichever half is relevant right now, without having to remember two URLs.

It does no CAN sampling of its own. tavascan_soc is already awake offroad and
already decodes the battery, odometer, locks, doors and 12V; it now writes that
state to /data/tavascan_state.json every cycle, and this process just serves it.
One sampler, two consumers.

Read-only. It never writes params, never sends CAN, and never touches the car.
"""
import json
import os
import threading
import time

from openpilot.common.realtime import Ratekeeper
from openpilot.tavascan_web import osm, server

PORT = int(os.getenv("TAVASCAN_WEB_PORT", "8088"))
STATE_PATH = os.getenv("TAVASCAN_STATE_PATH", "/data/tavascan_state.json")
PAGE = os.path.join(os.path.dirname(__file__), "parked.html")

# SoC rising by more than this over the recent window means it is charging.
# The reading is quantised to whole percent, so anything smaller is noise.
CHARGE_PCT = 0.5
CHARGE_WINDOW_S = 900

_snapshot: dict = {"ready": False}
_lock = threading.Lock()


def charging(history: list) -> dict:
  """Is the battery climbing? Returns direction and a rate when there is one."""
  out = {"charging": None, "rate_pct_h": None, "window_s": None}
  if not history or len(history) < 2:
    return out

  now = history[-1][0]
  recent = [h for h in history if now - h[0] <= CHARGE_WINDOW_S]
  if len(recent) < 2:
    return out

  dt = recent[-1][0] - recent[0][0]
  dv = recent[-1][1] - recent[0][1]
  if dt < 120:
    return out

  out["window_s"] = dt
  out["rate_pct_h"] = round(dv / (dt / 3600.0), 1)
  out["charging"] = bool(dv >= CHARGE_PCT)
  return out


def read_state() -> dict:
  try:
    with open(STATE_PATH) as f:
      return json.load(f)
  except (OSError, ValueError):
    return {}


def collector() -> None:
  rk = Ratekeeper(0.2)  # every 5 s; nothing here changes faster than that
  while True:
    blob = read_state()
    state = blob.get("state") or {}
    history = blob.get("history") or []

    try:
      file_age = time.time() - os.path.getmtime(STATE_PATH)
    except OSError:
      file_age = None

    snap = {
      "ready": bool(state),
      "t": round(time.time(), 1),
      "parked": True,
      "state": state,
      "history": history[-120:],
      "charge": charging(history),
      # How stale the whole picture is, as opposed to how old the CAN reading is.
      "file_age_s": round(file_age) if file_age is not None else None,
      "position": osm.read_params().get("LastGPSPosition"),
    }

    with _lock:
      _snapshot.clear()
      _snapshot.update(snap)
    rk.keep_time()


def main() -> None:
  threading.Thread(target=collector, daemon=True).start()
  server.serve(PORT, PAGE, _snapshot, _lock)


if __name__ == "__main__":
  main()
