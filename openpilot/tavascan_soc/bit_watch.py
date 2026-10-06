#!/usr/bin/env python3
"""Records what changes on the CAN bus while the car is parked, to find the
charge-plug signal.

Nothing in the DBC names it, and rlog only records while driving, so plugging
in and out has never been captured. Two things make it catchable here:

  * plugging in wakes the car, so the plug state is already set in the first
    frames after the bus wakes -- a snapshot of every message at wake-up holds it;
  * unplugging needs the car unlocked and awake, so it shows up as a bit that
    had been steady for a long time and then flips.

So this writes a snapshot of every message shortly after each wake-up and just
before the bus falls asleep, and in between every bit that flips after being
steady for QUIET_S. Bits that flip all the time -- counters, checksums -- never
stay quiet long enough to be written.

Offroad only, read-only, nothing is sent anywhere.
"""
import json
import os
import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

LOG_PATH = os.getenv("TAVASCAN_BITWATCH_PATH", "/data/tavascan_bitwatch.jsonl")
LOG_MAX_BYTES = 16 * 1024 * 1024
QUIET_S = 30.0          # a bit must have held this long before its flip is written
SLEEP_GAP_S = 5.0       # no frames for this long means the bus has gone to sleep
SNAPSHOT_AFTER_S = 3.0  # wait for every message to have arrived at least once


class BitWatch:
  """Pure logic, fed frame by frame, so it can be checked against a recording."""

  def __init__(self):
    self.value: dict[tuple[int, int], int] = {}
    self.raw: dict[tuple[int, int], bytes] = {}
    self.last_flip: dict[tuple[int, int], dict[int, float]] = {}
    self.awake_since: float | None = None
    self.last_frame: float | None = None
    self.snapshot_done = False

  def _snapshot(self, kind: str, t: float) -> dict:
    return {"kind": kind, "t": t,
            "msgs": {f"{b}:{a:x}": d.hex() for (b, a), d in sorted(self.raw.items())}}

  def feed(self, t: float, bus: int, addr: int, dat: bytes) -> list[dict]:
    out = []
    if self.last_frame is not None and t - self.last_frame > SLEEP_GAP_S:
      out.append(self._snapshot("sleep", self.last_frame))
      self.awake_since = None
    if self.awake_since is None:
      self.awake_since = t
      self.snapshot_done = False
      # everything is recent again: flips straight after waking are the bus
      # starting up, which the wake snapshot already covers
      for flips in self.last_flip.values():
        for b in flips:
          flips[b] = t
    self.last_frame = t

    key = (bus, addr)
    v = int.from_bytes(dat, "little")
    old = self.value.get(key)
    self.value[key] = v
    self.raw[key] = dat
    if old is not None and old != v:
      flips = self.last_flip.setdefault(key, {})
      x = old ^ v
      while x:
        low = x & -x
        bit = low.bit_length() - 1
        x ^= low
        prev = flips.get(bit, self.awake_since)
        if t - prev >= QUIET_S:
          out.append({"kind": "flip", "t": t, "bus": bus, "addr": f"{addr:x}", "bit": bit,
                      "to": (v >> bit) & 1, "held_s": round(t - prev, 1)})
        flips[bit] = t

    if not self.snapshot_done and t - self.awake_since >= SNAPSHOT_AFTER_S:
      out.append(self._snapshot("wake", t))
      self.snapshot_done = True
    return out


def write(events: list[dict]) -> None:
  try:
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > LOG_MAX_BYTES:
      os.replace(LOG_PATH, LOG_PATH + ".1")
    with open(LOG_PATH, "a") as f:
      for e in events:
        f.write(json.dumps(e) + "\n")
  except OSError:
    pass


def main() -> None:
  sock = messaging.sub_sock("can", timeout=1000)
  params = Params()
  w = BitWatch()
  cloudlog.info("tavascan_bitwatch: writing to %s", LOG_PATH)
  while True:
    # Manager only runs this offroad, but it can also be started by hand, and
    # then nothing stops it once the car drives off. Driving makes thousands of
    # flips a minute, none of them about the plug, so stand aside.
    if params.get_bool("IsOnroad"):
      messaging.drain_sock(sock)
      w = BitWatch()
      time.sleep(5)
      continue
    events = []
    for msg in messaging.drain_sock(sock, wait_for_one=True):
      now = time.time()
      for c in msg.can:
        if c.src <= 2:
          events += w.feed(now, c.src, c.address, bytes(c.dat))
    # The bus going to sleep is only noticed by the next frame, which may be
    # hours away. Check the gap here too, so the sleep snapshot is written now.
    if w.last_frame is not None and w.awake_since is not None and time.time() - w.last_frame > SLEEP_GAP_S:
      events.append(w._snapshot("sleep", w.last_frame))
      w.awake_since = None
      w.last_frame = None
    if events:
      write(events)


if __name__ == "__main__":
  main()
