#!/usr/bin/env python3
"""Decodes VW's Predictive Street Data (PSD) from the car's own CAN bus.

The car broadcasts an ADAS horizon on PSD_04/05/06 (0x462-0x464, bus 1): the
road ahead as a tree of segments, each with a length, a road category, a lane
count, curvature at both ends, and -- the interesting part -- a ramp flag and a
branch direction and angle. That is junction and slip-road information the mapd
channel never exposed, coming from the car's own navigation.

STOCK_PSD_PRESENT is set on this car, and the messages arrive at roughly 7 Hz.

Read-only. Nothing here transmits.

WHAT IS TRUSTED AND WHAT IS NOT
  Bit positions come from _vw_meb_common.dbc. Most fields have been seen taking
  plausible ranges on a real drive. Two have not earned trust yet:
    * PSD_Pos_Fahrspur, which should say which lane we are in, reads a constant
      0 -- it appears not to be populated, so it is reported but not used.
    * Fahrspuren_Anzahl showed values up to 5 on a day of single-lane driving,
      which is not obviously right. Treat lane counts as unverified until
      checked against a motorway.
"""
import time

ADDR_04, ADDR_05, ADDR_06 = 0x462, 0x463, 0x464
BUS = 1

# Segments stop being interesting once the car has driven past them.
SEGMENT_TTL_S = 20.0
MAX_PATH = 12

ROAD_CATEGORY = {0: "unknown", 1: "motorway", 2: "trunk", 3: "primary",
                 4: "secondary", 5: "local", 6: "minor", 7: "other"}


def _b(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


class PSD:
  def __init__(self):
    self.segments: dict[int, dict] = {}
    self.pos_segment: int | None = None
    self.pos_lane: int | None = None
    self.guidance: bool | None = None
    self.country: int | None = None
    self.last_seen: float = 0.0

  # --- ingestion -----------------------------------------------------------
  def feed(self, address: int, data: bytes) -> None:
    if len(data) < 8:
      return
    now = time.monotonic()
    if address == ADDR_04:
      sid = _b(data, 0, 6)
      if sid == 0:
        return
      self.segments[sid] = {
        "id": sid,
        "prev": _b(data, 6, 6),
        "length_m": _b(data, 12, 7) * 2,
        "category": _b(data, 19, 3),
        "lanes": _b(data, 40, 3),
        "ramp": _b(data, 45, 2),
        "branch_right": bool(_b(data, 56, 1)),
        "branch_angle": round(_b(data, 57, 7) * 1.417323, 1),
        "probable": bool(_b(data, 38, 1)),
        "straightest": bool(_b(data, 39, 1)),
        "quality": bool(_b(data, 37, 1)),
        "t": now,
      }
      self.last_seen = now
    elif address == ADDR_05:
      sid = _b(data, 0, 6)
      if sid:
        self.pos_segment = sid
      self.pos_lane = _b(data, 22, 3)
    elif address == ADDR_06 and _b(data, 0, 3) == 0:
      self.guidance = bool(_b(data, 26, 1))
      self.country = _b(data, 9, 8)

  def _expire(self) -> None:
    now = time.monotonic()
    for sid in [s for s, v in self.segments.items() if now - v["t"] > SEGMENT_TTL_S]:
      del self.segments[sid]

  # --- interpretation ------------------------------------------------------
  def path_ahead(self) -> dict:
    """Follows the most probable path from where we are, noting what branches off.

    Distances are measured from the START of the segment we are currently in.
    PSD gives no offset into that segment, so everything here is out by however
    far we are into it -- up to one segment length. Good enough to say "a ramp
    joins in about 200 m", not good enough for anything that needs metres.
    """
    self._expire()
    out = {"segments": [], "branches": [], "here": None}
    if self.pos_segment is None or self.pos_segment not in self.segments:
      return out

    by_prev: dict[int, list] = {}
    for s in self.segments.values():
      by_prev.setdefault(s["prev"], []).append(s)

    cur = self.segments[self.pos_segment]
    out["here"] = {"category": ROAD_CATEGORY.get(cur["category"], "?"),
                   "lanes": cur["lanes"], "segment": cur["id"]}
    dist = 0.0
    seen = set()
    for _ in range(MAX_PATH):
      if cur["id"] in seen:
        break
      seen.add(cur["id"])
      out["segments"].append({"id": cur["id"], "at_m": round(dist),
                              "length_m": cur["length_m"], "lanes": cur["lanes"],
                              "category": ROAD_CATEGORY.get(cur["category"], "?")})
      nxt = by_prev.get(cur["id"], [])
      if not nxt:
        break
      # The path we are expected to take; everything else leaving this point is
      # a side road, which is exactly what we want to know about.
      main = next((s for s in nxt if s["probable"]), None) or \
             next((s for s in nxt if s["straightest"]), None) or nxt[0]
      for s in nxt:
        if s["id"] == main["id"]:
          continue
        out["branches"].append({
          "at_m": round(dist + cur["length_m"]),
          "side": "right" if s["branch_right"] else "left",
          "angle": s["branch_angle"],
          "ramp": bool(s["ramp"]),
          "lanes": s["lanes"],
          "category": ROAD_CATEGORY.get(s["category"], "?"),
        })
      dist += cur["length_m"]
      cur = main

    out["branches"].sort(key=lambda b: b["at_m"])
    return out

  def snapshot(self) -> dict:
    p = self.path_ahead()
    return {
      "guidance": self.guidance,
      "country": self.country,
      "pos_lane_raw": self.pos_lane,      # reads 0 always; kept for visibility
      "segments_known": len(self.segments),
      "age_s": round(time.monotonic() - self.last_seen, 1) if self.last_seen else None,
      **p,
    }
