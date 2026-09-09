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
    * PSD_Abzweigerichtung, the side a branch leaves on, is read as 1 = left by
      inference rather than measurement -- see the note where it is decoded.
      The raw bit is reported alongside so it can be checked against a known
      exit.

  Fahrspuren_Anzahl has since been checked and is trustworthy: it reads 3 on
  the three-lane Oestjyske Motorvej and 1 on the single-lane roads around it.

  Curvature does NOT describe roundabouts or other tight corners, and the car's
  own route guidance being active makes no difference. Across two routes, of 39
  corners driven at 7-16 m radius only 12 carried a PSD value low enough to read
  as tight; the rest reported 49, 76, 132, 161, 168, 204 or the invalid 255 for
  the same kind of corner. The field is useful for the sweeping bends of a main
  road and nothing tighter, which is a real limit on how much this can be worth.

  The attribute list in PSD_05 was checked as an alternative and does not carry
  it either: the only candidate with enough samples predicted the distance to a
  roundabout with a slope of 0.18 and a median error of 70 m.
"""
import math
import time

ADDR_04, ADDR_05, ADDR_06 = 0x462, 0x463, 0x464
BUS = 1

# Segments stop being interesting once the car has driven past them.
SEGMENT_TTL_S = 20.0
MAX_PATH = 12

ROAD_CATEGORY = {0: "unknown", 1: "motorway", 2: "trunk", 3: "primary",
                 4: "secondary", 5: "local", 6: "minor", 7: "other"}

CURV_INVALID = 255
# PSD encodes curvature as an index that runs the other way from curvature: a
# low number is a tight bend and a high one is a straight road, with the radius
# roughly doubling every 40 counts. Fitted against the yaw rate actually driven
# through 346 segments:
#
#     PSD  0-15 -> 205 m      PSD 64-79  ->  712 m
#     PSD 32-47 -> 304 m      PSD 128-143-> 3480 m
#
# A least-squares fit on log radius gives R = 130 * exp(v / 43.5), which tracks
# the shape well (R^2 = 0.97 in log space) but is off by up to 25 % on any one
# bucket. The comparison pairs a segment's end curvature against a window of
# measured yaw rate, so treat the radius as an order of magnitude rather than a
# number to plan a manoeuvre with.
CURV_R0, CURV_V0, CURV_DECADE = 130.2, 0.0, 43.5
CURV_MAX_R = 5000.0
# The fit was built from ordinary road bends and only holds there. Checked
# against roundabouts, which we measured at 8-16 m radius, PSD reported values
# of 14, 17, 22, 76, 137, 175 and 255 for the same kind of corner -- no order at
# all. So tight corners are not described by this field, and a number computed
# for one would be invented. Outside the range the fit was measured over, no
# radius is reported and nothing downstream claims to know the speed.
CURV_VALID_LO, CURV_VALID_HI = 32, 145
# Lateral acceleration we are willing to take through a bend, matching the
# figure sunnypilot's curve speed control uses.
A_LAT_MAX = 2.0
# A bend only counts as a reason to slow down if it would actually make us slow
# down. Without this, the end of every straight segment is announced with the
# speed its 1000 m radius allows -- "bend in 14 m, 181 km/h" -- which is noise
# dressed up as a warning.
SLOWDOWN_MARGIN_KPH = 5
# Used when we do not know our own speed. Nothing above this constrains anyone.
SLOWDOWN_CEILING_KPH = 110


def radius_from_psd(value: int) -> float | None:
  """Metres, or None when the field is unset or says straight."""
  if value == CURV_INVALID or not (CURV_VALID_LO <= value <= CURV_VALID_HI):
    return None
  r = CURV_R0 * math.exp((value - CURV_V0) / CURV_DECADE)
  return None if r > CURV_MAX_R else round(r)


def speed_for_radius(radius: float | None) -> float | None:
  """km/h that keeps lateral acceleration at A_LAT_MAX through a bend."""
  if not radius or radius <= 0:
    return None
  return round(math.sqrt(A_LAT_MAX * radius) * 3.6)


def _b(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


class PSD:
  def __init__(self):
    self.segments: dict[int, dict] = {}
    self.pos_segment: int | None = None
    self.pos_remaining_m: int | None = None
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
        "branch_dir_bit": _b(data, 56, 1),
        "curv_start": _b(data, 47, 8),
        "curv_start_vz": _b(data, 55, 1),
        "curv_end": _b(data, 22, 8),
        "curv_end_vz": _b(data, 30, 1),
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
      # Not the segment's length, despite the name: it counts down to the end of
      # the segment we are in. Checked against distance travelled -- a 28 m
      # segment reads 28 and falls to 6 after 23 m driven, at 2 m resolution.
      # This is why distances here do not need the car's speed integrated: the
      # one number PSD appeared to be missing is the one it was sending.
      self.pos_remaining_m = _b(data, 6, 7) * 2
      self.pos_lane = _b(data, 22, 3)
    elif address == ADDR_06 and _b(data, 0, 3) == 0:
      self.guidance = bool(_b(data, 26, 1))
      self.country = _b(data, 9, 8)

  def _expire(self) -> None:
    now = time.monotonic()
    for sid in [s for s, v in self.segments.items() if now - v["t"] > SEGMENT_TTL_S]:
      del self.segments[sid]

  # --- interpretation ------------------------------------------------------
  def _is_slowdown(self, kph: float | None, v_ego_kph: float | None) -> bool:
    if not kph:
      return False
    if v_ego_kph is None:
      return kph <= SLOWDOWN_CEILING_KPH
    return kph <= max(v_ego_kph - SLOWDOWN_MARGIN_KPH, 30)

  def path_ahead(self, v_ego_kph: float | None = None) -> dict:
    """Follows the most probable path from where we are, noting what branches off.

    Distances are measured from the car. PSD_Pos_Segmentlaenge counts down to
    the end of the segment we are in, at 2 m resolution, so the first hop is
    however much of it is left rather than the whole thing. That removes the
    up-to-one-segment error the earlier version carried.
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
                   "lanes": cur["lanes"], "segment": cur["id"],
                   "remaining_m": self.pos_remaining_m}
    # Distances are measured from the car, not from the start of the segment it
    # happens to be in: the first hop is however much of this segment is left.
    remaining = self.pos_remaining_m
    if remaining is None or not (0 <= remaining <= cur["length_m"] + 4):
      remaining = cur["length_m"]        # stale or implausible: fall back
    out["remaining_trusted"] = remaining is self.pos_remaining_m
    dist = 0.0
    first = True
    seen = set()
    for _ in range(MAX_PATH):
      if cur["id"] in seen:
        break
      seen.add(cur["id"])
      r_end = radius_from_psd(cur["curv_end"])
      out["segments"].append({"id": cur["id"], "at_m": round(dist),
                              "ends_at_m": round(dist + (remaining if first else cur["length_m"])),
                              "length_m": cur["length_m"], "lanes": cur["lanes"],
                              "category": ROAD_CATEGORY.get(cur["category"], "?"),
                              "radius_m": r_end,
                              "radius_start_m": radius_from_psd(cur["curv_start"]),
                              # +1 left, -1 right. Verified against the yaw rate
                              # driven through 70 measurable bends: the bit set
                              # meant left in 36 of 37 cases and clear meant
                              # right in 29 of 33.
                              "bend_dir": 1 if cur["curv_end_vz"] else -1,
                              "bend_dir_start": 1 if cur["curv_start_vz"] else -1,
                              "curve_kph": speed_for_radius(r_end),
                              # How sharply this segment leaves the one before it.
                              # For a segment ahead of us that is the turn we are
                              # expected to make, at the distance the segment
                              # starts. The branch we take is followed as the main
                              # path, so it never appears in "branches" -- which is
                              # why the turn has to be read off the path itself.
                              "turn_angle": cur["branch_angle"],
                              "turn_side": "left" if cur["branch_dir_bit"] else "right"})
      nxt = by_prev.get(cur["id"], [])
      if not nxt:
        break
      # The path we are expected to take; everything else leaving this point is
      # a side road, which is exactly what we want to know about.
      main = next((s for s in nxt if s["probable"]), None) or \
             next((s for s in nxt if s["straightest"]), None) or \
             next((s for s in nxt if not s["ramp"]), None) or nxt[0]
      # If the only way on is a ramp that is neither the probable nor the
      # straightest path, the mainline segment simply has not arrived yet.
      # Following the ramp would invent a route we are not taking.
      if main["ramp"] and not (main["probable"] or main["straightest"]):
        for s2 in nxt:
          out["branches"].append({
            "at_m": round(dist + (remaining if first else cur["length_m"])),
            "side": "right" if s2["branch_dir_bit"] else "left",
            "dir_bit": s2["branch_dir_bit"], "angle": s2["branch_angle"],
            "ramp": bool(s2["ramp"]), "lanes": s2["lanes"],
            "category": ROAD_CATEGORY.get(s2["category"], "?"), "probable": s2["probable"],
            "radius_m": radius_from_psd(s2["curv_start"]),
            "curve_kph": speed_for_radius(radius_from_psd(s2["curv_start"])),
          })
        break
      hop = remaining if first else cur["length_m"]
      for s in nxt:
        if s["id"] == main["id"]:
          continue
        out["branches"].append({
          "at_m": round(dist + hop),
          # PSD_Abzweigerichtung. Read as 1 = left, on two lines of evidence
          # rather than a direct measurement: the neighbouring curvature sign
          # bit measured out as 1 = left, and with the opposite reading roughly
          # 80 % of motorway ramps came out on the left, where Danish ramps are
          # overwhelmingly on the right. Both flip the same way. What would
          # settle it directly is watching the bit on a branch we actually take
          # and comparing it with the yaw; until then the raw bit travels
          # alongside so the reading can be checked against a known exit.
          "side": "left" if s["branch_dir_bit"] else "right",
          "dir_bit": s["branch_dir_bit"],
          "angle": s["branch_angle"],
          "ramp": bool(s["ramp"]),
          "lanes": s["lanes"],
          "category": ROAD_CATEGORY.get(s["category"], "?"),
          # Whether the car expects us to take this one. Only meaningful with
          # the car's own route guidance running -- on a phone-navigated trip
          # PSD_Sys_Zielfuehrung reads false and nothing here is a prediction.
          "probable": s["probable"],
          "radius_m": radius_from_psd(s["curv_start"]),
          "curve_kph": speed_for_radius(radius_from_psd(s["curv_start"])),
        })
      dist += remaining if first else cur["length_m"]
      first = False
      cur = main

    out["branches"].sort(key=lambda b: b["at_m"])

    # The single thing worth putting in front of a driver: the next reason to
    # slow down, whether that is a bend on our own road or a turn off it.
    cand = []
    for seg in out["segments"]:
      if self._is_slowdown(seg["curve_kph"], v_ego_kph):
        cand.append({"kind": "bend", "at_m": seg["ends_at_m"],
                     "kph": seg["curve_kph"], "radius_m": seg["radius_m"],
                     "confirmed": True, "side": None, "angle": None})
    for br in out["branches"]:
      if self._is_slowdown(br["curve_kph"], v_ego_kph):
        cand.append({"kind": "ramp" if br["ramp"] else "turn", "at_m": br["at_m"],
                     "kph": br["curve_kph"], "radius_m": br["radius_m"],
                     "confirmed": bool(br["probable"] and self.guidance),
                     "side": br["side"], "angle": br["angle"]})
    # Nearest first, but a bend we will definitely drive through outranks a
    # turn we may not take.
    cand.sort(key=lambda c: (c["at_m"], not c["confirmed"]))
    out["next_slowdown"] = cand[0] if cand else None
    return out

  def snapshot(self, v_ego_kph: float | None = None) -> dict:
    p = self.path_ahead(v_ego_kph)
    return {
      "guidance": self.guidance,
      "country": self.country,
      "pos_lane_raw": self.pos_lane,      # reads 0 always; kept for visibility
      "segments_known": len(self.segments),
      "age_s": round(time.monotonic() - self.last_seen, 1) if self.last_seen else None,
      **p,
    }
