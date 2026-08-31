#!/usr/bin/env python3
"""Shadow-mode evaluation of driving rules that are NOT wired into control.

Every rule here answers one question: "what would this have done, right now?"
Nothing in this module is connected to the planner, the controller or CAN. It
computes a would-be speed cap and a human-readable reason, and that is all.

The point is to build confidence before anything is allowed to steer or brake.
A rule earns its way into the control path by being watched here first.

SIGN CONVENTION
  Everything in this module uses y POSITIVE TO THE RIGHT, matching modelV2's
  laneLines. The radar uses the opposite sign, so it is flipped once at
  ingestion in main.py and never again. Measured on a motorway recording:
  radar left-lane objects sit at y = +3.84 m and right-lane objects at
  y = -4.61 m, while the model's leftmost lane line sits at y = -4.37 m.
"""
from opendbc.car.common.conversions import Conversions as CV

# --- Undertaking guard ------------------------------------------------------
# Undertaking is illegal in Denmark. If a slower vehicle sits in the left lane,
# hold back rather than pass it on the right.
UT_MIN_SPEED = 70 * CV.KPH_TO_MS
UT_MAX_RANGE = 90.0
UT_MIN_RANGE = 3.0
UT_LAT = (-6.0, -2.0)                 # m, left lane (negative is left)
UT_SLIP = 4 * CV.KPH_TO_MS            # tolerated speed excess

# --- Merge yield ------------------------------------------------------------
# A vehicle on the right that is slower than us and close is most likely joining
# from a slip road. This rule is a first sketch and is deliberately conservative.
MG_MIN_SPEED = 60 * CV.KPH_TO_MS
MG_MAX_RANGE = 60.0
MG_LAT = (2.0, 6.0)                   # m, right lane
MG_SLIP = 2 * CV.KPH_TO_MS


def _closest(points: list, lat: tuple, max_range: float):
  """Closest point whose lateral offset falls inside the given lane window."""
  best = None
  for p in points:
    if not (lat[0] < p["y"] < lat[1]):
      continue
    if not (UT_MIN_RANGE < p["d"] < max_range):
      continue
    if best is None or p["d"] < best["d"]:
      best = p
  return best


def undertake(points: list, v_ego: float, rightmost_lane: bool | None) -> dict:
  """Would we be undertaking a slower vehicle in the left lane?"""
  o = _closest(points, UT_LAT, UT_MAX_RANGE)

  if v_ego < UT_MIN_SPEED:
    return {"active": False, "cap": None, "why": "below 70 km/h", "target": o}
  if o is None:
    return {"active": False, "cap": None, "why": "no vehicle in left lane", "target": None}
  if o["v_abs"] + UT_SLIP >= v_ego:
    return {"active": False, "cap": None, "why": "left lane is not slower", "target": o}

  cap = max(o["v_abs"] + UT_SLIP, UT_MIN_SPEED)
  why = f"left lane {o['v_abs'] * CV.MS_TO_KPH:.0f} km/h at {o['d']:.0f} m"
  # Undertaking only applies when we are actually in a lane to its right. We
  # cannot always tell, so this is reported rather than used to suppress.
  if rightmost_lane is False:
    why += " (but we do not appear to be rightmost)"
  return {"active": True, "cap": cap, "why": why, "target": o}


def merge_yield(points: list, v_ego: float) -> dict:
  """Would we be closing on a slower vehicle to our right, likely merging?"""
  o = _closest(points, MG_LAT, MG_MAX_RANGE)

  if v_ego < MG_MIN_SPEED:
    return {"active": False, "cap": None, "why": "below 60 km/h", "target": o}
  if o is None:
    return {"active": False, "cap": None, "why": "no vehicle on the right", "target": None}
  if o["v_abs"] + MG_SLIP >= v_ego:
    return {"active": False, "cap": None, "why": "right side is not slower", "target": o}

  cap = max(o["v_abs"] + MG_SLIP, MG_MIN_SPEED)
  return {"active": True, "cap": cap,
          "why": f"right {o['v_abs'] * CV.MS_TO_KPH:.0f} km/h at {o['d']:.0f} m", "target": o}


def lane_position(lane_line_probs, road_edges) -> dict:
  """Rough estimate of where we sit across the carriageway.

  laneLineProbs[1] and [2] are our own lane's markings; [0] and [3] are the outer
  markings of the neighbouring lanes. A low probability on [3] therefore means
  there is no lane to our right, not that our own edge line is missing.
  """
  out = {"rightmost": None, "leftmost": None, "right_edge_m": None, "left_edge_m": None}
  try:
    if len(lane_line_probs) >= 4:
      out["leftmost"] = bool(lane_line_probs[0] < 0.2)
      out["rightmost"] = bool(lane_line_probs[3] < 0.2)
    if road_edges is not None and len(road_edges) == 2:
      if len(road_edges[0].y) > 0:
        out["left_edge_m"] = round(abs(road_edges[0].y[0]), 2)
      if len(road_edges[1].y) > 0:
        out["right_edge_m"] = round(abs(road_edges[1].y[0]), 2)
  except (TypeError, IndexError, AttributeError):
    pass
  return out
