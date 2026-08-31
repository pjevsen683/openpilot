#!/usr/bin/env python3
"""Turns cereal messages into a plain bird's-eye scene the page can draw.

Everything leaving this module uses one frame: x metres AHEAD, y metres to the
RIGHT, origin at the car. That is modelV2's convention. The radar uses the
opposite lateral sign and is flipped here, once, so no drawing code ever has to
know that.

Measured on a motorway recording to establish it rather than assume it:
  radar   left-lane objects  y = +3.84 m   right-lane objects  y = -4.61 m
  model   leftmost lane line y = -4.37 m   rightmost           y = +2.32 m
"""

# Lane lines and edges are ~33 points out to 192 m. Every second point is plenty
# for a phone screen and keeps the JSON small.
STRIDE = 2
MAX_X = 160.0
# Below this probability a lane line is reported but marked as not confident, so
# the page can draw it faintly instead of pretending the lane is there.
LANE_PROB = 0.35


def _line(entry, prob=None) -> dict | None:
  try:
    xs, ys = entry.x, entry.y
  except AttributeError:
    return None
  pts = [[round(x, 1), round(y, 2)] for x, y in zip(xs[::STRIDE], ys[::STRIDE]) if x <= MAX_X]
  if len(pts) < 2:
    return None
  out = {"points": pts}
  if prob is not None:
    out["prob"] = round(float(prob), 2)
    out["confident"] = bool(prob >= LANE_PROB)
  return out


def scene(model) -> dict:
  """Lane lines and road edges as polylines in the car frame."""
  out: dict = {"lane_lines": [], "road_edges": []}
  try:
    probs = list(model.laneLineProbs)
    for i, ll in enumerate(model.laneLines):
      out["lane_lines"].append(_line(ll, probs[i] if i < len(probs) else None))
    for re in model.roadEdges:
      out["road_edges"].append(_line(re))
  except (AttributeError, TypeError, IndexError):
    pass
  return out


def radar_points(tracks, v_ego: float) -> list:
  """radarTracks points, flipped into the y-positive-right frame."""
  pts = []
  try:
    src = tracks.points
  except AttributeError:
    return pts
  for p in src:
    try:
      pts.append({
        "id": int(p.trackId),
        "d": round(float(p.dRel), 1),
        "y": round(-float(p.yRel), 2),   # radar is positive-left; flip it
        "v_rel": round(float(p.vRel), 1),
        "v_abs": round(v_ego + float(p.vRel), 1),
      })
    except (AttributeError, TypeError, ValueError):
      continue
  return pts
