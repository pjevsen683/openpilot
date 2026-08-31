#!/usr/bin/env python3
"""Reads whatever map data mapd actually makes available, without guessing.

The cereal message liveMapDataSP carries only six fields: speed limit, speed
limit ahead with its distance, and the road name. That is not enough to tell
whether a slip road is joining from the right.

But mapd writes considerably more than that straight into the params directory,
outside the cereal schema. Strings in the binary show it understands the full OSM
highway classification (motorway, motorway_link, trunk_link, primary_link,
tertiary_link, living_street, service, ...) and that it publishes params such as
MapTargetVelocities, MapCurvatures, MapTargetLatA, NextMapHazard and
MapAdvisoryLimit.

Rather than reverse-engineering the binary further, this module reads the params
directory as it is and reports everything it finds. What map data we really have
then becomes something we can look at on the page instead of something I guess
at. Once we know, we can decide whether slip-road detection is possible at all.

Params are read as files rather than through Params(), because most mapd keys are
not registered in params_keys.h and Params() refuses unknown keys.

Read-only. Nothing here writes a param.
"""
import json
import math
import os

MEM_PARAMS = "/dev/shm/params/d"
DISK_PARAMS = "/data/params/d"

# Prefixes worth showing. Deliberately broad -- the point is discovery.
PREFIXES = ("Map", "NextMap", "Road", "OSM", "LastGPSPosition")

# Values longer than this are summarised instead of shown in full, so a long
# MapTargetVelocities list does not swamp the page.
MAX_INLINE = 400


def _read(path: str) -> str | None:
  try:
    with open(path, "rb") as f:
      return f.read().decode("utf-8", "replace")
  except OSError:
    return None


def _summarise(key: str, raw: str):
  """Returns a JSON-safe view of one param value."""
  if len(raw) <= MAX_INLINE:
    try:
      return json.loads(raw)
    except (ValueError, TypeError):
      return raw
  # Long values are almost always JSON lists of points.
  try:
    parsed = json.loads(raw)
    if isinstance(parsed, list):
      return {"_list": True, "count": len(parsed), "first": parsed[:2], "last": parsed[-1:]}
    return {"_truncated": True, "bytes": len(raw)}
  except (ValueError, TypeError):
    return {"_truncated": True, "bytes": len(raw), "head": raw[:120]}


def read_params() -> dict:
  """Every map-related param currently set, from shm first then disk."""
  found: dict = {}
  for base in (MEM_PARAMS, DISK_PARAMS):
    try:
      keys = os.listdir(base)
    except OSError:
      continue
    for key in keys:
      if not key.startswith(PREFIXES):
        continue
      if key in found:
        continue
      raw = _read(os.path.join(base, key))
      if raw is None or raw == "":
        continue
      found[key] = _summarise(key, raw)
  return found


def read_live(sm) -> dict:
  """The structured half: liveMapDataSP, if the message is being published."""
  try:
    m = sm["liveMapDataSP"]
  except (KeyError, AttributeError):
    return {"valid": False}
  return {
    "valid": bool(sm.updated.get("liveMapDataSP", False)) or bool(sm.valid.get("liveMapDataSP", False)),
    "speed_limit_kph": round(m.speedLimit * 3.6, 1) if m.speedLimitValid else None,
    "ahead_kph": round(m.speedLimitAhead * 3.6, 1) if m.speedLimitAheadValid else None,
    "ahead_distance_m": round(m.speedLimitAheadDistance, 1) if m.speedLimitAheadValid else None,
    "road_name": m.roadName or None,
  }


def maps_installed() -> dict:
  """Whether an offline region has been downloaded at all."""
  root = "/data/media/0/osm/offline"
  try:
    n = sum(len(files) for _, _, files in os.walk(root))
  except OSError:
    n = 0
  return {"path": root, "present": n > 0, "files": n}


# --- map geometry -----------------------------------------------------------
# mapd publishes MapTargetVelocities as a list of {latitude, longitude, velocity}
# points along the road ahead. Projected into the car frame it becomes an actual
# drawable road, which is the only map data we have with real geometry.

EARTH_R = 6371000.0


def road_ahead(max_points: int = 60) -> dict:
  """The road ahead as a polyline in the car frame: x forward, y right, metres."""
  out = {"available": False, "points": [], "speeds": []}
  raw = _read(os.path.join(MEM_PARAMS, "MapTargetVelocities")) or \
        _read(os.path.join(DISK_PARAMS, "MapTargetVelocities"))
  pos = _read(os.path.join(MEM_PARAMS, "LastGPSPosition")) or \
        _read(os.path.join(DISK_PARAMS, "LastGPSPosition"))
  if not raw or not pos:
    return out

  try:
    pts = json.loads(raw)
    here = json.loads(pos)
    lat0 = float(here["latitude"])
    lon0 = float(here["longitude"])
    bearing = math.radians(float(here.get("bearing", 0.0)))
  except (ValueError, TypeError, KeyError):
    return out
  if not isinstance(pts, list) or not pts:
    return out

  coslat = math.cos(math.radians(lat0))
  cb, sb = math.cos(bearing), math.sin(bearing)
  for p in pts[:max_points]:
    try:
      dn = math.radians(float(p["latitude"]) - lat0) * EARTH_R          # north, m
      de = math.radians(float(p["longitude"]) - lon0) * EARTH_R * coslat  # east, m
    except (ValueError, TypeError, KeyError):
      continue
    # Rotate the north/east offset into the car frame. Bearing is degrees
    # clockwise from north, so ahead = north*cos + east*sin.
    ahead = dn * cb + de * sb
    right = -dn * sb + de * cb
    out["points"].append([round(ahead, 1), round(right, 1)])
    v = p.get("velocity")
    out["speeds"].append(round(float(v) * 3.6, 0) if v is not None else None)

  out["available"] = len(out["points"]) >= 2
  return out
