#!/usr/bin/env python3
"""Decodes the car's per-lane radar objects from Strukturen_01 (0x24F, bus 2).

openpilot sets VolkswagenFlags.DISABLE_RADAR on this car, so opendbc's own
radar_interface returns None and the message is never parsed. It also transmits
an empty Strukturen_01 of its own on bus 0 to silence stock AEB/FCW, so bus 2 is
the only place the real radar data appears.

Bit positions and scales are taken from _vw_meb_common.dbc. The radar reports two
objects in each of the same, left and right lanes, with longitudinal distance,
lateral distance and relative velocity. Lateral distance is positive to the LEFT,
which is the opposite sign convention to the model's laneLines.

This module only reads. It never transmits.
"""

RADAR_ADDR = 0x24F
RADAR_BUS = 2

DISTANCE_STATUS = (13, 2)
VALID_STATUS = 0

DIST_SCALE, DIST_OFF = 0.0625, -3.75
LAT_SCALE, LAT_OFF = 0.065, -33.28
VEL_SCALE, VEL_OFF = 0.25, -128.0

# name -> (object id bit, distance bit, lateral bit, velocity bit)
# Lengths are always 6 / 12 / 10 / 10.
OBJECTS = {
  "same_1":  (16, 64, 76, 86),
  "left_1":  (22, 96, 108, 118),
  "right_1": (28, 128, 140, 150),
  "same_2":  (34, 160, 172, 182),
  "left_2":  (40, 192, 204, 214),
  "right_2": (46, 224, 236, 246),
}

NO_OBJECT_ID = 0
# Plausibility limits. The radar pads unused slots with extreme values.
MIN_DIST, MAX_DIST = 0.5, 250.0


def _bits(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


def decode(data: bytes, v_ego: float) -> dict | None:
  """Returns {slot: {...}} for every occupied slot, or None if the frame is invalid.

  v_ego is used to turn the radar's relative velocity into an absolute speed,
  which is what makes the numbers readable on the page.
  """
  if len(data) < 32:
    return None
  if _bits(data, *DISTANCE_STATUS) != VALID_STATUS:
    return None

  out = {}
  for name, (id_bit, d_bit, y_bit, v_bit) in OBJECTS.items():
    obj_id = _bits(data, id_bit, 6)
    if obj_id == NO_OBJECT_ID:
      continue

    dist = _bits(data, d_bit, 12) * DIST_SCALE + DIST_OFF
    lat = _bits(data, y_bit, 10) * LAT_SCALE + LAT_OFF
    rel = _bits(data, v_bit, 10) * VEL_SCALE + VEL_OFF

    if not (MIN_DIST < dist < MAX_DIST):
      continue

    out[name] = {
      "id": obj_id,
      "d": round(dist, 1),      # m ahead
      "y": round(lat, 2),       # m, positive left
      "v_rel": round(rel, 1),   # m/s relative
      "v_abs": round(v_ego + rel, 1),
    }
  return out
