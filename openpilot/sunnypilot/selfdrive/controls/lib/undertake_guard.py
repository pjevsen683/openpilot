#!/usr/bin/env python3
"""Prevents undertaking by capping cruise speed behind the left lane.

Undertaking is illegal in Denmark. The car's own Travel Assist slows down when a
slower vehicle sits ahead in the left lane; openpilot does not, because the
planner only ever sees vehicles in our OWN lane
(modelV2.leadsV3 -> radarState.leadOne/leadTwo).

The car's radar, by contrast, reports objects grouped by lane in Strukturen_01
(0x24F, 25 Hz on bus 2): two objects in each of the same/left/right lanes, with
longitudinal distance, lateral distance and relative velocity. Verified against a
motorway recording: left-lane objects appeared at a lateral distance of about
+4.0 m with relative velocities matching the surrounding traffic.

openpilot sets VolkswagenFlags.DISABLE_RADAR on this car (networkLocation is
fwdCamera), so radar_interface returns None and the message is never parsed. We
therefore parse it ourselves. We transmit nothing -- this is listen-only.

DESIGN
  * OFF by default. Requires the UndertakeGuard parameter.
  * Only ever applies a CAP to the desired speed. It never brakes on its own and
    can never make the car accelerate. The existing MPC does the slowing down.
  * Only active above MIN_SPEED. In stop-and-go traffic passing on the right is
    legal, and the feature would only get in the way.
  * Does not touch stopping or pulling away. We have an unresolved fault in the
    EPB transition at low speed, and nothing here should be able to affect it.

SCOPE
  This module parses raw CAN inside plannerd, which is unusual for that process.
  The alternative was extending the cereal schema, opendbc and the car layer --
  three repositories and a schema change. This is easier to review and easier to
  remove again. When the feature is off, the socket is never even opened.
"""
import os

from openpilot.cereal import messaging
from openpilot.common.params import Params
from opendbc.car.common.conversions import Conversions as CV

RADAR_ADDR = 0x24F
RADAR_BUS = 2

# Bit positions from _vw_meb_common.dbc, Strukturen_01
_DISTANCE_STATUS = (13, 2)
_VALID_STATUS = 0
# (start bit, length) for distance / lateral distance / relative velocity
_LEFT_LANE = ((96, 12), (108, 10), (118, 10))

_DIST_SCALE, _DIST_OFF = 0.0625, -3.75
_LAT_SCALE, _LAT_OFF = 0.065, -33.28
_VEL_SCALE, _VEL_OFF = 0.25, -128.0

# Activation limits
MIN_SPEED = float(os.getenv("UG_MIN_SPEED_KPH", "70")) * CV.KPH_TO_MS
MAX_RANGE = float(os.getenv("UG_MAX_RANGE", "90"))       # m ahead
MIN_RANGE = float(os.getenv("UG_MIN_RANGE", "3"))        # m, filters out junk
LAT_MIN, LAT_MAX = 2.0, 6.0                              # m, plausible adjacent lane
# How much faster than the left lane we accept travelling. A small slip keeps the
# cap from engaging on insignificant differences.
SLIP = float(os.getenv("UG_SLIP_KPH", "4")) * CV.KPH_TO_MS
# The cap is released gradually once the object disappears, so speed does not jump.
RELEASE_S = 2.0


def _bits(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


class UndertakeGuard:
  def __init__(self):
    self.enabled = Params().get_bool("UndertakeGuard")
    self._sock = messaging.sub_sock("can", timeout=0) if self.enabled else None
    self.cap: float | None = None          # m/s, or None when inactive
    self.lead_v: float | None = None       # left-lane object's speed, m/s
    self.lead_d: float | None = None       # its distance, m
    self._age = 0.0

  def _decode(self, data: bytes, v_ego: float) -> tuple[float, float] | None:
    """Returns (distance, absolute speed) for a left-lane object, else None."""
    if _bits(data, *_DISTANCE_STATUS) != _VALID_STATUS:
      return None
    (ds, dl), (ls, ll), (vs, vl) = _LEFT_LANE
    dist = _bits(data, ds, dl) * _DIST_SCALE + _DIST_OFF
    lat = _bits(data, ls, ll) * _LAT_SCALE + _LAT_OFF
    rel = _bits(data, vs, vl) * _VEL_SCALE + _VEL_OFF

    if not (MIN_RANGE < dist < MAX_RANGE):
      return None
    # Lateral distance is positive to the left in the radar's frame -- the opposite
    # of the model's laneLines. Checked against a recording: the adjacent lane sat
    # at about +4.0 m.
    if not (LAT_MIN < lat < LAT_MAX):
      return None
    return dist, v_ego + rel

  def update(self, v_ego: float, dt: float) -> None:
    """Called every planner tick. Updates self.cap."""
    if self._sock is None:
      return

    found = None
    for msg in messaging.drain_sock(self._sock):
      for c in msg.can:
        if c.address == RADAR_ADDR and c.src == RADAR_BUS:
          found = self._decode(bytes(c.dat), v_ego) or found

    if found is not None:
      self.lead_d, self.lead_v = found
      self._age = 0.0
    else:
      self._age += dt
      if self._age > RELEASE_S:
        self.lead_d = self.lead_v = None

    # The cap only applies when we would otherwise undertake a slower vehicle. If
    # the left lane is faster -- the normal case -- nothing happens.
    if (self.lead_v is not None and v_ego > MIN_SPEED
        and self.lead_v + SLIP < v_ego):
      self.cap = max(self.lead_v + SLIP, MIN_SPEED)
    else:
      self.cap = None

  def apply(self, v_cruise: float) -> float:
    """Lowers the desired speed to the cap. Can never raise it."""
    if self.cap is None:
      return v_cruise
    return min(v_cruise, self.cap)
