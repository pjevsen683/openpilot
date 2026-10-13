#!/usr/bin/env python3
"""Keeps us from passing a slower car in the left lane on its right.

Undertaking is illegal in Denmark. The car's own Travel Assist holds back behind
a slower car in the left lane; openpilot does not, because its planner only
sees vehicles in our own lane.

The decision itself is tavascan_web.shadow.undertake, unchanged. That rule ran
in shadow mode on the live page from 2026-10-07 to 2026-10-13, and this module
wires the same function into the planner rather than a copy of it. What was
watched is what drives.

DESIGN
  * Off unless the UndertakeGuard parameter is set. Read once a second, so it
    can be switched off mid-drive.
  * Only ever CAPS the desired speed. It never raises it, and it never brakes
    on its own: the cap goes through the ordinary cruise acceleration limits.
  * Only above 70 km/h, and the cap never goes below 70 km/h, so it cannot
    touch stopping, pulling away or stop-and-go traffic.
  * The rule must hold for ENGAGE_S before the cap applies, so a radar track
    that flickers for a frame cannot make the car dip. Once the left lane
    clears, the last cap is kept for RELEASE_S before it is let go.
"""
from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.tavascan_web import geometry, shadow

ENGAGE_S = 1.0
RELEASE_S = 2.0
# radarTracks arrives at 20 Hz; older than this and we act as if it is empty.
RADAR_STALE_S = 0.5
PARAM_PERIOD_S = 1.0


class UndertakeGuard:
  def __init__(self, params=None):
    self._params = params or Params()
    self.enabled = False
    self._sm = None
    self._param_age = PARAM_PERIOD_S
    self._radar_age = RADAR_STALE_S
    self._points: list = []
    self._on_s = 0.0
    self._off_s = 0.0
    self.cap: float | None = None   # m/s, or None when not capping
    self.why = ""

  def _read_param(self, dt: float) -> None:
    self._param_age += dt
    if self._param_age < PARAM_PERIOD_S:
      return
    self._param_age = 0.0
    self.enabled = self._params.get_bool("UndertakeGuard")
    if self.enabled and self._sm is None:
      self._sm = messaging.SubMaster(["radarTracks"])

  def _reset(self) -> None:
    self._points = []
    self._on_s = self._off_s = 0.0
    self.cap = None
    self.why = ""

  def update(self, sm, v_ego: float, dt: float) -> None:
    """Called every planner tick with the planner's SubMaster."""
    self._read_param(dt)
    if not self.enabled:
      self._reset()
      return

    self._sm.update(0)
    if self._sm.updated["radarTracks"]:
      self._points = geometry.radar_points(self._sm["radarTracks"], v_ego)
      self._radar_age = 0.0
    else:
      self._radar_age += dt
      if self._radar_age > RADAR_STALE_S:
        self._points = []

    md = sm["modelV2"]
    lanes = shadow.lane_position(md.laneLineProbs, md.roadEdges)
    points = shadow.along_path(self._points, md.position.x, md.position.y)
    rule = shadow.undertake(points, v_ego, lanes["rightmost"])

    if rule["active"]:
      self._on_s += dt
      self._off_s = 0.0
      if self._on_s >= ENGAGE_S:
        self.cap = rule["cap"]
        self.why = rule["why"]
    else:
      self._on_s = 0.0
      if self.cap is not None:
        self._off_s += dt
        if self._off_s >= RELEASE_S:
          self.cap = None
          self.why = ""

  def apply(self, v_cruise: float) -> float:
    """Lowers the desired speed to the cap. Never raises it."""
    if self.cap is None:
      return v_cruise
    return min(v_cruise, self.cap)
