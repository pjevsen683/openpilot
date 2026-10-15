"""Stops STOP_OFFSET_M short of where the model plans to stop.

At red lights the model has chosen stop points just past the line: about
1 m on 2026-10-15 at 16:07, and the driver finds it is usually a little
long. This adds a planner candidate that brakes for a point STOP_OFFSET_M
before the model's own.

The model's stop point is where its planned speed reaches zero. It only
reaches zero within the horizon in roughly the last 40 m, and there it is
accurate: at 16:07 it was within 1 m of where the car stopped from 15 m
out, and 1-2 m long from 15-40 m. The end of the planned path, used
before zero appears, is not: it was off by up to 7 m, so it is not used.

Because the target is the model's own stop point, braking a little harder
does not move it: the model keeps aiming at the same place, and this
candidate keeps us STOP_OFFSET_M short of it.

It only runs while the stop latch holds (the model is braking for a stop,
see stop_latch.py) and lets go with it, so a light turning green releases
it. Once seen, the target is kept, in odometry, if later plans lose the
stop: near the line the lights leave the camera's view. Inside FREEZE_DIST
it is no longer updated, since a stopped or creeping plan is noisy.

The planner takes the lowest candidate, so this can only add braking. With
a lead in front of the stop point the stop is for the lead, which the MPC
handles.
"""
STOP_OFFSET_M = 1.5
ZERO_SPEED = 0.3        # m/s, planned speed at or below this counts as stopped
FREEZE_DIST = 3.0       # m
FILTER_S = 0.5          # s
LEAD_MARGIN = 3.0       # m
MIN_DIST = 0.3          # m
STOP_DIST = 0.5         # m, closer than this and slow: ask to stop
STOP_SPEED = 1.0        # m/s
A_MIN = -3.0            # m/s2
JERK_MAX = 2.0          # m/s3, how fast this may brake harder than the model


def planned_stop(plan_x, plan_v) -> float | None:
  """Distance along the plan at which the planned speed first reaches zero."""
  for k in range(1, min(len(plan_x), len(plan_v))):
    if plan_v[k] <= ZERO_SPEED:
      dv = plan_v[k - 1] - plan_v[k]
      f = (plan_v[k - 1] - ZERO_SPEED) / dv if dv > 1e-3 else 1.0
      return plan_x[k - 1] + min(max(f, 0.0), 1.0) * (plan_x[k] - plan_x[k - 1])
  return None


class StopOffset:
  def __init__(self):
    self._odo = 0.0
    self._target: float | None = None   # odometry of the point to stop at
    self._lead_blocked = False          # a lead sat at the stop point this stop
    self.active = False
    self.a_target = 0.0
    self.should_stop = False

  def _reset(self) -> None:
    self._target = None
    self.active = False
    self.a_target = 0.0
    self.should_stop = False

  def update(self, latched: bool, plan_x, plan_v, lead_present: bool, lead_d: float,
             v_ego: float, model_accel: float, dt: float) -> None:
    self._odo += v_ego * dt
    if not latched:
      self._reset()
      self._lead_blocked = False
      return
    if self._lead_blocked:
      return

    frozen = self._target is not None and self._target - self._odo < FREEZE_DIST
    stop = None if frozen else planned_stop(plan_x, plan_v)
    if stop is not None:
      target = self._odo + stop - STOP_OFFSET_M
      if self._target is None:
        self._target = target
      else:
        self._target += (target - self._target) * min(1.0, dt / FILTER_S)
    if self._target is None:
      return

    d = self._target - self._odo
    # Queueing behind a car: leave it to the MPC for the rest of this stop,
    # rather than switching on and off as the lead track comes and goes.
    if lead_present and lead_d < d + STOP_OFFSET_M + LEAD_MARGIN:
      self._reset()
      self._lead_blocked = True
      return

    # When the stop shows up late the extra braking it asks for can be a step;
    # ease into it from the model's own accel instead.
    a_req = max(-v_ego ** 2 / (2.0 * max(d, MIN_DIST)), A_MIN)
    prev = self.a_target if self.active else model_accel
    self.active = True
    self.a_target = max(a_req, min(prev, model_accel) - JERK_MAX * dt)
    self.should_stop = d < STOP_DIST and v_ego < STOP_SPEED
