"""Keeps the model in charge of a stop it has started, until it says go.

Dynamic Experimental Control hands longitudinal control to the model
("blended") when the model's planned trajectory comes up short, and back to
ACC when it no longer does. Near a red light those two can come apart: on
2026-10-13 at 07:30 the nearside lights left the camera's view about 6 m before
the stop, the trajectory lengthened, DEC went back to ACC, and ACC asked for
+1.4 m/s2 towards the set speed. The model's own desired acceleration at that
moment was still -1.9 m/s2. The driver braked.

So once the model is braking in blended mode, the latch keeps it as a
candidate until the model itself stops braking. The planner takes the lowest
of its candidates, so holding the model's acceleration can only add braking,
never take it away.

It lets go when
  * the model no longer wants to brake (green light: its acceleration rises),
  * we have stopped (DEC already hands standstill to the model, which is
    what pulls away on green),
  * the driver presses the accelerator, or longitudinal control is off,
  * MAX_HOLD_S has passed, as a backstop.
"""
from openpilot.common.realtime import DT_MDL

ARM_ACCEL = -0.3       # m/s2, model braking at least this hard while blended
HOLD_ACCEL = -0.1      # m/s2, still braking: keep holding
RELEASE_S = 0.5        # model above HOLD_ACCEL this long before letting go
MIN_SPEED = 0.3        # m/s, below this the stop is done
ARM_MIN_SPEED = 3.0    # m/s
MAX_HOLD_S = 20.0
RELEASE_FRAMES = round(RELEASE_S / DT_MDL)
MAX_HOLD_FRAMES = round(MAX_HOLD_S / DT_MDL)


class StopLatch:
  def __init__(self):
    self.active = False
    # counted in frames: summing DT_MDL in floats lands just short of the limits
    self._held = 0
    self._release = 0

  def _let_go(self) -> None:
    self.active = False
    self._held = 0
    self._release = 0

  def update(self, blended: bool, desired_accel: float, v_ego: float,
             gas_pressed: bool, long_active: bool) -> None:
    if gas_pressed or not long_active:
      self._let_go()
      return

    if not self.active:
      if blended and desired_accel < ARM_ACCEL and v_ego > ARM_MIN_SPEED:
        self.active = True
      return

    self._held += 1
    self._release = self._release + 1 if desired_accel > HOLD_ACCEL else 0
    if v_ego < MIN_SPEED or self._release >= RELEASE_FRAMES or self._held >= MAX_HOLD_FRAMES:
      self._let_go()
