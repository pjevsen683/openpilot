from openpilot.sunnypilot.selfdrive.controls.lib.stop_offset import StopOffset, planned_stop, STOP_OFFSET_M

DT = 0.05


def plan(stop_at):
  # planned speed reaches zero at stop_at
  return [0.0, stop_at / 2, stop_at, stop_at + 1], [5.0, 2.0, 0.3, 0.0]


def test_planned_stop():
  assert abs(planned_stop(*plan(20.0)) - 20.0) < 1e-6
  assert planned_stop([0.0, 10.0, 20.0], [10.0, 9.0, 8.0]) is None


def test_stops_short_of_the_model():
  o = StopOffset()
  o.update(True, *plan(20.0), False, 0.0, 6.0, -0.9, DT)
  assert o.active
  # a long run of updates settles on a stop STOP_OFFSET_M before the model's
  for _ in range(100):
    o.update(True, *plan(20.0), False, 0.0, 0.0, -0.9, DT)
  assert abs((o._target - o._odo) - (20.0 - STOP_OFFSET_M)) < 0.5


def test_braking_eases_in():
  o = StopOffset()
  o.update(True, *plan(4.0), False, 0.0, 8.0, -0.5, DT)
  assert o.a_target >= -0.5 - 2.0 * DT - 1e-6


def test_inactive_without_latch_or_stop_plan():
  o = StopOffset()
  o.update(False, *plan(20.0), False, 0.0, 6.0, -0.9, DT)
  assert not o.active
  o.update(True, [0.0, 50.0], [10.0, 9.0], False, 0.0, 6.0, -0.9, DT)
  assert not o.active


def test_target_held_when_stop_leaves_the_plan():
  o = StopOffset()
  o.update(True, *plan(20.0), False, 0.0, 5.0, -0.9, DT)
  o.update(True, [0.0, 60.0], [5.0, 4.0], False, 0.0, 5.0, -0.9, DT)
  assert o.active


def test_lead_at_the_stop_blocks_until_released():
  o = StopOffset()
  o.update(True, *plan(20.0), True, 15.0, 6.0, -0.9, DT)
  assert not o.active
  o.update(True, *plan(20.0), False, 0.0, 6.0, -0.9, DT)
  assert not o.active
  o.update(False, *plan(20.0), False, 0.0, 6.0, -0.9, DT)
  o.update(True, *plan(20.0), False, 0.0, 6.0, -0.9, DT)
  assert o.active
