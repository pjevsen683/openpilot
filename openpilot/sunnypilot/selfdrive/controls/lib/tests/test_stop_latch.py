from openpilot.sunnypilot.selfdrive.controls.lib.stop_latch import StopLatch, RELEASE_FRAMES, MAX_HOLD_FRAMES


def run(latch, n, **kw):
  args = dict(blended=False, desired_accel=-1.5, v_ego=10.0, gas_pressed=False, long_active=True)
  args.update(kw)
  for _ in range(n):
    latch.update(**args)
  return latch.active


def test_holds_braking_after_dec_leaves_blended():
  # 2026-10-13 07:30: blended and braking, then DEC flips to ACC while the model still brakes
  latch = StopLatch()
  assert run(latch, 1, blended=True)
  assert run(latch, 20, blended=False)


def test_not_armed_without_blended_braking():
  assert not run(StopLatch(), 20, blended=False)
  assert not run(StopLatch(), 20, blended=True, desired_accel=-0.1)
  assert not run(StopLatch(), 20, blended=True, v_ego=1.0)


def test_releases_when_model_says_go():
  latch = StopLatch()
  run(latch, 1, blended=True)
  frames = RELEASE_FRAMES
  assert run(latch, frames - 1, desired_accel=0.5)
  assert not run(latch, 1, desired_accel=0.5)


def test_brief_go_does_not_release():
  latch = StopLatch()
  run(latch, 1, blended=True)
  run(latch, RELEASE_FRAMES - 2, desired_accel=0.5)
  assert run(latch, 1)
  assert run(latch, RELEASE_FRAMES - 2, desired_accel=0.5)


def test_releases_at_standstill_gas_and_disengage():
  for kw in (dict(v_ego=0.1), dict(gas_pressed=True), dict(long_active=False)):
    latch = StopLatch()
    run(latch, 1, blended=True)
    assert not run(latch, 1, **kw)


def test_backstop():
  latch = StopLatch()
  run(latch, 1, blended=True)
  assert not run(latch, MAX_HOLD_FRAMES)
