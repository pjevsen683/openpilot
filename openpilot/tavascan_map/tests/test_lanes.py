from openpilot.tavascan_map.lanes import LaneFilter


def test_reset_is_uniform():
  f = LaneFilter(); f.reset(3)
  assert f.p == [1 / 3] * 3
  assert f.output()["of"] == 3


def test_lane_change_left_moves_one_lane():
  f = LaneFilter(); f.reset(3); f.p = [0.0, 0.0, 1.0]
  f.lane_change(-1, why="openpilot lane change left")
  out = f.output()
  assert out["best"] == 1
  assert abs(f.p[1] - 0.9) < 1e-9 and abs(f.p[2] - 0.1) < 1e-9
  assert out["why"] == "openpilot lane change left"


def test_lane_change_at_edge_keeps_probability():
  f = LaneFilter(); f.reset(2); f.p = [1.0, 0.0]
  f.lane_change(-1)
  assert abs(sum(f.p) - 1) < 1e-9 and f.p[0] == 1.0


def test_diffusion_is_by_distance_not_time():
  # Review focus 5: stopped in a queue must not flatten p
  f = LaneFilter(); f.reset(3); f.p = [0.0, 1.0, 0.0]
  for _ in range(6000):
    f.diffuse(0.0)
  assert f.p == [0.0, 1.0, 0.0]
  f.diffuse(15.0)   # one second at 15 m/s
  assert abs(f.p[0] - 0.005) < 1e-6 and abs(f.p[2] - 0.005) < 1e-6


def test_aborted_change_stays_mostly_on_original_lane():
  # Review focus 4: an aborted change is never reported as a change at all
  f = LaneFilter(); f.reset(3); f.p = [0.0, 1.0, 0.0]
  assert f.output()["best"] == 1
