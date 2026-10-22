from openpilot.tavascan_map.lanes import LaneFilter, lane_map, turn_only


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



def test_turn_only_parsing():
  assert turn_only("left|through|through;right", 3) == [-1, 0, 0]
  assert turn_only("none|none|right", 3) == [0, 0, 1]
  assert turn_only(None, 2) == [0, 0]
  assert turn_only("left|through", 3) == [0, 0, 0]   # count mismatch: unusable


def test_left_turn_lane_added_by_arrows_shifts_us_right_in_numbering():
  M = lane_map(2, "through|through;right", 3, "left|through|through;right")
  # old lane 0 -> new lane 1, old lane 1 -> new lane 2
  assert M[0][1] == 1.0 and M[1][2] == 1.0


def test_lane_added_unknown_side_splits():
  M = lane_map(2, None, 3, None, side_hint=0)
  assert abs(M[0][0] - 0.5) < 1e-9 and abs(M[0][1] - 0.5) < 1e-9


def test_lane_added_on_right_by_hint():
  M = lane_map(2, None, 3, None, side_hint=+1)
  assert M[0][0] == 1.0 and M[1][1] == 1.0


def test_lane_lost_merges_into_neighbour():
  f = LaneFilter(); f.reset(3); f.turns = None; f.p = [0.0, 0.0, 1.0]
  f.relayout(2, None, side_hint=+1)   # right lane ends
  assert f.L == 2 and abs(f.p[1] - 1.0) < 1e-9


def test_no_lane_count_never_indexes_out_of_range():
  # Review focus 1
  f = LaneFilter(); f.reset(1); f.turns = None
  f.relayout(1, "left|through", 0)
  assert f.p == [1.0]


def test_turn_lanes_without_indicating():
  f = LaneFilter(); f.reset(3); f.turns = "left|through|through"
  f.turn_lanes_without_indicating(seconds_since_left=999, seconds_since_right=999)
  assert f.p[0] < 0.06 and abs(sum(f.p) - 1) < 1e-9
  g = LaneFilter(); g.reset(3); g.turns = "left|through|through"
  g.turn_lanes_without_indicating(seconds_since_left=3, seconds_since_right=999)
  assert g.p[0] > 0.55
