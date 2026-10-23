from openpilot.tavascan_map.lane_runner import LaneRunner
from openpilot.tavascan_map.tests.test_lane_inputs import sweep

MODEL = {"lc_state": "off", "lc_dir": "none", "lane_ys": [-5.1, -1.7, 1.7, 5.1],
         "lane_probs": [0.5, 0.9, 0.9, 0.5], "edge_ys": [-8.0, 8.0], "edge_stds": [0.5, 0.5]}
CAR = {"left_blinker": False, "right_blinker": False}


def test_no_map_uses_model_lane_count():
  r = LaneRunner()
  out = r.step(0.0, 20.0, 0.1, None, MODEL, CAR, [])
  assert out["of"] == 3


def test_gps_dropout_keeps_last_layout():
  # Review focus 3
  r = LaneRunner()
  r.step(0.0, 20.0, 0.1, {"way": 1, "lanes": 3, "turn_lanes": None}, MODEL, CAR, [])
  out = r.step(0.1, 20.0, 0.1, None, MODEL, CAR, [])
  assert out["of"] == 3


def test_layout_change_mid_lane_change_applies_both():
  # Review focus 2: a left-turn lane opens while an openpilot change left completes
  r = LaneRunner()
  r.step(0.0, 15.0, 0.1, {"way": 1, "lanes": 2, "turn_lanes": None}, MODEL, CAR, [])
  r.f.p = [0.0, 1.0]
  # a real change left, a whole lane across: lines drift right and relabel on the way
  frames = sweep(-3.4)
  for i, ys in enumerate(frames):
    starting = dict(MODEL, lc_state="laneChangeStarting", lc_dir="left", lane_ys=ys)
    r.step(0.1 * (i + 1), 15.0, 0.1, {"way": 1, "lanes": 2, "turn_lanes": None}, starting, CAR, [])
  out = r.step(0.1 * (len(frames) + 1), 15.0, 0.1, {"way": 2, "lanes": 3, "turn_lanes": "left|through|through"},
               dict(MODEL, lane_ys=frames[-1]), CAR, [])
  # old lane 1 -> new lane 2 (turn lane added left), then the change moves us to lane 1
  assert out["best"] == 1


def test_joining_a_motorway_puts_the_new_lanes_on_our_left():
  # Danish on-ramps join from the right: lanes gained onto another motorway are to our left
  r = LaneRunner()
  r.step(0.0, 25.0, 0.1, {"way": 1, "lanes": 2, "turn_lanes": None, "class": "motorway", "ref": "501", "name": "A"}, MODEL, CAR, [])
  r.f.p = [0.0, 1.0]
  out = r.step(0.1, 25.0, 0.1, {"way": 2, "lanes": 4, "turn_lanes": None, "class": "motorway", "ref": "E 45", "name": "B"}, MODEL, CAR, [])
  assert out["best"] == 3


def test_more_lanes_on_the_same_road_stay_unknown_side():
  r = LaneRunner()
  r.step(0.0, 15.0, 0.1, {"way": 1, "lanes": 2, "turn_lanes": None, "class": "secondary", "ref": "O3", "name": "A"}, MODEL, CAR, [])
  r.f.p = [0.0, 1.0]
  r.step(0.1, 15.0, 0.1, {"way": 2, "lanes": 3, "turn_lanes": None, "class": "secondary", "ref": "O3", "name": "A"}, MODEL, CAR, [])
  assert 0.3 < r.f.p[1] < 0.7


def test_turning_onto_another_road_forgets_the_old_lane():
  # after a turn at a junction, the lane we came from says nothing about the new one
  r = LaneRunner()
  r.step(0.0, 10.0, 0.1, {"way": 1, "lanes": 4, "turn_lanes": None, "class": "secondary", "ref": "O3", "name": "A", "heading": 10.0}, MODEL, CAR, [])
  r.f.p = [1.0, 0.0, 0.0, 0.0]
  r.step(0.1, 10.0, 0.1, {"way": 2, "lanes": 2, "turn_lanes": None, "class": "secondary", "ref": "O3", "name": "B", "heading": 280.0}, MODEL, CAR, [])
  assert 0.3 < r.f.p[0] < 0.7


def test_keeping_straight_onto_another_road_keeps_the_lane():
  r = LaneRunner()
  r.step(0.0, 10.0, 0.1, {"way": 1, "lanes": 2, "turn_lanes": None, "class": "secondary", "ref": None, "name": "A", "heading": 10.0}, MODEL, CAR, [])
  r.f.p = [1.0, 0.0]
  r.step(0.1, 10.0, 0.1, {"way": 2, "lanes": 2, "turn_lanes": None, "class": "secondary", "ref": None, "name": "B", "heading": 15.0}, MODEL, CAR, [])
  assert r.f.p[0] > 0.9


def test_model_lane_count_does_not_flap():
  # Review focus 1: no lane count in the map; the model's outer line hovers around the threshold
  r = LaneRunner()
  m = {"way": 1, "lanes": None, "turn_lanes": None}
  r.step(0.0, 15.0, 0.1, m, MODEL, CAR, [])
  r.f.p = [0.9, 0.05, 0.05]
  for i in range(20):
    probs = [0.25 if i % 2 else 0.35, 0.9, 0.9, 0.5]
    out = r.step(0.1 * (i + 1), 15.0, 0.1, m, dict(MODEL, lane_probs=probs), CAR, [])
  assert out["of"] == 3


def test_lane_count_from_model_before_any_map_match():
  # Review focus 3: the first ticks come before modelV2 and GPS
  r = LaneRunner()
  r.step(0.0, 15.0, 0.1, None, None, CAR, [])
  for i in range(30):
    out = r.step(0.1 * (i + 1), 15.0, 0.1, None, MODEL, CAR, [])
  assert out["of"] == 3


def test_stale_model_is_not_observed_again():
  r = LaneRunner()
  r.step(0.0, 0.0, 0.1, {"way": 1, "lanes": 3, "turn_lanes": None}, MODEL, CAR, [])
  p0 = list(r.f.p)
  for i in range(100):
    r.step(0.1 * (i + 1), 0.0, 0.1, {"way": 1, "lanes": 3, "turn_lanes": None}, None, CAR, [])
  assert r.f.p == p0


def test_brief_match_to_a_parallel_ramp_keeps_the_lane():
  # GPS drifts onto an exit ramp beside the motorway for a few seconds and back
  mw = {"way": 1, "lanes": 3, "turn_lanes": None, "class": "motorway", "ref": "E 45", "name": "A", "heading": 10.0}
  ramp = {"way": 2, "lanes": 1, "turn_lanes": None, "class": "motorway_link", "ref": None, "name": None, "heading": 12.0}
  r = LaneRunner()
  r.step(0.0, 28.0, 0.1, mw, None, CAR, [])
  r.f.p = [0.0, 1.0, 0.0]
  t = 0.1
  for _ in range(90):
    r.step(t, 28.0, 0.1, ramp, None, CAR, []); t += 0.1
  out = r.step(t, 28.0, 0.1, mw, None, CAR, [])
  assert out["best"] == 1 and out["confidence"] > 0.8


def test_turning_back_onto_the_same_road_is_not_a_detour():
  # a U-turn: the same two-way way, but its lanes are now the other way round
  a = {"way": 1, "lanes": 2, "turn_lanes": None, "class": "secondary", "ref": None, "name": "A", "heading": 10.0}
  b = {"way": 2, "lanes": 1, "turn_lanes": None, "class": "tertiary", "ref": None, "name": "B", "heading": 100.0}
  r = LaneRunner()
  r.step(0.0, 8.0, 0.1, a, None, CAR, [])
  r.f.p = [1.0, 0.0]
  r.step(0.1, 8.0, 0.1, b, None, CAR, [])
  out = r.step(5.0, 8.0, 0.1, dict(a, heading=190.0), None, CAR, [])
  assert out["confidence"] < 0.8
