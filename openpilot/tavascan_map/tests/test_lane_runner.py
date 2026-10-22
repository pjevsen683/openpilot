from openpilot.tavascan_map.lane_runner import LaneRunner

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
  starting = dict(MODEL, lc_state="laneChangeStarting", lc_dir="left")
  r.step(0.1, 15.0, 0.1, {"way": 1, "lanes": 2, "turn_lanes": None}, starting, CAR, [])
  out = r.step(0.2, 15.0, 0.1, {"way": 2, "lanes": 3, "turn_lanes": "left|through|through"}, MODEL, CAR, [])
  # old lane 1 -> new lane 2 (turn lane added left), then the change moves us to lane 1
  assert out["best"] == 1
