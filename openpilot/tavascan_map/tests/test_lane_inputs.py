from openpilot.tavascan_map.lane_inputs import observation, LaneChangeWatch


def test_observation_edges_and_neighbours():
  obs = observation([-5.2, -1.8, 1.6, 5.0], [0.8, 0.95, 0.95, 0.1], [-5.6, 2.1], [0.3, 0.3],
                    [{"d": 5.0, "y": -3.4, "v_abs": 25.0}])
  assert abs(obs["left_edge_m"] - 5.6) < 1e-9 and abs(obs["right_edge_m"] - 2.1) < 1e-9
  assert obs["outer_left_prob"] == 0.8 and obs["car_left"] and not obs["car_right"]
  assert abs(obs["model_conf"] - 0.95) < 1e-9


def test_oncoming_or_far_cars_do_not_count():
  obs = observation([-5, -1.7, 1.7, 5], [0.5] * 4, [-6, 6], [0.3, 0.3],
                    [{"d": 5.0, "y": -3.4, "v_abs": -20.0}, {"d": 60.0, "y": 3.4, "v_abs": 25.0}])
  assert not obs["car_left"] and not obs["car_right"]


def test_openpilot_lane_change_reported_once_on_completion():
  w = LaneChangeWatch()
  assert w.update("laneChangeStarting", "left", True, False, [-5, -1.7, 1.7, 5], 0.0) is None
  assert w.update("laneChangeStarting", "left", False, False, [-5, -1.7, 1.7, 5], 3.0) is None
  assert w.update("off", "none", False, False, [-5, -1.7, 1.7, 5], 4.0) == (-1, "openpilot lane change left")
  assert w.update("off", "none", False, False, [-5, -1.7, 1.7, 5], 4.1) is None


def test_driver_lane_change_needs_indicator_and_crossing():
  w = LaneChangeWatch()
  # indicator right, our right line comes in towards 0 and the lines relabel
  w.update("off", "none", False, True, [-5.1, -1.7, 1.7, 5.1], 0.0)
  w.update("off", "none", False, True, [-4.0, -0.6, 0.4, 3.9], 1.0)
  ev = w.update("off", "none", False, True, [-5.0, -2.9, 0.6, 4.0], 1.1)
  assert ev == (1, "driver lane change right")


def test_indicator_alone_is_not_a_change():
  w = LaneChangeWatch()
  for i in range(30):
    assert w.update("off", "none", True, False, [-5.1, -1.7, 1.7, 5.1], i * 0.1) is None


def test_driver_lane_change_left():
  w = LaneChangeWatch()
  w.update("off", "none", True, False, [-5.1, -1.7, 1.7, 5.1], 0.0)
  w.update("off", "none", True, False, [-3.9, -0.4, 0.6, 4.0], 1.0)
  assert w.update("off", "none", True, False, [-4.0, -0.6, 2.9, 6.0], 1.1) == (-1, "driver lane change left")


def test_indicator_memory():
  w = LaneChangeWatch()
  w.update("off", "none", True, False, [-5, -1.7, 1.7, 5], 10.0)
  w.update("off", "none", False, False, [-5, -1.7, 1.7, 5], 11.0)
  assert abs(w.seconds_since(-1, 14.0) - 4.0) < 1e-9
  assert w.seconds_since(1, 14.0) > 1e6
