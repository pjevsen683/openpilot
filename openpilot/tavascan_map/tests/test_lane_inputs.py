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


# Frames as the model gives them: the four lines nearest us, our two in the
# middle. Moving across the road by `x` metres (+ = right), the lines sit at
# -1.7 - x + 3.4 n; when we cross one, the model relabels (index jumps a lane).
def frame(x):
  lines = sorted(-1.7 - x + 3.4 * n for n in range(-3, 4))
  left = max(i for i, y in enumerate(lines) if y <= 0)
  return [round(y, 2) for y in lines[left - 1:left + 3]]


def sweep(x_end, n=25):
  return [frame(x_end * i / n) for i in range(n + 1)]


RIGHT_CHANGE = sweep(3.4)
LEFT_CHANGE = sweep(-3.4)


def test_frames_relabel_like_the_recording():
  # recorded right change, 2026-10-21: [-5.9, -2.9, 0.2, 3.4] -> [-3.8, -0.8, 2.4, 4.4]
  a, b = frame(1.5), frame(1.9)
  assert b[1] - a[1] > 2 and b[2] - a[2] > 2


def run_watch(w, frames, lc_state="off", lc_dir="none", left=False, right=False, t0=0.0):
  evs = []
  for i, ys in enumerate(frames):
    ev = w.update(lc_state, lc_dir, left, right, ys, t0 + 0.1 * i, v_ego=20.0, lane_probs=[0.5, 0.9, 0.9, 0.5])
    if ev:
      evs.append(ev)
  return evs


def test_driver_lane_change_right_from_recorded_relabel():
  assert run_watch(LaneChangeWatch(), RIGHT_CHANGE, right=True) == [(1, "driver lane change right")]


def test_driver_lane_change_left():
  assert run_watch(LaneChangeWatch(), LEFT_CHANGE, left=True) == [(-1, "driver lane change left")]


def test_indicator_alone_is_not_a_change():
  assert run_watch(LaneChangeWatch(), [[-5.1, -1.7, 1.7, 5.1]] * 30, left=True) == []


def test_relabel_without_indicator_is_not_a_change():
  assert run_watch(LaneChangeWatch(), RIGHT_CHANGE) == []


def test_aborted_openpilot_change_is_not_a_change():
  # Review focus 4: this car goes laneChangeStarting -> off for completed and aborted
  # changes alike; without the lines moving a lane, it is an abort
  w = LaneChangeWatch()
  frames = [[-5.1, -1.7, 1.7, 5.1], [-4.8, -1.4, 2.0, 5.4], [-5.0, -1.6, 1.8, 5.2]]
  evs = run_watch(w, frames, lc_state="laneChangeStarting", lc_dir="left", left=True)
  evs += run_watch(w, [[-5.1, -1.7, 1.7, 5.1]], lc_state="off", t0=1.0)
  assert evs == []


def test_completed_openpilot_change_is_reported_once():
  w = LaneChangeWatch()
  evs = run_watch(w, LEFT_CHANGE, lc_state="laneChangeStarting", lc_dir="left", left=True)
  evs += run_watch(w, [[-6.7, -3.3, 0.1, 3.5]] * 3, lc_state="off", t0=1.0)
  assert evs == [(-1, "openpilot lane change left")]


def test_no_events_when_slow():
  w = LaneChangeWatch()
  for i, ys in enumerate(RIGHT_CHANGE):
    assert w.update("off", "none", False, True, ys, 0.1 * i, v_ego=1.0) is None


def test_indicator_memory():
  w = LaneChangeWatch()
  w.update("off", "none", True, False, [-5, -1.7, 1.7, 5], 10.0)
  w.update("off", "none", False, False, [-5, -1.7, 1.7, 5], 11.0)
  assert abs(w.seconds_since(-1, 14.0) - 4.0) < 1e-9
  assert w.seconds_since(1, 14.0) > 1e6


def test_aborted_change_tail_still_not_a_change():
  w = LaneChangeWatch()
  run_watch(w, sweep(-0.6, 6), lc_state="laneChangeStarting", lc_dir="left", left=True)
  assert run_watch(w, [frame(-0.6)] * 15, lc_state="off", t0=1.0) == []


def test_change_finishing_after_the_state_ends_counts():
  # the state ends with only part of the lane crossed; the rest follows
  w = LaneChangeWatch()
  evs = run_watch(w, sweep(-1.0, 10), lc_state="laneChangeStarting", lc_dir="left", left=True)
  evs += run_watch(w, [frame(-1.0 - 0.1 * i) for i in range(1, 8)], lc_state="off", t0=1.1)
  assert evs == [(-1, "openpilot lane change left")]


def test_turning_at_a_junction_is_not_a_driver_change():
  w = LaneChangeWatch()
  for i, ys in enumerate(LEFT_CHANGE):
    assert w.update("off", "none", True, False, ys, 0.1 * i, v_ego=7.0, lane_probs=[0.5, 0.9, 0.9, 0.5]) is None
