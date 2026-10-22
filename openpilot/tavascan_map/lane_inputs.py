"""Turns what openpilot sees into the plain values the lane filter takes.

Frame: y metres to the RIGHT of the car, as in modelV2 and tavascan_web.
"""
BESIDE_D = (-6.0, 12.0)     # m, radar distance window that counts as beside us
BESIDE_Y = (2.0, 5.5)       # m, lateral window of the neighbouring lane
LANE_W = 3.4
# Measured on four drives, openpilot changes moved the lines 1.6-3.6 m (typically
# ~2 m) while laneChangeStarting lasted; the rest of the move comes after it ends.
SHIFT_DONE = 0.4            # a change counts once the lines moved this share of a lane
OP_TAIL_S = 1.0             # keep counting this long after laneChangeStarting ends
MIN_SPEED = 5.0             # m/s; below this the lines jump about
DRIVER_MIN_SPEED = 10.0     # m/s; turning at a junction with the indicator on is not a lane change
DRIVER_MIN_PROB = 0.3       # our own two lines must be this confident
DRIVER_CHANGE_BLINK_S = 3.0  # indicator on this recently counts for a crossing


def observation(lane_ys, lane_probs, edge_ys, edge_stds, radar) -> dict:
  """lane_ys: y at the car of the model's four lane lines; edge_ys/edge_stds: its
  two road edges; radar: points {"d", "y", "v_abs"}."""
  obs = {}
  if len(lane_probs) == 4:
    obs["outer_left_prob"] = float(lane_probs[0])
    obs["outer_right_prob"] = float(lane_probs[3])
    obs["model_conf"] = (float(lane_probs[1]) + float(lane_probs[2])) / 2
  if len(edge_ys) == 2 and len(edge_stds) == 2:
    obs["left_edge_m"] = abs(float(edge_ys[0]))
    obs["right_edge_m"] = abs(float(edge_ys[1]))
    obs["left_edge_prob"] = max(0.0, min(1.0, 1.0 - float(edge_stds[0])))
    obs["right_edge_prob"] = max(0.0, min(1.0, 1.0 - float(edge_stds[1])))
  left = right = False
  for p in radar:
    if p.get("v_abs", 0.0) <= 2.0 or not (BESIDE_D[0] <= p["d"] <= BESIDE_D[1]):
      continue   # oncoming, standing or not beside us
    if BESIDE_Y[0] <= -p["y"] <= BESIDE_Y[1]:
      left = True
    elif BESIDE_Y[0] <= p["y"] <= BESIDE_Y[1]:
      right = True
  obs["car_left"], obs["car_right"] = left, right
  return obs


def lines_moved(prev, ys) -> float:
  """How far the lane lines moved sideways between two frames, in metres (+ = right).

  Each line is matched to the nearest line of the previous frame, whatever
  index the model gave it: when we cross a line the model relabels them, and
  matching by index would read that as a jump of a whole lane.
  """
  d = []
  for y in ys:
    nearest = min(prev, key=lambda q: abs(y - q))
    if abs(y - nearest) < LANE_W / 2:
      d.append(y - nearest)
  if not d:
    return 0.0
  d.sort()
  return d[len(d) // 2]


class LaneChangeWatch:
  """Reports each completed lane change once, from openpilot or from the driver.

  The evidence is the lane lines moving sideways under us by most of a lane
  width (they move the opposite way to us). openpilot's own lane change goes
  laneChangeStarting -> off on this car whether it completed or was aborted
  (laneChangeFinishing never shows), so it only counts if the lines moved. A
  driver change needs the indicator on that side within the last few seconds.
  """

  def __init__(self):
    self._op = None              # direction of an openpilot change in progress
    self._op_moved = 0.0         # lines' sideways movement since it started
    self._op_end = None          # when laneChangeStarting ended, for the tail
    self._moved = 0.0            # since the last driver change or quiet spell
    self._last_blink = {-1: -1e9, 1: -1e9}
    self._prev_ys = None

  def seconds_since(self, side: int, t: float) -> float:
    """Seconds since the indicator on that side (-1 left, +1 right) was last on."""
    return t - self._last_blink[side]

  def update(self, lc_state, lc_dir, left_blinker, right_blinker, lane_ys, t, v_ego=DRIVER_MIN_SPEED,
             lane_probs=None):
    if left_blinker:
      self._last_blink[-1] = t
    if right_blinker:
      self._last_blink[1] = t
    ys = list(lane_ys) if len(lane_ys) == 4 else None
    step = 0.0
    if ys is not None and self._prev_ys is not None and v_ego >= MIN_SPEED:
      step = lines_moved(self._prev_ys, ys)
    if ys is not None:
      self._prev_ys = ys
    ev = None

    if lc_state == "laneChangeStarting" and lc_dir in ("left", "right"):
      d = -1 if lc_dir == "left" else 1
      if self._op != d:
        self._op, self._op_moved = d, 0.0
      self._op_moved += step
      self._op_end = None
      self._moved = 0.0
      return None
    if self._op is not None:
      self._op_moved += step
      if self._op_end is None:
        self._op_end = t
      # lines move right (+) when we move left
      if -self._op * self._op_moved >= SHIFT_DONE * LANE_W:
        ev = (self._op, f"openpilot lane change {'left' if self._op < 0 else 'right'}")
      if ev or t - self._op_end >= OP_TAIL_S:
        self._op = None
        self._moved = 0.0
      return ev

    blink_recent = min(self.seconds_since(-1, t), self.seconds_since(1, t)) <= DRIVER_CHANGE_BLINK_S
    confident = lane_probs is None or (len(lane_probs) == 4 and min(lane_probs[1], lane_probs[2]) >= DRIVER_MIN_PROB)
    if not blink_recent or v_ego < DRIVER_MIN_SPEED or not confident:
      self._moved = 0.0          # drift without indicating is lane keeping, not a change
      return None
    self._moved += step
    side = -1 if self._moved > 0 else 1
    if abs(self._moved) >= SHIFT_DONE * LANE_W and self.seconds_since(side, t) <= DRIVER_CHANGE_BLINK_S:
      ev = (side, f"driver lane change {'left' if side < 0 else 'right'}")
      # counted at part of a lane; the next change needs a whole lane more
      self._moved += side * LANE_W
    return ev
