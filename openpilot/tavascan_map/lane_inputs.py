"""Turns what openpilot sees into the plain values the lane filter takes.

Frame: y metres to the RIGHT of the car, as in modelV2 and tavascan_web.
"""
BESIDE_D = (-6.0, 12.0)     # m, radar distance window that counts as beside us
BESIDE_Y = (2.0, 5.5)       # m, lateral window of the neighbouring lane
RELABEL_JUMP = 2.0          # m, our nearest line jumping this far = the model relabelled
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


class LaneChangeWatch:
  """Reports each completed lane change once, from openpilot or from the driver."""

  def __init__(self):
    self._op = None              # direction of an openpilot change in progress
    self._last_blink = {-1: -1e9, 1: -1e9}
    self._prev_ys = None

  def seconds_since(self, side: int, t: float) -> float:
    """Seconds since the indicator on that side (-1 left, +1 right) was last on."""
    return t - self._last_blink[side]

  def update(self, lc_state, lc_dir, left_blinker, right_blinker, lane_ys, t):
    if left_blinker:
      self._last_blink[-1] = t
    if right_blinker:
      self._last_blink[1] = t
    ev = None
    if lc_state == "laneChangeStarting" and lc_dir in ("left", "right"):
      self._op = -1 if lc_dir == "left" else 1
    elif lc_state == "off" and self._op is not None:
      ev = (self._op, f"openpilot lane change {'left' if self._op < 0 else 'right'}")
      self._op = None
    elif self._op is None and self._prev_ys is not None and len(lane_ys) == 4:
      # A driver change shows as the model relabelling its lines once we are
      # over: moving right, our left line (index 1) jumps left by about a lane
      # (measured -0.6 -> -2.9 m); moving left, our right line (index 2) jumps
      # right by about a lane.
      for side, idx in ((-1, 2), (1, 1)):
        jump = lane_ys[idx] - self._prev_ys[idx]
        recent = t - self._last_blink[side] <= DRIVER_CHANGE_BLINK_S
        if recent and ((side < 0 and jump > RELABEL_JUMP) or (side > 0 and jump < -RELABEL_JUMP)):
          ev = (side, f"driver lane change {'left' if side < 0 else 'right'}")
    self._prev_ys = list(lane_ys) if len(lane_ys) == 4 else self._prev_ys
    return ev
