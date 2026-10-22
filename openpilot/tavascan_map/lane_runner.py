"""Feeds the lane filter once per tick: map layout, lane changes, observations."""
from openpilot.tavascan_map.lane_inputs import LaneChangeWatch, observation
from openpilot.tavascan_map.lanes import LaneFilter

MODEL_LANE_PROB = 0.3


class LaneRunner:
  def __init__(self):
    self.f = LaneFilter()
    self.watch = LaneChangeWatch()
    self._layout = None          # (way, lanes, turn_lanes) last applied

  def _model_lanes(self, model) -> int:
    """Lanes in our direction as far as the model can see: ours plus confident neighbours."""
    probs = model.get("lane_probs") or []
    if len(probs) != 4:
      return self.f.L
    return 1 + int(probs[0] > MODEL_LANE_PROB) + int(probs[3] > MODEL_LANE_PROB)

  def step(self, t, v_ego, dt, match, model, car, radar) -> dict:
    # 1. layout: from the map when matched, otherwise keep what we had
    if match:
      layout = (match.get("way"), match.get("lanes") or self._model_lanes(model), match.get("turn_lanes"))
    elif self._layout is None:
      layout = (None, self._model_lanes(model), None)
    else:
      layout = self._layout
    if self._layout is None:
      self.f.reset(layout[1])
      self.f.turns = layout[2]
    elif layout != self._layout:
      self.f.relayout(layout[1], layout[2])
      if layout[2] != self._layout[2]:
        self.f.turn_lanes_without_indicating(self.watch.seconds_since(-1, t), self.watch.seconds_since(1, t))
    self._layout = layout
    # 2. lane changes
    ev = self.watch.update(model.get("lc_state", "off"), model.get("lc_dir", "none"),
                           car.get("left_blinker", False), car.get("right_blinker", False),
                           model.get("lane_ys") or [], t)
    if ev:
      self.f.lane_change(ev[0], why=ev[1])
    # 3. diffusion by distance, then observations
    self.f.diffuse(max(0.0, v_ego) * dt)
    self.f.observe(observation(model.get("lane_ys") or [], model.get("lane_probs") or [],
                               model.get("edge_ys") or [], model.get("edge_stds") or [], radar))
    return self.f.output()
