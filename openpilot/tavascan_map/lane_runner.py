"""Feeds the lane filter once per tick: map layout, lane changes, observations."""
from openpilot.tavascan_map.lane_inputs import LaneChangeWatch, observation
from openpilot.tavascan_map.lanes import LaneFilter

MODEL_LANE_PROB = 0.3
MOTORWAY = {"motorway", "trunk"}
TURN_DEG = 45.0       # a change of road with this much change of heading is a turn
TURN_WINDOW_S = 6.0   # ... compared with our heading this long ago


def _side_hint(old_road, new_road, old_L, new_L) -> int:
  """Where lanes gained at a change of road most likely are.

  Danish on-ramps join from the right, so joining another motorway or trunk
  road with more lanes puts the new ones to our left (-1). Otherwise unknown.
  """
  if new_L > old_L and old_road and new_road and new_road[0] in MOTORWAY and old_road[1:] != new_road[1:]:
    return -1
  return 0


class LaneRunner:
  def __init__(self):
    self.f = LaneFilter()
    self.watch = LaneChangeWatch()
    self._layout = None          # (way, lanes, turn_lanes) last applied
    self._road = None            # (class, ref, name) of the last matched way
    self._headings: list = []    # (t, heading) over the last TURN_WINDOW_S

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
      road = (match.get("class"), match.get("ref"), match.get("name")) if match else self._road
      heading = match.get("heading") if match else None
      before = self._headings[0][1] if self._headings else None
      turned = (road != self._road and heading is not None and before is not None and
                abs((heading - before + 180.0) % 360.0 - 180.0) > TURN_DEG)
      if turned:
        # Lanes before and after a turn at a junction are not the same lanes;
        # start over and let what we see decide.
        self.f.reset(layout[1])
        self.f.turns = layout[2]
        self.f.why = "turned onto another road"
      else:
        self.f.relayout(layout[1], layout[2], _side_hint(self._road, road, self._layout[1], layout[1]))
      if layout[2] != self._layout[2]:
        self.f.turn_lanes_without_indicating(self.watch.seconds_since(-1, t), self.watch.seconds_since(1, t))
    self._layout = layout
    if match:
      self._road = (match.get("class"), match.get("ref"), match.get("name"))
      if match.get("heading") is not None:
        self._headings.append((t, match["heading"]))
    self._headings = [h for h in self._headings if t - h[0] <= TURN_WINDOW_S]
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
