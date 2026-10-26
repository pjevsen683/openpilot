"""Feeds the lane filter once per tick: map layout, lane changes, observations."""
from openpilot.tavascan_map.lane_inputs import LaneChangeWatch, observation
from openpilot.tavascan_map.lanes import LaneFilter, turn_only

MODEL_LANE_PROB = 0.3
MODEL_L_HOLD_S = 2.0   # the model's lane count must hold this long before it is used
MOTORWAY = {"motorway", "trunk"}
TURN_DEG = 45.0       # a change of road with this much change of heading is a turn
TURN_WINDOW_S = 6.0   # ... compared with our heading this long ago
# Back on the way we left within this long: the match strayed (GPS drift onto a
# ramp beside us) and the lane we had there still holds.
DETOUR_S = 15.0
LANE_TURN_CONFIDENCE = 0.6   # this sure of a turn-only lane, the map's path ahead takes its turn


def _side_hint(old_road, new_road, old_L, new_L) -> int:
  """Where lanes gained at a change of road most likely are.

  Danish on-ramps join from the right, so joining another motorway or trunk
  road with more lanes puts the new ones to our left (-1). Otherwise unknown.
  """
  if new_L > old_L and old_road and new_road and new_road[0] in MOTORWAY and old_road[1:] != new_road[1:]:
    return -1
  return 0


def lane_turn(match: dict | None, lane: dict | None) -> str | None:
  """"left"/"right" when the lane we are most likely in only turns that way."""
  if not match or not lane or lane.get("confidence", 0) < LANE_TURN_CONFIDENCE:
    return None
  L = match.get("lanes") or 0
  if lane.get("of") != L:
    return None
  k = turn_only(match.get("turn_lanes"), L)[lane["best"]]
  return "left" if k < 0 else "right" if k > 0 else None


class LaneRunner:
  def __init__(self):
    self.f = LaneFilter()
    self.watch = LaneChangeWatch()
    self._layout = None          # (way, lanes, turn_lanes) last applied
    self._road = None            # (class, ref, name) of the last matched way
    self._headings: list = []    # (t, heading) over the last TURN_WINDOW_S
    self._model_L = None         # the model's lane count, once it has held
    self._model_cand = (None, 0.0)
    self._left = None            # (layout, p, t, heading) of the way we last left
    self.source = "none"         # where L came from: "map", "model" or "none"

  def _model_lanes(self, model, t) -> int | None:
    """Lanes in our direction as far as the model can see (ours plus confident
    neighbours), once that count has held for MODEL_L_HOLD_S."""
    probs = (model or {}).get("lane_probs") or []
    if len(probs) == 4:
      n = 1 + int(probs[0] > MODEL_LANE_PROB) + int(probs[3] > MODEL_LANE_PROB)
      if self._model_L is None:
        self._model_L = n
      elif n == self._model_L:
        self._model_cand = (None, t)
      elif n != self._model_cand[0]:
        self._model_cand = (n, t)
      elif t - self._model_cand[1] >= MODEL_L_HOLD_S:
        self._model_L, self._model_cand = n, (None, t)
    return self._model_L

  def _is_detour_back(self, layout, t, heading) -> bool:
    if self._left is None or self._left[0] != layout or t - self._left[2] > DETOUR_S:
      return False
    # the same way the other way round (a U-turn) has its lanes mirrored
    before = self._left[3]
    return heading is None or before is None or abs((heading - before + 180.0) % 360.0 - 180.0) <= TURN_DEG

  def step(self, t, v_ego, dt, match, model, car, radar) -> dict:
    # 1. layout: from the map when matched; the model's count where the map has
    # none, or before any match; after a match is lost, what we had
    model_L = self._model_lanes(model, t)
    if match:
      layout = (match.get("way"), match.get("lanes") or model_L or self.f.L, match.get("turn_lanes"))
      self.source = "map" if match.get("lanes") else ("model" if model_L else "none")
    elif self._layout is None or self._layout[0] is None:
      layout = (None, model_L or self.f.L, None)
      self.source = "model" if model_L else "none"
    else:
      layout = self._layout
    if self._layout is None:
      self.f.reset(layout[1])
      self.f.turns = layout[2]
    elif layout != self._layout and self._is_detour_back(layout, t, (match or {}).get("heading")):
      self.f.L, self.f.p, self.f.turns = layout[1], list(self._left[1]), layout[2]
      self.f.why = "back on the road we left"
      self._left = None
    elif layout != self._layout:
      self._left = (self._layout, list(self.f.p), t, self._headings[-1][1] if self._headings else None)
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
    # 2. diffusion by distance
    self.f.diffuse(max(0.0, v_ego) * dt)
    # 3. lane changes and observations, only on a fresh model frame (None = stale)
    if model is not None:
      ev = self.watch.update(model.get("lc_state", "off"), model.get("lc_dir", "none"),
                             car.get("left_blinker", False), car.get("right_blinker", False),
                             model.get("lane_ys") or [], t, v_ego, model.get("lane_probs"))
      if ev:
        self.f.lane_change(ev[0], why=ev[1])
        self._left = None        # the lane we had on the way we left no longer holds
      # Mid-change the model already shows the lane we are moving into; seen
      # then and counted again when the change completes, it moved us two lanes.
      if not self.watch.changing:
        self.f.observe(observation(model.get("lane_ys") or [], model.get("lane_probs") or [],
                                   model.get("edge_ys") or [], model.get("edge_stds") or [], radar))
    out = self.f.output()
    out["source"] = self.source
    return out
