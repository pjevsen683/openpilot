"""Which lane of our carriageway we are in, as a probability per lane.

A discrete Bayes (histogram) filter: the exact form of Monte Carlo
Localization for a state with a handful of values. Lanes are counted from
the left of our carriageway, in our direction of travel. Predict steps move
probability (lane changes, the map's lanes changing under us, slow
diffusion); update steps weigh it by what the model and radar see.

Pure logic, fed plain values, so recorded drives replay through it exactly.
"""
LANE_CHANGE_WEIGHT = 0.9
# 0.5 % per second to each neighbour at 15 m/s, applied per metre travelled
# so that standing in a queue does not wash out what we know.
DIFFUSE_PER_M = 0.005 / 15.0
TURN_ONLY_FACTOR = 0.1
TURN_SIGNALLED_FACTOR = 3.0
INDICATOR_MEMORY_S = 10.0
LANE_W = 3.4
EDGE_NEAR_M = 2.5
MODEL_CONF_FLOOR = 0.3   # below this the model's terms count for little
# Per-tick likelihood strengths; at 10 Hz a steady observation takes about a
# second to dominate a uniform prior.
EDGE_WEIGHT = 0.25
OUTER_WEIGHT = 0.15
RADAR_WEIGHT = 0.3


def _soft(strength: float, yes: bool) -> float:
  return 1.0 + strength if yes else 1.0 / (1.0 + strength)


def turn_only(turns: str | None, L: int) -> list[int]:
  """Per lane, from the left: -1 a left-turn-only lane, +1 right-turn only, 0 otherwise."""
  if not turns:
    return [0] * L
  parts = turns.split("|")
  if len(parts) != L:
    return [0] * L
  out = []
  for lane in parts:
    k = {x.strip() for x in lane.split(";") if x.strip()}
    if k and k <= {"left", "slight_left", "sharp_left", "reverse"}:
      out.append(-1)
    elif k and k <= {"right", "slight_right", "sharp_right"}:
      out.append(1)
    else:
      out.append(0)
  return out


def _added_side(old_L, old_turns, new_L, new_turns) -> int:
  """-1 / +1 when the arrows show where the extra lanes came in, else 0."""
  n = turn_only(new_turns, new_L)
  extra = new_L - old_L
  if extra <= 0 or not new_turns:
    return 0
  if all(x == -1 for x in n[:extra]) and not all(x == -1 for x in n[-extra:]):
    return -1
  if all(x == 1 for x in n[-extra:]) and not all(x == 1 for x in n[:extra]):
    return 1
  return 0


def lane_map(old_L: int, old_turns: str | None, new_L: int, new_turns: str | None,
             side_hint: int = 0) -> list[list[float]]:
  """M[i][j]: share of old lane i that is new lane j."""
  M = [[0.0] * new_L for _ in range(old_L)]
  if new_L == old_L:
    for i in range(old_L):
      M[i][i] = 1.0
    return M
  side = _added_side(old_L, old_turns, new_L, new_turns) or side_hint
  d = new_L - old_L
  for i in range(old_L):
    if d > 0:     # lanes added
      if side < 0:
        M[i][i + d] = 1.0
      elif side > 0:
        M[i][i] = 1.0
      else:
        M[i][i] += 0.5
        M[i][i + d] += 0.5
    else:         # lanes lost: those beyond the new count merge into the nearest
      if side < 0:
        M[i][max(0, i + d)] = 1.0
      else:
        M[i][min(new_L - 1, i)] = 1.0
  return M


class LaneFilter:
  def __init__(self):
    self.L = 1
    self.p = [1.0]
    self.why = ""
    self.turns: str | None = None

  def reset(self, L: int) -> None:
    self.L = max(1, int(L))
    self.p = [1.0 / self.L] * self.L

  def _norm(self) -> None:
    s = sum(self.p)
    self.p = [x / s for x in self.p] if s > 0 else [1.0 / self.L] * self.L

  def lane_change(self, direction: int, weight: float = LANE_CHANGE_WEIGHT, why: str = "") -> None:
    q = [0.0] * self.L
    for i, x in enumerate(self.p):
      j = i + direction
      if 0 <= j < self.L:
        q[j] += x * weight
        q[i] += x * (1.0 - weight)
      else:
        q[i] += x       # no lane there: we cannot have moved into it
    self.p = q
    self._norm()
    if why:
      self.why = why

  def diffuse(self, distance_m: float) -> None:
    a = DIFFUSE_PER_M * max(0.0, distance_m)
    if a <= 0 or self.L < 2:
      return
    q = [0.0] * self.L
    for i, x in enumerate(self.p):
      out = 0.0
      for j in (i - 1, i + 1):
        if 0 <= j < self.L:
          q[j] += x * a
          out += a
      q[i] += x * (1.0 - out)
    self.p = q
    self._norm()

  def relayout(self, new_L: int, new_turns: str | None, side_hint: int = 0) -> None:
    """The map's lanes changed under us: carry each lane's probability over."""
    new_L = max(1, int(new_L))
    if new_L != self.L:
      M = lane_map(self.L, self.turns, new_L, new_turns, side_hint)
      q = [0.0] * new_L
      for i, x in enumerate(self.p):
        for j, share in enumerate(M[i]):
          q[j] += x * share
      self.L, self.p = new_L, q
      self._norm()
      self.why = f"map: {new_L} lanes"
    self.turns = new_turns

  def turn_lanes_without_indicating(self, seconds_since_left: float, seconds_since_right: float) -> None:
    """On entering a stretch with turn-only lanes: unlikely unless we indicated that way."""
    kinds = turn_only(self.turns, self.L)
    if not any(kinds):
      return
    for i, k in enumerate(kinds):
      if k == 0:
        continue
      since = seconds_since_left if k < 0 else seconds_since_right
      self.p[i] *= TURN_SIGNALLED_FACTOR if since <= INDICATOR_MEMORY_S else TURN_ONLY_FACTOR
    self._norm()

  def observe(self, obs: dict) -> None:
    """Weigh the lanes by what the model and radar see this tick.

    obs keys, all optional: left_edge_m/right_edge_m (distance to the road
    edge), left_edge_prob/right_edge_prob, outer_left_prob/outer_right_prob
    (the model's outer lane lines), car_left/car_right (a vehicle beside us
    going our way), model_conf (our own lane lines' probability).
    """
    L = self.L
    if L < 2 or not obs:
      return
    conf = obs.get("model_conf")
    k_model = 1.0 if conf is None else min(1.0, max(0.0, (conf - 0.1) / (MODEL_CONF_FLOOR + 0.3)))
    lik = [1.0] * L
    for side, edge_key, prob_key in ((-1, "left_edge_m", "left_edge_prob"), (1, "right_edge_m", "right_edge_prob")):
      d, pr = obs.get(edge_key), obs.get(prob_key)
      if d is None or pr is None:
        continue
      n = max(0.0, (d - LANE_W / 2) / LANE_W)    # lanes between us and that edge
      s = EDGE_WEIGHT * pr * k_model
      for i in range(L):
        lanes_to_edge = i if side < 0 else L - 1 - i
        if d <= EDGE_NEAR_M:
          lik[i] *= _soft(s, lanes_to_edge == 0)
        else:
          lik[i] *= 1.0 + s * max(0.0, 1.0 - abs(lanes_to_edge - n))
    for side, key in ((-1, "outer_left_prob"), (1, "outer_right_prob")):
      pr = obs.get(key)
      if pr is None:
        continue
      s = OUTER_WEIGHT * k_model * max(0.0, pr - 0.3) / 0.7
      outermost = 0 if side < 0 else L - 1
      for i in range(L):
        lik[i] *= _soft(s, i != outermost)
    for side, key in ((-1, "car_left"), (1, "car_right")):
      if obs.get(key):
        outermost = 0 if side < 0 else L - 1
        lik[outermost] *= 1.0 / (1.0 + RADAR_WEIGHT * 3)
    self.p = [x * l for x, l in zip(self.p, lik)]
    self._norm()

  def output(self) -> dict:
    best = max(range(self.L), key=lambda i: self.p[i])
    return {"p": [round(x, 3) for x in self.p], "best": best, "of": self.L,
            "confidence": round(self.p[best], 3), "why": self.why}
