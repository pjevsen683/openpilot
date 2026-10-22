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


class LaneFilter:
  def __init__(self):
    self.L = 1
    self.p = [1.0]
    self.why = ""

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

  def output(self) -> dict:
    best = max(range(self.L), key=lambda i: self.p[i])
    return {"p": [round(x, 3) for x in self.p], "best": best, "of": self.L,
            "confidence": round(self.p[best], 3), "why": self.why}
