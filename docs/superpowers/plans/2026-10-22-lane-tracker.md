# Lane Tracker and Map-First 3D View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate, as a probability per lane with history, which lane of our carriageway we are in, publish it from `tavascan_map`, and draw the 3D page around the map with our car in the most probable lane.

**Architecture:** A pure histogram filter (`tavascan_map/lanes.py`) with predict steps (lane changes, map topology, turn lanes without indicating, diffusion) and update steps (road edges, neighbouring lane lines, radar neighbours). A small feature extractor (`tavascan_map/lane_inputs.py`) turns cereal messages into plain values. `tavascan_map/main.py` runs both at 10 Hz next to the 2 Hz map match and writes the result into its JSON. The 3D page lays the map road out around `lane.best`.

**Tech Stack:** Python 3.12 on the device (3.10 locally), SQLite map file, pytest (installed into the local venv `/home/peter/privat/openpilot/undertake/venv`), three.js page.

**Spec:** `docs/superpowers/specs/2026-10-22-lane-tracker-design.md`

## Global Constraints

- Lanes are counted from the LEFT of our carriageway, in our direction of travel, `0..L-1`.
- Predict and update run at 10 Hz; map matching stays at 2 Hz.
- openpilot lane change moves probability one lane with weight 0.9; a driver lane change (indicator + model line crossing) also 0.9; indicator alone does not move `p`.
- Diffusion 0.5 % per second to each neighbour.
- Turn-only lanes without an indicator towards their side within the last 10 s: probability × 0.1; with one: × 3.
- Road-edge evidence only from an edge within ~2.5 m of its lane side; model terms weakened when lane line probabilities are below ~0.3.
- Output: `lane = {p: [...], best: i, of: L, confidence: max(p), why: "<last change>"}` in `/dev/shm/tavascan_map.json` and the trace.
- Read-only towards the car: no cereal publishing, no CAN, no params written.
- All repo text in English. No street names or places from the driver's commute in code, comments or commits.
- Branch `lane-tracker`, merged into `shadow-view`; deploy only via git, only while the car is parked at home (device on `10.0.1.x`, no `plannerd` running).

## Review Focus

1. Map has no lane count for the way (most residential roads): the filter must still run with `L` from the model and never index out of range.
2. `L` changes while a lane change is in progress: the pending change must be applied to the new lane layout, not lost or doubled.
3. GPS drop-out or no fix (first 2–4 minutes in the morning): the filter keeps predicting from model/car inputs; output says `of` from the last known `L` and confidence decays, no crash.
4. Lane change aborted half-way (indicator off, model returns to the original lane): `p` must stay mostly on the original lane.
5. Stationary in a queue for minutes: diffusion must not flatten `p` to uniform while stopped (diffusion scales with distance travelled, not time).

Each is pinned by a test in the task that owns the code (Tasks 1, 2, 4, 5).

---

## File Structure

- Create `openpilot/tavascan_map/lanes.py` — `LaneFilter`: state, predict, update, output. Pure; no openpilot imports.
- Create `openpilot/tavascan_map/lane_inputs.py` — `LaneInputs` dataclass and `extract(model, car, radar_points)`, plus `LaneChangeWatch` that turns per-tick model/car values into lane-change events.
- Create `openpilot/tavascan_map/tests/test_lanes.py`, `openpilot/tavascan_map/tests/test_lane_inputs.py`.
- Modify `openpilot/tavascan_map/matcher.py` — expose the directed lane layout (`lanes`, `turn_lanes`) of the current stretch and of the stretch ahead in `match()` output (already present as `lanes`/`turn_lanes`; add `way` change detection helper `layout_key`).
- Modify `openpilot/tavascan_map/main.py` — 10 Hz loop, subscribe `modelV2`, `carState`, `radarTracks`, run the filter, add `lane` to output.
- Modify `openpilot/tavascan_web/page3d.html` — lay road out by `lane.best`, cars by lane, model layer toggle (off), HUD lane text.
- Create (outside repo) `/home/peter/privat/openpilot/web3d_dev/lane_replay.py` — replay the filter over recorded drives.

---

### Task 1: LaneFilter core — state, lane change, diffusion

**Files:**
- Create: `openpilot/tavascan_map/lanes.py`
- Test: `openpilot/tavascan_map/tests/test_lanes.py`

**Interfaces:**
- Produces:
  - `class LaneFilter` with `__init__(self)`, attribute `p: list[float]`, `L: int`
  - `reset(self, L: int) -> None` (uniform)
  - `lane_change(self, direction: int, weight: float = 0.9, why: str = "") -> None` (`direction` −1 left, +1 right)
  - `diffuse(self, distance_m: float) -> None` (0.5 % per second at 15 m/s ⇒ rate per metre `DIFFUSE_PER_M = 0.005 / 15.0`)
  - `output(self) -> dict` → `{"p": [..], "best": int, "of": int, "confidence": float, "why": str}`

- [ ] **Step 1: Install pytest into the local venv**

Run: `/home/peter/privat/openpilot/undertake/venv/bin/pip install -q pytest`
Expected: no error.

- [ ] **Step 2: Write the failing tests**

```python
# openpilot/tavascan_map/tests/test_lanes.py
from openpilot.tavascan_map.lanes import LaneFilter


def test_reset_is_uniform():
  f = LaneFilter(); f.reset(3)
  assert f.p == [1 / 3] * 3
  assert f.output()["of"] == 3


def test_lane_change_left_moves_one_lane():
  f = LaneFilter(); f.reset(3); f.p = [0.0, 0.0, 1.0]
  f.lane_change(-1, why="openpilot lane change left")
  out = f.output()
  assert out["best"] == 1
  assert abs(f.p[1] - 0.9) < 1e-9 and abs(f.p[2] - 0.1) < 1e-9
  assert out["why"] == "openpilot lane change left"


def test_lane_change_at_edge_keeps_probability():
  f = LaneFilter(); f.reset(2); f.p = [1.0, 0.0]
  f.lane_change(-1)
  assert abs(sum(f.p) - 1) < 1e-9 and f.p[0] == 1.0


def test_diffusion_is_by_distance_not_time():
  # Review focus 5: stopped in a queue must not flatten p
  f = LaneFilter(); f.reset(3); f.p = [0.0, 1.0, 0.0]
  for _ in range(6000):
    f.diffuse(0.0)
  assert f.p == [0.0, 1.0, 0.0]
  f.diffuse(15.0)   # one second at 15 m/s
  assert abs(f.p[0] - 0.005) < 1e-6 and abs(f.p[2] - 0.005) < 1e-6


def test_aborted_change_stays_mostly_on_original_lane():
  # Review focus 4: an aborted change is never reported as a change at all
  f = LaneFilter(); f.reset(3); f.p = [0.0, 1.0, 0.0]
  assert f.output()["best"] == 1
```

- [ ] **Step 3: Run to see them fail**

Run: `cd /home/peter/privat/openpilot/openpilot && /home/peter/privat/openpilot/undertake/venv/bin/python -m pytest openpilot/tavascan_map/tests/test_lanes.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'openpilot.tavascan_map.lanes'`.

- [ ] **Step 4: Implement**

```python
# openpilot/tavascan_map/lanes.py
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
```

Also create an empty `openpilot/tavascan_map/tests/__init__.py`.

- [ ] **Step 5: Run the tests**

Run: same as Step 3. Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add openpilot/tavascan_map/lanes.py openpilot/tavascan_map/tests/__init__.py openpilot/tavascan_map/tests/test_lanes.py
git commit -m "tavascan_map: lane filter core (state, lane change, diffusion by distance)"
```

---

### Task 2: Map topology — mapping lanes between stretches, turn lanes without indicating

**Files:**
- Modify: `openpilot/tavascan_map/lanes.py`
- Test: `openpilot/tavascan_map/tests/test_lanes.py`

**Interfaces:**
- Consumes: `LaneFilter` from Task 1.
- Produces:
  - `lane_map(old_L: int, old_turns: str | None, new_L: int, new_turns: str | None, side_hint: int = 0) -> list[list[float]]` — matrix `M[i][j]` = share of old lane `i` going to new lane `j`; `side_hint` −1 (extra lanes on the left), +1 (right), 0 unknown.
  - `LaneFilter.relayout(self, new_L: int, new_turns: str | None, side_hint: int = 0) -> None` — applies `lane_map` from the filter's current `(L, turns)`, stores the new ones (`self.turns`).
  - `LaneFilter.turn_lanes_without_indicating(self, seconds_since_left: float, seconds_since_right: float) -> None`
  - Helper `turn_only(turns: str | None, L: int) -> list[int]` — per lane: −1 left-turn only, +1 right-turn only, 0 otherwise.

- [ ] **Step 1: Write the failing tests**

```python
from openpilot.tavascan_map.lanes import LaneFilter, lane_map, turn_only


def test_turn_only_parsing():
  assert turn_only("left|through|through;right", 3) == [-1, 0, 0]
  assert turn_only("none|none|right", 3) == [0, 0, 1]
  assert turn_only(None, 2) == [0, 0]
  assert turn_only("left|through", 3) == [0, 0, 0]   # count mismatch: unusable


def test_left_turn_lane_added_by_arrows_shifts_us_right_in_numbering():
  M = lane_map(2, "through|through;right", 3, "left|through|through;right")
  # old lane 0 -> new lane 1, old lane 1 -> new lane 2
  assert M[0][1] == 1.0 and M[1][2] == 1.0


def test_lane_added_unknown_side_splits():
  M = lane_map(2, None, 3, None, side_hint=0)
  assert abs(M[0][0] - 0.5) < 1e-9 and abs(M[0][1] - 0.5) < 1e-9


def test_lane_added_on_right_by_hint():
  M = lane_map(2, None, 3, None, side_hint=+1)
  assert M[0][0] == 1.0 and M[1][1] == 1.0


def test_lane_lost_merges_into_neighbour():
  f = LaneFilter(); f.reset(3); f.turns = None; f.p = [0.0, 0.0, 1.0]
  f.relayout(2, None, side_hint=+1)   # right lane ends
  assert f.L == 2 and abs(f.p[1] - 1.0) < 1e-9


def test_no_lane_count_never_indexes_out_of_range():
  # Review focus 1
  f = LaneFilter(); f.reset(1); f.turns = None
  f.relayout(1, "left|through", 0)
  assert f.p == [1.0]


def test_turn_lanes_without_indicating():
  f = LaneFilter(); f.reset(3); f.turns = "left|through|through"
  f.turn_lanes_without_indicating(seconds_since_left=999, seconds_since_right=999)
  assert f.p[0] < 0.06 and abs(sum(f.p) - 1) < 1e-9
  g = LaneFilter(); g.reset(3); g.turns = "left|through|through"
  g.turn_lanes_without_indicating(seconds_since_left=3, seconds_since_right=999)
  assert g.p[0] > 0.55
```

- [ ] **Step 2: Run to see them fail**

Run: `cd /home/peter/privat/openpilot/openpilot && /home/peter/privat/openpilot/undertake/venv/bin/python -m pytest openpilot/tavascan_map/tests/test_lanes.py -q`
Expected: FAIL, `ImportError: cannot import name 'lane_map'`.

- [ ] **Step 3: Implement** (append to `lanes.py`; add `self.turns: str | None = None` in `__init__`)

```python
TURN_ONLY_FACTOR = 0.1
TURN_SIGNALLED_FACTOR = 3.0
INDICATOR_MEMORY_S = 10.0


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
  o, n = turn_only(old_turns, old_L), turn_only(new_turns, new_L)
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
        M[i][i] += 0.5; M[i][i + d] += 0.5
    else:         # lanes lost: those beyond the new count merge into the nearest
      if side < 0:
        M[i][max(0, i + d)] = 1.0
      else:
        M[i][min(new_L - 1, i)] = 1.0
  return M
```

Methods on `LaneFilter`:

```python
  def relayout(self, new_L: int, new_turns: str | None, side_hint: int = 0) -> None:
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
    kinds = turn_only(self.turns, self.L)
    if not any(kinds):
      return
    for i, k in enumerate(kinds):
      if k == 0:
        continue
      since = seconds_since_left if k < 0 else seconds_since_right
      self.p[i] *= TURN_SIGNALLED_FACTOR if since <= INDICATOR_MEMORY_S else TURN_ONLY_FACTOR
    self._norm()
```

Note: `turn_lanes_without_indicating` is called once when a stretch with turn-only lanes is entered (Task 5 calls it on layout change only), not every tick.

- [ ] **Step 4: Run the tests** — Expected: all passed (12).

- [ ] **Step 5: Commit**

```bash
git add openpilot/tavascan_map/lanes.py openpilot/tavascan_map/tests/test_lanes.py
git commit -m "tavascan_map: lane filter map topology and turn lanes without indicating"
```

---

### Task 3: Observation update — edges, neighbouring lines, radar

**Files:**
- Modify: `openpilot/tavascan_map/lanes.py`
- Test: `openpilot/tavascan_map/tests/test_lanes.py`

**Interfaces:**
- Consumes: `LaneFilter`.
- Produces: `LaneFilter.observe(self, obs: dict) -> None` with `obs` keys (all optional, `None` = unknown):
  `left_edge_m`, `right_edge_m` (distance from car centre to the edge, metres, positive), `left_edge_prob`, `right_edge_prob` (0..1),
  `outer_left_prob`, `outer_right_prob` (model lane line probabilities of lines 0 and 3),
  `car_left`, `car_right` (bool: a vehicle moving our way beside us on that side),
  `model_conf` (0..1: mean probability of our own two lane lines).
- Constants: `LANE_W = 3.4`, `EDGE_NEAR_M = 2.5`.

- [ ] **Step 1: Write the failing tests**

```python
def test_near_left_edge_favours_lane_0():
  f = LaneFilter(); f.reset(3)
  for _ in range(10):
    f.observe({"left_edge_m": 1.9, "left_edge_prob": 0.9, "model_conf": 0.9})
  assert f.output()["best"] == 0


def test_car_on_the_left_rules_out_lane_0():
  f = LaneFilter(); f.reset(2); f.p = [0.6, 0.4]
  for _ in range(10):
    f.observe({"car_left": True})
  assert f.output()["best"] == 1


def test_confident_outer_right_line_means_not_rightmost():
  f = LaneFilter(); f.reset(2); f.p = [0.4, 0.6]
  for _ in range(10):
    f.observe({"outer_right_prob": 0.9, "model_conf": 0.9})
  assert f.output()["best"] == 0


def test_low_model_confidence_weakens_model_terms():
  f = LaneFilter(); f.reset(3); g = LaneFilter(); g.reset(3)
  f.observe({"left_edge_m": 1.9, "left_edge_prob": 0.9, "model_conf": 0.9})
  g.observe({"left_edge_m": 1.9, "left_edge_prob": 0.9, "model_conf": 0.1})
  assert f.p[0] > g.p[0] >= 1 / 3 - 1e-9


def test_empty_observation_changes_nothing():
  f = LaneFilter(); f.reset(3); f.p = [0.2, 0.5, 0.3]
  f.observe({})
  assert f.p == [0.2, 0.5, 0.3]
```

- [ ] **Step 2: Run to see them fail** — Expected: `AttributeError: 'LaneFilter' object has no attribute 'observe'`.

- [ ] **Step 3: Implement** (append constants and method)

```python
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

  # (method of LaneFilter)
  def observe(self, obs: dict) -> None:
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
      # lanes between us and that edge, from its distance
      n = max(0.0, (d - LANE_W / 2) / LANE_W)
      s = EDGE_WEIGHT * pr * k_model
      for i in range(L):
        lanes_to_edge = i if side < 0 else L - 1 - i
        near = d <= EDGE_NEAR_M
        if near:
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
```

(`_soft` is a module-level function; `observe` is indented inside the class.)

- [ ] **Step 4: Run the tests** — Expected: all passed (17).

- [ ] **Step 5: Commit**

```bash
git add openpilot/tavascan_map/lanes.py openpilot/tavascan_map/tests/test_lanes.py
git commit -m "tavascan_map: lane filter observations (edges, outer lines, radar neighbours)"
```

---

### Task 4: Inputs from cereal — features and lane-change events

**Files:**
- Create: `openpilot/tavascan_map/lane_inputs.py`
- Test: `openpilot/tavascan_map/tests/test_lane_inputs.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (produces the dict `LaneFilter.observe` takes).
- Produces:
  - `observation(lane_ys: list[float], lane_probs: list[float], edge_ys: list[float], edge_stds: list[float], radar: list[dict]) -> dict` — `lane_ys`: y at the car (x≈0) of the four model lane lines (right positive); `edge_ys`: y of the two road edges; `edge_stds` their std; `radar`: points `{"d", "y", "v_abs"}` (y right positive).
  - `class LaneChangeWatch` with `update(self, lc_state: str, lc_dir: str, left_blinker: bool, right_blinker: bool, lane_ys: list[float], t: float) -> tuple[int, str] | None` — returns `(direction, why)` once per completed change, else `None`. Also `seconds_since(self, side: int, t: float) -> float` (indicator memory).

- [ ] **Step 1: Write the failing tests**

```python
# openpilot/tavascan_map/tests/test_lane_inputs.py
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
```

- [ ] **Step 2: Run to see them fail** — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# openpilot/tavascan_map/lane_inputs.py
"""Turns what openpilot sees into the plain values the lane filter takes.

Frame: y metres to the RIGHT of the car, as in modelV2 and tavascan_web.
"""
BESIDE_D = (-6.0, 12.0)     # m, radar distance window that counts as beside us
BESIDE_Y = (2.0, 5.5)       # m, lateral window of the neighbouring lane
RELABEL_JUMP = 2.0          # m, our nearest line jumping this far = the model relabelled


def observation(lane_ys, lane_probs, edge_ys, edge_stds, radar) -> dict:
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
      continue
    if BESIDE_Y[0] <= -p["y"] <= BESIDE_Y[1]:
      left = True
    elif BESIDE_Y[0] <= p["y"] <= BESIDE_Y[1]:
      right = True
  obs["car_left"], obs["car_right"] = left, right
  return obs


class LaneChangeWatch:
  def __init__(self):
    self._op = None              # direction of an openpilot change in progress
    self._last_blink = {-1: -1e9, 1: -1e9}
    self._prev_ys = None

  def seconds_since(self, side: int, t: float) -> float:
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
        recent = t - self._last_blink[side] <= 3.0
        if recent and ((side < 0 and jump > RELABEL_JUMP) or (side > 0 and jump < -RELABEL_JUMP)):
          ev = (side, f"driver lane change {'left' if side < 0 else 'right'}")
    self._prev_ys = list(lane_ys) if len(lane_ys) == 4 else self._prev_ys
    return ev
```

- [ ] **Step 4: Run the tests**

Run: `cd /home/peter/privat/openpilot/openpilot && /home/peter/privat/openpilot/undertake/venv/bin/python -m pytest openpilot/tavascan_map/tests -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add openpilot/tavascan_map/lane_inputs.py openpilot/tavascan_map/tests/test_lane_inputs.py
git commit -m "tavascan_map: lane filter inputs from model, car and radar"
```

---

### Task 5: Run it in tavascan_map at 10 Hz, publish `lane`

**Files:**
- Modify: `openpilot/tavascan_map/main.py`
- Create: `openpilot/tavascan_map/lane_runner.py` (glue, testable without cereal)
- Test: `openpilot/tavascan_map/tests/test_lane_runner.py`

**Interfaces:**
- Consumes: `LaneFilter` (Tasks 1–3), `observation`, `LaneChangeWatch` (Task 4), matcher output keys `way`, `lanes`, `turn_lanes`.
- Produces: `class LaneRunner` with `step(self, t: float, v_ego: float, dt: float, match: dict | None, model: dict, car: dict, radar: list[dict]) -> dict` returning `LaneFilter.output()`. `model` = `{"lc_state", "lc_dir", "lane_ys", "lane_probs", "edge_ys", "edge_stds"}`, `car` = `{"left_blinker", "right_blinker"}`.

- [ ] **Step 1: Write the failing tests**

```python
# openpilot/tavascan_map/tests/test_lane_runner.py
from openpilot.tavascan_map.lane_runner import LaneRunner

MODEL = {"lc_state": "off", "lc_dir": "none", "lane_ys": [-5.1, -1.7, 1.7, 5.1],
         "lane_probs": [0.5, 0.9, 0.9, 0.5], "edge_ys": [-8.0, 8.0], "edge_stds": [0.5, 0.5]}
CAR = {"left_blinker": False, "right_blinker": False}


def test_no_map_uses_model_lane_count():
  r = LaneRunner()
  out = r.step(0.0, 20.0, 0.1, None, MODEL, CAR, [])
  assert out["of"] >= 1


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
```

- [ ] **Step 2: Run to see them fail** — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# openpilot/tavascan_map/lane_runner.py
"""Feeds the lane filter once per tick: map layout, lane changes, observations."""
from openpilot.tavascan_map.lanes import LaneFilter
from openpilot.tavascan_map.lane_inputs import LaneChangeWatch, observation

MODEL_LANE_PROB = 0.3


class LaneRunner:
  def __init__(self):
    self.f = LaneFilter()
    self.watch = LaneChangeWatch()
    self._layout = None          # (way, lanes, turn_lanes) last applied

  def _model_lanes(self, model) -> int:
    probs = model.get("lane_probs") or []
    if len(probs) != 4:
      return self.f.L
    return 1 + (probs[0] > MODEL_LANE_PROB) + (probs[3] > MODEL_LANE_PROB)

  def step(self, t, v_ego, dt, match, model, car, radar) -> dict:
    # 1. layout: from the map when matched, otherwise keep what we had
    if match:
      layout = (match.get("way"), match.get("lanes") or self._model_lanes(model), match.get("turn_lanes"))
    elif self._layout is None:
      layout = (None, self._model_lanes(model), None)
    else:
      layout = self._layout
    if self._layout is None:
      self.f.reset(layout[1]); self.f.turns = layout[2]
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
```

Then in `main.py`:
- `sm = messaging.SubMaster([gps_service, "modelV2", "carState", "radarTracks"])`, `rk = Ratekeeper(10.0)`.
- Keep map matching at 2 Hz: `if now - last_match >= 0.5` around the existing match block; keep `st` between matches.
- Each tick build `model` from `sm["modelV2"]` (`meta.laneChangeState`, `meta.laneChangeDirection` via `str()`, `[ll.y[0] for ll in laneLines]`, `list(laneLineProbs)`, `[e.y[0] for e in roadEdges]`, `[e for e in roadEdgeStds]`), `car` from `sm["carState"].leftBlinker/rightBlinker`, `radar` via `openpilot.tavascan_web.geometry.radar_points(sm["radarTracks"], v_ego)`; call `lane = runner.step(time.monotonic(), v_ego, 0.1, st, model, car, radar)`; `out["lane"] = lane`.
- Write the JSON every tick only when it changed or every 0.5 s (keep `/dev/shm` writes ~2 Hz): `if now - last_write >= 0.5`.
- Wrap the runner in `try/except Exception: cloudlog.exception(...)` so a filter bug cannot stop map output.

- [ ] **Step 4: Run all tests** — `... -m pytest openpilot/tavascan_map/tests -q`. Expected: all passed. Also `python3 -m py_compile openpilot/tavascan_map/main.py`.

- [ ] **Step 5: Commit**

```bash
git add openpilot/tavascan_map/lane_runner.py openpilot/tavascan_map/main.py openpilot/tavascan_map/tests/test_lane_runner.py
git commit -m "tavascan_map: run the lane filter at 10 Hz and publish lane"
```

---

### Task 6: Replay over recorded drives and check against camera episodes

**Files:**
- Create (outside repo): `/home/peter/privat/openpilot/web3d_dev/lane_replay.py`

**Interfaces:**
- Consumes: `LaneRunner` (Task 5), `Map` (matcher), rlog reader `/home/peter/privat/openpilot/undertake/rlog.py`.
- Produces: a timeline printed at every change of `best` or `why`, and `frames` regenerated by `gen_frames.py` with `map.lane` included (modify `gen_frames.py` to run a `LaneRunner` per modelV2 tick and add `mp["lane"]`).

- [ ] **Step 1: Write `lane_replay.py`**

```python
"""lane_replay.py <route dir> [HH:MM:SS HH:MM:SS] -- lane filter timeline over a recorded drive."""
import sys, glob, datetime
sys.path[:0] = ["/home/peter/privat/openpilot/undertake", "/home/peter/privat/openpilot/openpilot"]
import rlog
from openpilot.tavascan_map.matcher import Map
from openpilot.tavascan_map.lane_runner import LaneRunner
from openpilot.tavascan_web import geometry
route = sys.argv[1]; a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("00:00:00", "99")
M = Map("/home/peter/privat/openpilot/osm/tavascan_map.sqlite"); R = LaneRunner()
segs = sorted(glob.glob(route + "/*/rlog.zst"), key=lambda s: int(s.split("--")[-1].split("/")[0]))
v = 0.0; car = {"left_blinker": False, "right_blinker": False}; radar = []; gps = None; st = None; last_m = 0; prev = None
for p in segs:
  wall = None
  for e in rlog.read(p):
    w = e.which(); t = e.logMonoTime / 1e9
    if w == "clocks" and wall is None: wall = e.clocks.wallTimeNanos / 1e9 - t
    elif w == "carState": cs = e.carState; v = cs.vEgo; car = {"left_blinker": cs.leftBlinker, "right_blinker": cs.rightBlinker}
    elif w == "radarTracks": radar = geometry.radar_points(e.radarTracks, v)
    elif w == "gpsLocationExternal": gps = e.gpsLocationExternal
    elif w == "modelV2" and wall:
      s = datetime.datetime.fromtimestamp(t + wall).strftime("%H:%M:%S")
      if not (a <= s <= b): continue
      if gps is not None and gps.hasFix and t - last_m >= 0.5:
        st = M.match(gps.latitude, gps.longitude, gps.bearingDeg, gps.speed); last_m = t
      md = e.modelV2
      model = {"lc_state": str(md.meta.laneChangeState), "lc_dir": str(md.meta.laneChangeDirection),
               "lane_ys": [ll.y[0] for ll in md.laneLines], "lane_probs": list(md.laneLineProbs),
               "edge_ys": [r.y[0] for r in md.roadEdges], "edge_stds": list(md.roadEdgeStds)}
      out = R.step(t, v, 0.05, st, model, car, radar)
      key = (out["best"], out["of"], out["why"])
      if key != prev:
        print(s, "%5.1f km/h" % (v * 3.6), "lane %d of %d (%.0f %%)" % (out["best"], out["of"], 100 * out["confidence"]),
              out["p"], out["why"], "|", (st or {}).get("turn_lanes"))
        prev = key
```

(The replay calls `step` at the modelV2 rate, 20 Hz, with `dt=0.05`; on the device it runs at 10 Hz with `dt=0.1`. The filter is rate-independent except for the per-tick observation strength, so the replay is slightly more confident than the car — note that when comparing.)

- [ ] **Step 2: Run over the home drive of 2026-10-21 and check three episodes against the camera**

Run: `cd /home/peter/privat/openpilot/web3d_dev && /home/peter/privat/openpilot/undertake/venv/bin/python lane_replay.py ../routes/00000162--fe0d3dcccd 17:06:30 17:12:30`
Expected, checked against camera frames (`frames_at.py`):
- the driver's change into a left-turn lane before a signalised junction (around 17:09:21, indicator left): `best` moves one lane left with `why` "driver lane change left" or "openpilot lane change left", and the turn lane carries most of `p` afterwards;
- the four-lane approach with `left|none|none|right` (around 17:07:44): `best` is a through lane, not 0 or 3, without an indicator;
- the motorway exit lane appearing (4 → 3 lanes around 17:11:42): `best` stays on our through lane.
If any is wrong, adjust the constant responsible (`TURN_ONLY_FACTOR`, `EDGE_WEIGHT`, `RADAR_WEIGHT`), re-run, and record the before/after in the commit message of Task 7.

- [ ] **Step 3: Add `lane` to `gen_frames.py`** (run a `LaneRunner` per modelV2 tick as above; `mp["lane"] = out`) and regenerate `frames_162.json` and `frames_162b.json`.

(No repo commit in this task; the replay lives outside the repo. Tuning changes to `lanes.py` are committed in Task 7 with the replay evidence.)

---

### Task 7: 3D page laid out around the tracked lane

**Files:**
- Modify: `openpilot/tavascan_web/page3d.html`

**Interfaces:**
- Consumes: `d.map.lane = {p, best, of, confidence, why}` (Task 5), `d.map.match.path` (existing).

- [ ] **Step 1: Lane index from the tracker** — in `buildRoad`, replace the model-based guess:

```js
  // Which lane are we in: the lane tracker's most probable lane; with no
  // tracker output, the middle one.
  const lane = (d_lane && d_lane.of === L0) ? d_lane : null;
  let idx = lane ? lane.best : Math.floor((L0 - 1) / 2);
```

Pass `d.map && d.map.lane` as a new last argument `d_lane` to `buildRoad(mm, lanePos, modelReach, modelHeading, d_lane)`. Draw our lane highlight with opacity `0.05 + 0.15 * (lane ? lane.confidence : 0)`.

- [ ] **Step 2: Radar cars in lanes** — in `updateOthers`, snap lateral position to the lane grid around our lane centre so cars sit in lanes even when the radar's y wanders:

```js
    const laneOff = Math.round(p.y / LANE_W);                // lanes right of ours
    const ySnap = laneOff * LANE_W + 0.35 * (p.y - laneOff * LANE_W);
    o.target = P(p.d + CAR_L / 2, ySnap);
```

- [ ] **Step 3: Model lane lines as an optional layer, off by default** — add a button `<button class="btn" id="lay">lines</button>` next to `camera`; keep `buildModel` computing `reach`/`heading` every update but set `modelGroup.visible = showModel` (default `false`, remembered in `localStorage` key `lines3d`, inside try/catch). Draw map markings from `s0` (not from `modelReach`) when the model layer is hidden.

- [ ] **Step 4: HUD** — after the road info line, append lane text:

```js
  const ln = map && map.lane;
  if (ln && ln.of > 1) $("rinfo").textContent += ` · lane ${ln.best + 1} of ${ln.of} · ${Math.round(100 * ln.confidence)} %`;
```

- [ ] **Step 5: Check with screenshots** — regenerate frames (Task 6 Step 3), start `dev_server.py`, take `shot.sh` screenshots at the three episodes, and look at them: our car in the tracked lane, cars in lanes, HUD text present. Also check `?debug` draw calls stay under 200.

- [ ] **Step 6: Commit**

```bash
git add openpilot/tavascan_web/page3d.html openpilot/tavascan_map/lanes.py
git commit -m "tavascan_web 3D: map-first layout around the tracked lane"
```

---

### Task 8: Merge and deploy (parked at home), then review drives

- [ ] **Step 1: Run all tests** — `... -m pytest openpilot/tavascan_map/tests -q` — all pass.
- [ ] **Step 2: Merge** — `git checkout shadow-view && git merge --no-ff lane-tracker && git push origin lane-tracker shadow-view`.
- [ ] **Step 3: Deploy only when parked at home** — check `ssh comma@10.0.1.164 'ip -4 addr show wlan0 | grep -c "inet 10.0.1."; ps -eo args | grep -c "[p]lannerd"'` gives `1` and `0`; then `ssh comma@10.0.1.164 'cd /data/openpilot && git pull --ff-only'`. `tavascan_map` and `tavascan_web` are onroad processes and pick the code up on the next drive; no reboot.
- [ ] **Step 4: On device, import check** — `ssh comma@10.0.1.164 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python -c "import openpilot.tavascan_map.main, openpilot.tavascan_map.lane_runner; print(1)"'` prints `1`.
- [ ] **Step 5: After the next drives** — pull `/data/tavascan_map.jsonl`, list lane changes and low-confidence stretches, and go through them with the driver.
