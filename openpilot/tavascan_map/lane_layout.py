"""Where our lanes lie across the road along the path ahead, so they join up.

Each way's own geometry puts our lanes about its middle (one-way) or right of
its centre line (two-way). Drawn way by way, lanes jump sideways wherever the
lane count changes: a left-turn pocket pushes the through lane right, and the
two lanes past a junction line up with its left-turn lanes rather than the
through lanes. Here the layout is carried across each change instead, the way
the lane filter maps lanes (turn-only lanes at the ends are the ones that
come and go), opened over a short taper, and let back onto the map's geometry
some way on.

Per point: (left, right, base) in metres right of the way's line. left/right
are the edge of the asphalt, tapering where a lane opens or closes; base is
the left side of lane 0 of the lanes at that point, which lane markings and
lane centres are measured from.
"""
from openpilot.tavascan_map.lanes import LANE_W, _added_left, turn_only

TAPER_M = 20.0    # a lane opens (or closes) over this, ending where it starts in the map
HOLD_M = 120.0    # past a change, the layout carried across it is kept this far (turn pockets run 50-100 m) ...
DECAY_M = 100.0   # ... then eased back onto the map's geometry over this


def _default_left(lanes: int, oneway: bool, total) -> float:
  if oneway:
    return -lanes * LANE_W / 2
  t = total if total and total >= lanes else 2 * lanes
  return -t * LANE_W / 2 + (t - lanes) * LANE_W


GOES = {"left": {"left", "slight_left", "sharp_left"}, "right": {"right", "slight_right", "sharp_right"},
        "through": {"through", "none", ""}}


def _carried_left(old_left, old_L, old_turns, new_L, new_turns, via=None) -> float:
  """Left side of the new lanes, so the lanes that carry on stay where they were.

  Where no lanes are gained and the old arrows are known, the lanes that go
  the way we went (via, or straight on) are the ones that carry on.
  """
  d = new_L - old_L
  parts = old_turns.split("|") if old_turns else []
  if d <= 0 and len(parts) == old_L and new_turns != old_turns:
    goes = GOES.get(via or "through", GOES["through"])
    kept = [i for i, lane in enumerate(parts) if {x.strip() for x in lane.split(";")} & goes]
    if kept:
      return old_left + kept[0] * LANE_W
  if d > 0:
    k = _added_left(old_L, new_L, new_turns)
    return old_left - (d / 2 if k is None else k) * LANE_W
  if d < 0:
    k = _added_left(new_L, old_L, old_turns)
    return old_left + (-d / 2 if k is None else k) * LANE_W
  return old_left


def _ending_lanes_turn_off(old_L, old_turns, new_L) -> bool:
  """The lanes that end are turn-only lanes: they turned off at a junction, so no taper."""
  if new_L >= old_L or not old_turns:
    return False
  kinds = turn_only(old_turns, old_L)
  lost = old_L - new_L
  k = _added_left(new_L, old_L, old_turns)
  return k is not None and all(kinds[i] != 0 for i in list(range(k)) + list(range(old_L - (lost - k), old_L)))


def edges(pts: list[dict]) -> list[tuple[float, float, float]]:
  """pts: per point "s" (m along the path), "lanes" (our direction, None if unknown),
  "turns" (turn:lanes or None), "oneway", "total" (lanes both ways, or None)."""
  out = []
  changes = []        # (index, old (left, right), taper)
  key = None
  base = corr = s_change = None
  L = 1
  for i, p in enumerate(pts):
    new_L = max(1, int(p.get("lanes") or 1))
    k = (new_L, p.get("turns"), bool(p.get("oneway")), p.get("total"), p.get("via"))
    default = _default_left(new_L, k[2], k[3])
    if key is None:
      corr, s_change = 0.0, p["s"]
    elif k != key:
      old = out[-1]
      left = _carried_left(old[2], L, key[1], new_L, k[1], p.get("via"))
      corr, s_change = left - default, p["s"]
      if new_L != L:
        changes.append((i, old, not _ending_lanes_turn_off(L, key[1], new_L)))
    key, L = k, new_L
    run = p["s"] - s_change
    c = corr if run <= HOLD_M else corr * max(0.0, 1.0 - (run - HOLD_M) / DECAY_M)
    base = default + c
    out.append((base, base + new_L * LANE_W, base))
  # open (or close) the asphalt over TAPER_M before each change in lane count
  for i, (old_l, old_r, _), taper in changes:
    if not taper:
      continue
    s_c = pts[i]["s"]
    new_l, new_r = out[i][0], out[i][1]
    for j in range(i - 1, -1, -1):
      f = (pts[j]["s"] - (s_c - TAPER_M)) / TAPER_M
      if f <= 0:
        break
      l, r, b = out[j]
      out[j] = (l + f * (new_l - l) if new_l < l else l, r + f * (new_r - r) if new_r > r else r, b)
  return out
