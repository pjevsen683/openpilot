from openpilot.tavascan_map.lane_layout import LANE_W, edges


def road(*stretches):
  """Points every 5 m: stretches of (length m, lanes, turn lanes, oneway, lanes total)."""
  pts, s = [], -40.0
  for length, lanes, turns, oneway, total in stretches:
    end = s + length
    while s < end:
      pts.append({"s": s, "lanes": lanes, "turns": turns, "oneway": oneway, "total": total})
      s += 5.0
  return pts


def centre(e, lane):
  """Middle of our lane `lane` (from the left) at a point, metres right of the way."""
  return e[2] + (lane + 0.5) * LANE_W


def at(pts, out, s):
  i = min(range(len(pts)), key=lambda k: abs(pts[k]["s"] - s))
  return out[i]


def test_no_change_follows_the_map():
  pts = road((200, 1, None, False, 2))
  out = edges(pts)
  assert all(abs(left) < 1e-9 and abs(right - LANE_W) < 1e-9 for left, right, _ in out)


def test_one_way_road_is_centred_on_the_way():
  pts = road((200, 2, None, True, 2))
  assert all(abs(left + LANE_W) < 1e-9 for left, _, _ in edges(pts))


def test_left_turn_pocket_keeps_the_through_lane_straight():
  # 1 + 1 two-way, then a left-turn pocket (2 our way + 1), then 1 + 1 again
  pts = road((100, 1, None, False, 2), (60, 2, "left|through", False, 3), (100, 1, None, False, 2))
  out = edges(pts)
  before = centre(at(pts, out, 50), 0)
  pocket = centre(at(pts, out, 100), 1)      # the through lane of the pocket, just after it opens
  after = centre(at(pts, out, 160), 0)       # just past the junction
  assert abs(pocket - before) < 0.2 and abs(after - before) < 0.2


def test_through_lanes_carry_on_across_a_junction():
  # Torshoejvej: left|left|none|none|right, then two lanes on the far side
  pts = road((100, 5, "left|left|none|none|right", True, 5), (100, 2, None, True, 2))
  out = edges(pts)
  assert abs(centre(at(pts, out, 55), 2) - centre(at(pts, out, 65), 0)) < 0.2
  assert abs(centre(at(pts, out, 55), 3) - centre(at(pts, out, 65), 1)) < 0.2


def test_a_new_lane_opens_gradually():
  pts = road((100, 1, None, True, 1), (100, 2, "left|through", True, 2))
  out = edges(pts)
  lefts = [at(pts, out, s)[0] for s in (35, 45, 55, 65)]
  steps = [abs(b - a) for a, b in zip(lefts, lefts[1:])]
  assert max(steps) < 0.6 * LANE_W     # not the whole lane at one point
  # while the asphalt widens, the lane we are in stays where it is
  assert abs(centre(at(pts, out, 45), 0) - centre(at(pts, out, 65), 1)) < 1e-6


def test_far_from_a_change_the_road_returns_to_the_map():
  pts = road((100, 5, "left|left|none|none|right", True, 5), (300, 2, None, True, 2))
  out = edges(pts)
  assert abs(at(pts, out, 340)[0] - (-LANE_W)) < 1e-6


def test_unknown_lane_counts_are_drawn_as_one_lane():
  pts = road((100, None, None, False, None))
  assert all(abs(right - left - LANE_W) < 1e-9 for left, right, _ in edges(pts))
