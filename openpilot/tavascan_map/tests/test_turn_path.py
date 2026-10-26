import array
import sqlite3

from openpilot.tavascan_map.lane_layout import LANE_W, edges
from openpilot.tavascan_map.tests.test_nearby import ll


def make_junction(path, left=((250, -60), (400, -120)), right=((110, 150), (120, 300))):
  """An approach with left|right lanes running north into a node, a branch bending
  slightly left, and one turning sharp right: the motorway exit of 2026-10-26."""
  db = sqlite3.connect(path)
  db.executescript("""
    CREATE TABLE ways (id INTEGER PRIMARY KEY, hw TEXT, name TEXT, ref TEXT, oneway INTEGER,
      lanes INTEGER, lanes_fwd INTEGER, lanes_bwd INTEGER, turn_fwd TEXT, turn_bwd TEXT,
      maxspeed INTEGER, maxspeed_fwd INTEGER, maxspeed_bwd INTEGER, junction TEXT,
      bridge INTEGER, tunnel INTEGER, destination TEXT, lit TEXT, surface TEXT,
      lat BLOB, lon BLOB, nodes BLOB);
    CREATE VIRTUAL TABLE way_rt USING rtree(id, minlat, maxlat, minlon, maxlon);
    CREATE TABLE points (id INTEGER PRIMARY KEY, kind TEXT, lat REAL, lon REAL, tags TEXT);
    CREATE VIRTUAL TABLE point_rt USING rtree(id, minlat, maxlat, minlon, maxlon);
    CREATE TABLE junction (node INTEGER, way INTEGER);
  """)
  J = 999
  ways = [  # id, lanes, turn lanes, points (n, e), node ids
    (1, 2, "left|right", [(-150, 0), (0, 0), (100, 0)], [101, 102, J]),
    (2, 1, None, [(100, 0), *left], [J, 201, 202]),      # slight left
    (3, 1, None, [(100, 0), *right], [J, 301, 302]),     # sharp right
  ]
  for wid, lanes, turns, pts, nodes in ways:
    lat = array.array("i", [round(ll(n, e)[0] * 1e7) for n, e in pts])
    lon = array.array("i", [round(ll(n, e)[1] * 1e7) for n, e in pts])
    db.execute("INSERT INTO ways VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (wid, "motorway_link", None, None, 1, lanes, lanes, None, turns, None, None, None, None, None,
                0, 0, None, None, None, lat.tobytes(), lon.tobytes(), array.array("q", nodes).tobytes()))
    db.execute("INSERT INTO way_rt VALUES (?,?,?,?,?)", (wid, min(lat) / 1e7, max(lat) / 1e7, min(lon) / 1e7, max(lon) / 1e7))
    for n in nodes:
      db.execute("INSERT INTO junction VALUES (?,?)", (n, wid))
  db.commit(); db.close()


def _after_junction(m):
  # points well past the node at 100 m north (s ~ 100), as [x ahead, y right]
  return [p for p in m["path"]["pts"] if p[4] > 140]


def test_without_a_turn_the_path_bends_least(tmp_path):
  from openpilot.tavascan_map.matcher import Map
  p = str(tmp_path / "j.sqlite"); make_junction(p)
  st = Map(p).match(*ll(0, 0), 0.0, 10.0)
  assert all(pt[1] < 0 for pt in _after_junction(st))      # the slight left branch


def test_in_a_right_turn_lane_the_path_turns_right(tmp_path):
  from openpilot.tavascan_map.matcher import Map
  p = str(tmp_path / "j.sqlite"); make_junction(p)
  st = Map(p).match(*ll(0, 0), 0.0, 10.0, turn="right")
  after = _after_junction(st)
  assert after and all(pt[1] > 20 for pt in after[2:])


def test_the_turn_lane_carries_on_into_the_road_it_turns_into():
  # left|right, then right: the right lane is the one that goes on
  pts = [{"s": float(s), "lanes": 2, "turns": "left|right", "oneway": True, "total": 2} for s in range(-40, 100, 5)]
  pts += [{"s": float(s), "lanes": 1, "turns": None, "oneway": True, "total": 1, "via": "right"} for s in range(100, 200, 5)]
  out = edges(pts)
  before = out[[p["s"] for p in pts].index(95.0)]
  after = out[[p["s"] for p in pts].index(100.0)]
  assert abs((before[2] + 1.5 * LANE_W) - (after[2] + 0.5 * LANE_W)) < 1e-6


def test_a_way_split_with_the_same_lanes_keeps_them_where_they_are():
  # the map splits a road (a bridge, a speed limit) without its lanes changing
  pts = [{"s": float(s), "lanes": 2, "turns": "left|none", "oneway": True, "total": 2} for s in range(-40, 100, 5)]
  pts += [{"s": float(s), "lanes": 2, "turns": "left|none", "oneway": True, "total": 2, "via": "through"} for s in range(100, 200, 5)]
  out = edges(pts)
  assert abs(out[[p["s"] for p in pts].index(95.0)][2] - out[[p["s"] for p in pts].index(100.0)][2]) < 1e-6


def test_at_a_fork_the_turn_lane_takes_its_branch(tmp_path):
  # the exit of 2026-10-26: the branches part by only -11 and +17 degrees
  from openpilot.tavascan_map.matcher import Map
  p = str(tmp_path / "j.sqlite")
  make_junction(p, left=((250, -29), (400, -58)), right=((250, 46), (400, 92)))
  st = Map(p).match(*ll(0, 0), 0.0, 10.0, turn="right")
  assert all(pt[1] > 0 for pt in _after_junction(st)[4:])
