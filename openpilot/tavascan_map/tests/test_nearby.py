import array
import math
import sqlite3

from openpilot.tavascan_map.matcher import Map

LAT0, LON0 = 56.0, 10.0
M_PER_DEG_LAT = 111320.0


def ll(n, e):
  """metres north/east of (LAT0, LON0) -> lat, lon"""
  return LAT0 + n / M_PER_DEG_LAT, LON0 + e / (M_PER_DEG_LAT * math.cos(math.radians(LAT0)))


def make_db(path, ways):
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
  for wid, hw, lanes, junction, pts in ways:
    lat = array.array("i", [round(ll(n, e)[0] * 1e7) for n, e in pts])
    lon = array.array("i", [round(ll(n, e)[1] * 1e7) for n, e in pts])
    nodes = array.array("q", [wid * 100 + i for i in range(len(pts))])
    db.execute("INSERT INTO ways VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (wid, hw, None, None, 0, lanes, None, None, None, None, None, None, None, junction,
                0, 0, None, None, None, lat.tobytes(), lon.tobytes(), nodes.tobytes()))
    db.execute("INSERT INTO way_rt VALUES (?,?,?,?,?)", (wid, min(lat) / 1e7, max(lat) / 1e7, min(lon) / 1e7, max(lon) / 1e7))
  db.commit(); db.close()


def test_nearby_returns_roads_in_view_in_metres_from_the_anchor(tmp_path):
  p = str(tmp_path / "m.sqlite")
  make_db(p, [
    (1, "secondary", 2, None, [(-100, 0), (0, 0), (200, 0)]),        # our road, running north
    (2, "residential", None, None, [(100, -80), (100, 0)]),          # side road joining from the west
    (3, "tertiary", None, None, [(2000, 0), (2100, 0)]),             # far ahead: out of view
  ])
  m = Map(p)
  lat, lon = ll(0, 0)
  out = m.nearby(lat, lon, heading=0.0)
  ids = sorted(w["id"] for w in out["ways"])
  assert ids == [1, 2]
  side = next(w for w in out["ways"] if w["id"] == 2)
  assert side["class"] == "residential" and side["lanes"] is None
  n, e = side["pts"][0]
  assert abs(n - 100) < 1.5 and abs(e + 80) < 1.5   # metres north/east of the anchor (our position)


def test_nearby_looks_ahead_more_than_behind(tmp_path):
  p = str(tmp_path / "m.sqlite")
  make_db(p, [
    (1, "residential", None, None, [(250, -10), (250, 10)]),          # 250 m ahead: in view
    (2, "residential", None, None, [(-250, -10), (-250, 10)]),        # 250 m behind: not
  ])
  m = Map(p)
  lat, lon = ll(0, 0)
  assert [w["id"] for w in m.nearby(lat, lon, heading=0.0)["ways"]] == [1]
  assert [w["id"] for w in m.nearby(lat, lon, heading=180.0)["ways"]] == [2]


def test_nearby_thins_points(tmp_path):
  p = str(tmp_path / "m.sqlite")
  make_db(p, [(1, "residential", None, None, [(i * 0.5, 0) for i in range(200)])])   # a point every 0.5 m
  w = Map(p).nearby(*ll(0, 0), heading=0.0)["ways"][0]
  assert len(w["pts"]) < 40 and w["pts"][0] == [0.0, 0.0] and abs(w["pts"][-1][0] - 99.5) < 0.6


def test_roundabout_flag(tmp_path):
  p = str(tmp_path / "m.sqlite")
  make_db(p, [(1, "secondary", 1, "roundabout", [(50, 0), (60, 10), (70, 0), (60, -10), (50, 0)])])
  assert Map(p).nearby(*ll(0, 0), heading=0.0)["ways"][0]["roundabout"] is True


def test_long_road_crossing_the_view_without_a_point_in_it(tmp_path):
  p = str(tmp_path / "m.sqlite")
  make_db(p, [(1, "primary", 2, None, [(100, -1000), (100, 1000)])])   # one straight segment across our path
  assert [w["id"] for w in Map(p).nearby(*ll(0, 0), heading=0.0)["ways"]] == [1]
