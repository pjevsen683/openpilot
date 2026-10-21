#!/usr/bin/env python3
"""Builds the map file tavascan_map reads on the device, from an OSM extract.

mapd's own offline tiles keep only name, ref, speed limits, a lane count (on
about four ways in ten) and the geometry. Road class, one-way, turn lanes,
traffic signals and roundabouts are dropped. This keeps them.

Run on a PC, not on the device -- it needs pyosmium and a few GB of RAM:

  python build_map.py denmark-latest.osm.pbf tavascan_map.sqlite

The result is a single SQLite file:

  ways      one row per drivable OSM way: class, name, ref, one-way, lanes
            in total and per direction, turn lanes per direction, speed
            limits, junction (roundabout), bridge/tunnel, and the geometry
            as packed int32 lat/lon (degrees * 1e7) plus the node ids
  way_rt    R*tree over the ways' bounding boxes
  points    signals, stop and give-way signs, crossings, speed cameras,
            motorway exits, traffic calming, level crossings
  point_rt  R*tree over the points
  junction  node id -> way ids, for every node shared by two or more ways,
            so the matcher can follow the road onto the next way
  meta      source file and build time

OSM data is (c) OpenStreetMap contributors, ODbL.
"""
import array
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict

import osmium

DRIVABLE = {
  "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
  "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
  "living_street", "service", "road",
}
# Service roads are mostly car parks and drive-ins; these subtypes never matter
# for driving through.
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access"}

POINT_HIGHWAY = {"traffic_signals", "stop", "give_way", "crossing", "speed_camera",
                 "motorway_junction", "mini_roundabout"}
POINT_KEEP_TAGS = ("highway", "traffic_signals", "traffic_signals:direction", "direction",
                   "crossing", "ref", "name", "exit_to", "traffic_calming", "railway", "maxspeed")


def speed_kph(v: str | None) -> int | None:
  """OSM maxspeed to km/h, or None. Handles '50', '80 mph', 'DK:urban' and the like."""
  if not v:
    return None
  v = v.strip().lower()
  zones = {"dk:urban": 50, "dk:rural": 80, "dk:motorway": 130}
  if v in zones:
    return zones[v]
  try:
    if v.endswith("mph"):
      return round(float(v[:-3]) * 1.609)
    return round(float(v.split(";")[0]))
  except ValueError:
    return None


def to_int(v: str | None) -> int | None:
  try:
    return int(v.split(";")[0]) if v else None
  except ValueError:
    return None


class Handler(osmium.SimpleHandler):
  def __init__(self, db):
    super().__init__()
    self.db = db
    self.node_use = defaultdict(int)
    self.n_ways = 0
    self.n_points = 0

  def node(self, n):
    t = n.tags
    kind = None
    if t.get("highway") in POINT_HIGHWAY:
      kind = t.get("highway")
    elif "traffic_calming" in t:
      kind = "traffic_calming"
    elif t.get("railway") == "level_crossing":
      kind = "level_crossing"
    if kind is None or not n.location.valid():
      return
    tags = {k: t.get(k) for k in POINT_KEEP_TAGS if k in t}
    lat, lon = n.location.lat, n.location.lon
    self.db.execute("INSERT INTO points VALUES (?,?,?,?,?)", (n.id, kind, lat, lon, json.dumps(tags, ensure_ascii=False)))
    self.db.execute("INSERT INTO point_rt VALUES (?,?,?,?,?)", (n.id, lat, lat, lon, lon))
    self.n_points += 1

  def way(self, w):
    t = w.tags
    hw = t.get("highway")
    if hw not in DRIVABLE or (hw == "service" and t.get("service") in SKIP_SERVICE):
      return
    if t.get("area") == "yes":
      return
    lats, lons, ids = array.array("i"), array.array("i"), array.array("q")
    for nd in w.nodes:
      if not nd.location.valid():
        return
      lats.append(round(nd.location.lat * 1e7))
      lons.append(round(nd.location.lon * 1e7))
      ids.append(nd.ref)
    if len(ids) < 2:
      return
    for i in ids:
      self.node_use[i] += 1

    oneway = {"yes": 1, "true": 1, "1": 1, "-1": -1, "reverse": -1}.get(t.get("oneway", ""), 0)
    if t.get("junction") in ("roundabout", "circular") or hw == "motorway":
      oneway = oneway or 1
    lanes = to_int(t.get("lanes"))
    lanes_fwd, lanes_bwd = to_int(t.get("lanes:forward")), to_int(t.get("lanes:backward"))
    turn_fwd, turn_bwd = t.get("turn:lanes:forward"), t.get("turn:lanes:backward")
    if oneway:
      turn_fwd = turn_fwd or t.get("turn:lanes")
      lanes_fwd = lanes_fwd or lanes
    maxspeed = speed_kph(t.get("maxspeed"))
    row = (w.id, hw, t.get("name"), t.get("ref"), oneway, lanes, lanes_fwd, lanes_bwd,
           turn_fwd, turn_bwd, maxspeed, speed_kph(t.get("maxspeed:forward")), speed_kph(t.get("maxspeed:backward")),
           t.get("junction"), 1 if t.get("bridge") not in (None, "no") else 0, 1 if t.get("tunnel") not in (None, "no") else 0,
           t.get("destination:lanes") or t.get("destination"), t.get("lit"), t.get("surface"),
           lats.tobytes(), lons.tobytes(), ids.tobytes())
    self.db.execute("INSERT INTO ways VALUES (" + ",".join("?" * len(row)) + ")", row)
    self.db.execute("INSERT INTO way_rt VALUES (?,?,?,?,?)",
                    (w.id, min(lats) / 1e7, max(lats) / 1e7, min(lons) / 1e7, max(lons) / 1e7))
    self.n_ways += 1


def main(src: str, out: str) -> None:
  if os.path.exists(out):
    os.remove(out)
  db = sqlite3.connect(out)
  db.executescript("""
    PRAGMA journal_mode = OFF; PRAGMA synchronous = OFF;
    CREATE TABLE ways (id INTEGER PRIMARY KEY, hw TEXT, name TEXT, ref TEXT, oneway INTEGER,
      lanes INTEGER, lanes_fwd INTEGER, lanes_bwd INTEGER, turn_fwd TEXT, turn_bwd TEXT,
      maxspeed INTEGER, maxspeed_fwd INTEGER, maxspeed_bwd INTEGER, junction TEXT,
      bridge INTEGER, tunnel INTEGER, destination TEXT, lit TEXT, surface TEXT,
      lat BLOB, lon BLOB, nodes BLOB);
    CREATE VIRTUAL TABLE way_rt USING rtree(id, minlat, maxlat, minlon, maxlon);
    CREATE TABLE points (id INTEGER PRIMARY KEY, kind TEXT, lat REAL, lon REAL, tags TEXT);
    CREATE VIRTUAL TABLE point_rt USING rtree(id, minlat, maxlat, minlon, maxlon);
    CREATE TABLE junction (node INTEGER, way INTEGER);
    CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT);
  """)
  t0 = time.time()
  h = Handler(db)
  h.apply_file(src, locations=True, idx="flex_mem")
  print(f"{h.n_ways} ways, {h.n_points} points in {time.time() - t0:.0f} s")

  shared = {n for n, c in h.node_use.items() if c >= 2}
  rows = []
  for wid, blob in db.execute("SELECT id, nodes FROM ways"):
    ids = array.array("q"); ids.frombytes(blob)
    rows.extend((n, wid) for n in set(ids) if n in shared)
  db.executemany("INSERT INTO junction VALUES (?,?)", rows)
  db.execute("CREATE INDEX junction_node ON junction(node)")
  db.executemany("INSERT INTO meta VALUES (?,?)", [
    ("source", os.path.basename(src)), ("built", time.strftime("%Y-%m-%d %H:%M:%S")),
    ("license", "(c) OpenStreetMap contributors, ODbL"),
  ])
  db.commit()
  db.execute("VACUUM")
  db.close()
  print(f"{len(rows)} junction links; {os.path.getsize(out) / 1e6:.0f} MB in {time.time() - t0:.0f} s")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
