#!/usr/bin/env python3
"""Finds the road we are on in the map file, and what is coming up on it.

Pure logic on top of the SQLite file build_map.py makes, so it can be run
against a recorded drive on a PC exactly as it runs in the car.

Matching: every way within SEARCH_M of the GPS fix is a candidate. Each is
scored by how far its nearest segment is from us, plus a penalty for the
angle between that segment and our heading, plus a bonus when it is the way
we were on or one joined to it. One-way roads only count in their own
direction. While stopped the heading is unreliable, so the last match is
kept.

Looking ahead: from where we are, the road is followed to the end of the
way, then onto the joining way that bends least (and that we may drive in
that direction), until AHEAD_M. Points on that path -- signals, crossings,
give-way and stop signs, traffic calming -- are reported with their
distance, and so are changes in lanes, speed limit, road class and the start
of a roundabout.
"""
import array
import math
import sqlite3

EARTH_R = 6371000.0
SEARCH_M = 40.0
MAX_ANGLE = 60.0        # deg, a segment further off our heading is not ours
ANGLE_COST = 0.4        # m of distance per degree off our heading
STAY_BONUS = 8.0        # m, for the way we were on
JOINED_BONUS = 4.0      # m, for a way joined to it
MIN_HEADING_SPEED = 2.0  # m/s, below this the GPS bearing is noise
AHEAD_M = 500.0
BEHIND_M = 40.0
PATH_STEP_M = 5.0
POINT_ON_PATH_M = 12.0


def _bearing(lat1, lon1, lat2, lon2) -> float:
  dl = math.radians(lon2 - lon1)
  p1, p2 = math.radians(lat1), math.radians(lat2)
  y = math.sin(dl) * math.cos(p2)
  x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
  return math.degrees(math.atan2(y, x)) % 360.0


def _angle(a, b) -> float:
  return abs((a - b + 180.0) % 360.0 - 180.0)


class Way:
  __slots__ = ("id", "hw", "name", "ref", "oneway", "lanes", "lanes_fwd", "lanes_bwd", "turn_fwd", "turn_bwd",
               "maxspeed", "maxspeed_fwd", "maxspeed_bwd", "junction", "bridge", "tunnel", "destination",
               "lit", "surface", "lat", "lon", "nodes")

  def __init__(self, row):
    (self.id, self.hw, self.name, self.ref, self.oneway, self.lanes, self.lanes_fwd, self.lanes_bwd,
     self.turn_fwd, self.turn_bwd, self.maxspeed, self.maxspeed_fwd, self.maxspeed_bwd, self.junction,
     self.bridge, self.tunnel, self.destination, self.lit, self.surface, la, lo, no) = row
    a = array.array("i"); a.frombytes(la); self.lat = [v / 1e7 for v in a]
    a = array.array("i"); a.frombytes(lo); self.lon = [v / 1e7 for v in a]
    a = array.array("q"); a.frombytes(no); self.nodes = list(a)

  def directed(self, d: int) -> dict:
    """What a driver going in direction d (+1 along the node order, -1 against) sees."""
    if self.oneway:
      lanes, turn, vmax = self.lanes_fwd or self.lanes, self.turn_fwd, self.maxspeed_fwd or self.maxspeed
    elif d > 0:
      lanes, turn, vmax = self.lanes_fwd, self.turn_fwd, self.maxspeed_fwd or self.maxspeed
    else:
      lanes, turn, vmax = self.lanes_bwd, self.turn_bwd, self.maxspeed_bwd or self.maxspeed
    if lanes is None and not self.oneway and self.lanes:
      lanes = self.lanes // 2 if self.lanes >= 2 else None
    return {"lanes": lanes, "turn_lanes": turn, "maxspeed": vmax}

  def summary(self, d: int) -> dict:
    out = {"way": self.id, "class": self.hw, "name": self.name, "ref": self.ref, "oneway": bool(self.oneway),
           "lanes_total": self.lanes, "roundabout": self.junction in ("roundabout", "circular"),
           "bridge": bool(self.bridge), "tunnel": bool(self.tunnel)}
    out.update(self.directed(d))
    if self.destination:
      out["destination"] = self.destination
    if self.surface:
      out["surface"] = self.surface
    if self.lit:
      out["lit"] = self.lit
    return out


class Map:
  def __init__(self, path: str):
    self.db = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
    self._cache: dict[int, Way] = {}
    self.cur: Way | None = None
    self.dir = 1
    self.seg = 0
    self._heading: float | None = None

  def way(self, wid: int) -> Way | None:
    w = self._cache.get(wid)
    if w is None:
      row = self.db.execute("SELECT id, hw, name, ref, oneway, lanes, lanes_fwd, lanes_bwd, turn_fwd, turn_bwd, "
                            "maxspeed, maxspeed_fwd, maxspeed_bwd, junction, bridge, tunnel, destination, lit, surface, "
                            "lat, lon, nodes FROM ways WHERE id=?", (wid,)).fetchone()
      if row is None:
        return None
      if len(self._cache) > 3000:
        self._cache.clear()
      w = self._cache[wid] = Way(row)
    return w

  def _near(self, lat, lon, r) -> list[Way]:
    dlat = r / 111320.0
    dlon = r / (111320.0 * math.cos(math.radians(lat)))
    ids = self.db.execute("SELECT id FROM way_rt WHERE maxlat>=? AND minlat<=? AND maxlon>=? AND minlon<=?",
                          (lat - dlat, lat + dlat, lon - dlon, lon + dlon)).fetchall()
    return [w for (i,) in ids if (w := self.way(i)) is not None]

  def _joined(self, w: Way) -> set[int]:
    ends = (w.nodes[0], w.nodes[-1])
    rows = self.db.execute("SELECT way FROM junction WHERE node IN (?,?)", ends).fetchall()
    return {r[0] for r in rows}

  @staticmethod
  def _project(w: Way, lat, lon):
    """Nearest segment of w to (lat, lon): (distance m, segment index, fraction, segment bearing)."""
    kx = 111320.0 * math.cos(math.radians(lat)); ky = 111320.0
    best = (1e9, 0, 0.0, 0.0)
    for i in range(len(w.lat) - 1):
      ax, ay = (w.lon[i] - lon) * kx, (w.lat[i] - lat) * ky
      bx, by = (w.lon[i + 1] - lon) * kx, (w.lat[i + 1] - lat) * ky
      dx, dy = bx - ax, by - ay
      L2 = dx * dx + dy * dy
      f = 0.0 if L2 < 1e-6 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
      px, py = ax + f * dx, ay + f * dy
      dist = math.hypot(px, py)
      if dist < best[0]:
        best = (dist, i, f, math.degrees(math.atan2(dx, dy)) % 360.0)
    return best

  def match(self, lat: float, lon: float, heading: float, speed: float) -> dict | None:
    if speed >= MIN_HEADING_SPEED or self._heading is None:
      self._heading = heading
    if speed < MIN_HEADING_SPEED and self.cur is not None:
      dist, seg, f, _ = self._project(self.cur, lat, lon)
      self.seg = seg
      return self._state(lat, lon, dist, seg, f)

    joined = self._joined(self.cur) if self.cur is not None else set()
    best = None
    for w in self._near(lat, lon, SEARCH_M):
      dist, seg, f, brg = self._project(w, lat, lon)
      if dist > SEARCH_M:
        continue
      for d in ((1,) if w.oneway == 1 else (-1,) if w.oneway == -1 else (1, -1)):
        a = _angle(heading, brg if d > 0 else (brg + 180.0) % 360.0)
        if speed >= MIN_HEADING_SPEED and a > MAX_ANGLE:
          continue
        score = dist + ANGLE_COST * a
        if self.cur is not None and w.id == self.cur.id:
          score -= STAY_BONUS
        elif w.id in joined:
          score -= JOINED_BONUS
        if best is None or score < best[0]:
          best = (score, w, d, dist, seg, f)
    if best is None:
      self.cur = None
      return None
    _, self.cur, self.dir, dist, self.seg, f = best
    return self._state(lat, lon, dist, self.seg, f)

  def _state(self, lat, lon, dist, seg, f) -> dict:
    out = self.cur.summary(self.dir)
    out["offset_m"] = round(dist, 1)
    out["heading"] = round(self._heading, 1) if self._heading is not None else None
    ahead = self._ahead(seg, f)
    path = ahead.pop("_path")
    out["ahead"] = ahead
    out["path"] = self._car_frame(lat, lon, self._behind(seg, f) + path)
    return out

  def _car_frame(self, lat, lon, path) -> dict:
    """The road as [x ahead, y right] in metres from the GPS fix, every PATH_STEP_M or so,
    with our direction's lane count, an index into `turns` (or -1) and the distance
    along the road from us (negative behind) per point."""
    b = math.radians(self._heading or 0.0)
    cb, sb = math.cos(b), math.sin(b)
    kx = 111320.0 * math.cos(math.radians(lat))
    turns: list[str] = []
    pts = []
    last_s = None
    for la, lo, s, lanes, turn in path:
      if last_s is not None and s - last_s < PATH_STEP_M and s > 0:
        continue
      last_s = s
      dn, de = (la - lat) * 111320.0, (lo - lon) * kx
      if turn and turn not in turns:
        turns.append(turn)
      pts.append([round(dn * cb + de * sb, 1), round(-dn * sb + de * cb, 1), lanes or 0,
                  turns.index(turn) if turn else -1, round(s)])
    return {"pts": pts, "turns": turns}

  def _behind(self, seg: int, f: float) -> list:
    """BEHIND_M of the current way behind us, nearest last, for drawing."""
    w, d = self.cur, self.dir
    info = w.directed(d)
    idx = seg if d > 0 else seg + 1
    out = []
    s = 0.0
    lat0 = w.lat[seg] + f * (w.lat[seg + 1] - w.lat[seg]); lon0 = w.lon[seg] + f * (w.lon[seg + 1] - w.lon[seg])
    prev = (lat0, lon0)
    while 0 <= idx < len(w.lat) and s < BEHIND_M:
      p = (w.lat[idx], w.lon[idx])
      s += self._dist(prev, p)
      out.append((p[0], p[1], -s, info["lanes"], info["turn_lanes"]))
      prev = p
      idx -= d
    out.reverse()
    return out + [(lat0, lon0, 0.0, info["lanes"], info["turn_lanes"])]

  # --- looking ahead -----------------------------------------------------------
  def _ahead(self, seg: int, f: float) -> dict:
    path = []          # (lat, lon, distance from us)
    events = []
    w, d = self.cur, self.dir
    idx = seg + (1 if d > 0 else 0)          # next node in our direction
    lat0, lon0 = w.lat[seg] + f * (w.lat[seg + 1] - w.lat[seg]), w.lon[seg] + f * (w.lon[seg + 1] - w.lon[seg])
    s = 0.0
    prev = (lat0, lon0)
    seen = {w.id}
    last_info = w.directed(d) | {"class": w.hw}
    while s < AHEAD_M:
      while 0 <= idx < len(w.lat) and s < AHEAD_M:
        p = (w.lat[idx], w.lon[idx])
        step = self._dist(prev, p)
        # long straight segments are split so the drawn road can bend smoothly
        # into the next one and lane changes land where they are
        n = max(1, int(step // PATH_STEP_M))
        for j in range(1, n + 1):
          q = (prev[0] + (p[0] - prev[0]) * j / n, prev[1] + (p[1] - prev[1]) * j / n)
          path.append((q[0], q[1], s + step * j / n, last_info["lanes"], last_info["turn_lanes"]))
        s += step
        prev = p
        idx += d
      if s >= AHEAD_M:
        break
      end_node = w.nodes[-1] if d > 0 else w.nodes[0]
      nxt = self._continue(w, d, end_node, seen)
      if nxt is None:
        # Inside a roundabout the exit we take is not knowable from the map.
        if w.junction not in ("roundabout", "circular"):
          events.append({"at_m": round(s), "kind": "end_of_mapped_road"})
        break
      w, d, start = nxt
      seen.add(w.id)
      if w.junction in ("roundabout", "circular"):
        events.append({"at_m": round(s), "kind": "roundabout"})
        break
      info = w.directed(d) | {"class": w.hw}
      for k, label in (("lanes", "lanes"), ("maxspeed", "maxspeed"), ("class", "class")):
        if info[k] != last_info[k] and info[k] is not None:
          events.append({"at_m": round(s), "kind": label, "from": last_info[k], "to": info[k]})
      if info["turn_lanes"] and info["turn_lanes"] != last_info.get("turn_lanes"):
        events.append({"at_m": round(s), "kind": "turn_lanes", "to": info["turn_lanes"]})
      last_info = info
      idx = start + d
    events += self._points_on([(p[0], p[1], p[2]) for p in path])
    events.sort(key=lambda e: e["at_m"])
    return {"horizon_m": round(s), "events": events[:20], "_path": path}

  def _continue(self, w: Way, d: int, node: int, seen: set):
    """Where the road goes on from node, the end of w: (way, direction, index of node in it).

    A roundabout the road runs into is returned whatever the angle, so the
    caller can report it. Otherwise the joining way that bends least, driven
    in a direction it allows. A way may join mid-way: where a road splits
    into a dual carriageway, or carries straight on past a side street.
    """
    end_brg = _bearing(w.lat[-2], w.lon[-2], w.lat[-1], w.lon[-1]) if d > 0 else \
              _bearing(w.lat[1], w.lon[1], w.lat[0], w.lon[0])
    best = None
    for (wid,) in self.db.execute("SELECT way FROM junction WHERE node=?", (node,)).fetchall():
      if wid in seen:
        continue
      o = self.way(wid)
      if o is None:
        continue
      try:
        k = o.nodes.index(node)
      except ValueError:
        continue
      if o.junction in ("roundabout", "circular"):
        return o, 1, k
      options = []
      if k < len(o.nodes) - 1 and o.oneway != -1:
        options.append((1, _bearing(o.lat[k], o.lon[k], o.lat[k + 1], o.lon[k + 1])))
      if k > 0 and o.oneway != 1:
        options.append((-1, _bearing(o.lat[k], o.lon[k], o.lat[k - 1], o.lon[k - 1])))
      for nd, brg in options:
        a = _angle(end_brg, brg)
        if a < 70 and (best is None or a < best[0]):
          best = (a, o, nd, k)
    return None if best is None else best[1:]

  def _points_on(self, path) -> list:
    if len(path) < 2:
      return []
    lats = [p[0] for p in path]; lons = [p[1] for p in path]
    pad = POINT_ON_PATH_M / 111320.0
    rows = self.db.execute(
      "SELECT p.id, p.kind, p.lat, p.lon, p.tags FROM point_rt r JOIN points p ON p.id = r.id "
      "WHERE r.maxlat>=? AND r.minlat<=? AND r.maxlon>=? AND r.minlon<=?",
      (min(lats) - pad, max(lats) + pad, min(lons) - 2 * pad, max(lons) + 2 * pad)).fetchall()
    out = []
    for pid, kind, plat, plon, tags in rows:
      best = None
      for la, lo, s in path:
        dd = self._dist((la, lo), (plat, plon))
        if best is None or dd < best[0]:
          best = (dd, s)
      if best and best[0] <= POINT_ON_PATH_M:
        e = {"at_m": round(best[1]), "kind": kind}
        if tags and tags != "{}":
          e["tags"] = tags
        out.append(e)
    # A signal is often mapped once per direction, a few metres apart.
    out.sort(key=lambda e: e["at_m"])
    kept = []
    for e in out:
      if not any(k["kind"] == e["kind"] and e["at_m"] - k["at_m"] < 10 for k in kept):
        kept.append(e)
    return kept

  @staticmethod
  def _dist(a, b) -> float:
    kx = 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot((b[1] - a[1]) * kx, (b[0] - a[0]) * 111320.0)
