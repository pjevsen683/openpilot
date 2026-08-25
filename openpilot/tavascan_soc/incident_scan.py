#!/usr/bin/env python3
"""Gennemgår afsluttede ruter for det EPB-mønster der sendte bilen i P 25/8,
og opgør samtidig hvor stor en del af tiden longitudinal styres af e2e.

Baggrund, se TAVASCAN-PROGRESS.md: openpilot kan ikke kommandere gear eller
parkeringsbremse, men dens holde-request ved stilstand indgår i kæden.
mebcan.get_acc_hold_type() har eksplicit beskyttelse mod "car error with EPB at
low speed", og den slog ikke til. Mønsteret var:

    parkingBrake False -> True   ved v < 5 km/t mens gear == drive
    accFaulted   -> True         inden for faa sekunder
    gear         -> park

Kører kun offroad, og kun paa segmenter der ikke er scannet foer. Resultatet
sendes til MQTT, saa det dukker op i Home Assistant frem for at skulle opdages
i trafikken.
"""
import glob
import json
import os
import time

from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import Ratekeeper
from openpilot.tavascan_soc.mqtt_min import publish

MQTT_HOST = os.getenv("TAVASCAN_MQTT_HOST", "10.0.1.198")
MQTT_PORT = int(os.getenv("TAVASCAN_MQTT_PORT", "1883"))
INTERVAL_S = float(os.getenv("TAVASCAN_SCAN_INTERVAL", "300"))
SEGMENTS_PER_CYCLE = int(os.getenv("TAVASCAN_SCAN_BATCH", "3"))

STATE_PATH = "/data/tavascan_scan_state.json"
REALDATA = "/data/media/0/realdata"

TOPIC = "tavascan/incidents"
STATE_TOPIC = TOPIC + "/state"
AVAIL_TOPIC = TOPIC + "/availability"
DEV_ID = "tavascan_soc"          # samme enhed i HA som SoC-sensorerne

LOW_SPEED_MS = 5 / 3.6           # EPB-faelden ses kun ved meget lav fart
LINK_WINDOW_S = 5.0              # hvor taet accFaulted skal foelge parkingBrake


def discovery_messages() -> list[tuple[str, str]]:
  device = {
    "identifiers": [DEV_ID],
    "name": "Cupra Tavascan",
    "manufacturer": "CUPRA",
    "model": "Tavascan (via comma CAN)",
  }
  msgs = []
  for key, name, unit, field, sclass in [
    ("epb_events", "Tavascan EPB-haendelser", None, "total_epb_events", "total_increasing"),
    ("e2e_share", "Tavascan e2e-andel", "%", "last_e2e_pct", "measurement"),
    ("scanned", "Tavascan scannede segmenter", None, "scanned", "total_increasing"),
  ]:
    cfg = {
      "name": name,
      "unique_id": f"{DEV_ID}_{key}",
      "state_topic": STATE_TOPIC,
      "availability_topic": AVAIL_TOPIC,
      "value_template": "{{ value_json.%s }}" % field,
      "device": device,
      "state_class": sclass,
    }
    if unit:
      cfg["unit_of_measurement"] = unit
    msgs.append((f"homeassistant/sensor/{DEV_ID}_{key}/config", json.dumps(cfg)))

  cfg = {
    "name": "Tavascan EPB-haendelse sidste tur",
    "unique_id": f"{DEV_ID}_epb_last",
    "state_topic": STATE_TOPIC,
    "availability_topic": AVAIL_TOPIC,
    "value_template": "{{ value_json.last_route_had_epb }}",
    "payload_on": "True",
    "payload_off": "False",
    "device_class": "problem",
    "device": device,
  }
  msgs.append((f"homeassistant/binary_sensor/{DEV_ID}_epb_last/config", json.dumps(cfg)))
  return msgs


def load_state() -> dict:
  try:
    with open(STATE_PATH) as f:
      return json.load(f)
  except (OSError, ValueError):
    return {"done": [], "total_epb_events": 0, "scanned": 0, "incidents": []}


def save_state(st: dict) -> None:
  st["done"] = st["done"][-4000:]
  st["incidents"] = st["incidents"][-40:]
  tmp = STATE_PATH + ".tmp"
  with open(tmp, "w") as f:
    json.dump(st, f)
  os.replace(tmp, STATE_PATH)


def scan_segment(path: str) -> dict:
  """Returnerer fund for eet segment. Importeres dovent: LogReader er tung."""
  from openpilot.tools.lib.logreader import LogReader

  res = {"epb": [], "engaged_n": 0, "e2e_n": 0, "src_n": 0}
  t0 = None
  v = 0.0
  gear = ""
  pb_prev = None
  pb_t = None
  active = False

  for msg in LogReader(path):
    try:
      w = msg.which()
      t = msg.logMonoTime / 1e9
      if t0 is None:
        t0 = t
      rel = t - t0

      if w == "selfdriveState":
        active = msg.selfdriveState.active
        if active:
          res["engaged_n"] += 1
      elif w == "longitudinalPlan" and active:
        res["src_n"] += 1
        if str(msg.longitudinalPlan.longitudinalPlanSource) == "e2e":
          res["e2e_n"] += 1
      elif w == "carState":
        cs = msg.carState
        v = cs.vEgo
        gear = str(cs.gearShifter)
        pb = cs.parkingBrake
        if pb_prev is False and pb and v < LOW_SPEED_MS and gear == "drive":
          pb_t = rel
          res["epb"].append({"t": round(rel, 2), "kmh": round(v * 3.6, 1),
                             "engaged": active, "accfault": False, "to_park": False})
        pb_prev = pb
        if pb_t is not None and rel - pb_t <= LINK_WINDOW_S:
          if gear == "park":
            res["epb"][-1]["to_park"] = True
          if cs.accFaulted:
            res["epb"][-1]["accfault"] = True
    except Exception:
      continue
  return res


def main() -> None:
  st = load_state()
  done = set(st["done"])
  rk = Ratekeeper(1.0 / INTERVAL_S)
  discovery_sent = False
  last_route = None
  last_e2e = None
  last_epb = False

  cloudlog.info("tavascan incident_scan: startet")

  while True:
    segs = sorted(glob.glob(f"{REALDATA}/*--*/rlog.zst"))
    todo = [s for s in segs if s not in done][:SEGMENTS_PER_CYCLE]

    for path in todo:
      try:
        r = scan_segment(path)
      except Exception as e:
        cloudlog.warning(f"incident_scan: {path} fejlede ({e})")
        done.add(path)
        continue

      done.add(path)
      st["scanned"] += 1
      route = path.split("/")[-2].rsplit("--", 1)[0]
      last_route = route

      if r["src_n"] > 20:
        last_e2e = round(100.0 * r["e2e_n"] / r["src_n"], 1)

      for hit in r["epb"]:
        st["total_epb_events"] += 1
        hit["route"] = route
        hit["segment"] = path.split("/")[-2]
        st["incidents"].append(hit)
        last_epb = True
        cloudlog.error(f"tavascan incident_scan: EPB-moenster i {hit['segment']} "
                       f"t={hit['t']}s v={hit['kmh']}km/t engaged={hit['engaged']} "
                       f"accfault={hit['accfault']} to_park={hit['to_park']}")

    if todo:
      st["done"] = list(done)
      save_state(st)

    recent = st["incidents"][-1] if st["incidents"] else None
    state = {
      "total_epb_events": st["total_epb_events"],
      "scanned": st["scanned"],
      "last_route": last_route,
      "last_e2e_pct": last_e2e,
      "last_route_had_epb": last_epb,
      "queue": max(0, len(segs) - len(done)),
      "latest_incident": recent,
    }

    msgs = []
    if not discovery_sent:
      msgs += discovery_messages()
    msgs.append((AVAIL_TOPIC, "online"))
    msgs.append((STATE_TOPIC, json.dumps(state)))
    try:
      publish(MQTT_HOST, MQTT_PORT, "tavascan-scan", msgs)
      discovery_sent = True
    except OSError:
      discovery_sent = False

    last_epb = False
    rk.keep_time()


if __name__ == "__main__":
  main()
