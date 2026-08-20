#!/usr/bin/env python3
"""Publicerer data fra Cupra Tavascan til MQTT for Home Assistant.

Kører på comma-enheden og SENDER data ud. Der er bevidst ingen vej ind i
enheden: ingen SSH, ingen lyttende port, ingen credentials — brokeren er
anonym.

Processen kører KUN offroad (only_offroad), så den ikke belaster enheden
under kørsel. Det koster ingenting: alle vinduer hvor bussen er vågen og
værdierne ændrer sig — opladning og de ~15 min efter parkering — er offroad.

Bilens CAN-bus sover når den holder parkeret. Sidste kendte værdi bevares
derfor pr. signal, og en age-sensor viser hvor gammel aflæsningen er.
12V-spændingen er undtagelsen: den kommer fra panda'en, ikke fra CAN, og er
frisk også mens bussen sover.

SoC: forholdet mellem den rå værdi og bilens viste SoC er IKKE proportionalt,
så hverken DBC'ens 50 Wh eller de 62,5 Wh vi først gættede på giver en
troværdig kapacitet (68,9 hhv. 86,1 kWh mod bilens 77). SoC beregnes derfor af
en lineær kalibrering mod displayet, se SOC_A/SOC_B. Se BATTERI-SOC-NOTER.md.
"""
import json
import os
import time

from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import Ratekeeper
from openpilot.cereal import messaging
from openpilot.tavascan_soc.mqtt_min import publish
from openpilot.tavascan_soc import signals

MQTT_HOST = os.getenv("TAVASCAN_MQTT_HOST", "10.0.1.198")
MQTT_PORT = int(os.getenv("TAVASCAN_MQTT_PORT", "1883"))
INTERVAL_S = float(os.getenv("TAVASCAN_INTERVAL", "60"))
SAMPLE_S = float(os.getenv("TAVASCAN_SAMPLE_S", "6"))

# SoC kalibreres lineaert mod bilens eget display: soc_pct = SOC_A * raw + SOC_B.
# Punkterne: raw 483 -> ~40 % (upraecist aflaest), raw 785 -> 57 % (praecist).
# PROVISORISK: to punkter definerer altid en linje. Et tredje punkt over 80 %
# eller under 25 % vil vise om offsettet paa +12,8 % er reelt.
SOC_A = float(os.getenv("TAVASCAN_SOC_A", "0.056291"))
SOC_B = float(os.getenv("TAVASCAN_SOC_B", "12.81"))

TOPIC = "tavascan/soc"
STATE_TOPIC = TOPIC + "/state"
AVAIL_TOPIC = TOPIC + "/availability"
DEV_ID = "tavascan_soc"

DEVICE = {
  "identifiers": [DEV_ID],
  "name": "Cupra Tavascan",
  "manufacturer": "CUPRA",
  "model": "Tavascan (via comma CAN)",
}

# key, navn, enhed, device_class, felt i state, state_class
SENSORS = [
  ("soc", "Tavascan SoC", "%", "battery", "soc_pct", "measurement"),
  ("odometer", "Tavascan Kilometerstand", "km", "distance", "odometer_km", "total_increasing"),
  ("climate_power", "Tavascan Klimaeffekt", "W", "power", "climate_w", "measurement"),
  ("volt12", "Tavascan 12V batteri", "V", "voltage", "volt12", "measurement"),
  ("age", "Tavascan CAN alder", "s", "duration", "age_s", None),
]

# key, navn, felt i state, device_class
BINARY_SENSORS = [
  ("awake", "Tavascan CAN vaagen", "awake", None),
  ("locked", "Tavascan laast", "locked", "lock"),
  ("door_open", "Tavascan doer aaben", "any_door_open", "door"),
]


def discovery_messages() -> list[tuple[str, str]]:
  msgs = []
  for key, name, unit, dclass, field, sclass in SENSORS:
    cfg = {
      "name": name,
      "unique_id": f"{DEV_ID}_{key}",
      "state_topic": STATE_TOPIC,
      "availability_topic": AVAIL_TOPIC,
      "unit_of_measurement": unit,
      "device_class": dclass,
      "value_template": "{{ value_json.%s }}" % field,
      "device": DEVICE,
    }
    if sclass:
      cfg["state_class"] = sclass
    msgs.append((f"homeassistant/sensor/{DEV_ID}_{key}/config", json.dumps(cfg)))

  for key, name, field, dclass in BINARY_SENSORS:
    # HA's lock-klasse er omvendt af vores felt: on betyder ULAAST.
    on, off = ("False", "True") if dclass == "lock" else ("True", "False")
    cfg = {
      "name": name,
      "unique_id": f"{DEV_ID}_{key}",
      "state_topic": STATE_TOPIC,
      "availability_topic": AVAIL_TOPIC,
      "value_template": "{{ value_json.%s }}" % field,
      "payload_on": on,
      "payload_off": off,
      "device": DEVICE,
    }
    if dclass:
      cfg["device_class"] = dclass
    msgs.append((f"homeassistant/binary_sensor/{DEV_ID}_{key}/config", json.dumps(cfg)))
  return msgs


def sample(sock, duration: float) -> tuple[bool, dict]:
  """Returnerer (bussen_var_vaagen, dekodede_signaler)."""
  frames: dict[int, list[bytes]] = {a: [] for a in signals.WANTED_ADDRS}
  seen = 0
  t0 = time.monotonic()
  while time.monotonic() - t0 < duration:
    for msg in messaging.drain_sock(sock):
      for c in msg.can:
        seen += 1
        if c.address in frames:
          frames[c.address].append(bytes(c.dat))
    time.sleep(0.01)
  return seen > 0, signals.decode(frames)


def main() -> None:
  can_sock = messaging.sub_sock("can", timeout=100)
  sm = messaging.SubMaster(["pandaStates"])
  discovery_sent = False
  latest: dict = {}
  last_ts: float | None = None
  rk = Ratekeeper(1.0 / INTERVAL_S)

  cloudlog.info(f"tavascan_soc: sender til {MQTT_HOST}:{MQTT_PORT} hvert {INTERVAL_S:.0f}s")

  while True:
    awake, decoded = sample(can_sock, SAMPLE_S)
    if decoded:
      latest.update(decoded)
      last_ts = time.time()

    volt = None
    try:
      sm.update(0)
      for ps in sm["pandaStates"]:
        if ps.voltage:
          volt = round(ps.voltage / 1000.0, 2)
    except Exception:
      pass

    raw = latest.get("energy_raw")
    doors = [latest.get(k) for k in ("door_driver", "door_pass", "door_rear_l", "door_rear_r", "tailgate")]
    known = [d for d in doors if d is not None]

    state = {
      "awake": awake,
      "raw": raw,
      "soc_pct": round(min(100.0, max(0.0, SOC_A * raw + SOC_B)), 1) if raw else None,
      "odometer_km": int(latest["odometer_km"]) if "odometer_km" in latest else None,
      "climate_w": int(latest["climate_w"]) if "climate_w" in latest else None,
      "locked": latest.get("locked"),
      "any_door_open": (any(known) if known else None),
      "volt12": volt,
      "age_s": int(time.time() - last_ts) if last_ts else None,
      "fresh": bool(decoded),
    }

    msgs = []
    if not discovery_sent:
      msgs += discovery_messages()
    msgs.append((AVAIL_TOPIC, "online"))
    msgs.append((STATE_TOPIC, json.dumps(state)))

    try:
      publish(MQTT_HOST, MQTT_PORT, "tavascan-comma", msgs)
      discovery_sent = True
    except OSError as e:
      # Forventet naar bilen ikke er paa hjemme-WiFi. Ikke en fejl.
      cloudlog.debug(f"tavascan_soc: broker ikke naaet ({e})")
      discovery_sent = False

    rk.keep_time()


if __name__ == "__main__":
  main()
