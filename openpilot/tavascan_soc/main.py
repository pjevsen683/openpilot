#!/usr/bin/env python3
"""Publicerer Cupra Tavascans HV-batteri-SoC til MQTT for Home Assistant.

Kører på comma-enheden og SENDER data ud. Der er bevidst ingen vej ind i
enheden: ingen SSH, ingen lyttende port, ingen credentials — brokeren er
anonym.

Processen kører KUN offroad (only_offroad), så den ikke belaster enheden
under kørsel. Det koster ingenting: alle vinduer hvor bussen er vågen og
SoC ændrer sig — opladning og de ~15 min efter parkering — er offroad.

Bilens CAN-bus sover når den holder parkeret. Sidste kendte værdi bevares
derfor, og en age-sensor viser hvor gammel den er.

Signalet er HVEM_02 (0x5AC) / HVEM_Nutzbare_Energie: bit 32, 11 bit,
little endian.

energy_kwh er kun vejledende. Forholdet mellem den rå værdi og bilens viste
SoC er IKKE proportionalt, så hverken DBC'ens 50 Wh eller de 62,5 Wh vi først
gættede på giver en troværdig kapacitet (68,9 hhv. 86,1 kWh mod bilens 77).
SoC beregnes derfor af en lineær kalibrering mod displayet, se SOC_A/SOC_B.
Se BATTERI-SOC-NOTER.md.
"""
import json
import os
import time

from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import Ratekeeper
from openpilot.cereal import messaging
from openpilot.tavascan_soc.mqtt_min import publish

MQTT_HOST = os.getenv("TAVASCAN_MQTT_HOST", "10.0.1.198")
MQTT_PORT = int(os.getenv("TAVASCAN_MQTT_PORT", "1883"))
INTERVAL_S = float(os.getenv("TAVASCAN_INTERVAL", "60"))
SAMPLE_S = float(os.getenv("TAVASCAN_SAMPLE_S", "6"))
WH_PER_COUNT = float(os.getenv("TAVASCAN_WH_PER_COUNT", "62.5"))
BATTERY_KWH = float(os.getenv("TAVASCAN_BATTERY_KWH", "77"))

# SoC kalibreres lineaert mod bilens eget display: soc_pct = SOC_A * raw + SOC_B.
# En ren proportional model (energi/kapacitet) passer IKKE - to maalepunkter mod
# displayet gav forholdet 1.625 i raa vaerdi men kun 1.425 i SoC.
# Punkterne: raw 483 -> ~40 % (upraecist aflaest), raw 785 -> 57 % (praecist).
# PROVISORISK: to punkter definerer altid en linje. Et tredje punkt ved en meget
# anden ladetilstand (over 80 % eller under 25 %) vil vise om offsettet er reelt.
SOC_A = float(os.getenv("TAVASCAN_SOC_A", "0.056291"))
SOC_B = float(os.getenv("TAVASCAN_SOC_B", "12.81"))

ENERGY_ADDR = 0x5AC          # HVEM_02
RAW_INVALID_MIN = 2040       # 2046/2047 ses som ugyldige udfald

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

SENSORS = [
  ("soc", "Tavascan SoC", "%", "battery", "{{ value_json.soc_pct }}", True),
  ("energy", "Tavascan Batterienergi", "kWh", "energy_storage", "{{ value_json.energy_kwh }}", True),
  ("age", "Tavascan SoC alder", "s", "duration", "{{ value_json.age_s }}", False),
]
BINARY_SENSORS = [
  ("awake", "Tavascan CAN vaagen", "{{ value_json.awake }}"),
]


def discovery_messages() -> list[tuple[str, str]]:
  msgs = []
  for key, name, unit, dclass, tpl, measurement in SENSORS:
    cfg = {
      "name": name,
      "unique_id": f"{DEV_ID}_{key}",
      "state_topic": STATE_TOPIC,
      "availability_topic": AVAIL_TOPIC,
      "unit_of_measurement": unit,
      "device_class": dclass,
      "value_template": tpl,
      "device": DEVICE,
    }
    if measurement:
      cfg["state_class"] = "measurement"
    msgs.append((f"homeassistant/sensor/{DEV_ID}_{key}/config", json.dumps(cfg)))

  for key, name, tpl in BINARY_SENSORS:
    cfg = {
      "name": name,
      "unique_id": f"{DEV_ID}_{key}",
      "state_topic": STATE_TOPIC,
      "availability_topic": AVAIL_TOPIC,
      "value_template": tpl,
      "payload_on": "True",
      "payload_off": "False",
      "device": DEVICE,
    }
    msgs.append((f"homeassistant/binary_sensor/{DEV_ID}_{key}/config", json.dumps(cfg)))
  return msgs


def sample_energy_raw(sock, duration: float) -> tuple[bool, int | None]:
  """Returnerer (bussen_var_vaagen, median_raa_vaerdi_eller_None)."""
  raw_vals: list[int] = []
  frames = 0
  t0 = time.monotonic()
  while time.monotonic() - t0 < duration:
    for msg in messaging.drain_sock(sock):
      for c in msg.can:
        frames += 1
        if c.address == ENERGY_ADDR:
          d = int.from_bytes(bytes(c.dat), "little")
          v = (d >> 32) & 0x7FF
          if 0 < v < RAW_INVALID_MIN:
            raw_vals.append(v)
    time.sleep(0.01)

  if not raw_vals:
    return frames > 0, None
  raw_vals.sort()
  return True, raw_vals[len(raw_vals) // 2]


def main() -> None:
  sock = messaging.sub_sock("can", timeout=100)
  discovery_sent = False
  last_raw: int | None = None
  last_ts: float | None = None
  rk = Ratekeeper(1.0 / INTERVAL_S)

  cloudlog.info(f"tavascan_soc: sender til {MQTT_HOST}:{MQTT_PORT} hvert {INTERVAL_S:.0f}s")

  while True:
    awake, raw = sample_energy_raw(sock, SAMPLE_S)
    if raw is not None:
      last_raw = raw
      last_ts = time.time()

    energy_kwh = round(last_raw * WH_PER_COUNT / 1000.0, 2) if last_raw else None
    state = {
      "awake": awake,
      "raw": last_raw,
      "energy_kwh": energy_kwh,
      "soc_pct": round(min(100.0, max(0.0, SOC_A * last_raw + SOC_B)), 1) if last_raw else None,
      "age_s": int(time.time() - last_ts) if last_ts else None,
      "fresh": raw is not None,
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
      # Forventet når bilen ikke er på hjemme-WiFi. Ikke en fejl.
      cloudlog.debug(f"tavascan_soc: broker ikke naaet ({e})")
      discovery_sent = False

    rk.keep_time()


if __name__ == "__main__":
  main()
