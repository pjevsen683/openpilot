#!/usr/bin/env python3
"""Publishes data from a Cupra Tavascan to MQTT for Home Assistant.

Runs on the comma device and PUSHES data out. There is deliberately no way in:
no SSH, no listening port, no credentials -- the broker is anonymous.

The process runs ONLY offroad (only_offroad) so it does not load the device
while driving. That costs nothing: every window in which the bus is awake and
the values change -- charging, and the ~15 min after parking -- is offroad.

The car's CAN bus sleeps while parked. The last known value is therefore kept
per signal, and an age sensor shows how old the reading is. The 12V voltage is
the exception: it comes from the panda, not from CAN, and stays fresh even
while the bus sleeps.

SoC: the ratio between the raw value and the car's displayed SoC is NOT
proportional, so neither the DBC's 50 Wh nor the 62.5 Wh we first guessed give
a believable capacity (68.9 and 86.1 kWh respectively, against the car's 77).
SoC is therefore computed from a linear calibration against the display, see
SOC_A/SOC_B. See BATTERI-SOC-NOTER.md.
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

# SoC is calibrated linearly against the car's own display: soc_pct = SOC_A * raw + SOC_B.
# Fitted on six readings of the car's display during a single long charge at
# constant power, which gave a clean span from 40 to 85 %:
#   raw 493->40, 699->53, 785->57, 925->67, 1114->80, 1195->85
# Largest deviation 1.3 pp. The slopes between neighbouring points (0.047-0.071)
# vary without any systematic trend, so a straight line is the right model -- the
# scatter is rounding noise from the display's whole percentages.
# The offset of about +11 % is real, not an artefact: ratios in the raw value do
# not match ratios in SoC, which they would if SoC were simply energy/capacity.
# There is a reserve below the display's zero.
# OPEN: the extrapolation above 85 % and below 25 % is untested -- which is why
# the raw value is also exposed as its own sensor, so it can be logged and
# refitted over a wider range.
SOC_A = float(os.getenv("TAVASCAN_SOC_A", "0.064640"))
SOC_B = float(os.getenv("TAVASCAN_SOC_B", "7.53"))

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

# key, name, unit, device_class, field in state, state_class
SENSORS = [
  ("soc", "Tavascan SoC", "%", "battery", "soc_pct", "measurement"),
  # Uncalibrated raw counter from HVEM_02. Exposed on purpose so it can be logged
  # in HA's long-term statistics and compared against the car's own display over a
  # wide SoC range.
  ("raw", "Tavascan SoC raw", None, None, "raw", "measurement"),
  ("odometer", "Tavascan Odometer", "km", "distance", "odometer_km", "total_increasing"),
  ("climate_power", "Tavascan Climate Power", "W", "power", "climate_w", "measurement"),
  ("volt12", "Tavascan 12V Battery", "V", "voltage", "volt12", "measurement"),
  ("age", "Tavascan CAN Age", "s", "duration", "age_s", None),
]

# key, name, field in state, device_class
BINARY_SENSORS = [
  ("awake", "Tavascan CAN Awake", "awake", None),
  ("locked", "Tavascan Locked", "locked", "lock"),
  ("door_open", "Tavascan Door Open", "any_door_open", "door"),
]


def discovery_messages() -> list[tuple[str, str]]:
  msgs = []
  for key, name, unit, dclass, field, sclass in SENSORS:
    cfg = {
      "name": name,
      "unique_id": f"{DEV_ID}_{key}",
      "state_topic": STATE_TOPIC,
      "availability_topic": AVAIL_TOPIC,
      "value_template": "{{ value_json.%s }}" % field,
      "device": DEVICE,
    }
    if unit:
      cfg["unit_of_measurement"] = unit
    if dclass:
      cfg["device_class"] = dclass
    if sclass:
      cfg["state_class"] = sclass
    msgs.append((f"homeassistant/sensor/{DEV_ID}_{key}/config", json.dumps(cfg)))

  for key, name, field, dclass in BINARY_SENSORS:
    # HA's lock class is inverted relative to our field: on means UNLOCKED.
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
  """Returns (bus_was_awake, decoded_signals)."""
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

  cloudlog.info(f"tavascan_soc: publishing to {MQTT_HOST}:{MQTT_PORT} every {INTERVAL_S:.0f}s")

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
      # Expected when the car is not on the home WiFi. Not an error.
      cloudlog.debug(f"tavascan_soc: broker unreachable ({e})")
      discovery_sent = False

    rk.keep_time()


if __name__ == "__main__":
  main()
