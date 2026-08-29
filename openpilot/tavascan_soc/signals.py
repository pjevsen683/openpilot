"""Signal definitions for the CAN messages we read on a Cupra Tavascan (MEB GEN2).

All bit positions are little endian, as in the DBC.

Several of the signals occasionally send their maximum value to mean "invalid" --
we saw Nutzbare_Energie send 2046 and Kapazitaet send 409.4 Ah (= 2047 * 0.2).
Each numeric signal therefore has a validity range, and the median is taken over
the sampling window. Without that you get jumps of several percent for no reason.
"""

ADDR_HVEM_02 = 0x5AC       # energy and climate power
ADDR_DIAGNOSE_01 = 0x6B2   # odometer
ADDR_ZV_02 = 0x583         # central locking and doors


def _bits(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


# name -> (address, start bit, bit count, scale, min_valid_raw, max_valid_raw)
# Climate power has min 0: zero watts means "climate off" and is a valid reading.
# Energy and odometer have min 1, since zero there is implausible and in practice
# means the signal has not been populated yet.
NUMERIC = {
  "energy_raw":   (ADDR_HVEM_02, 32, 11, 1.0, 1, 2040),
  "climate_w":    (ADDR_HVEM_02, 24, 8, 50.0, 0, 254),
  "odometer_km":  (ADDR_DIAGNOSE_01, 8, 20, 1.0, 1, 1048570),
}

# Bit signals: name -> (address, bit)
BOOLEAN = {
  # ZV_verriegelt_extern_ist. Verified in BOTH directions on 2025-08-20: unlocking
  # from the app gave 1->0, the car's auto-relock 45 s later gave 0->1. Two
  # transitions across 1602 messages, no spurious flips.
  "locked":       (ADDR_ZV_02, 17),
  "door_driver":  (ADDR_ZV_02, 24),
  "door_pass":    (ADDR_ZV_02, 25),
  "door_rear_l":  (ADDR_ZV_02, 26),
  "door_rear_r":  (ADDR_ZV_02, 27),
  "tailgate":     (ADDR_ZV_02, 28),
}

WANTED_ADDRS = {ADDR_HVEM_02, ADDR_DIAGNOSE_01, ADDR_ZV_02}


def decode(frames: dict[int, list[bytes]]) -> dict:
  """frames: address -> list of payloads seen during the sampling window.

  Numeric values are returned as the median over valid readings.
  Bit values are returned as the most common value.
  Signals with no valid data are omitted entirely, so the caller can keep a
  previous value rather than publish something wrong.
  """
  out: dict = {}

  for name, (addr, start, length, scale, min_valid, max_valid) in NUMERIC.items():
    vals = [_bits(d, start, length) for d in frames.get(addr, [])]
    vals = [v for v in vals if min_valid <= v <= max_valid]
    if vals:
      vals.sort()
      out[name] = vals[len(vals) // 2] * scale

  for name, (addr, bit) in BOOLEAN.items():
    vals = [(int.from_bytes(d, "little") >> bit) & 1 for d in frames.get(addr, [])]
    if vals:
      out[name] = bool(round(sum(vals) / len(vals)))

  return out
