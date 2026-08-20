"""Signaldefinitioner for de CAN-beskeder vi aflæser på Cupra Tavascan (MEB GEN2).

Alle bitpositioner er little endian, som i DBC'en.

Flere af signalerne sender indimellem deres maksimalværdi som "ugyldig" — vi så
Nutzbare_Energie sende 2046 og Kapazitaet sende 409,4 Ah (= 2047 × 0,2). Derfor
har hvert numerisk signal en gyldighedsgrænse, og der tages median over
måleperioden. Uden det får man spring på flere procent uden grund.
"""

ADDR_HVEM_02 = 0x5AC       # energi og klima-effekt
ADDR_DIAGNOSE_01 = 0x6B2   # kilometerstand
ADDR_ZV_02 = 0x583         # centrallås og døre


def _bits(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


# navn -> (adresse, startbit, antal bit, skala, min_gyldig_raa, maks_gyldig_raa)
# Klimaeffekt har min 0: nul watt betyder "klima slukket" og er en gyldig
# aflaesning. Energi og kilometerstand har min 1, da nul dér er usandsynligt
# og i praksis betyder at signalet ikke er udfyldt endnu.
NUMERIC = {
  "energy_raw":   (ADDR_HVEM_02, 32, 11, 1.0, 1, 2040),
  "climate_w":    (ADDR_HVEM_02, 24, 8, 50.0, 0, 254),
  "odometer_km":  (ADDR_DIAGNOSE_01, 8, 20, 1.0, 1, 1048570),
}

# Bit-signaler: navn -> (adresse, bit)
BOOLEAN = {
  # ZV_verriegelt_extern_ist. Verificeret i BEGGE retninger 20/8: app-oplaasning
  # gav 1->0, bilens auto-genlaasning 45 s senere gav 0->1. To skift i 1602
  # beskeder, ingen falske udslag.
  "locked":       (ADDR_ZV_02, 17),
  "door_driver":  (ADDR_ZV_02, 24),
  "door_pass":    (ADDR_ZV_02, 25),
  "door_rear_l":  (ADDR_ZV_02, 26),
  "door_rear_r":  (ADDR_ZV_02, 27),
  "tailgate":     (ADDR_ZV_02, 28),
}

WANTED_ADDRS = {ADDR_HVEM_02, ADDR_DIAGNOSE_01, ADDR_ZV_02}


def decode(frames: dict[int, list[bytes]]) -> dict:
  """frames: adresse -> liste af payloads set i maaleperioden.

  Numeriske vaerdier returneres som median over gyldige aflaesninger.
  Bit-vaerdier returneres som den hyppigste vaerdi.
  Signaler uden gyldige data udelades helt, saa kalderen kan beholde en
  tidligere vaerdi frem for at publicere noget forkert.
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
