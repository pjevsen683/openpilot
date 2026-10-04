#!/usr/bin/env python3
"""The part of MEB_Side_Assist_02 (0x24D, bus 0) that we can read.

Decoded against 89 cars overtaking us on 2026-10-01, whose position and speed
were known from the front radar after they passed. What came out:

  bytes 0..15   header, not understood
  bytes 16..63  six 8-byte object slots, at 16, 24, 32, 40, 48 and 56

A slot is empty when it is all 0xFF, or when byte +4 is 0xFF and byte +1 has its
top bit clear. In an occupied slot:

  +0  closing speed, (value - 143) * 0.2 m/s, positive = gaining on us. Median
      error +0.1 km/h against the front radar, half within 2 km/h, and it does
      not follow our own speed.
  +4  a code whose high nibble follows the slot pair (3, 2, 1), not the side
  +5  a per-object counter, one or two per frame -- NOT distance, although it
      tracks distance at r=0.99 within any one pass

Distance and side are not in any slot or header field: every 4-16 bit field,
both byte orders, signed and unsigned, was tested against range, azimuth and
rotated coordinates from the rear corner. The other slot bytes barely move
while a car covers 40 m, so there is nowhere for a position to be.

With one car behind us there are usually two to four occupied slots. The extra
ones are not clutter: they close at 5-24 km/h, so they are traffic further back.
"""

SLOTS = (16, 24, 32, 40, 48, 56)
SPEED_ZERO = 143
SPEED_STEP_MS = 0.2


def objects(raw: bytes) -> list[dict]:
  """Occupied slots in one 0x24D frame, each with its closing speed."""
  out = []
  for s in SLOTS:
    if len(raw) < s + 8:
      break
    sl = raw[s:s + 8]
    if not (sl[1] & 0x80) or sl[4] == 0xFF:
      continue
    out.append({
      "slot": s,
      "closing_kph": round((sl[0] - SPEED_ZERO) * SPEED_STEP_MS * 3.6, 1),
      "code": sl[4],
    })
  return out
