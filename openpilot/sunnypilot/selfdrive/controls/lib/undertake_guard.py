#!/usr/bin/env python3
"""Forhindrer overhaling indenom ved at lægge et fartloft efter venstre vognbane.

I Danmark må man ikke overhale indenom. Bilens egen Travel Assist sænker farten
når der ligger et langsommere køretøj i venstre spor foran; openpilot gør ikke,
fordi planlæggeren udelukkende ser køretøjer i ens EGEN bane
(modelV2.leadsV3 -> radarState.leadOne/leadTwo).

Bilens radar rapporterer derimod objekter opdelt paa vognbane i Strukturen_01
(0x24F, 25 Hz paa bus 2): to objekter i hver af samme/venstre/hoejre bane, med
afstand, sideafstand og relativ hastighed. Verificeret mod en optagelse fra
motorvejen: venstre-bane-objekter dukkede op med sideafstand ca. +4,0 m og
relative hastigheder der stemte med trafikken.

openpilot saetter VolkswagenFlags.DISABLE_RADAR paa denne bil (networkLocation
er fwdCamera), saa radar_interface returnerer None og beskeden aldrig laeses.
Vi laeser den derfor selv. Vi sender intet -- kun aflytning.

DESIGN
  * Slaaet FRA som standard. Kraever parameteren UndertakeGuard.
  * Saetter kun et LOFT paa den oenskede fart. Bremser ikke selv, og kan ikke
    faa bilen til at accelerere. Den eksisterende MPC klarer nedbremsningen.
  * Virker kun over MIN_SPEED. I kø er indenom-passage tilladt, og funktionen
    ville vaere til gene.
  * Roerer ikke stop og igangsaetning. Vi har en uafklaret fejl i EPB-overgangen
    ved lave hastigheder, og den skal ikke kunne paavirkes herfra.

AFGRAENSNING
  Modulet laeser raa CAN i plannerd, hvilket ikke er saedvanligt der. Alternativet
  var at udvide cereal-skemaet, opendbc og bilaget -- tre repoer og en
  skemaaendring. Det her er nemmere at gennemgaa og nemmere at fjerne igen.
  Naar funktionen er slaaet fra, oprettes socket'en slet ikke.
"""
import os

from openpilot.cereal import messaging
from openpilot.common.params import Params
from opendbc.car.common.conversions import Conversions as CV

RADAR_ADDR = 0x24F
RADAR_BUS = 2

# Bitpositioner fra _vw_meb_common.dbc, Strukturen_01
_DISTANCE_STATUS = (13, 2)
_VALID_STATUS = 0
# (startbit, laengde) for afstand / sideafstand / relativ hastighed
_LEFT_LANE = ((96, 12), (108, 10), (118, 10))

_DIST_SCALE, _DIST_OFF = 0.0625, -3.75
_LAT_SCALE, _LAT_OFF = 0.065, -33.28
_VEL_SCALE, _VEL_OFF = 0.25, -128.0

# Aktiveringsgraenser
MIN_SPEED = float(os.getenv("UG_MIN_SPEED_KPH", "70")) * CV.KPH_TO_MS
MAX_RANGE = float(os.getenv("UG_MAX_RANGE", "90"))       # m fremad
MIN_RANGE = float(os.getenv("UG_MIN_RANGE", "3"))        # m, filtrerer skrald fra
LAT_MIN, LAT_MAX = 2.0, 6.0                              # m, plausibel nabobane
# Hvor meget hurtigere end venstre spor vi accepterer at koere. Et lille slip
# undgaar at loftet slaar til ved ubetydelige forskelle.
SLIP = float(os.getenv("UG_SLIP_KPH", "4")) * CV.KPH_TO_MS
# Loftet slippes gradvist naar objektet forsvinder, saa farten ikke springer.
RELEASE_S = 2.0


def _bits(data: bytes, start: int, length: int) -> int:
  return (int.from_bytes(data, "little") >> start) & ((1 << length) - 1)


class UndertakeGuard:
  def __init__(self):
    self.enabled = Params().get_bool("UndertakeGuard")
    self._sock = messaging.sub_sock("can", timeout=0) if self.enabled else None
    self.cap: float | None = None          # m/s, eller None naar inaktiv
    self.lead_v: float | None = None       # venstre-bane-objektets fart, m/s
    self.lead_d: float | None = None       # dets afstand, m
    self._age = 0.0

  def _decode(self, data: bytes, v_ego: float) -> tuple[float, float] | None:
    """Returnerer (afstand, absolut fart) for venstre-bane-objekt, ellers None."""
    if _bits(data, *_DISTANCE_STATUS) != _VALID_STATUS:
      return None
    (ds, dl), (ls, ll), (vs, vl) = _LEFT_LANE
    dist = _bits(data, ds, dl) * _DIST_SCALE + _DIST_OFF
    lat = _bits(data, ls, ll) * _LAT_SCALE + _LAT_OFF
    rel = _bits(data, vs, vl) * _VEL_SCALE + _VEL_OFF

    if not (MIN_RANGE < dist < MAX_RANGE):
      return None
    # Sideafstanden er positiv mod venstre i radarens billede -- modsat modellens
    # laneLines. Kontrolleret mod optagelse: nabobane laa paa ca. +4,0 m.
    if not (LAT_MIN < lat < LAT_MAX):
      return None
    return dist, v_ego + rel

  def update(self, v_ego: float, dt: float) -> None:
    """Kaldes hver planlaegger-tick. Opdaterer self.cap."""
    if self._sock is None:
      return

    found = None
    for msg in messaging.drain_sock(self._sock):
      for c in msg.can:
        if c.address == RADAR_ADDR and c.src == RADAR_BUS:
          found = self._decode(bytes(c.dat), v_ego) or found

    if found is not None:
      self.lead_d, self.lead_v = found
      self._age = 0.0
    else:
      self._age += dt
      if self._age > RELEASE_S:
        self.lead_d = self.lead_v = None

    # Loftet gaelder kun naar vi ellers ville passere et langsommere koeretoej
    # indenom. Er venstre spor hurtigere -- det normale tilfaelde -- sker intet.
    if (self.lead_v is not None and v_ego > MIN_SPEED
        and self.lead_v + SLIP < v_ego):
      self.cap = max(self.lead_v + SLIP, MIN_SPEED)
    else:
      self.cap = None

  def apply(self, v_cruise: float) -> float:
    """Saenker den oenskede fart til loftet. Kan aldrig haeve den."""
    if self.cap is None:
      return v_cruise
    return min(v_cruise, self.cap)
