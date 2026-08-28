#!/usr/bin/env python3
"""Holder en WireGuard-tunnel til hjemmenettet oppe.

Kernen paa AGNOS (4.9) har ikke WireGuard-modulet, saa vi bruger wireguard-go i
userspace via /dev/net/tun. Binaeren ligger i /data og er bygget fra
git.zx2c4.com/wireguard-go.

Split tunnel med vilje: kun hjemmenettet og VPN-subnettet routes gennem
tunnelen. Al oevrig trafik -- openpilots uploads, modeldownloads -- gaar
udenom, saa bilens dataforbrug ikke loeber gennem hjemmeforbindelsen og en
nedadgaaende tunnel ikke tager internettet med sig.

Ruten til hjemmenettet tilfoejes kun naar enheden IKKE allerede er paa det.
Ellers ville trafik til maskiner vi kan naa direkte tage vejen rundt gennem
VPN'en -- og i vaerste fald kappe den SSH-session man sidder paa.
"""
import base64
import os
import re
import socket
import subprocess
import time

from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import Ratekeeper

CONF = os.getenv("WG_CONF", "/data/wg_client.conf")
BINARY = os.getenv("WG_BINARY", "/data/wireguard-go")
IFACE = os.getenv("WG_IFACE", "wg0")
HOME_NET = os.getenv("WG_HOME_NET", "10.0.1.0/24")
VPN_NET = os.getenv("WG_VPN_NET", "10.6.0.0/24")
CHECK_S = float(os.getenv("WG_CHECK_INTERVAL", "60"))
SOCK = f"/var/run/wireguard/{IFACE}.sock"


def conf_get(text: str, key: str) -> str | None:
  m = re.search(rf"^{key}\s*=\s*(.+)$", text, re.M | re.I)
  return m.group(1).strip() if m else None


def sh(*args, check: bool = False) -> subprocess.CompletedProcess:
  return subprocess.run(["sudo", *args], capture_output=True, text=True, check=check)


def iface_exists() -> bool:
  return subprocess.run(["ip", "link", "show", IFACE], capture_output=True).returncode == 0


def ensure_sock_access() -> bool:
  """wireguard-go koerer som root og laver socketen 0600 root:root, men denne proces
  koerer som comma. Uden det her fejler enhver konfiguration med Permission denied --
  hvilket er praecis hvad der skete ved foerste forsoeg paa at gore tunnelen permanent."""
  if not os.path.exists(SOCK):
    return False
  if os.access(SOCK, os.R_OK | os.W_OK):
    return True
  sh("chown", f"{os.getuid()}:{os.getgid()}", SOCK)
  return os.access(SOCK, os.R_OK | os.W_OK)


def uapi(payload: str) -> str:
  s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  s.connect(SOCK)
  s.sendall(payload.encode())
  s.shutdown(socket.SHUT_WR)
  out = s.recv(65536).decode()
  s.close()
  return out


def last_handshake_age() -> float | None:
  """Sekunder siden sidste handshake, eller None hvis aldrig / utilgaengelig."""
  if not ensure_sock_access():
    return None
  try:
    d = uapi("get=1\n\n")
  except OSError:
    return None
  for line in d.splitlines():
    if line.startswith("last_handshake_time_sec="):
      ts = int(line.split("=", 1)[1])
      return (time.time() - ts) if ts else None
  return None


def on_home_network() -> bool:
  r = subprocess.run(["ip", "route", "get", HOME_NET.split("/")[0].rsplit(".", 1)[0] + ".1"],
                     capture_output=True, text=True)
  return IFACE not in r.stdout and r.returncode == 0


def start_tunnel() -> bool:
  conf = open(CONF).read()
  priv, peer = conf_get(conf, "PrivateKey"), conf_get(conf, "PublicKey")
  addr, endpoint = conf_get(conf, "Address"), conf_get(conf, "Endpoint")
  keep = conf_get(conf, "PersistentKeepalive") or "25"
  if not all((priv, peer, addr, endpoint)):
    cloudlog.error("wg_tunnel: mangler felter i %s", CONF)
    return False

  if not iface_exists():
    sh("rm", "-f", SOCK)
    subprocess.Popen(["sudo", "setsid", BINARY, "-f", IFACE],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(80):
      if os.path.exists(SOCK):
        break
      time.sleep(0.1)
    else:
      cloudlog.error("wg_tunnel: wireguard-go startede ikke")
      return False

  if not ensure_sock_access():
    cloudlog.error("wg_tunnel: kan ikke tilgaa %s", SOCK)
    return False

  host, _, port = endpoint.rpartition(":")
  try:
    ep_ip = socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
  except socket.gaierror as e:
    cloudlog.warning("wg_tunnel: kan ikke slaa %s op (%s)", host, e)
    return False

  hexk = lambda b: base64.b64decode(b).hex()  # noqa: E731
  payload = (
    "set=1\n"
    f"private_key={hexk(priv)}\n"
    "replace_peers=true\n"
    f"public_key={hexk(peer)}\n"
    f"endpoint={ep_ip}:{port}\n"
    f"persistent_keepalive_interval={keep}\n"
    "replace_allowed_ips=true\n"
    f"allowed_ip={HOME_NET}\n"
    f"allowed_ip={VPN_NET}\n"
    "\n"
  )
  try:
    if "errno=0" not in uapi(payload):
      cloudlog.error("wg_tunnel: UAPI afviste konfigurationen")
      return False
  except OSError as e:
    cloudlog.error("wg_tunnel: UAPI utilgaengelig (%s)", e)
    return False

  sh("ip", "address", "add", addr, "dev", IFACE)
  sh("ip", "link", "set", "mtu", "1420", "up", "dev", IFACE)
  sh("ip", "route", "add", VPN_NET, "dev", IFACE)
  if not on_home_network():
    sh("ip", "route", "add", HOME_NET, "dev", IFACE)

  cloudlog.info("wg_tunnel: oppe mod %s (%s:%s) som %s", host, ep_ip, port, addr)
  return True


def main() -> None:
  if not os.path.exists(CONF) or not os.path.exists(BINARY):
    cloudlog.info("wg_tunnel: ikke konfigureret, springer over")
    return

  rk = Ratekeeper(1.0 / CHECK_S)
  while True:
    age = last_handshake_age() if iface_exists() else None
    # Uden handshake i over tre keepalive-perioder er tunnelen reelt doed.
    if age is None or age > 90:
      cloudlog.info("wg_tunnel: rejser tunnel (handshake: %s)",
                    "aldrig" if age is None else f"{age:.0f}s siden")
      start_tunnel()
    rk.keep_time()


if __name__ == "__main__":
  main()
