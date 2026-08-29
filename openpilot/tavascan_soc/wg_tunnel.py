#!/usr/bin/env python3
"""Keeps a WireGuard tunnel to the home network up.

The AGNOS kernel (4.9) does not have the WireGuard module, so we use wireguard-go
in userspace via /dev/net/tun. The binary lives in /data and is built from
git.zx2c4.com/wireguard-go.

Split tunnel by design: only the home network and the VPN subnet are routed
through the tunnel. All other traffic -- openpilot's uploads, model downloads --
goes around it, so the car's data usage does not run through the home connection
and a tunnel going down does not take the internet with it.

The route to the home network is only added when the device is NOT already on it.
Otherwise traffic to machines we can reach directly would take the long way round
through the VPN -- and in the worst case cut the SSH session you are sitting on.
"""
import base64
import os
import re
import socket
import subprocess
import sys
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

_wg_proc: subprocess.Popen | None = None


def conf_get(text: str, key: str) -> str | None:
  m = re.search(rf"^{key}\s*=\s*(.+)$", text, re.M | re.I)
  return m.group(1).strip() if m else None


def sh(*args, check: bool = False) -> subprocess.CompletedProcess:
  return subprocess.run(["sudo", *args], capture_output=True, text=True, check=check)


def iface_exists() -> bool:
  return subprocess.run(["ip", "link", "show", IFACE], capture_output=True).returncode == 0


# wireguard-go runs as root and creates the socket 0700 root:root, while this
# process runs as comma. Attempts to chown it were unreliable -- the socket could
# disappear between chown and use -- so we talk to it as root instead.
_UAPI_HELPER = (
  "import socket,sys\n"
  "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)\n"
  "s.connect(sys.argv[1])\n"
  "s.sendall(sys.stdin.buffer.read())\n"
  "s.shutdown(socket.SHUT_WR)\n"
  "sys.stdout.write(s.recv(65536).decode())\n"
)


def uapi(payload: str) -> str:
  r = subprocess.run(["sudo", sys.executable, "-c", _UAPI_HELPER, SOCK],
                     input=payload, capture_output=True, text=True, timeout=10)
  if r.returncode != 0:
    raise OSError(r.stderr.strip()[:200] or "uapi fejlede")
  return r.stdout


def last_handshake_age() -> float | None:
  """Seconds since the last handshake, or None if never / unavailable."""
  if not os.path.exists(SOCK):
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
    cloudlog.error("wg_tunnel: missing fields in %s", CONF)
    return False

  global _wg_proc
  if _wg_proc is None or _wg_proc.poll() is not None:
    # Run wireguard-go as OUR child process in the foreground (-f) rather than
    # detaching it. A detached process did not survive the parent going away, and
    # the interface was then left unconfigured. If the supervisor dies now,
    # manager restarts it and the tunnel is brought up from scratch.
    sh("ip", "link", "delete", IFACE)
    sh("rm", "-f", SOCK)
    _wg_proc = subprocess.Popen(["sudo", BINARY, "-f", IFACE],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL)
    for _ in range(80):
      if os.path.exists(SOCK):
        break
      if _wg_proc.poll() is not None:
        cloudlog.error("wg_tunnel: wireguard-go exited with %s", _wg_proc.returncode)
        return False
      time.sleep(0.1)
    else:
      cloudlog.error("wg_tunnel: wireguard-go did not start")
      return False

  host, _, port = endpoint.rpartition(":")
  try:
    ep_ip = socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
  except socket.gaierror as e:
    cloudlog.warning("wg_tunnel: cannot resolve %s (%s)", host, e)
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
    cloudlog.error("wg_tunnel: UAPI unavailable (%s)", e)
    return False

  sh("ip", "address", "add", addr, "dev", IFACE)
  sh("ip", "link", "set", "mtu", "1420", "up", "dev", IFACE)
  sh("ip", "route", "add", VPN_NET, "dev", IFACE)
  if not on_home_network():
    sh("ip", "route", "add", HOME_NET, "dev", IFACE)

  cloudlog.info("wg_tunnel: up to %s (%s:%s) as %s", host, ep_ip, port, addr)
  return True


def main() -> None:
  global _wg_proc
  if not os.path.exists(CONF) or not os.path.exists(BINARY):
    cloudlog.info("wg_tunnel: not configured, skipping")
    return

  rk = Ratekeeper(1.0 / CHECK_S)
  while True:
    alive = _wg_proc is not None and _wg_proc.poll() is None
    age = last_handshake_age() if (alive and iface_exists()) else None
    # With no handshake for over three keepalive periods the tunnel is dead.
    if age is None or age > 90:
      cloudlog.info("wg_tunnel: bringing up tunnel (handshake: %s)",
                    "aldrig" if age is None else f"{age:.0f}s siden")
      start_tunnel()
    rk.keep_time()


if __name__ == "__main__":
  main()
