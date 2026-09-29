#!/usr/bin/env python3
"""The HTTP half, shared by the onroad and offroad pages.

Both pages bind the same port. That is safe because they are registered with
opposite predicates -- one only_onroad, the other only_offroad -- so manager
never has both alive. SO_REUSEADDR covers the handover, where the outgoing
socket may still sit in TIME_WAIT, and bind is retried rather than crashing if
the old process has not quite let go.

Serving only. Nothing here reads the car or writes anything.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn

from openpilot.common.swaglog import cloudlog

BIND_RETRY_S = 2.0
BIND_ATTEMPTS = 30


class _Server(ThreadingHTTPServer):
  allow_reuse_address = True
  daemon_threads = True


def make_handler(page_path: str, snapshot, lock):
  class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
      pass  # do not spam the openpilot logs with request lines

    def _send(self, code: int, body: bytes, ctype: str) -> None:
      self.send_response(code)
      self.send_header("Content-Type", ctype)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      if self.path.startswith("/data.json"):
        with lock:
          body = json.dumps(snapshot).encode()
        self._send(200, body, "application/json")
      elif self.path in ("/", "/index.html"):
        try:
          with open(page_path, "rb") as f:
            self._send(200, f.read(), "text/html; charset=utf-8")
        except OSError:
          self._send(500, b"page missing", "text/plain")
      else:
        self._send(404, b"not found", "text/plain")

  return Handler


def serve(port: int, page_path: str, snapshot: dict, lock: threading.Lock) -> None:
  """Blocks. Retries the bind so a handover from the other page is not fatal."""
  handler = make_handler(page_path, snapshot, lock)
  for attempt in range(BIND_ATTEMPTS):
    try:
      srv = _Server(("0.0.0.0", port), handler)
      break
    except OSError as e:
      cloudlog.info("tavascan_web: port %d busy (%s), retrying", port, e)
      time.sleep(BIND_RETRY_S)
  else:
    cloudlog.error("tavascan_web: could not bind port %d", port)
    return

  cloudlog.info("tavascan_web: serving on :%d", port)
  srv.serve_forever()
