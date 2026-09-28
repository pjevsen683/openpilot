#!/usr/bin/env python3
"""Live top-down (inverse perspective) view of the road for the shadow page.

Every output pixel is a fixed point on the ground in front of the car, so the
mapping from ground metres to camera pixels never changes while the calibration
holds. That table is built once and each frame is then a single gather plus a
JPEG encode -- measured on the device at 3.5 ms for 320x272, which at 4 Hz is
1.4 % of one core out of eight.

It costs nothing at all when nobody is looking: the worker only touches the
camera if /topdown.jpg has been asked for in the last few seconds, so closing
the page on the phone stops the work on the car.

Read-only, like the rest of tavascan_web. It subscribes and it serves; it never
publishes, writes params or sends CAN.
"""
import io
import os
import threading
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.swaglog import cloudlog

# Ground patch we render, in the calibrated frame: x ahead, y right, metres.
# The near edge is set by the dashboard, which hides everything closer than
# about 6 m from the camera.
#
# How far the far edge can usefully go is set by the camera, not by us. One
# image row covers this much ground:
#
#            20 m      30 m      50 m     100 m
#   wide     0.97 m    1.94 m    4.85 m   17.76 m
#   narrow   0.38 m    0.76 m    1.90 m    6.94 m
#
# so the wide camera has nothing left past about 40 m, while the narrow one
# reaches roughly three times further. Hence both: the wide camera near, where
# its field of view is what we need, and the narrow one beyond the crossover,
# where its reach is. Every output pixel still comes from exactly one camera,
# so the warp costs the same as before.
X_NEAR, X_FAR = 5.0, float(os.getenv("TAVASCAN_TOPDOWN_FAR", "100"))
Y_HALF = 14.0

# Where we stop using the wide camera. Below this the narrow camera cannot see
# wide enough -- it covers +/- 0.589 x, so it only spans Y_HALF from 23.8 m out
# -- and above it the wide camera has no resolution left. The two happen to
# cross at the same place, which is why a single split works.
X_CROSS = 24.0

# First image row that is dashboard rather than road, measured across the frame
# on this car. Everything below it is masked black instead of being smeared
# across the near field, where it would look like road surface. In rows of a
# 760-row frame, scaled to whatever the buffer turns out to be. The narrow
# camera looks past the dashboard entirely, so it has no such cut.
V_DASH = 490.0

OUT_W = int(os.getenv("TAVASCAN_TOPDOWN_W", "256"))
# Not tied to the aspect ratio. The page draws this into a rectangle it works
# out from the extent, so it stretches to fit whatever shape the panel is.
# Matching the aspect would mean 869 rows for a 95 m range, four fifths of them
# spent on the far field where one image row already covers several metres of
# ground. Fewer rows there cost nothing and save the warp two thirds of its work.
OUT_H = int(os.getenv("TAVASCAN_TOPDOWN_H", "384"))
HZ = float(os.getenv("TAVASCAN_TOPDOWN_HZ", "4"))
QUALITY = int(os.getenv("TAVASCAN_TOPDOWN_QUALITY", "55"))
# How long a single request keeps the worker awake. Comfortably longer than the
# page's own poll interval, so a dropped frame does not stall the stream.
DEMAND_S = 4.0

# The wide camera is not a pinhole. openpilot models it as one and the source
# says so itself next to the number: "focal length probably wrong? magnification
# is not consistent across frame". Registering its top-down view against the
# narrow camera's, which is close to a true pinhole over its 61 degree field,
# moves the normalised cross-correlation from 0.41 to 0.96. Straight ahead the
# pinhole model puts a feature at 12 m half a metre too far; 12 m out to the
# side it is off by 4 m.
#
# Fitted on one frame of 000000dd--00f1d6e7ca--1, so the lens terms are solid
# (they are physical) while the small pitch and yaw offsets are a residual
# alignment term from that one fit. Set TAVASCAN_TOPDOWN_PINHOLE=1 to fall back
# to openpilot's own numbers.
FISHEYE = dict(fl=446.15, k1=-0.12258, k2=0.33397, dpitch=0.0095126, dyaw=0.0013963)
PINHOLE = dict(fl=567.0 / 4 * 3, k1=0.0, k2=0.0, dpitch=0.0, dyaw=0.0)
# The narrow camera needs none of that -- it is close enough to a true pinhole
# over its 61 degree field that it served as the reference the wide one was
# fitted against, so openpilot's own focal length is used as it stands.
NARROW = dict(fl=1522.0 * 3 / 4, k1=0.0, k2=0.0, dpitch=0.0, dyaw=0.0)
USE_PINHOLE = os.getenv("TAVASCAN_TOPDOWN_PINHOLE") == "1"
# Both focal lengths are in pixels of a 1344-wide frame, so they only mean what
# they say at that size. Scaled by the buffer we actually get.
FIT_WIDTH, FIT_HEIGHT = 2688 // 2, 1520 // 2

# view frame: x right, y down, z forward.  device/calib frame: x fwd, y right, z down
VIEW_FROM_DEVICE = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def rot_from_euler(rpy) -> np.ndarray:
  r, p, y = rpy
  cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
  return (np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]]) @
          np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]]) @
          np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]]))


def row_split() -> int:
  """First output row taken from the wide camera. Row 0 is the far edge."""
  xs = np.linspace(X_FAR, X_NEAR, OUT_H)
  return int(np.searchsorted(-xs, -X_CROSS))


def build_table(cam, rpy_calib, wide_from_device, height, width, rows, stride):
  """Flat indices into one camera's Y plane, for the output rows it covers.

  Returns (idx, bad) where idx is int32 and bad marks output pixels that fall
  outside the image or behind the camera. The two cameras cover disjoint bands
  of rows, so each output pixel is sampled exactly once.
  """
  if cam == "narrow":
    m = NARROW
    R = VIEW_FROM_DEVICE @ rot_from_euler(np.asarray(rpy_calib))
  else:
    m = PINHOLE if USE_PINHOLE else FISHEYE
    extra = np.array([0.0, m["dpitch"], m["dyaw"]])
    R = (VIEW_FROM_DEVICE @ rot_from_euler(np.asarray(wide_from_device) + extra)
         @ rot_from_euler(np.asarray(rpy_calib)))

  # top row is the far edge, so the image reads the same way up as the drive
  split = row_split()
  xs = np.linspace(X_FAR, X_NEAR, OUT_H)[:split] if cam == "narrow" else \
       np.linspace(X_FAR, X_NEAR, OUT_H)[split:]
  ys = np.linspace(-Y_HALF, Y_HALF, OUT_W)
  yy, xx = np.meshgrid(ys, xs)
  pts = np.stack([xx, yy, np.full_like(xx, height)], -1)

  pv = pts @ R.T
  z = pv[..., 2]
  with np.errstate(divide="ignore", invalid="ignore"):
    xn, yn = pv[..., 0] / z, pv[..., 1] / z
  rn = np.hypot(xn, yn)
  th = np.arctan(rn)
  rd = th + m["k1"] * th ** 3 + m["k2"] * th ** 5
  s = np.where(rn > 1e-9, rd / np.maximum(rn, 1e-9), 1.0)
  fl = m["fl"] * width / FIT_WIDTH
  u = fl * xn * s + width / 2.0
  v = fl * yn * s + rows / 2.0

  # only the wide camera has the dashboard in shot
  v_dash = rows - 1.0 if cam == "narrow" else min(rows - 1.0, V_DASH * rows / FIT_HEIGHT)
  bad = ((z <= 0) | ~np.isfinite(u) | ~np.isfinite(v) |
         (u < 0) | (u > width - 1) | (v < 0) | (v > v_dash))
  ui = np.clip(np.rint(u), 0, width - 1).astype(np.int32)
  vi = np.clip(np.rint(v), 0, rows - 1).astype(np.int32)
  return (vi * stride + ui).ravel().astype(np.int32), bad.ravel()


class TopDown:
  """Warps both road cameras onto the ground plane, on demand only."""

  def __init__(self) -> None:
    self.jpeg: bytes | None = None
    self.ts: float = 0.0
    self.extent = dict(x_near=X_NEAR, x_far=X_FAR, y_half=Y_HALF, x_cross=X_CROSS)
    self._lock = threading.Lock()
    self._wanted = 0.0
    self._tables: dict = {}
    self._out: np.ndarray | None = None

  def request(self) -> tuple[bytes | None, float]:
    """Marks demand and hands back the newest frame."""
    self._wanted = time.monotonic()
    with self._lock:
      return self.jpeg, self.ts

  @property
  def wanted(self) -> bool:
    return (time.monotonic() - self._wanted) < DEMAND_S

  def _table_for(self, cam, cal, buf):
    key = (round(cal.rpyCalib[1], 5), round(cal.rpyCalib[2], 5),
           round(cal.height[0], 3), buf.width, buf.height, buf.stride)
    have = self._tables.get(cam)
    if have is None or have[0] != key:
      idx, bad = build_table(cam, cal.rpyCalib, cal.wideFromDeviceEuler,
                             cal.height[0], buf.width, buf.height, buf.stride)
      self._tables[cam] = (key, idx, bad)
      self._out = None
      cloudlog.info("tavascan_web: top-down %s table rebuilt for %s", cam, key)
    return self._tables[cam][1], self._tables[cam][2]

  def run(self) -> None:
    while True:
      try:
        self._run()
      except Exception:
        cloudlog.exception("tavascan_web: top-down worker failed, retrying")
        time.sleep(3.0)

  def _run(self) -> None:
    # Imported here so that a device without these -- or an offroad run -- costs
    # nothing until the view is actually asked for.
    from openpilot.cereal.visionipc import VisionStreamType
    from msgq.visionipc import VisionIpcClient
    from PIL import Image

    if HZ <= 0:
      cloudlog.info("tavascan_web: top-down disabled")
      return

    streams = {"narrow": VisionStreamType.VISION_STREAM_NARROW_ROAD,
               "wide": VisionStreamType.VISION_STREAM_WIDE_ROAD}
    split = row_split()
    bands = {"narrow": (0, split * OUT_W), "wide": (split * OUT_W, OUT_H * OUT_W)}
    sm = None
    clients: dict = {}
    period = 1.0 / HZ
    next_frame = 0.0

    while True:
      if not self.wanted:
        if clients:
          clients = {}
          self._out = None
          with self._lock:
            self.jpeg, self.ts = None, 0.0
        time.sleep(0.5)
        continue

      if sm is None:
        sm = messaging.SubMaster(["extrinsicsCalibration"])
      sm.update(0)

      for cam, stream in streams.items():
        c = clients.get(cam)
        if c is None or not c.is_connected():
          c = VisionIpcClient("camerad", stream, True)
          clients[cam] = c if c.connect(False) else None
          if clients[cam] is not None:
            cloudlog.info("tavascan_web: top-down connected to the %s camera", cam)
      if not any(clients.values()):
        time.sleep(1.0)
        continue

      bufs = {cam: (c.recv(timeout_ms=200) if c is not None else None)
              for cam, c in clients.items()}
      if not any(b is not None for b in bufs.values()):
        continue

      now = time.monotonic()
      if now < next_frame:
        continue
      next_frame = now + period

      cal = sm["extrinsicsCalibration"]
      if len(cal.rpyCalib) != 3 or len(cal.height) != 1 or str(cal.calStatus) != "calibrated":
        continue

      if self._out is None or self._out.size != OUT_H * OUT_W:
        self._out = np.zeros(OUT_H * OUT_W, np.uint8)
      for cam, buf in bufs.items():
        if buf is None:
          continue       # keep whatever that band held, rather than flashing black
        idx, bad = self._table_for(cam, cal, buf)
        # The Y plane alone: a grey road reads fine and colour would mean
        # carrying the interleaved UV plane through the same gather for no real
        # gain. buf.data is a view on the shared buffer, so nothing is copied
        # here; the gather below is the first time any pixel is touched.
        data = buf.data
        flat = data if isinstance(data, np.ndarray) else np.frombuffer(data, dtype=np.uint8)
        band = flat[idx]
        band[bad] = 0
        lo, hi = bands[cam]
        self._out[lo:hi] = band

      # The two cameras expose independently, which leaves a visible step right
      # across the road at the crossover. Match the narrow band to the wide one
      # using a few rows either side of the seam: a single gain, so the far
      # field keeps its own relative shading and only the step goes away.
      if split > 8 and bufs.get("narrow") is not None and bufs.get("wide") is not None:
        near_seam = self._out[(split - 8) * OUT_W:split * OUT_W]
        far_seam = self._out[split * OUT_W:(split + 8) * OUT_W]
        a, c = float(np.median(near_seam)), float(np.median(far_seam))
        if a > 4.0 and c > 4.0:
          gain = float(np.clip(c / a, 0.6, 1.7))
          head = self._out[:split * OUT_W]
          # via float, because scaling a uint8 buffer in place wraps at 255
          # instead of clipping, and a bright sky would come back black
          self._out[:split * OUT_W] = np.clip(head * gain, 0, 255).astype(np.uint8)

      b = io.BytesIO()
      Image.fromarray(self._out.reshape(OUT_H, OUT_W), "L").save(b, "JPEG", quality=QUALITY)
      with self._lock:
        self.jpeg = b.getvalue()
        self.ts = time.time()


def start() -> TopDown:
  td = TopDown()
  threading.Thread(target=td.run, daemon=True, name="tavascan_topdown").start()
  return td
