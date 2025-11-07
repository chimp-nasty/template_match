import time
import cv2
import numpy as np

class ProcessManager:
    def __init__(self, *, need_rgb: bool = True, need_hsv: bool = True):
        self.need_rgb = need_rgb
        self.need_hsv = need_hsv

        self._hsv_buf: np.ndarray | None = None   # HxWx3 uint8
        self._rgb_buf: np.ndarray | None = None   # HxWx3 uint8

    def _ensure_bufs(self, frame: np.ndarray) -> None:
        H, W = frame.shape[:2]
        if self.need_hsv:
            if (self._hsv_buf is None) or (self._hsv_buf.shape[:2] != (H, W)):
                self._hsv_buf = np.empty_like(frame)
        if self.need_rgb:
            if (self._rgb_buf is None) or (self._rgb_buf.shape[:2] != (H, W)):
                self._rgb_buf = np.empty_like(frame)

    def process(self, frame: np.ndarray) -> dict | None:
        if frame is None:
            return None

        self._ensure_bufs(frame)

        # Convert on demand, using preallocated buffers
        hsv = None
        if self.need_hsv:
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self._hsv_buf)
            hsv = self._hsv_buf

        rgb = None
        if self.need_rgb:
            # If you truly need RGB (not BGR), convert once into the scratch buffer.
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
            rgb = self._rgb_buf
            # (Alternative without cv2: self._rgb_buf[...] = frame[:, :, ::-1])

        return {
            "ts": time.perf_counter(),
            "bgr": frame,  # drop if not needed downstream
            "hsv": hsv,
            "rgb": rgb,
        }
