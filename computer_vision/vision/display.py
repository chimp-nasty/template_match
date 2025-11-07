import cv2
import time
import numpy as np

class Display:
    def __init__(self, window_name="Capture", scale=1.0, hotkey="q", fps_overlay=True):
        self.window_name = window_name
        self.scale = float(scale)
        self.hk = (hotkey or "q").lower()

        self._sized = False
        self._fps_ema = None
        self._last_ts = time.time()

        self.fps_overlay = fps_overlay

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    # --- helpers ---
    @staticmethod
    def _to_bgr(frame: np.ndarray) -> np.ndarray:
        """Ensure frame is BGR for imshow (accepts GRAY/BGR/BGRA)."""
        if frame is None:
            return None
        if frame.ndim == 2:  # GRAY
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:  # BGRA -> BGR
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame  # already BGR (or unexpected but pass-through)

    def _ensure_window_size(self, w, h):
        if not self._sized:
            sw = max(1, int(round(w * self.scale)))
            sh = max(1, int(round(h * self.scale)))
            cv2.resizeWindow(self.window_name, sw, sh)
            self._sized = True

    def _update_fps(self):
        now = time.time()
        dt = max(1e-6, now - self._last_ts)
        inst = 1.0 / dt
        self._last_ts = now
        # Exponential moving average for stable readout
        if self._fps_ema is None:
            self._fps_ema = inst
        else:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst
        return self._fps_ema

    def _overlay_fps(self, frame: np.ndarray) -> np.ndarray:
        fps = self._update_fps()
        cv2.putText(
            frame, f"{fps:.1f} FPS", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        return frame

    # --- public API ---
    def show(self, frame) -> bool:
        """Show a frame. Returns False if window closed or hotkey pressed."""
        if frame is None:
            # still pump events so window remains responsive
            key = cv2.waitKey(1) & 0xFF
            if key == ord(self.hk) or key == 27:  # quit hotkey or ESC
                return False
            # also detect user closing the window (property becomes < 0)
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            return True

        frame = self._to_bgr(frame)

        h, w = frame.shape[:2]
        self._ensure_window_size(w, h)

        if self.scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(round(w * self.scale)), int(round(h * self.scale))),
                interpolation=cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
            )

        if self.fps_overlay:
            frame = self._overlay_fps(frame)
        cv2.imshow(self.window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(self.hk) or key == 27:  # 'hotkey' or ESC
            return False

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        return True

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            cv2.destroyAllWindows()
