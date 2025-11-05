import win32gui
import mss
import dxcam
import numpy as np
import win32api, win32con


from computer_vision.utils.pacer import Pacer


def _monitor_rect_from_hwnd(hwnd: int) -> tuple[int,int,int,int]:
    hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
    info = win32api.GetMonitorInfo(hmon)
    return info["Monitor"]  # (l,t,r,b) in desktop coords

def _enum_monitor_rects_sorted() -> list[tuple[int,int,int,int]]:
    rects = [m[2] for m in win32api.EnumDisplayMonitors()]
    rects.sort(key=lambda r: (r[0], r[1]))  # left→right, then top→bottom
    return rects

def _find_monitor_index(rect, monitors) -> int:
    lx, ty, rx, by = rect
    best_i, best_area = 0, -1
    for i, (ml, mt, mr, mb) in enumerate(monitors):
        ix = max(0, min(rx, mr) - max(lx, ml))
        iy = max(0, min(by, mb) - max(ty, mt))
        area = ix * iy
        if area > best_area:
            best_area, best_i = area, i
    return best_i

def _to_local_and_clamp(ax: int, ay: int, w: int, h: int, mon_rect):
    ml, mt, mr, mb = mon_rect
    W, H = mr - ml, mb - mt
    x = ax - ml
    y = ay - mt
    # clamp to monitor bounds
    x = max(0, min(x, W))
    y = max(0, min(y, H))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return x, y, w, h

def fast_grab(cam, region=None, out: np.ndarray | None = None):
    """
    Compatibility wrapper for old dxcam (no `output=`).
    Calls cam.grab(region). If `out` is provided and shape/dtype match,
    copies into it; otherwise returns the grabbed frame as-is.
    Handles None frames gracefully.
    """
    frame = cam.grab(region=region)

    # Stream not ready / transient failure
    if frame is None:
        return None

    if out is None:
        return frame

    # Only copy when shapes/types match exactly; otherwise just return frame
    if out.shape == frame.shape and out.dtype == frame.dtype:
        np.copyto(out, frame)
        return out
    else:
        return frame


class Capture:
    def __init__(self, monitor=0, hwnd=None, pace_fps: float | None = None):
        self.hwnd = hwnd if (hwnd and win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)) else None
        self._monitors = _enum_monitor_rects_sorted()

        # choose which monitor to start on
        if self.hwnd:
            wl, wt, wr, wb = win32gui.GetWindowRect(self.hwnd)
            self._mon_idx = _find_monitor_index((wl, wt, wr, wb), self._monitors)
        else:
            self._mon_idx = int(monitor or 0)

        self._buf = None
        self.cam = dxcam.create(output_idx=self._mon_idx, output_color="BGRA")
        self._pacer = Pacer(pace_fps, high_res=True) if (pace_fps and pace_fps > 0) else None

    def _get_monitor_resolution(self):
        ml, mt, mr, mb = self._monitors[self._mon_idx]
        return mr - ml, mb - mt
    
    def is_valid_hwnd(self, hwnd):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd):
            return hwnd
        return None

    def _get_region(self, region=None):
        if region is None:
            region = {}
        # client size
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        width_default = right - left
        height_default = bottom - top
        # client origin in screen coords (absolute)
        base_x, base_y = win32gui.ClientToScreen(self.hwnd, (0, 0))
        # relative → absolute
        rel_x = int(region.get("x", 0))
        rel_y = int(region.get("y", 0))
        width = int(region.get("width", width_default))
        height = int(region.get("height", height_default))
        abs_x = base_x + rel_x
        abs_y = base_y + rel_y
        return abs_x, abs_y, width, height

    def _ensure_cam_on_hwnd_monitor(self):
        """If the window moved monitors, recreate dxcam on the correct output."""
        win_mon_rect = _monitor_rect_from_hwnd(self.hwnd)
        desired_idx = _find_monitor_index(win_mon_rect, self._monitors)
        if desired_idx != self._mon_idx:
            self._mon_idx = desired_idx
            self.cam = dxcam.create(output_idx=self._mon_idx, output_color="BGRA")

    def _ensure_buf(self, w, h, channels=4):
        need_shape = (h, w, channels)
        if self._buf is None or self._buf.shape != need_shape:
            self._buf = np.empty(need_shape, dtype=np.uint8)
        return self._buf
    
    def mss_grab(self, region: dict = None):
        if self._pacer:
            self._pacer.pace()

        if self.hwnd:
            x, y, w, h = self._get_region(region)
            with mss.mss() as sct:
                frame = sct.grab({"left": x, "top": y, "width": w, "height": h})
        else:
            with mss.mss() as sct:
                frame = sct.grab(sct.monitors[1])
        return np.array(frame, copy=False)

    def dxc_grab(self, region: dict = None):
        if self._pacer:
            self._pacer.pace()

        if self.hwnd:
            ax, ay, w, h = self._get_region(region)
            self._ensure_cam_on_hwnd_monitor()
            mon_rect = self._monitors[self._mon_idx]
            x, y, w, h = _to_local_and_clamp(ax, ay, w, h, mon_rect)
            if w <= 0 or h <= 0:
                return None

            out = self._ensure_buf(w, h, 4)  # BGRA
            try:
                return fast_grab(self.cam, region=(x, y, w, h), out=out)
            except ValueError:
                full = fast_grab(self.cam, out=None)  # fallback: full grab
                if full is None:
                    return None
                return full[y:y+h, x:x+w]
        else:
            # full monitor
            ml, mt, mr, mb = self._monitors[self._mon_idx]
            W, H = mr - ml, mb - mt
            out = self._ensure_buf(W, H, 4)
            return fast_grab(self.cam, region=None, out=out)
