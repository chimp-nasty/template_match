from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import numpy as np
import cv2
from mss import mss

user32 = C.windll.user32
try:
    dwmapi = C.windll.dwmapi
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
except Exception:
    dwmapi = None

# --- Use wintypes.RECT / wintypes.POINT consistently ---
user32.GetWindowRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
user32.GetWindowRect.restype  = W.BOOL

user32.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
user32.GetClientRect.restype  = W.BOOL

user32.MapWindowPoints.argtypes = [W.HWND, W.HWND, C.POINTER(W.POINT), W.UINT]
user32.MapWindowPoints.restype  = W.INT
# ------------------------------------------------------

class ScreenCapture:
    """
    Thin wrapper around mss for fast ROI grabs.
    Use grab(rect) where rect = (left, top, width, height).
    """
    def __init__(self):
        self._sct = mss()

    def grab(self, rect: tuple[int, int, int, int], gray: bool = True) -> np.ndarray:
        left, top, width, height = rect
        raw = np.array(self._sct.grab({"left": left, "top": top, "width": width, "height": height}))
        if gray:
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        return raw  # BGRA

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass

def get_window_rect(hwnd: W.HWND, client_only: bool = False):
    """
    Return (left, top, width, height) in screen coordinates.
    If client_only is True, returns the client area; otherwise the window frame
    (prefers DWM extended frame bounds when available).
    """
    if client_only:
        r = W.RECT()
        if not user32.GetClientRect(hwnd, C.byref(r)):
            raise OSError("GetClientRect failed")
        tl, br = W.POINT(0, 0), W.POINT(r.right, r.bottom)
        user32.MapWindowPoints(hwnd, None, C.byref(tl), 1)
        user32.MapWindowPoints(hwnd, None, C.byref(br), 1)
        left, top, right, bottom = tl.x, tl.y, br.x, br.y
    else:
        if dwmapi:
            r2 = W.RECT()
            hr = dwmapi.DwmGetWindowAttribute(
                hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, C.byref(r2), C.sizeof(r2)
            )
            if hr == 0:  # S_OK
                left, top, right, bottom = r2.left, r2.top, r2.right, r2.bottom
            else:
                r = W.RECT()
                if not user32.GetWindowRect(hwnd, C.byref(r)):
                    raise OSError("GetWindowRect failed")
                left, top, right, bottom = r.left, r.top, r.right, r.bottom
        else:
            r = W.RECT()
            if not user32.GetWindowRect(hwnd, C.byref(r)):
                raise OSError("GetWindowRect failed")
            left, top, right, bottom = r.left, r.top, r.right, r.bottom

    return left, top, right - left, bottom - top
