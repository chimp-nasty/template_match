import ctypes as C
from ctypes import wintypes as W
import numpy as np
import cv2

# =========================
#  Win32 setup (ctypes)
# =========================
user32 = C.windll.user32
gdi32 = C.windll.gdi32
kernel32 = C.windll.kernel32

winmm = C.windll.winmm
winmm.timeBeginPeriod.argtypes = [W.UINT]
winmm.timeBeginPeriod.restype = W.UINT
winmm.timeEndPeriod.argtypes = [W.UINT]
winmm.timeEndPeriod.restype = W.UINT

SW_HIDE = 0  # for ShowWindow
IS_64 = C.sizeof(C.c_void_p) == 8
ULONG_PTR = C.c_ulonglong if IS_64 else C.c_ulong
LONG_PTR = C.c_longlong if IS_64 else C.c_long

# Fill missing wintypes aliases
if not hasattr(W, "WPARAM"):
    W.WPARAM = ULONG_PTR
if not hasattr(W, "LPARAM"):
    W.LPARAM = LONG_PTR
if not hasattr(W, "LRESULT"):
    W.LRESULT = LONG_PTR
if not hasattr(W, "HWND"):
    W.HWND = W.HANDLE
if not hasattr(W, "HINSTANCE"):
    W.HINSTANCE = W.HANDLE
if not hasattr(W, "HICON"):
    W.HICON = W.HANDLE
if not hasattr(W, "HCURSOR"):
    W.HCURSOR = W.HANDLE
if not hasattr(W, "HBRUSH"):
    W.HBRUSH = W.HANDLE
if not hasattr(W, "LPCWSTR"):
    W.LPCWSTR = C.c_wchar_p
if not hasattr(W, "HDC"):
    W.HDC = W.HANDLE
if not hasattr(W, "COLORREF"):
    W.COLORREF = W.DWORD

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000

ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

SW_SHOWNOACTIVATE = 4
WM_DESTROY = 0x0002
WM_QUIT = 0x0012

GW_OWNER = 4
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# For more robust root-window check and title/class queries
GA_ROOT = 2

class SIZE(C.Structure):
    _fields_ = [("cx", W.LONG), ("cy", W.LONG)]

class BLENDFUNCTION(C.Structure):
    _fields_ = [
        ("BlendOp", W.BYTE),
        ("BlendFlags", W.BYTE),
        ("SourceConstantAlpha", W.BYTE),
        ("AlphaFormat", W.BYTE),
    ]

WNDPROC = C.WINFUNCTYPE(W.LRESULT, W.HWND, W.UINT, W.WPARAM, W.LPARAM)
class WNDCLASS(C.Structure):
    _fields_ = [
        ("style", W.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", C.c_int),
        ("cbWndExtra", C.c_int),
        ("hInstance", W.HINSTANCE),
        ("hIcon", W.HICON),
        ("hCursor", W.HCURSOR),
        ("hbrBackground", W.HBRUSH),
        ("lpszMenuName", W.LPCWSTR),
        ("lpszClassName", W.LPCWSTR),
    ]

if hasattr(W, "MSG"):
    MSG = W.MSG
else:
    class MSG(C.Structure):
        _fields_ = [
            ("hwnd", W.HWND),
            ("message", W.UINT),
            ("wParam", W.WPARAM),
            ("lParam", W.LPARAM),
            ("time", W.DWORD),
            ("pt", W.POINT),
        ]

# Prototypes
user32.DefWindowProcW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
user32.DefWindowProcW.restype = W.LRESULT
user32.RegisterClassW.argtypes = [C.POINTER(WNDCLASS)]
user32.RegisterClassW.restype = W.ATOM
user32.CreateWindowExW.argtypes = [W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD, W.INT, W.INT, W.INT, W.INT, W.HWND, W.HMENU, W.HINSTANCE, C.c_void_p]
user32.CreateWindowExW.restype = W.HWND
user32.ShowWindow.argtypes = [W.HWND, W.INT]
user32.ShowWindow.restype = W.BOOL
user32.GetSystemMetrics.argtypes = [C.c_int]
user32.GetSystemMetrics.restype = C.c_int

# IMPORTANT: use wintypes.RECT / wintypes.POINT
user32.GetWindowRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
user32.GetWindowRect.restype = W.BOOL
user32.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
user32.GetClientRect.restype = W.BOOL
user32.MapWindowPoints.argtypes = [W.HWND, W.HWND, C.POINTER(W.POINT), W.UINT]
user32.MapWindowPoints.restype = W.INT

user32.SetWindowPos.argtypes = [W.HWND, W.HWND, W.INT, W.INT, W.INT, W.INT, W.UINT]
user32.SetWindowPos.restype = W.BOOL
user32.UpdateLayeredWindow.argtypes = [W.HWND, W.HDC, C.POINTER(W.POINT), C.POINTER(SIZE), W.HDC, C.POINTER(W.POINT), W.COLORREF, C.POINTER(BLENDFUNCTION), W.DWORD]
user32.UpdateLayeredWindow.restype = W.BOOL

user32.PeekMessageW.argtypes = [C.POINTER(MSG), W.HWND, W.UINT, W.UINT, W.UINT]
user32.PeekMessageW.restype = W.BOOL
user32.TranslateMessage.argtypes = [C.POINTER(MSG)]
user32.TranslateMessage.restype = W.BOOL
user32.DispatchMessageW.argtypes = [C.POINTER(MSG)]
user32.DispatchMessageW.restype = W.LRESULT
user32.PostQuitMessage.argtypes = [C.c_int]

user32.IsWindowVisible.argtypes = [W.HWND]
user32.IsWindowVisible.restype = W.BOOL
user32.GetWindow.argtypes = [W.HWND, W.UINT]
user32.GetWindow.restype = W.HWND
user32.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
user32.GetWindowThreadProcessId.restype = W.DWORD
user32.GetDC.argtypes = [W.HWND]
user32.GetDC.restype = W.HDC
user32.ReleaseDC.argtypes = [W.HWND, W.HDC]
user32.ReleaseDC.restype = W.INT
user32.DestroyWindow.argtypes = [W.HWND]
user32.DestroyWindow.restype = W.BOOL

# Title/class/ancestry
user32.GetWindowTextLengthW.argtypes = [W.HWND]
user32.GetWindowTextLengthW.restype = W.INT
user32.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, W.INT]
user32.GetWindowTextW.restype = W.INT
user32.GetClassNameW.argtypes = [W.HWND, W.LPWSTR, W.INT]
user32.GetClassNameW.restype = W.INT
user32.GetAncestor.argtypes = [W.HWND, W.UINT]
user32.GetAncestor.restype = W.HWND
user32.GetForegroundWindow.restype = W.HWND  # add restype for visibility helper

gdi32.CreateCompatibleDC.argtypes = [W.HDC]
gdi32.CreateCompatibleDC.restype = W.HDC
gdi32.SelectObject.argtypes = [W.HDC, W.HGDIOBJ]
gdi32.SelectObject.restype = W.HGDIOBJ
gdi32.DeleteObject.argtypes = [W.HGDIOBJ]
gdi32.DeleteObject.restype = W.BOOL
gdi32.DeleteDC.argtypes = [W.HDC]
gdi32.DeleteDC.restype = W.BOOL

class BITMAPINFOHEADER(C.Structure):
    _fields_ = [
        ("biSize", W.DWORD),
        ("biWidth", W.LONG),
        ("biHeight", W.LONG),
        ("biPlanes", W.WORD),
        ("biBitCount", W.WORD),
        ("biCompression", W.DWORD),
        ("biSizeImage", W.DWORD),
        ("biXPelsPerMeter", W.LONG),
        ("biYPelsPerMeter", W.LONG),
        ("biClrUsed", W.DWORD),
        ("biClrImportant", W.DWORD),
    ]

class BITMAPINFO(C.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", W.DWORD * 3)]

gdi32.CreateDIBSection.argtypes = [W.HDC, C.POINTER(BITMAPINFO), W.UINT, C.POINTER(C.c_void_p), W.HANDLE, W.DWORD]
gdi32.CreateDIBSection.restype = W.HBITMAP

# Optional DWM for nicer bounds
try:
    dwmapi = C.windll.dwmapi
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    dwmapi.DwmGetWindowAttribute.argtypes = [W.HWND, W.DWORD, C.c_void_p, W.DWORD]
    dwmapi.DwmGetWindowAttribute.restype = W.HRESULT
except Exception:
    dwmapi = None

# DPI awareness
try:
    C.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

# =========================
#  Helpers for target hwnd
# =========================
ENUMPROC = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
kernel32.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
kernel32.OpenProcess.restype = W.HANDLE
kernel32.CloseHandle.argtypes = [W.HANDLE]
kernel32.CloseHandle.restype = W.BOOL
kernel32.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
kernel32.QueryFullProcessImageNameW.restype = W.BOOL
try:
    psapi = C.windll.psapi
    psapi.GetProcessImageFileNameW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD]
    psapi.GetProcessImageFileNameW.restype = W.DWORD
except Exception:
    psapi = None

def _norm_exe(name: str) -> str:
    name = name.lower()
    return name if name.endswith(".exe") else (name + ".exe")

def _is_root_window(hwnd: W.HWND) -> bool:
    # Root (no owner/parent), visible
    return (user32.GetAncestor(hwnd, GA_ROOT) == hwnd) and bool(user32.IsWindowVisible(hwnd))

def _get_text(hwnd: W.HWND) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = C.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value

def _get_class(hwnd: W.HWND) -> str:
    buf = C.create_unicode_buffer(256)
    if user32.GetClassNameW(hwnd, buf, 256):
        return buf.value
    return ""

def _pid_to_exe(pid: int) -> str | None:
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if h:
        try:
            buf_len = W.DWORD(32768)
            buf = C.create_unicode_buffer(buf_len.value)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, C.byref(buf_len)):
                return buf.value.split("\\")[-1].lower()
        finally:
            kernel32.CloseHandle(h)
    if psapi:
        h2 = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if h2:
            try:
                buf = C.create_unicode_buffer(32768)
                n = psapi.GetProcessImageFileNameW(h2, buf, 32768)
                if n:
                    return buf.value.split("\\")[-1].lower()
            finally:
                kernel32.CloseHandle(h2)
    return None

def find_hwnd(process_name: str | None = None,
              title_substr: str | None = None,
              class_substr: str | None = None) -> W.HWND | None:
    """Try (1) process name, then (2) window title substring, then (3) class substring."""
    want_proc = _norm_exe(process_name) if process_name else None
    want_title = title_substr.lower() if title_substr else None
    want_class = class_substr.lower() if class_substr else None

    found = {"hwnd": None}

    @ENUMPROC
    def enum_cb(hwnd, lparam):
        if not _is_root_window(hwnd):
            return True

        title = _get_text(hwnd).lower()
        cls = _get_class(hwnd).lower()

        if want_proc:
            pid = W.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, C.byref(pid))
            exe = _pid_to_exe(pid.value)
            if exe and exe == want_proc:
                found["hwnd"] = hwnd
                return False

        if want_title and title and (want_title in title):
            found["hwnd"] = hwnd
            return False

        if want_class and cls and (want_class in cls):
            found["hwnd"] = hwnd
            return False

        return True

    user32.EnumWindows(enum_cb, 0)
    return found["hwnd"]

def debug_list_top_windows(limit=50):
    rows = []
    @ENUMPROC
    def cb(hwnd, lparam):
        if _is_root_window(hwnd):
            pid = W.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, C.byref(pid))
            exe = _pid_to_exe(pid.value) or "?"
            rows.append((hex(hwnd), exe, _get_class(hwnd), _get_text(hwnd)))
        return True
    user32.EnumWindows(cb, 0)
    for i, r in enumerate(rows[:limit]):
        print(f"{i:02} hwnd={r[0]} exe={r[1]} class={r[2]} title={r[3]}")

# =========================
#  Overlay class
# =========================
@WNDPROC
def _wndproc(hWnd, msg, wParam, lParam):
    if msg == WM_DESTROY:
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hWnd, msg, wParam, lParam)

class Overlay:
    """
    You set `overlay_img` (BGR/BGRA/GRAY). The class only uploads it.
    Optionally attach to a specific app via process/title/class matching.
    """
    def __init__(self,
                 target_process: str | None = None,
                 title_substr: str | None = None,
                 class_substr: str | None = None,
                 client_only: bool = False,
                 only_when_foreground: bool = True):
        self.client_only = client_only
        self.target_hwnd = find_hwnd(target_process, title_substr, class_substr)
        self.overlay_hwnd: W.HWND | None = None
        self.overlay_img: np.ndarray | None = None  # you set this externally

        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)

        self.only_when_foreground = only_when_foreground
        self._visible = True

        # updated each tick by _align_overlay(); useful to size your frame
        self.last_w = self.screen_w if self.target_hwnd is None else 1
        self.last_h = self.screen_h if self.target_hwnd is None else 1

        self._register_window_class()
        self._create_window()

    def update_img(self, img):
        """Set the next frame (BGR/BGRA/GRAY)."""
        self.overlay_img = img

    def display_overlay(self):
        """
        Upload the currently set overlay_img (if any). If no image is set,
        it uploads an empty transparent frame of the current size.
        """
        w, h = self.last_w, self.last_h
        if w <= 0 or h <= 0:
            return
        if self.overlay_img is not None:
            frame = self._prepare_overlay_frame(self.overlay_img, w, h)
        else:
            frame = np.zeros((h, w, 4), dtype=np.uint8)
        self._update_layered_window(frame)

    def tick(self) -> bool:
        """
        Process the message queue, align the overlay, and update visibility/size.
        Returns False if the window is quitting; True otherwise.
        """
        if not self._pump_messages():
            return False

        # Respect foreground-only visibility (if enabled); still keep alignment current
        if not self._ensure_visibility():
            self._align_overlay()
            return True

        w, h = self._align_overlay()
        self.last_w, self.last_h = w, h
        return True

    def _ensure_visibility(self):
        """Hide overlay unless the target is the foreground root (when enabled)."""
        if not self.target_hwnd or not self.only_when_foreground:
            # If we’re global or not restricting, ensure visible
            if not self._visible:
                user32.ShowWindow(self.overlay_hwnd, SW_SHOWNOACTIVATE)
                self._visible = True
            return True  # allowed to draw

        fg = user32.GetForegroundWindow()
        root = user32.GetAncestor(fg, GA_ROOT)
        if root == self.target_hwnd:
            if not self._visible:
                user32.ShowWindow(self.overlay_hwnd, SW_SHOWNOACTIVATE)
                self._visible = True
            return True
        else:
            if self._visible:
                user32.ShowWindow(self.overlay_hwnd, SW_HIDE)
                self._visible = False
            return False

    def close(self):
        if self.overlay_hwnd:
            user32.DestroyWindow(self.overlay_hwnd)
            self.overlay_hwnd = None

    # ---------- window setup ----------
    def _register_window_class(self):
        hInstance = kernel32.GetModuleHandleW(None)
        className = "cv_clickthrough_overlay"

        wc = WNDCLASS()
        wc.style = 0
        wc.lpfnWndProc = _wndproc
        wc.cbClsExtra = 0
        wc.cbWndExtra = 0
        wc.hInstance = hInstance
        wc.hIcon = None
        wc.hCursor = user32.LoadCursorW(None, C.c_wchar_p(32512))  # IDC_ARROW
        wc.hbrBackground = None
        wc.lpszMenuName = None
        wc.lpszClassName = className

        atom = user32.RegisterClassW(C.byref(wc))
        if not atom and kernel32.GetLastError() not in (0, 1410):
            raise OSError("RegisterClassW failed")

    def _create_window(self):
        hInstance = kernel32.GetModuleHandleW(None)
        className = "cv_clickthrough_overlay"

        ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        owner = None
        if self.target_hwnd is None:
            ex_style |= WS_EX_TOPMOST
            w, h = self.screen_w, self.screen_h
        else:
            owner = self.target_hwnd
            w, h = 1, 1  # will be aligned immediately

        self.overlay_hwnd = user32.CreateWindowExW(
            ex_style, className, "OpenCV Overlay", WS_POPUP,
            0, 0, w, h, owner, None, hInstance, None
        )
        if not self.overlay_hwnd:
            raise OSError("CreateWindowExW failed")

        user32.ShowWindow(self.overlay_hwnd, SW_SHOWNOACTIVATE)

    # ---------- alignment ----------
    def _get_target_bounds(self, hwnd: W.HWND):
        r = W.RECT()
        if self.client_only:
            if not user32.GetClientRect(hwnd, C.byref(r)):
                raise OSError("GetClientRect failed")
            tl, br = W.POINT(0, 0), W.POINT(r.right, r.bottom)
            user32.MapWindowPoints(hwnd, None, C.byref(tl), 1)
            user32.MapWindowPoints(hwnd, None, C.byref(br), 1)
            return tl.x, tl.y, br.x, br.y
        else:
            if dwmapi:
                r2 = W.RECT()
                hr = dwmapi.DwmGetWindowAttribute(
                    hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, C.byref(r2), C.sizeof(r2)
                )
                if hr == 0:
                    return r2.left, r2.top, r2.right, r2.bottom
            if not user32.GetWindowRect(hwnd, C.byref(r)):
                raise OSError("GetWindowRect failed")
            return r.left, r.top, r.right, r.bottom

    def _align_overlay(self):
        """Position/size overlay. Returns (w, h)."""
        if self.target_hwnd:
            l, t, r, b = self._get_target_bounds(self.target_hwnd)
            w, h = max(1, r - l), max(1, b - t)
            SWP_NOSENDCHANGING = 0x0400
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            user32.SetWindowPos(
                self.overlay_hwnd, None, l, t, w, h,
                SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOSENDCHANGING
            )
            return w, h
        else:
            return self.screen_w, self.screen_h

    # ---------- image prep & upload ----------
    @staticmethod
    def _to_bgra(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            a = np.full(bgr.shape[:2] + (1,), 255, dtype=np.uint8)
            return np.concatenate([bgr, a], axis=-1)
        if img.shape[2] == 3:
            a = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
            return np.concatenate([img, a], axis=-1)
        if img.shape[2] == 4:
            return img
        raise ValueError("overlay_img must have 1, 3, or 4 channels")

    @staticmethod
    def _premultiply_alpha(bgra: np.ndarray) -> np.ndarray:
        a = bgra[..., 3:4].astype(np.float32) / 255.0
        if np.all(a == 1.0):
            return bgra
        rgb = bgra[..., :3].astype(np.float32) * a
        bgra[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        return bgra

    def _prepare_overlay_frame(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        bgra = self._to_bgra(img)
        if (bgra.shape[1] != w) or (bgra.shape[0] != h):
            bgra = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_LINEAR)
        return self._premultiply_alpha(bgra)

    def _np_bgra_to_hbitmap(self, img_bgra: np.ndarray) -> W.HBITMAP:
        h, w = img_bgra.shape[:2]
        BI_RGB = 0
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = C.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        bmi.bmiHeader.biSizeImage = w * h * 4

        hdc = user32.GetDC(None)
        ppvBits = C.c_void_p()
        hBitmap = gdi32.CreateDIBSection(hdc, C.byref(bmi), 0, C.byref(ppvBits), None, 0)
        user32.ReleaseDC(None, hdc)
        if not hBitmap:
            raise OSError("CreateDIBSection failed")

        C.memmove(ppvBits, img_bgra.ctypes.data, w * h * 4)
        return hBitmap

    def _update_layered_window(self, frame_bgra: np.ndarray) -> None:
        h, w = frame_bgra.shape[:2]
        hBitmap = self._np_bgra_to_hbitmap(frame_bgra)

        hdcScreen = user32.GetDC(None)
        hdcMem = gdi32.CreateCompatibleDC(hdcScreen)
        old = gdi32.SelectObject(hdcMem, hBitmap)

        size = SIZE(w, h)
        ptSrc = W.POINT(0, 0)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)

        ok = user32.UpdateLayeredWindow(
            self.overlay_hwnd,
            hdcScreen,
            None,                 # keep position from SetWindowPos (prevents cross-monitor flicker)
            C.byref(size),
            hdcMem,
            C.byref(ptSrc),
            0,
            C.byref(blend),
            ULW_ALPHA
        )

        gdi32.SelectObject(hdcMem, old)
        gdi32.DeleteObject(hBitmap)
        gdi32.DeleteDC(hdcMem)
        user32.ReleaseDC(None, hdcScreen)

        if not ok:
            raise OSError("UpdateLayeredWindow failed")

    # ---------- message pump ----------
    def _pump_messages(self) -> bool:
        msg = MSG()
        PM_REMOVE = 0x0001
        while user32.PeekMessageW(C.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(C.byref(msg))
            user32.DispatchMessageW(C.byref(msg))
            if msg.message == WM_QUIT:
                return False
        return True
