import subprocess, csv, io
import win32con, win32gui, win32process, win32api


def _top_level_visible(hwnd: int) -> bool:
    try:
        return win32gui.IsWindowVisible(hwnd) and win32gui.GetWindow(hwnd, win32con.GW_OWNER) == 0
    except Exception:
        return False

def _pids_for_process_names(names: set[str]) -> set[int]:
    """
    Uses 'tasklist /fo csv /nh' for robust parsing (no locale-dependent spacing).
    Returns PIDs for any process whose Image Name matches one of `names` (case-insensitive).
    """
    out = subprocess.check_output(["tasklist", "/fo", "csv", "/nh"], text=True, encoding="utf-8", errors="ignore")
    rdr = csv.reader(io.StringIO(out))
    pids = set()
    for row in rdr:
        if not row or len(row) < 2:
            continue
        image = (row[0] or "").strip().lower()
        if image in names:
            try:
                pids.add(int((row[1] or "0").replace(",", "")))
            except ValueError:
                pass
    return pids

def hwnds_for(names: set[str]) -> list[int]:
    """Return a list of HWNDs for the given process image names (e.g., {'wow.exe', 'wowclassic.exe'})."""
    pids = _pids_for_process_names({n.lower() for n in names})
    if not pids:
        return []

    results: list[int] = []

    def enum_cb(hwnd, _):
        try:
            if not _top_level_visible(hwnd):
                return True
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid in pids:
                results.append(hwnd)
        except Exception:
            pass
        return True  # keep enumerating

    win32gui.EnumWindows(enum_cb, None)
    return results

def _enum_monitor_infos():
    """Native order list used by dxcam: [(hmon, hdc, (l,t,r,b)), ...]."""
    return win32api.EnumDisplayMonitors()

def _index_for_hmonitor(hmon, infos) -> int | None:
    """Find the index of a given HMONITOR inside EnumDisplayMonitors() results."""
    for i, info in enumerate(infos):
        if info[0] == hmon:
            return i
    return None

def monitor_index_for_hwnd(hwnd: int) -> int | None:
    """
    Returns the dxcam output_idx for the monitor containing (or nearest to) hwnd.
    None if hwnd is invalid or the monitor can't be resolved.
    """
    if not (hwnd and win32gui.IsWindow(hwnd)):
        return None
    try:
        hmon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
        infos = _enum_monitor_infos()
        return _index_for_hmonitor(hmon, infos)
    except Exception:
        return None