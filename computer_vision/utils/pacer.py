import time, atexit, threading
from typing import Optional, Tuple
import ctypes

# ---------- process-wide high-res timer manager ----------
class _HighResTimers:
    _lock   = threading.Lock()
    _count  = 0
    _ms_set = 0
    _winmm  = ctypes.windll.winmm

    @classmethod
    def enable(cls, ms: int = 1) -> None:
        with cls._lock:
            if cls._count == 0:
                cls._winmm.timeBeginPeriod(ms)
                cls._ms_set = ms
            cls._count += 1

    @classmethod
    def disable(cls) -> None:
        with cls._lock:
            if cls._count > 0:
                cls._count -= 1
                if cls._count == 0:
                    cls._winmm.timeEndPeriod(cls._ms_set)
                    cls._ms_set = 0

# Ensure we clean up on interpreter exit even if __del__ isn’t called
atexit.register(_HighResTimers.disable)

# ---------- precise pacer ----------
class Pacer:
    """
    Precise pacer for fixed-rate loops.
    Call pace() once per iteration.

    Args:
        fps: target iterations per second (None/<=0 disables pacing)
        spin_seconds: short busy-wait window for sub-ms precision
        catch_up: keep phase by advancing next time in fixed steps
        high_res: if True, request 1 ms system timer resolution (process-wide)
    """
    def __init__(self, fps: Optional[float] = None,
                 spin_seconds: float = 0.0005,
                 catch_up: bool = True,
                 high_res: bool = False) -> None:
        self._interval: Optional[float] = None
        self._next: Optional[float] = None
        self._spin = max(0.0, float(spin_seconds))
        self._catch_up = bool(catch_up)
        self._using_high_res = False

        if high_res:
            _HighResTimers.enable()
            self._using_high_res = True

        self.set_fps(fps)

    def __del__(self):
        if self._using_high_res:
            _HighResTimers.disable()

    def set_fps(self, fps: Optional[float]) -> None:
        self._interval = (1.0 / float(fps)) if fps and fps > 0 else None
        self._next = None

    def enabled(self) -> bool:
        return self._interval is not None

    def reset(self) -> None:
        self._next = None

    def pace(self) -> Tuple[float, float]:
        if self._interval is None:
            return (0.0, 0.0)

        now = time.perf_counter()
        if self._next is None:
            self._next = now + self._interval
            return (0.0, 0.0)

        rem = self._next - now
        slept = 0.0

        if rem > self._spin:
            to_sleep = rem - self._spin
            time.sleep(to_sleep)
            slept += to_sleep
            now = time.perf_counter()

        # short spin for precision
        while True:
            rem = self._next - now
            if rem <= 0.0:
                break
            now = time.perf_counter()

        overrun = max(0.0, now - self._next)

        if self._catch_up:
            k = 1 + int((now - self._next) // self._interval) if overrun > 0 else 1
            self._next += k * self._interval
        else:
            self._next = now + self._interval

        return (slept, overrun)
