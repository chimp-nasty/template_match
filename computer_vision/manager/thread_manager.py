import threading
from queue import Queue, Full, Empty
import time

from utils.pacer import Pacer


def _ema_update(old, x, a=0.2):
    return x if old is None else (1.0 - a) * old + a * x


class ThreadManager:
    def __init__(
        self,
        capture_func,
        process_func,
        logic_func=None,
        is_paused=lambda: False,
        on_pause=None,
        on_resume=None,
        target_fps: float = 100.0,
        # --- debug controls ---
        debug: bool = False,
        debug_interval: float = 1.0,
        debug_capture: bool = True,
        debug_process: bool = True,
        debug_logic: bool = True,
        stage_names=("capture", "process", "logic"),
    ):
        """
        Two-thread (producer–consumer) manager for continuous capture + processing.

        capture_func:  () -> frame | None
        process_func:  (frame) -> data
        logic_func:    (data) -> None
        logic_hz:      paced logic rate (Hz) using Pacer

        Debug output prints once per 'debug_interval' seconds:
        [dbg:capture] <FPS> FPS | <last_ms> ms (EMA: <ema_ms> ms)
        """
        self.capture_func = capture_func
        self.process_func = process_func
        self.logic_func   = logic_func or (lambda data: None)

        self.is_paused = is_paused
        self.on_pause  = on_pause
        self.on_resume = on_resume

        self.queue     = Queue(maxsize=1)  # latest-only
        self.stop_evt  = threading.Event()
        self.threads   = []
        self._was_paused = False

        self.data = None

        # Logic pacer (precise cadence)
        self.target_fps = target_fps
        self._logic_pacer = Pacer(
            fps=target_fps,
            high_res=True,
            spin_seconds=0.0015,
            catch_up=True,
        )

        # Debug state
        self._dbg_enabled   = bool(debug)
        self._dbg_interval  = float(debug_interval)
        self._dbg_capture   = bool(debug_capture)
        self._dbg_process   = bool(debug_process)
        self._dbg_logic     = bool(debug_logic)
        self._dbg_names     = stage_names

        self._dbg_lock      = threading.Lock()
        now = time.perf_counter()
        self._dbg_last_report = now

        # counts for FPS and EMA for ms
        self._dbg = {
            "cap_count": 0,
            "cap_ms_last": 0.0,
            "cap_ms_ema": None,

            "proc_count": 0,
            "proc_ms_last": 0.0,
            "proc_ms_ema": None,

            "logic_count": 0,
            "logic_ms_last": 0.0,
            "logic_ms_ema": None,
        }

    # --- lifecycle controls ---
    def start(self):
        self.stop_evt.clear()
        producer = threading.Thread(target=self._capture_loop,  name="CaptureThread",  daemon=True)
        consumer = threading.Thread(target=self._consumer_loop, name="ConsumerThread", daemon=True)
        self.threads = [producer, consumer]
        for t in self.threads:
            t.start()

    def stop(self):
        self.stop_evt.set()
        for t in self.threads:
            if t.is_alive():
                t.join()

    def _sleep_while_paused(self):
        while not self.stop_evt.is_set() and self.is_paused():
            time.sleep(0.05)

    def _maybe_signal_hooks(self):
        p = self.is_paused()
        if p and not self._was_paused and self.on_pause:
            try: self.on_pause()
            except Exception: pass
        if (not p) and self._was_paused and self.on_resume:
            try: self.on_resume()
            except Exception: pass
        self._was_paused = p

    # --- debug helpers ---
    def _dbg_capture_sample(self, dt_ms: float, enqueued: bool):
        if not self._dbg_enabled: return
        with self._dbg_lock:
            self._dbg["cap_ms_last"] = dt_ms
            self._dbg["cap_ms_ema"]  = _ema_update(self._dbg["cap_ms_ema"], dt_ms)
            if enqueued:
                self._dbg["cap_count"] += 1

    def _dbg_process_sample(self, dt_ms: float):
        if not self._dbg_enabled: return
        with self._dbg_lock:
            self._dbg["proc_ms_last"] = dt_ms
            self._dbg["proc_ms_ema"]  = _ema_update(self._dbg["proc_ms_ema"], dt_ms)
            self._dbg["proc_count"]  += 1

    def _dbg_logic_sample(self, dt_ms: float):
        if not self._dbg_enabled: return
        with self._dbg_lock:
            self._dbg["logic_ms_last"] = dt_ms
            self._dbg["logic_ms_ema"]  = _ema_update(self._dbg["logic_ms_ema"], dt_ms)
            self._dbg["logic_count"]  += 1

    def _dbg_maybe_report(self):
        if not self._dbg_enabled: return
        now = time.perf_counter()
        with self._dbg_lock:
            elapsed = now - self._dbg_last_report
            if elapsed < self._dbg_interval:
                return

            # snapshot + reset
            cap_ms  = self._dbg["cap_ms_last"]
            cap_ema = self._dbg["cap_ms_ema"]

            proc_ms  = self._dbg["proc_ms_last"]
            proc_ema = self._dbg["proc_ms_ema"]

            logic_ms  = self._dbg["logic_ms_last"]
            logic_ema = self._dbg["logic_ms_ema"]

            self._dbg_last_report = now

        # print outside the lock
        cap_name, proc_name, logic_name = self._dbg_names
        if self._dbg_capture:
            fps = min(int(1000/cap_ms), self.target_fps)
            print(f"[dbg:{cap_name}] {fps:.0f} FPS | {cap_ms:.2f} ms (EMA: { (cap_ema or 0.0):.2f} ms)")
        if self._dbg_process:
            fps = min(int(1000/proc_ms), self.target_fps)
            print(f"[dbg:{proc_name}] {fps:.0f} FPS | {proc_ms:.2f} ms (EMA: { (proc_ema or 0.0):.2f} ms)")
        if self._dbg_logic:
            fps = min(int(1000/logic_ms), self.target_fps)
            print(f"[dbg:{logic_name}] {fps:.0f} FPS | {logic_ms:.4f} ms (EMA: { (logic_ema or 0.0):.4f} ms)")

    # --- internal loops ---
    def _capture_loop(self):
        while not self.stop_evt.is_set():
            self._maybe_signal_hooks()
            if self.is_paused():
                self._sleep_while_paused()
                continue

            t0 = time.perf_counter()
            frame = self.capture_func()   # may be None when no desktop change
            dt_ms = (time.perf_counter() - t0) * 1000.0

            enqueued = False
            if frame is not None:
                try:
                    self.queue.put_nowait(frame)
                    enqueued = True
                except Full:
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                    self.queue.put_nowait(frame)
                    enqueued = True

            self._dbg_capture_sample(dt_ms, enqueued)

        # Signal end-of-stream
        try: self.queue.put_nowait(None)
        except Full: pass

    def _consumer_loop(self):
        while not self.stop_evt.is_set():
            self._maybe_signal_hooks()
            if self.is_paused():
                self._sleep_while_paused()
                continue

            # Non-blocking fetch to avoid delaying logic ticks
            got_frame = False
            try:
                frame = self.queue.get_nowait()
                if frame is None:
                    break  # shutdown signal
                t0 = time.perf_counter()
                self.data = self.process_func(frame)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                self.queue.task_done()
                got_frame = True
                self._dbg_process_sample(dt_ms)
            except Empty:
                pass

            # Always run logic on the pacer cadence (even if no new frame)
            t0 = time.perf_counter()
            self.logic_func(self.data)   # your logic; keep it lightweight
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._dbg_logic_sample(dt_ms)

            # per-interval debug print (once, here)
            self._dbg_maybe_report()

            # precise logic pacing
            self._logic_pacer.pace()
