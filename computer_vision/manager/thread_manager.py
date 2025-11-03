import threading
from queue import Queue, Full, Empty
import time


class ThreadManager:
    def __init__(self, capture_func, process_func, logic_func=None, is_paused=lambda: False, on_pause=None, on_resume=None):
        """
        Two-thread (producer–consumer) manager for continuous capture + processing.

        Args:
            capture_func: callable → returns a frame (NumPy array or similar).
            process_func: callable(frame) → processed data.
            logic_func: optional callable(data) → executes logic step.
        """
        self.capture_func = capture_func
        self.process_func = process_func
        self.logic_func = logic_func or (lambda data: None)

        self.is_paused = is_paused
        self.on_pause = on_pause
        self.on_resume = on_resume

        self.queue = Queue(maxsize=1)
        self.stop_evt = threading.Event()
        self.threads = []
        self._was_paused = False

        self.data = None

    # --- lifecycle controls ---
    def start(self):
        """Start producer and consumer threads."""
        self.stop_evt.clear()
        producer = threading.Thread(target=self._capture_loop, name="CaptureThread", daemon=True)
        consumer = threading.Thread(target=self._consumer_loop, name="ConsumerThread", daemon=True)
        self.threads = [producer, consumer]

        for t in self.threads:
            t.start()

    def stop(self):
        """Signal both threads to stop and wait for them."""
        self.stop_evt.set()
        for t in self.threads:
            if t.is_alive():
                t.join()

    def _sleep_while_paused(self):
        # Idle while paused without burning CPU
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

    # --- internal loops ---
    def _capture_loop(self):
        while not self.stop_evt.is_set():
            self._maybe_signal_hooks()
            if self.is_paused():
                self._sleep_while_paused()
                continue

            frame = self.capture_func()
            if frame is None:
                continue
            try:
                self.queue.put_nowait(frame)
            except Full:
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
                self.queue.put_nowait(frame)
        # signal end-of-stream
        try: self.queue.put_nowait(None)
        except Full:
            pass

    def _consumer_loop(self):
        while not self.stop_evt.is_set():
            self._maybe_signal_hooks()
            if self.is_paused():
                self._sleep_while_paused()
                continue

            try:
                frame = self.queue.get(timeout=0.5)
            except Empty:
                continue
            if frame is None:
                break

            self.data = self.process_func(frame)
            self.logic_func(self.data)
            self.queue.task_done()