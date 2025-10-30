import threading
from queue import Queue, Full, Empty
import time

from proto.scr_cap import ImgShow, ScreenGrab, ColorConvertor


class ThreadedPipelineManager:
    def __init__(self, capture_func, process_func, logic_func=None):
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

        self.queue = Queue(maxsize=1)
        self.stop_evt = threading.Event()
        self.threads = []
        self.cycles = None

    # --- lifecycle controls ---
    def start(self, num_cycles=None):
        """Start producer and consumer threads."""
        self.stop_evt.clear()
        self.cycles = num_cycles

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

    # --- internal loops ---
    def _capture_loop(self):
        count = 0
        while not self.stop_evt.is_set():
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

            count += 1
            if self.cycles and count >= self.cycles:
                break
        self.queue.put(None)  # signal end-of-stream

    def _consumer_loop(self):
        while not self.stop_evt.is_set():
            try:
                frame = self.queue.get(timeout=0.5)
            except Empty:
                continue
            if frame is None:
                break

            data = self.process_func(frame)
            self.logic_func(data)
            self.queue.task_done()


if __name__ == "__main__":
    cap = ScreenGrab()
    pro = ColorConvertor()
    viewer = ImgShow(scale=0.5, hotkey="q")

    latest_frame = [None]  # simple shared reference

    def capture():
        frame = cap.dxc_grab()
        if frame is not None:
            latest_frame[0] = frame
        return frame

    def process(frame):
        return pro.colors_all_pixels_rgb(frame)

    mgr = ThreadedPipelineManager(
        capture_func=capture,
        process_func=process,
        logic_func=lambda data: None,
    )

    running = True

    if running:
        mgr.start()

        try:
            while True:
                frame = latest_frame[0]
                if frame is not None:
                    cont = viewer.show(frame)
                    if not cont:
                        break
                time.sleep(0.01)
        finally:
            mgr.stop()
            viewer.close()
    else:
        print("shilling")

    