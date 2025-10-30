import threading
from queue import Queue, Full, Empty
from time import perf_counter
import time
import random
from proto.scr_cap import ScreenGrab, ImgShow, ColorConvertor


# --- Logic function ---
def do_logic(data):
    idx = random.randint(0, len(data) - 1)
    print(idx)
    return f"logic_{data}"


# --- Threaded pipeline benchmark ---
def threaded_benchmark(num_cycles, get_screenshot, process_img):
    q = Queue(maxsize=1)
    stop_evt = threading.Event()
    count = 0

    def capture_loop():
        nonlocal count
        while count < num_cycles and not stop_evt.is_set():
            frame = safe_get_screenshot(get_screenshot)
            if frame is None:
                continue
            try:
                q.put_nowait(frame)
            except Full:
                try:
                    q.get_nowait()
                except Empty:
                    pass
                q.put_nowait(frame)
            count += 1
        stop_evt.set()
        producer.join()
        consumer.join()
        try:
            q.put_nowait(None)
        except Full:
            pass

    def consumer_loop():
        while not stop_evt.is_set():
            try:
                frame = q.get(timeout=0.5)
            except Empty:
                continue
            if frame is None:
                break
            data = process_img(frame)
            do_logic(data)
            q.task_done()

    # Benchmark threaded version
    t0 = perf_counter()
    producer = threading.Thread(target=capture_loop, daemon=True)
    consumer = threading.Thread(target=consumer_loop, daemon=True)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    t1 = perf_counter()

    return t1 - t0


# --- Single-threaded benchmark ---
def single_loop_benchmark(num_cycles, get_screenshot, process_img):
    t0 = perf_counter()
    for _ in range(num_cycles):
        frame = safe_get_screenshot(get_screenshot)
        if frame is None:
            continue  # skip if still invalid after retries
        data = process_img(frame)
        do_logic(data)
    t1 = perf_counter()
    return t1 - t0

def safe_get_screenshot(get_fn, retries=3, delay=0.01):
    """Try multiple times to get a valid frame; return None if all fail."""
    for _ in range(retries):
        frame = get_fn()
        if frame is not None and hasattr(frame, "shape") and frame.ndim == 3:
            return frame
        time.sleep(delay)
    return None


# --- Run comparison ---
if __name__ == "__main__":
    cap = ScreenGrab()
    pro = ColorConvertor()

    NUM_CYCLES = 60

    # Define the bound methods
    get_screenshot = cap.dxc_grab
    process_img = pro.colors_all_pixels_rgb

    # Run both
    t_single = single_loop_benchmark(NUM_CYCLES, get_screenshot, process_img)
    t_threaded = threaded_benchmark(NUM_CYCLES, get_screenshot, process_img)

    print(f"\nBenchmark Results ({NUM_CYCLES} cycles):")
    print(f"Single-threaded : {t_single:.4f} sec")
    print(f"Threaded pipeline: {t_threaded:.4f} sec")
    print(f"Speed-up factor  : {t_single / t_threaded:.2f}×")
