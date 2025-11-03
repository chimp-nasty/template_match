import time


from computer_vision.utils.hwnd_utils import hwnds_for
from computer_vision.manager.keyboard_controller import Controller
from computer_vision.manager.thread_manager import ThreadManager
from computer_vision.vision.capture import Capture
from computer_vision.vision.display import Display
from computer_vision.processing.color_processor import ColorConvertor


PROCESS = ["lofi"]

if __name__ == "__main__":
    hwnds = hwnds_for(PROCESS)
    hwnd = None
    if hwnds:
        hwnd = hwnds[0]

    # --- Controller setup (pause / terminate hotkeys) ---
    controller = Controller(pause_key="SPACE", terminate_key="c")
    next_tick = time.perf_counter()
    tick_interval = 0.01  # target ~100 Hz main-loop pacing

    # --- Vision pipeline components ---
    capture_device = Capture(hwnd=hwnd)
    color_processor = ColorConvertor()
    display_window = Display(scale=0.5, hotkey="c")

    # Shared reference for most-recent frame
    latest_frame = [None]

    # --- Pipeline stage wrappers ---
    def capture_frame():
        """Grab a frame from the capture device and store the latest copy."""
        frame = capture_device.dxc_grab()
        if frame is not None:
            latest_frame[0] = frame
        return frame

    def process_frame(frame):
        """Convert raw frame to color-processed data."""
        return color_processor.colors_all_pixels_rgb(frame)

    # --- Threaded producer/consumer manager ---
    manager = ThreadManager(
        capture_func=capture_frame,
        process_func=process_frame,
        logic_func=lambda data: None,  # placeholder for future logic
        is_paused=lambda: controller.paused,
        on_pause=lambda: getattr(capture_device, "pause", lambda: None)(),
        on_resume=lambda: getattr(capture_device, "resume", lambda: None)(),
    )

    # --- Main loop ---
    try:
        # [Capture] / [Process + Logic] threads run independently
        manager.start()

        # Display loop needs to be the "main" loop
        while not controller.terminated:
            controller.poll()

            if controller.paused:
                time.sleep(0.05)
                continue

            frame = latest_frame[0]
            if frame is not None:
                if not display_window.show(frame):  # BGRA/GRAY safe inside
                    break
            
            # if manager.data is not None:
            #     print(manager.data[0])

            next_tick = controller.pace(next_tick, tick_interval)

    except KeyboardInterrupt:
        controller.terminated = True

    finally:
        manager.stop()
        display_window.close()