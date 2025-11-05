import time

from computer_vision.utils.hwnd_utils import hwnds_for
from computer_vision.utils.pacer import Pacer
from computer_vision.manager.keyboard_controller import Controller
from computer_vision.manager.thread_manager import ThreadManager
from computer_vision.vision.capture import Capture
from computer_vision.vision.display import Display
from computer_vision.processing.color_processor import ColorConvertor


PROCESS = ["lofi.exe"]
FPS = 200

class Client:
    def __init__(self, target_fps, hwnd, debug=True, display=False):
        self.target_fps = target_fps
        self.hwnd = hwnd
        self.debug = debug
        self.display = display

        # --- Controller setup (pause / terminate hotkeys) ---
        self.controller = Controller(pause_key="SPACE", terminate_key="c")

        # Use Pacer instead of next_tick/tick_interval (target ~200 Hz)
        self.main_pacer = Pacer(fps=FPS, high_res=True)

        # --- Vision pipeline components ---
        self.capture_device = Capture(monitor=1, hwnd=self.hwnd, pace_fps=FPS)  # capture pacing stays as you set it
        self.color_processor = ColorConvertor()
        if self.display:
            self.display_window = Display(scale=0.5, hotkey="c")

        # Shared reference for most-recent frame (for display)
        self.latest_frame = [None]

        self.manager = ThreadManager(
            capture_func=self.capture_func,
            process_func=self.process_func,
            logic_func=self.logic_func,
            is_paused=lambda: self.controller.paused,
            on_pause=lambda: getattr(self.capture_device, "pause", lambda: None)(),
            on_resume=lambda: getattr(self.capture_device, "resume", lambda: None)(),
            target_fps=FPS,
            debug=self.debug,                 # <- master debug on/off
            debug_interval=1.0,         # <- print every 1s
            debug_capture=self.debug,
            debug_process=self.debug,
            debug_logic=self.debug,
            stage_names=("capture", "process", "logic"),
        )
        
    def capture_func(self):
        frame = self.capture_device.dxc_grab()
        if frame is not None:
            self.latest_frame[0] = frame
        return frame

    def process_func(self, frame):
        if frame is None:
            return None
        return self.color_processor.opencv_color(frame)

    def logic_func(self, data):
        # super light: just touch one value (or do your real logic)
        _ = data[0] if (data is not None and len(data) > 0) else None

    def start(self):
        try:
            self.manager.start()
            while not self.controller.terminated:
                self.controller.poll()

                if self.controller.paused:
                    time.sleep(0.05)
                    continue
                
                if self.display:
                    frame = self.latest_frame[0]
                    if frame is not None:
                        if not self.display_window.show(frame):
                            break

                # Precise main-loop pacing via Pacer (replaces controller.pace/next_tick)
                self.main_pacer.pace()

        except KeyboardInterrupt:
            self.controller.terminated = True
        finally:
            self.manager.stop()
            if self.display:
                self.display_window.close()


if __name__ == "__main__":
    hwnds = hwnds_for(PROCESS)
    hwnd = hwnds[0] if hwnds else None

    clients = []
    
    if len(hwnds) > 1:
        for hwnd in hwnds:
            client = Client(fps=FPS, hwnd=hwnd, debug=False, display=False)
            clients.append(client)
    
    else:
        client = Client(fps=FPS, hwnd=hwnd, debug=True, display=True)
        clients.append(client)

    for i, client in enumerate(clients, start=1):
        print(f"Starting Client {i}/{len(clients)} -> hwnd={client.hwnd}")
        client.start()