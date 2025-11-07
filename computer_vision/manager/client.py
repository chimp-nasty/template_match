import time

from utils.pacer import Pacer
from manager.keyboard_controller import Controller
from manager.thread_manager import ThreadManager
from vision.capture import Capture
from vision.display import Display
from processing.process_manager import ProcessManager
import config as cfg


class Client:
    def __init__(self, hwnd, logic, target_fps=200, debug=True, display=False):
        self.target_fps = target_fps
        self.hwnd = hwnd
        self.debug = debug
        self.display = display
        self.logic = logic
        
        self.need_rgb = bool(getattr(self.logic, "need_rgb", True))
        self.need_hsv = bool(getattr(self.logic, "need_hsv", True))

        # --- Controller setup (pause / terminate hotkeys) ---
        self.controller = Controller(pause_key=cfg.PAUSE, terminate_key=cfg.CLOSE)

        # Use Pacer instead of next_tick/tick_interval (target ~200 Hz)
        self.main_pacer = Pacer(fps=cfg.FPS, high_res=True)

        # --- Vision pipeline components ---
        self.capture_device = Capture(monitor=1, hwnd=self.hwnd, pace_fps=cfg.FPS)  # capture pacing stays as you set it
        self.img_processor = ProcessManager(need_rgb=self.need_rgb, need_hsv=self.need_hsv)
        if self.display:
            self.display_window = Display(scale=0.5, hotkey=cfg.CLOSE)

        # Shared reference for most-recent frame (for display)
        self.latest_frame = [None]

        # Multi threads the capture / consumer loops
        self.manager = ThreadManager(
            capture_func=self.capture_func,
            process_func=self.img_processor.process,
            logic=self.logic,
            is_paused=lambda: self.controller.paused,
            on_pause=lambda: getattr(self.capture_device, "pause", lambda: None)(),
            on_resume=lambda: getattr(self.capture_device, "resume", lambda: None)(),
            target_fps=cfg.FPS,
            debug=self.debug,
            debug_interval=cfg.DEBUG_INTERVAL,
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

    def initialize(self, *, retries: int = 5, delay: float = 0.02) -> dict:
        """
        Try an initial capture, run one process pass, feed logic, and call logic.start().
        Returns: {"status": bool, "message": str}
        """
        try:
            # 1) Warm-up capture with a few retries
            frame = None
            for _ in range(max(1, retries)):
                frame = self.capture_device.dxc_grab()
                if frame is not None:
                    break
                time.sleep(delay)
            if frame is None:
                return {"status": False, "message": "Calibration failed: no frame from capture."}

            # 2) Sync feature flags into the processor (in case logic changed them pre-init)
            self.img_processor.need_rgb = bool(getattr(self.logic, "need_rgb", self.img_processor.need_rgb))
            self.img_processor.need_hsv = bool(getattr(self.logic, "need_hsv", self.img_processor.need_hsv))

            # 3) One processing pass
            data = self.img_processor.process(frame)
            if not data:
                return {"status": False, "message": "Calibration failed: processing returned no data."}

            # Keep latest frame for UI if you want
            self.latest_frame[0] = frame

            # 4) Seed logic and start
            self.logic.update_data(data)

            res = self.logic.start()  # can return dict or bool
            if isinstance(res, dict):
                ok  = bool(res.get("status", False))
                msg = res.get("message", "Calibration " + ("Success" if ok else "Failed"))
            else:
                ok  = bool(res)
                msg = "Calibration " + ("Success" if ok else "Failed")

            return {"status": ok, "message": msg}

        except Exception as e:
            return {"status": False, "message": f"Calibration error: {e.__class__.__name__}: {e}"}

    def start(self):
        try:
            res = self.initialize()
            print(res["message"])
            if not res["status"]:
                return
            
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