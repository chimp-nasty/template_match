import win32gui
import mss
import dxcam
import numpy as np
import cv2
import time
from time import perf_counter

class ScreenGrab:
    def __init__(self, monitor=1, hwnd=None):
        self.hwnd = self.is_valid_hwnd(hwnd)
        self.cam = dxcam.create(output_idx=monitor, output_color="BGRA")
    
    def is_valid_hwnd(self, hwnd):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd):
            return hwnd
        return None

    def _get_region(self, region=None):
        if region is None:
            region = {}

        # Get client area size (width/height)
        left, top, right, bottom = win32gui.GetClientRect(self.hwnd)
        width_default = right - left
        height_default = bottom - top

        # Convert client-area origin to screen coords (removes border/title bar)
        base_x, base_y = win32gui.ClientToScreen(self.hwnd, (0, 0))

        # Relative region inside the client area
        rel_x = int(region.get("x", 0))
        rel_y = int(region.get("y", 0))
        width = int(region.get("width", width_default))
        height = int(region.get("height", height_default))

        # Absolute coords for screen grabbing
        abs_x = base_x + rel_x
        abs_y = base_y + rel_y

        return abs_x, abs_y, width, height

    def mss_grab(self, region: dict = None):
        if self.hwnd:
            x, y, w, h = self._get_region(region)
            with mss.mss() as sct:
                frame = sct.grab({"left": x, "top": y, "width": w, "height": h})
        else:
            with mss.mss() as sct:
                frame = sct.grab(sct.monitors[1])
        return np.array(frame, copy=False)

    def dxc_grab(self, region: dict = None):
        if self.hwnd:
            x, y, w, h = self._get_region(region)
            frame = self.cam.grab(region=(x, y, w, h))
        else:
            frame = self.cam.grab()
        return frame


class ImgShow:
    def __init__(self, window_name="Capture", scale=1.0, hotkey="q"):
        self.window_name = window_name
        self.scale = float(scale)
        self.last_time = time.time()
        self.fps = 0.0
        self._sized = False
        self.hk = hotkey  # ← fixed variable name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _ensure_window_size(self, w, h):
        if not self._sized:
            sw = max(1, int(w * self.scale))
            sh = max(1, int(h * self.scale))
            cv2.resizeWindow(self.window_name, sw, sh)
            self._sized = True

    def draw_fps(self, frame):
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        cv2.putText(frame, f"{self.fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def show(self, frame):
        if frame is None:
            return True

        # Convert BGRA → BGR if needed
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        self._ensure_window_size(w, h)

        # Apply scaling
        if self.scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(w * self.scale), int(h * self.scale)),
                interpolation=cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
            )

        frame = self.draw_fps(frame)
        cv2.imshow(self.window_name, frame)

        # Return False when the hotkey is pressed
        key = cv2.waitKey(1) & 0xFF
        return key != ord(self.hk.lower())

    def close(self):
        cv2.destroyAllWindows()



class ColorConvertor:
    def __init__(self):
        ...

    def colors_all_pixels_rgb(self, screenshot: np.ndarray) -> np.ndarray:
        """Return RGB colors for the entire image in row-major order. Shape: (H*W, 3)."""
        arr = np.asarray(screenshot)
        bgr = arr[..., :3]            
        rgb = bgr[..., ::-1].copy()  
        return rgb.reshape(-1, 3)

    def colors_all_pixels_rgb_packed(self, screenshot: np.ndarray) -> np.ndarray:
        """Return packed 24-bit RGB integers (0xRRGGBB) in row-major order."""
        rgb = self.colors_all_pixels_rgb(screenshot)
        r = rgb[:, 0].astype(np.uint32)
        g = rgb[:, 1].astype(np.uint32)
        b = rgb[:, 2].astype(np.uint32)
        return (r << 16) | (g << 8) | b

    def get_colors_all_pixels_slow(self, screenshot: np.ndarray) -> list[tuple[int, int, int]]:
        """Naive per-pixel getter using Python loops on a NumPy array. Very slow."""
        h, w = screenshot.shape[:2]
        out = []
        for y in range(h):
            for x in range(w):
                # screenshot[y, x] is [B, G, R, A] or [B, G, R]
                px = screenshot[y, x]
                if px.shape[0] >= 3:
                    b, g, r = px[:3]
                    out.append((int(r), int(g), int(b)))
                else:
                    raise ValueError("Pixel missing color channels.")
        return out


if __name__ == "__main__":
    grabber = ScreenGrab()
    display = ImgShow(scale=0.5)
    color_conv = ColorConvertor()

    # frame = grabber.dxc_grab()  # shape: (H, W, 4) BGRA

    # h, w = frame.shape[:2]
    # flat = frame.reshape(-1, frame.shape[2])      # (H*W, 4)
    
    # sample = frame
    # # sample = flat[:2000].reshape(2000, 1, 4)      # example slice

    # # Fast RGB
    # t0 = perf_counter()
    # rgb_array = color_conv.colors_all_pixels_rgb(sample)
    # t1 = perf_counter()

    # # Packed RGB
    # packed_array = color_conv.colors_all_pixels_rgb_packed(sample)
    # t2 = perf_counter()

    # # Slow version for comparison
    # t3 = perf_counter()
    # slow_list = color_conv.get_colors_all_pixels_slow(sample)
    # t4 = perf_counter()

    # print("\nBenchmark Results:")
    # print(f"Fast (RGB ndarray)      : {(t1 - t0):.6f} sec  | pixels={rgb_array.shape[0]}")
    # print(f"Fast (packed ndarray)   : {(t2 - t1):.6f} sec  | pixels={packed_array.shape[0]}")
    # print(f"Slow (pixel-by-pixel)   : {(t4 - t3):.6f} sec  | pixels={len(slow_list)}")
    

    try:
        while True:
            frame = grabber.dxc_grab()
            if not display.show(frame):
                break
    finally:
        display.close()