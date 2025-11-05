import numpy as np
import cv2


class ColorConvertor:
    def __init__(self):
        ...

    def colors_all_pixels_rgb(self, screenshot: np.ndarray) -> np.ndarray:
        """Return RGB colors for the entire image in row-major order. Shape: (H*W, 3)."""
        arr = np.asarray(screenshot)
        bgr = arr[..., :3]            
        rgb = bgr[..., ::-1]
        return rgb.reshape(-1, 3)

    def colors_all_pixels_rgb_packed(self, screenshot: np.ndarray) -> np.ndarray:
        """Return packed 24-bit RGB integers (0xRRGGBB) in row-major order."""
        rgb = self.colors_all_pixels_rgb(screenshot)
        r = rgb[:, 0].astype(np.uint32)
        g = rgb[:, 1].astype(np.uint32)
        b = rgb[:, 2].astype(np.uint32)
        return (r << 16) | (g << 8) | b

    def opencv_color(self, screenshot: np.ndarray) -> np.ndarray:
        arr = np.asarray(screenshot, dtype=np.uint8)
        if arr.shape[-1] == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return rgb.reshape(-1, 3)