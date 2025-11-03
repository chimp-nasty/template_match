import numpy as np


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
