from pathlib import Path
import cv2
import numpy as np

def load_gray_template(png_path: Path, trim_border: bool = True, bg_thresh: int = 0) -> np.ndarray:
    """
    Load a PNG template and return a single-channel uint8 grayscale image.
    - Trims fully-transparent or near-black borders if trim_border=True.
    """
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Template not found: {png_path}")

    # If it has alpha, drop to BGR using alpha as a mask (keeps opaque content)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3:
        bgr = img
    else:
        # already single-channel
        gray = img
        if trim_border:
            gray = _trim_empty_border(gray, bg_thresh)
        return _as_gray_u8(gray)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if trim_border:
        gray = _trim_empty_border(gray, bg_thresh)
    return _as_gray_u8(gray)

def _as_gray_u8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

def _trim_empty_border(gray: np.ndarray, bg_thresh: int) -> np.ndarray:
    """
    Remove rows/cols that are entirely <= bg_thresh (e.g., transparent/black padding from export).
    """
    mask = gray > bg_thresh
    if not np.any(mask):
        return gray  # nothing to trim
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y0, y1 = ys[0], ys[-1] + 1
    x0, x1 = xs[0], xs[-1] + 1
    return gray[y0:y1, x0:x1]
