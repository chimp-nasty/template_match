from __future__ import annotations
import numpy as np
import cv2

def draw_matches_mask(
    size_wh: tuple[int, int],
    matches: list[tuple[int, int, int, int]],
    color_bgra: tuple[int, int, int, int] = (0, 255, 255, 160),
    thickness: int = 2,
    fill: bool = False,
) -> np.ndarray:
    """
    Return a BGRA mask (H,W,4) with rectangles drawn for each match.
    - size_wh: (width, height) of the target window/client area
    - matches: [(x, y, w, h), ...] in *window-local* coordinates
    - color_bgra: (B,G,R,A)
    - thickness: rectangle border width; ignored if fill=True
    - fill: if True, filled rectangles (thickness=-1)
    """
    w, h = size_wh
    bgra = np.zeros((h, w, 4), dtype=np.uint8)  # C-contiguous
    b, g, r, a = color_bgra
    t = -1 if fill else int(thickness)

    # Draw directly on the 4-channel image to avoid non-contiguous BGR views.
    for (x, y, ww, hh) in matches:
        if ww <= 0 or hh <= 0:
            continue
        x0 = max(0, min(w - 1, x))
        y0 = max(0, min(h - 1, y))
        x1 = max(0, min(w - 1, x + ww - 1))
        y1 = max(0, min(h - 1, y + hh - 1))
        if x1 < x0 or y1 < y0:
            continue
        cv2.rectangle(
            bgra,
            (x0, y0),
            (x1, y1),
            (b, g, r, a),
            t,
            lineType=cv2.LINE_AA,
        )

    return bgra