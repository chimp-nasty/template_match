from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import Dict, List, Tuple
from vision.template_manager import MatchRecord
from vision.capture import ScreenCapture
import ctypes as C

user32 = C.windll.user32
SM_XVIRTUALSCREEN  = 76
SM_YVIRTUALSCREEN  = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

def _virtual_screen_bounds() -> Tuple[int, int, int, int]:
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    return vx, vy, vw, vh

def _clamp_roi(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    vx, vy, vw, vh = _virtual_screen_bounds()
    x1 = max(x, vx)
    y1 = max(y, vy)
    x2 = min(x + w, vx + vw)
    y2 = min(y + h, vy + vh)
    gw = max(0, x2 - x1)
    gh = max(0, y2 - y1)
    return x1, y1, gw, gh

def _normalize_digit_patch(patch: np.ndarray) -> np.ndarray:
    """Make foreground white on black, denoise, keep size."""
    g = patch.astype(np.uint8, copy=False)

    # light blur to stabilize thresholding
    g = cv2.GaussianBlur(g, (3,3), 0)

    # Otsu binarize
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Decide foreground color by looking at the center area
    h, w = bw.shape
    cy0, cy1 = h//4, 3*h//4
    cx0, cx1 = w//4, 3*w//4
    center_mean = bw[cy0:cy1, cx0:cx1].mean()

    # If background is white (center bright), digits are dark → invert
    if center_mean > 127:
        bw = 255 - bw

    # optional: thin / clean
    # bw = cv2.medianBlur(bw, 3)

    return bw.astype(np.float32)

def _split_digits_by_projection(gray: np.ndarray, max_digits=6) -> list[tuple[int,int,int,int]]:
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # make digits white on black for projection
    if bw[h//4:3*h//4, w//4:3*w//4].mean() > 127:
        bw = 255 - bw

    col = (bw > 0).sum(axis=0).astype(np.int32)  # ink per column
    gap = col <= max(1, int(0.05 * h))           # columns with little ink = gap

    boxes = []
    x0 = None
    for x in range(w):
        if not gap[x] and x0 is None:
            x0 = x
        if (gap[x] or x == w-1) and x0 is not None:
            x1 = x if gap[x] else x+1
            if x1 - x0 >= 3:                     # min width
                boxes.append((x0, 0, x1 - x0, h))
            x0 = None

    # keep the left-most `max_digits` largest segments
    boxes.sort(key=lambda b: b[2], reverse=True)
    boxes = boxes[:max_digits]
    boxes.sort(key=lambda b: b[0])               # left→right

    if not boxes:
        boxes = [(0,0,w,h)]
    return boxes

@dataclass
class DigitMatch:
    char: str
    score: float

class DigitBank:
    """
    Builds templates 0..9 from a known TTF/OTF at a fixed pixel height.
    Fast per-box classification using cv2.matchTemplate (TM_CCOEFF_NORMED).
    """
    def __init__(self, ttf_path: str, pixel_height: int, blur_ksize: int = 3):
        self.pixel_height = int(pixel_height)
        self.bank: Dict[str, np.ndarray] = self._build_bank(ttf_path, self.pixel_height, blur_ksize)

    @staticmethod
    def _build_bank(ttf_path: str, pixel_height: int, blur_ksize: int) -> Dict[str, np.ndarray]:
        font = ImageFont.truetype(ttf_path, pixel_height)
        out: Dict[str, np.ndarray] = {}
        for ch in "0123456789":
            l, t, r, b = font.getbbox(ch)
            w = (r - l) + 4
            h = (b - t) + 4
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).text((2 - l, 2 - t), ch, 255, font=font)
            arr = np.array(im, dtype=np.uint8)
            if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
                arr = cv2.GaussianBlur(arr, (blur_ksize, blur_ksize), 0)
            out[ch] = np.ascontiguousarray(arr.astype(np.float32))
        return out

    def classify_patch(self, patch_gray: np.ndarray) -> DigitMatch:
        """
        patch_gray: grayscale uint8 or float32 patch of a SINGLE digit, already roughly
        normalized to this bank's height (we'll center/pad to template size).
        """
        if patch_gray.dtype != np.float32:
            patch = patch_gray.astype(np.float32, copy=False)
        else:
            patch = patch_gray
        best_c, best_s = "?", -1.0
        for ch, tmpl in self.bank.items():
            th, tw = tmpl.shape
            H, W = patch.shape
            if H != th or W != tw:
                canvas = np.zeros((th, tw), np.float32)
                y0 = max(0, (th - H) // 2)
                x0 = max(0, (tw - W) // 2)
                y1 = min(th, y0 + H)
                x1 = min(tw, x0 + W)
                canvas[y0:y1, x0:x1] = patch[0:(y1 - y0), 0:(x1 - x0)]
                roi = canvas
            else:
                roi = patch
            s = float(cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            if s > best_s:
                best_c, best_s = ch, s
        return DigitMatch(best_c, best_s)

    def read_fixed_boxes(self, gray_img: np.ndarray, boxes: List[Tuple[int,int,int,int]], score_thresh: float = 0.65) -> Tuple[str, List[float]]:
        out_chars, scores = [], []
        for (x, y, w, h) in boxes:
            patch = gray_img[y:y+h, x:x+w]
            if patch.size == 0:
                out_chars.append("?"); continue

            patch = _normalize_digit_patch(patch)  # <-- add this

            H, W = patch.shape
            if H != self.pixel_height:
                scale = self.pixel_height / max(1, H)
                patch = cv2.resize(patch, (max(1, int(W*scale)), self.pixel_height), interpolation=cv2.INTER_AREA)

            m = self.classify_patch(patch)
            out_chars.append(m.char if m.score >= score_thresh else "?")
            scores.append(float(m.score))
        return "".join(out_chars), scores


class DigitReader:
    def __init__(
        self,
        ttf_path: str,
        pixel_height: int = 18,
        blur_ksize: int = 3,
        roi_offset: Tuple[int, int] = (50, 0),   # offset from match box
        roi_size: Tuple[int, int] = (80, 20),    # region width × height
        score_thresh: float = 0.65,
    ):
        self.bank = DigitBank(ttf_path, pixel_height, blur_ksize)
        self.cap = ScreenCapture()
        self.roi_offset = roi_offset
        self.roi_size = roi_size
        self.score_thresh = score_thresh
        self.last_results: List[Tuple[int, int, int, int, int, str, float]] = []

    def process_matches(self, matches: List[MatchRecord]) -> None:
        self.last_results.clear()
        if not matches:
            return

        rw, rh = map(int, self.roi_size)

        for m in matches:
            rx = int(m.x + self.roi_offset[0])
            ry = int(m.y + self.roi_offset[1])

            gx, gy, gw, gh = _clamp_roi(rx, ry, rw, rh)
            if gw <= 0 or gh <= 0:
                self.last_results.append((m.tmpl_idx, gx, gy, gw, gh, "", 0.0))
                continue

            gray = self.cap.grab((gx, gy, gw, gh), gray=True)

            # Robust split by projection (find real digit boundaries)
            boxes = _split_digits_by_projection(gray, max_digits=4)

            # Read digits
            text, scores = self.bank.read_fixed_boxes(gray, boxes, self.score_thresh)
            avg_score = float(np.mean(scores)) if scores else 0.0

            # Store (tmpl_idx, x, y, w, h, text, avg_score)
            self.last_results.append((m.tmpl_idx, gx, gy, gw, gh, text, avg_score))
            
    def close(self):
        self.cap.close()