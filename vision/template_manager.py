from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import ctypes as C
import numpy as np
import cv2

from vision.capture import ScreenCapture, get_window_rect
from vision.matchers import TemplateMatcher
from utils.prof import Profiler

user32 = C.windll.user32

# ---- helpers ----------------------------------------------------------------

def _rgb_to_bgra(
    color_rgb: Tuple[int, int, int] | Tuple[int, int, int, int],
    default_a: int,
) -> Tuple[int, int, int, int]:
    """
    Convert RGB/RGBA (0..255) to BGRA. If alpha not provided, use default_a.
    """
    if len(color_rgb) == 4:
        r, g, b, a = color_rgb
    else:
        r, g, b = color_rgb
        a = default_a
    return (b, g, r, a)

# Nicely spaced defaults (RGB). We cycle if fewer than #templates.
_DEFAULT_PALETTE: List[Tuple[int, int, int]] = [
    (255, 99, 71),     # tomato
    (30, 144, 255),    # dodger blue
    (60, 179, 113),    # medium sea green
    (255, 215, 0),     # gold
    (186, 85, 211),    # medium orchid
    (255, 140, 0),     # dark orange
    (0, 206, 209),     # dark turquoise
    (220, 20, 60),     # crimson
]

# ---- config / records -------------------------------------------------------

@dataclass
class TMConfig:
    score_thresh: float = 0.90
    nms_iou: float = 0.40
    border_thickness: int = 2
    fill: bool = False
    alpha: int = 160
    max_hits_per_template: int = 50
    max_total_hits: int = 200
    peak_kernel: int = 5
    debug: bool = False  # enable per-frame profiling

@dataclass
class MatchRecord:
    x: int
    y: int
    w: int
    h: int
    score: float
    tmpl_idx: int

# ---- manager ----------------------------------------------------------------

class TemplateManager:
    """
    Manages multi-template matching against a target window and draws all matches
    onto a single BGRA mask (window-sized).

    - templates: list of grayscale templates (or BGR/BGRA convertible), uint8
    - colors: optional list of per-template RGB/RGBA; auto-assigned if omitted
    - capture: ScreenCapture instance
    - hwnd/client_only: target window and whether to use client rect
    - config: TMConfig thresholds & drawing params

    Usage each frame:
        mask = template_manager.update()  # does capture + match + draw
        ov.update_img(mask)
    """
    def __init__(
        self,
        templates: Sequence[np.ndarray],
        colors: Optional[Sequence[Tuple[int, int, int] | Tuple[int, int, int, int]]] = None,
        capture: Optional[ScreenCapture] = None,
        hwnd: Optional[int] = None,
        client_only: bool = False,
        config: Optional[TMConfig] = None,
    ):
        if len(templates) == 0:
            raise ValueError("TemplateManager requires at least one template.")

        # Ensure all templates are single-channel uint8
        self.templates = [self._to_gray_uint8(t) for t in templates]
        self.matcher = TemplateMatcher(self.templates)

        self.capture = capture or ScreenCapture()
        self.hwnd = hwnd
        self.client_only = client_only
        self.cfg = config or TMConfig()

        # Normalize per-template colors (BGRA)
        self.colors_bgra: List[Tuple[int, int, int, int]] = []
        if colors is not None:
            if len(colors) != len(self.templates):
                raise ValueError("If provided, colors must match number of templates.")
            for c in colors:
                self.colors_bgra.append(_rgb_to_bgra(c, self.cfg.alpha))
        else:
            for i in range(len(self.templates)):
                self.colors_bgra.append(
                    _rgb_to_bgra(_DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)], self.cfg.alpha)
                )

        # Fallback to full screen if no hwnd
        self._screen_w = user32.GetSystemMetrics(0)
        self._screen_h = user32.GetSystemMetrics(1)

        # Expose last profiler for external printing
        self.last_profiler: Optional[Profiler] = None

        # Save matches coords
        self.last_matches: list = []

    # -- public API -----------------------------------------------------------

    def set_target(self, hwnd: Optional[int], client_only: bool = False) -> None:
        self.hwnd = hwnd
        self.client_only = client_only

    def update(self) -> np.ndarray:
        """
        1) Capture grayscale image of the target window (or full screen)
        2) Run multi-template match with optional profiling
        3) Draw all matches to a single BGRA mask and return it
        """
        # 1) Determine capture rect (window or fullscreen)
        if self.hwnd:
            left, top, width, height = get_window_rect(self.hwnd, client_only=self.client_only)
        else:
            left, top, width, height = 0, 0, self._screen_w, self._screen_h

        # 2) Capture as grayscale for matching
        gray = self.capture.grab((left, top, width, height), gray=True)

        # 3) Match all templates in one pass
        prof = Profiler(enabled=self.cfg.debug)
        raw_matches = self.matcher.match(
            gray,
            score_thresh=self.cfg.score_thresh,
            nms_iou=self.cfg.nms_iou,
            max_hits_per_template=self.cfg.max_hits_per_template,
            max_total_hits=self.cfg.max_total_hits,
            peak_kernel=self.cfg.peak_kernel,
            prof=prof,
        )
        self.last_profiler = prof  # make available for the caller

        # 4) Draw one BGRA mask containing all matches (per-template colors)
        mask = np.zeros((height, width, 4), dtype=np.uint8)
        thickness = -1 if self.cfg.fill else int(max(1, self.cfg.border_thickness))

        for m in raw_matches:
            x0, y0, ww, hh = int(m.x), int(m.y), int(m.w), int(m.h)
            if ww <= 0 or hh <= 0:
                continue

            # clamp to mask bounds
            x1 = max(0, min(width - 1, x0 + ww - 1))
            y1 = max(0, min(height - 1, y0 + hh - 1))
            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            if x1 < x0 or y1 < y0:
                continue

            b, g, r, a = self.colors_bgra[m.tmpl_idx]

            # Draw directly onto the 4-channel mask (B,G,R,A), all args positional.
            # This matches your working approach in mask.py.
            cv2.rectangle(
                mask,
                (x0, y0),
                (x1, y1),
                (int(b), int(g), int(r), int(a)),
                int(thickness),
                cv2.LINE_AA,
            )
        
        if self.hwnd:
            left, top, width, height = get_window_rect(self.hwnd, client_only=self.client_only)
            for m in raw_matches:
                m.x += left
                m.y += top

        self.last_matches = raw_matches

        return mask

    def get_absolute_coords(self, match) -> tuple[int, int, int, int]:
        """
        Convert match-relative coords (x, y, w, h) to absolute screen coords.
        Returns (abs_x, abs_y, abs_w, abs_h)
        """
        if self.hwnd:
            left, top, width, height = get_window_rect(self.hwnd, client_only=self.client_only)
        else:
            left, top, width, height = 0, 0, self._screen_w, self._screen_h

        abs_x = left + match.x
        abs_y = top  + match.y
        return abs_x, abs_y, match.w, match.h


    def get_absolute_center(self, match) -> tuple[int, int]:
        """
        Return the center point of a match in absolute desktop coordinates.
        """
        abs_x, abs_y, w, h = self.get_absolute_coords(match)
        cx = abs_x + w // 2
        cy = abs_y + h // 2
        return cx, cy

    def get_last_profile_summary(self) -> Optional[str]:
        """
        Returns a human-friendly per-frame timing summary or None if profiling is off/absent.
        """
        if not self.last_profiler or not self.last_profiler.enabled:
            return None
        # Group all per-template timings under "tm.t"
        return self.last_profiler.summary_str(group_prefixes=["tm.t"])

    # -- utils ----------------------------------------------------------------

    @staticmethod
    def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Template is None")
        if img.ndim == 2:
            out = img
        elif img.ndim == 3:
            # Convert BGR/BGRA -> Gray
            if img.shape[2] == 4:
                img = img[..., :3]
            out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported template shape")
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        return out
