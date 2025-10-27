from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
import cv2
from utils.prof import Profiler

@dataclass
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float
    tmpl_idx: int

def _vectorized_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (4,) best box, b: (N,4) other boxes; coords are [x1,y1,x2,y2]
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def fast_nms(matches: List[Match], iou_thresh: float, max_total_hits: int) -> List[Match]:
    if not matches:
        return []

    # Build arrays
    boxes = []
    scores = []
    for m in matches:
        x1, y1 = m.x, m.y
        x2, y2 = m.x + m.w - 1, m.y + m.h - 1
        boxes.append((x1, y1, x2, y2))
        scores.append(m.score)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # Sort by score desc
    order = np.argsort(-scores)
    boxes = boxes[order]
    keep_idx = []

    while boxes.shape[0] > 0 and len(keep_idx) < max_total_hits:
        keep_idx.append(order[0])
        if boxes.shape[0] == 1:
            break
        best = boxes[0]
        rest = boxes[1:]
        iou = _vectorized_iou_xyxy(best, rest)
        mask = iou < iou_thresh
        # Keep only boxes with IoU below threshold
        boxes = boxes[1:][mask]
        order = order[1:][mask]

    # Rebuild kept matches
    kept = [matches[i] for i in keep_idx]
    # Keep original sort (optional): already approx by score
    return kept

class TemplateMatcher:
    """
    Multi-template matcher. Expects grayscale uint8 templates.
    """
    def __init__(self, templates: Sequence[np.ndarray]):
        if len(templates) == 0:
            raise ValueError("Need at least one template")
        self.templates = [self._to_gray_uint8(t) for t in templates]
        self.method = cv2.TM_CCOEFF_NORMED

    @staticmethod
    def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            out = img
        else:
            if img.shape[2] == 4:
                img = img[..., :3]
            out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return out if out.dtype == np.uint8 else np.clip(out, 0, 255).astype(np.uint8)

    def match(
        self,
        gray_img: np.ndarray,
        score_thresh: float = 0.9,
        nms_iou: float = 0.4,
        max_hits_per_template: int = 50,
        max_total_hits: int = 200,
        peak_kernel: int = 5,
        prof: Profiler | None = None,     # <<< add this
    ) -> List[Match]:
        if gray_img.ndim != 2:
            raise ValueError("match expects grayscale image")

        if prof is None:
            # tiny stub so we can use prof.section(...) safely
            class _Noop:
                def section(self, *a, **k):
                    from contextlib import nullcontext
                    return nullcontext()
            prof = _Noop()  # type: ignore

        all_hits: List[Match] = []
        H, W = gray_img.shape

        with prof.section("tm.total"):
            for ti, tmpl in enumerate(self.templates):
                th, tw = tmpl.shape[:2]
                if H < th or W < tw:
                    continue

                with prof.section(f"tm.t{ti}.matchTemplate"):
                    resp = cv2.matchTemplate(gray_img, tmpl, self.method)

                with prof.section(f"tm.t{ti}.peaks"):
                    k = int(peak_kernel)
                    if k < 1:
                        k = 1
                    if (k % 2) == 0:
                        k += 1
                    resp = resp.astype(np.float32, copy=False)
                    dil = cv2.dilate(resp, np.ones((k, k), np.uint8))
                    peaks = (resp == dil) & (resp >= score_thresh)
                    ys, xs = np.where(peaks)
                    if xs.size:
                        vals = resp[ys, xs]
                        if xs.size > max_hits_per_template:
                            sel = np.argpartition(-vals, max_hits_per_template - 1)[:max_hits_per_template]
                            xs = xs[sel]; ys = ys[sel]; vals = vals[sel]
                        for x, y, v in zip(xs.tolist(), ys.tolist(), vals.tolist()):
                            all_hits.append(Match(x=x, y=y, w=tw, h=th, score=float(v), tmpl_idx=ti))

            with prof.section("tm.nms"):
                out = fast_nms(all_hits, iou_thresh=nms_iou, max_total_hits=max_total_hits)

        return out
