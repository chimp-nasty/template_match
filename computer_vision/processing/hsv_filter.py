import numpy as np
import cv2
from typing import List, Tuple, Dict


def _clip_int(v, lo, hi):
    return max(lo, min(int(v), hi))

def _frac_to_px(v, maxdim: int, name: str) -> int:
    """
    Convert a fractional dimension (0..1] to pixels.
    - v must be float/int in (0, 1]; ints like 1 are allowed but treated as 1.0
    """
    if v is None:
        v = 1.0  # default to full dimension if missing
    try:
        vf = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a fraction in (0, 1], got {v!r}")
    if not (0.0 < vf <= 1.0):
        raise ValueError(f"{name} must be a fraction in (0, 1], got {vf}")
    px = int(round(vf * maxdim))
    return max(1, min(px, maxdim))


class HSVFilter:
    """
    Finds ROIs via HSV color bands. Color bands are stored once as NumPy arrays:
      self.bands = {'R': [(lo_u8, hi_u8), ...], 'G': [...], 'B': [...]}
    """

    def __init__(self, filters: Dict[str, List[Tuple[int, int, int]]]):
        self.filters = filters        # human HSV (deg 0..360, S/V 0..100)
        self.bands: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"R": [], "G": [], "B": []}
        # reusable morphology kernels
        self.K_OPEN  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.K_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

    def hsv_convert(self, h, s, v):
        # human → OpenCV HSV
        h = _clip_int(round(h / 2), 0, 179)
        s = _clip_int(round(s * 2.55), 0, 255)
        v = _clip_int(round(v * 2.55), 0, 255)
        return h, s, v

    def create_filters(self) -> None:
        """Build NumPy lo/hi pairs once per color key ('R','G','B')."""
        def norm_band(lo_hsv, hi_hsv):
            loH, loS, loV = lo_hsv
            hiH, hiS, hiV = hi_hsv
            loH = _clip_int(loH % 360, 0, 360)
            hiH = _clip_int(hiH % 360, 0, 360)
            loS = _clip_int(loS, 0, 100); loV = _clip_int(loV, 0, 100)
            hiS = _clip_int(hiS, 0, 100); hiV = _clip_int(hiV, 0, 100)

            if loH <= hiH:
                bands_human = [((loH, loS, loV), (hiH, hiS, hiV))]
            else:
                # wrap around 360
                bands_human = [((loH, loS, loV), (360, hiS, hiV)),
                               ((0,   loS, loV), (hiH, hiS, hiV))]
            bands_cv = []
            for lo, hi in bands_human:
                lo_cv = self.hsv_convert(*lo)
                hi_cv = self.hsv_convert(*hi)
                bands_cv.append((np.array(lo_cv, np.uint8), np.array(hi_cv, np.uint8)))
            return bands_cv

        # Build R/G/B lists if present in self.filters
        self.bands["R"] = norm_band(*self.filters["red"])    if "red"   in self.filters else []
        self.bands["G"] = norm_band(*self.filters["green"])  if "green" in self.filters else []
        self.bands["B"] = norm_band(*self.filters["blue"])   if "blue"  in self.filters else []

    # --- Mask helpers ---
    def create_mask_from_hsv(self, hsv: np.ndarray, bands_np: List[Tuple[np.ndarray, np.ndarray]], *, do_morph=True):
        """Build a binary mask from a precomputed HSV image and prebuilt bands."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_arr, hi_arr in bands_np:
            mask |= cv2.inRange(hsv, lo_arr, hi_arr)
        if do_morph:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.K_OPEN,  iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.K_CLOSE, iterations=1)
        return mask

    def create_mask(self, bgr: np.ndarray, bands_np: List[Tuple[np.ndarray, np.ndarray]], *, do_morph=True):
        """Convenience: converts BGR→HSV then defers to create_mask_from_hsv."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return self.create_mask_from_hsv(hsv, bands_np, do_morph=do_morph)

    # --- Cropping ---
    def crop_region(self, img: np.ndarray, region: dict):
        """
        region = {"anchor": "tl"|"tr"|"bl"|"br"|"center", "w": (0,1], "h": (0,1]}
        Returns (crop, (x0,y0,w,h)) in full-image coords.
        """
        H, W = img.shape[:2]
        anchor = (region.get("anchor") or "tl").lower()
        if anchor not in ("tl", "tr", "bl", "br", "center"):
            anchor = "tl"
        w = _frac_to_px(region.get("w"), W, "w")
        h = _frac_to_px(region.get("h"), H, "h")

        if anchor == "tl":
            x0, y0 = 0, 0
        elif anchor == "tr":
            x0, y0 = W - w, 0
        elif anchor == "bl":
            x0, y0 = 0, H - h
        elif anchor == "br":
            x0, y0 = W - w, H - h
        else:
            x0 = max(0, (W - w) // 2)
            y0 = max(0, (H - h) // 2)

        crop = img[y0:y0+h, x0:x0+w]
        return crop, (x0, y0, w, h)

    # --- ROI finding ---
    def _roi_passes_size(self, w:int, h:int, area:int, *,
                         min_area:int = 300, max_area:int | None = None,
                         min_w:int | None = None, max_w:int | None = None,
                         min_h:int | None = None, max_h:int | None = None,
                         min_ar:float | None = None, max_ar:float | None = None) -> bool:
        if area < (min_area or 0): return False
        if max_area is not None and area > max_area: return False
        if min_w is not None and w < min_w: return False
        if max_w is not None and w > max_w: return False
        if min_h is not None and h < min_h: return False
        if max_h is not None and h > max_h: return False
        if h > 0 and (min_ar is not None or max_ar is not None):
            ar = w / h
            if min_ar is not None and ar < min_ar: return False
            if max_ar is not None and ar > max_ar: return False
        return True

    def find_roi(self, mask: np.ndarray, *,
                 min_area: int = 300,
                 max_area: int | None = None,
                 min_w: int | None = None, max_w: int | None = None,
                 min_h: int | None = None, max_h: int | None = None,
                 min_ar: float | None = None, max_ar: float | None = None):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not self._roi_passes_size(w, h, area,
                                         min_area=min_area, max_area=max_area,
                                         min_w=min_w, max_w=max_w,
                                         min_h=min_h, max_h=max_h,
                                         min_ar=min_ar, max_ar=max_ar):
                continue
            rois.append((x, y, w, h))
        return sorted(rois, key=lambda r: r[2], reverse=True)  # by width

    def find_roi_polygons(self, mask: np.ndarray, *,
                          min_area: int = 300,
                          max_area: int | None = None,
                          min_w: int | None = None, max_w: int | None = None,
                          min_h: int | None = None, max_h: int | None = None,
                          min_ar: float | None = None, max_ar: float | None = None,
                          epsilon_frac: float = 0.02) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: List[np.ndarray] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not self._roi_passes_size(w, h, area,
                                         min_area=min_area, max_area=max_area,
                                         min_w=min_w, max_w=max_w,
                                         min_h=min_h, max_h=max_h,
                                         min_ar=min_ar, max_ar=max_ar):
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon_frac * peri, True)
            polys.append(approx)
        return polys

    # --- Offset helpers (crop → full image) ---
    def offset_rois(self,
                    rois: List[Tuple[int, int, int, int]],
                    offset: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        ox, oy = offset
        return [(x + ox, y + oy, w, h) for (x, y, w, h) in rois]

    def offset_polygons(self, polys: List[np.ndarray], offset: Tuple[int, int]) -> List[np.ndarray]:
        ox, oy = offset
        out = []
        for p in polys:
            q = p.copy()
            q[:, 0, 0] += ox
            q[:, 0, 1] += oy
            out.append(q)
        return out

    # Stores high fidelity polys cheaper for runtime ops
    def simplify_polys(self, polys, epsilon_frac=0.01):
        out = []
        for p in polys:
            peri = cv2.arcLength(p, True)
            out.append(cv2.approxPolyDP(p, epsilon_frac * peri, True))
        return out

    # --- Drawing (testing/visualization only) ---
    def draw_roi(self, img: np.ndarray,
                 rois: List[Tuple[int, int, int, int]],
                 *, color: Tuple[int, int, int] = (0, 255, 255),
                 thickness: int = 2, copy: bool = True, labels: bool = False) -> np.ndarray:
        if not rois:
            return img
        canvas = img.copy() if copy else img
        for i, (x, y, w, h) in enumerate(rois):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
            if labels:
                cv2.putText(canvas, f"{i}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return canvas

    def draw_polygons(self, img: np.ndarray, polys: List[np.ndarray], *,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 2, copy: bool = True,
                      fill: bool = False, labels: bool = False) -> np.ndarray:
        if not polys:
            return img
        canvas = img.copy() if copy else img
        if fill:
            cv2.fillPoly(canvas, polys, color)
        else:
            cv2.polylines(canvas, polys, isClosed=True, color=color, thickness=thickness)
        if labels:
            for i, poly in enumerate(polys):
                x, y, w, h = cv2.boundingRect(poly)
                cv2.putText(canvas, str(i), (x, max(0, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return canvas

    # --- Preview ---
    def display_img(self, img, title: str = "preview", scale: float = 1.0, wait: int = 0, destroy: bool = False):
        if img is None:
            return
        if img.dtype == bool:
            img = (img.astype("uint8") * 255)
        vis = img
        if scale != 1.0:
            h, w = vis.shape[:2]
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=interp)
        cv2.imshow(title, vis)
        cv2.waitKey(wait)
        if destroy:
            cv2.destroyWindow(title)

    # --- Simple ROI fill helpers (area based) ---
    def roi_mask(self, frame_bgr, roi, bands_np):
        """Return binary mask for a color inside a given ROI (converts that ROI to HSV)."""
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            return None
        crop = frame_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            return None
        return self.create_mask(crop, bands_np)

    def roi_fill_ratio(self, frame_bgr, roi, bands_np):
        mask = self.roi_mask(frame_bgr, roi, bands_np)
        if mask is None:
            return 0.0
        filled = cv2.countNonZero(mask)
        total  = mask.shape[0] * mask.shape[1]
        return (filled / total) if total else 0.0
    



# -------- Example usage --------
if __name__ == "__main__":
    from pathlib import Path
    import sys

    def resource_path(rel: str) -> Path:
        if hasattr(sys, "_MEIPASS"):
            base = Path(sys._MEIPASS)
        elif "__file__" in globals():
            base = Path(__file__).resolve().parent
        else:
            base = Path.cwd()
        return (base / rel).resolve()

    IMG_PATH = resource_path("epoch.jpeg")
    img = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)

    FILTERS = {
        "red":   [(340, 24, 16), (20, 100, 100)],
        "green": [(70,  24, 39), (170, 100, 100)],
        "blue":  [(160, 20, 27), (240, 100, 100)],
    }

    ROI = "poly"  # 'rect' or 'poly'

    hsv_filter = HSVFilter(FILTERS)
    hsv_filter.create_filters()

    # Crop for detection (bottom-left 100% width, 15% height)
    cropped_img, (x0, y0, w, h) = hsv_filter.crop_region(img, {"anchor": "bl", "w": 1.0, "h": 0.15})

    # Convert HSV ONCE for the crop; then build masks from prebuilt bands
    crop_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    mask_r = hsv_filter.create_mask_from_hsv(crop_hsv, hsv_filter.bands['R'])
    mask_b = hsv_filter.create_mask_from_hsv(crop_hsv, hsv_filter.bands['B'])

    if ROI == "rect":
        rois_r = hsv_filter.find_roi(mask_r, min_area=10000, max_area=18000)
        rois_b = hsv_filter.find_roi(mask_b, min_area=10000, max_area=18000)
        
        abs_rois = hsv_filter.offset_rois(rois_r + rois_b, (x0, y0))
        
    else:  # ROI == "poly"
        polys_r = hsv_filter.find_roi_polygons(mask_r, min_area=10000, max_area=18000, epsilon_frac=0.001)
        polys_b = hsv_filter.find_roi_polygons(mask_b, min_area=10000, max_area=18000, epsilon_frac=0.001)
        
        abs_rois = hsv_filter.offset_polygons(polys_r + polys_b, (x0, y0))
        abs_polys_rt = hsv_filter.simplify_polys(abs_rois, epsilon_frac=0.01)
        
    vis = hsv_filter.draw_polygons(img, abs_rois, color=(0, 200, 0), thickness=2, copy=True)
    hsv_filter.display_img(vis, scale=0.5)
