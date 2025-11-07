import cv2
import time

from utils.hwnd_utils import hwnds_for
from utils.pacer import Pacer
from utils.helpers import resource_path

from vision.capture import Capture
from vision.display import Display

from manager.keyboard_controller import Controller
from processing.hsv_filter import HSVFilter

import config as cfg
import numpy as np

_SLIDER_NAMES = ("H min","H max","S min","S max","V min","V max")
_SLIDER_DEFAULTS = (0, 179, 0, 255, 0, 255)

def _make_hsv_controls():
    cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Controls", 360, 260)
    # create with a no-op callback to avoid C→Python calls while dragging
    for name, default, maxv in zip(
        _SLIDER_NAMES,
        _SLIDER_DEFAULTS,
        (179, 179, 255, 255, 255, 255),
    ):
        cv2.createTrackbar(name, "HSV Controls", default, maxv, lambda *_: None)

def _read_hsv_controls():
    g = cv2.getTrackbarPos
    lo = (g("H min","HSV Controls"), g("S min","HSV Controls"), g("V min","HSV Controls"))
    hi = (g("H max","HSV Controls"), g("S max","HSV Controls"), g("V max","HSV Controls"))
    return lo, hi

def _pump_gui(times=2):
    # Pump UI events for all windows; do this each loop
    for _ in range(times):
        cv2.waitKeyEx(1)


class HSVCLI:
    def __init__(self):
        self.hwnds = hwnds_for(cfg.PROCESS)
        self.hwnd = self.hwnds[0] if self.hwnds else None
        self.cap = Capture(monitor=1, hwnd=self.hwnd, pace_fps=cfg.FPS)
        self.dis = Display(scale=0.5, hotkey=cfg.CLOSE, fps_overlay=False)
        self.dis_mask = Display(window_name="HSV Mask", scale=0.5, hotkey=cfg.CLOSE, fps_overlay=False)
        self.con = Controller(pause_key=cfg.PAUSE, terminate_key=cfg.CLOSE)
        self.main_pacer = Pacer(fps=cfg.FPS, high_res=True)
        self.hsv_filter = HSVFilter()

        _make_hsv_controls()

    def start(self):
        try:
            while not self.con.terminated:
                self.con.poll()
                if self.con.paused:
                    # keep UI snappy even when paused
                    cv2.waitKeyEx(1); cv2.waitKeyEx(1)
                    time.sleep(0.02)
                    continue

                frame = self.cap.dxc_grab()
                if frame is None:
                    cv2.waitKeyEx(1); cv2.waitKeyEx(1)
                    self.main_pacer.pace()
                    continue

                # --- read sliders
                lo, hi = _read_hsv_controls()
                lo_np = np.array(lo, dtype=np.uint8)
                hi_np = np.array(hi, dtype=np.uint8)

                # --- ensure BGR before HSV
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame_bgr = frame

                # --- build mask once per frame
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lo_np, hi_np)

                # --- optional: clean mask a touch (cheap 3x3 open)
                # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

                # --- visualize: overlay mask color on original for context
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(frame_bgr, 0.65, mask_bgr, 0.35, 0.0)

                # --- show both
                if not self.dis.show(overlay):
                    break
                if not self.dis_mask.show(mask):
                    break

                # pump GUI a bit for both windows
                cv2.waitKeyEx(1); cv2.waitKeyEx(1)

                self.main_pacer.pace()

        except KeyboardInterrupt:
            self.con.terminated = True
        finally:
            self.dis.close()
            self.dis_mask.close()
            cv2.destroyWindow("HSV Controls")


if __name__ == "__main__":
    hsv_client = HSVCLI()
    hsv_client.start()

    # IMG_PATH = resource_path("tooling/example_imgs/epoch.jpeg")
    # img = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # FILTERS = {
    #     "red":   [(340, 24, 16), (20, 100, 100)],
    #     "green": [(70,  24, 39), (170, 100, 100)],
    #     "blue":  [(160, 20, 27), (240, 100, 100)],
    # }

    # ROI = "rect"  # 'rect' or 'poly'

    # hsv_filter = HSVFilter()

    # # add in our custom filters
    # hsv_filter.create_filters(name="Health", hsv_range=FILTERS["red"])
    # hsv_filter.create_filters(name="G", hsv_range=FILTERS["green"])
    # hsv_filter.create_filters(name="Mana", hsv_range=FILTERS["blue"])

    # # Crop for detection (bottom-left 100% width, 15% height)
    # cropped_img, (x0, y0, w, h) = hsv_filter.crop_region(img, {"anchor": "bl", "w": 1.0, "h": 0.15})

    # # Convert HSV ONCE for the crop; then build masks from prebuilt bands
    # crop_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # mask_r = hsv_filter.create_mask_from_hsv(crop_hsv, hsv_filter.bands['Health'])
    # mask_b = hsv_filter.create_mask_from_hsv(crop_hsv, hsv_filter.bands['Mana'])

    # if ROI == "rect":
    #     rois_r = hsv_filter.find_roi(mask_r, min_area=10000, max_area=18000)
    #     rois_b = hsv_filter.find_roi(mask_b, min_area=10000, max_area=18000)
        
    #     abs_rois = hsv_filter.offset_rois(rois_r + rois_b, (x0, y0))

    #     vis = hsv_filter.draw_roi(img, abs_rois, color=(0, 200, 0), thickness=2, copy=True)

    #     import time
    #     roi  = abs_rois[0]
    #     band = hsv_filter.bands['Health']
    #     t0 = time.perf_counter()

    #     # benchmarking the fill ratio (1000 cycles)
    #     for _ in range(1000):
    #         fill_ratio = hsv_filter.fill_ratio_any(
    #             hsv,
    #             roi,
    #             band,
    #             is_hsv=True,
    #             do_morph=False
    #         )
    #     print(f"fill_ratio: {round(fill_ratio*100, 2)} %")
    #     t1 = time.perf_counter()

    #     avg_ms = (t1 - t0) * 1000 / 1000
    #     print(f"Avg per call: {avg_ms:.5f} ms")
        
    # else:  # ROI == "poly"
    #     polys_r = hsv_filter.find_roi_polygons(mask_r, min_area=10000, max_area=18000, epsilon_frac=0.001)
    #     polys_b = hsv_filter.find_roi_polygons(mask_b, min_area=10000, max_area=18000, epsilon_frac=0.001)
        
    #     abs_rois = hsv_filter.offset_polygons(polys_r + polys_b, (x0, y0))
    #     abs_polys_rt = hsv_filter.simplify_polys(abs_rois, epsilon_frac=0.01)
        
    #     vis = hsv_filter.draw_polygons(img, abs_rois, color=(0, 200, 0), thickness=2, copy=True)

    #     import time
    #     roi  = abs_polys_rt[0]
    #     band = hsv_filter.bands['Health']

    #     # no hsv
    #     # t0 = time.perf_counter()
    #     # for _ in range(1000):
    #     #     # Simulates: new BGR frame arrives, we call fill_ratio_any on it.
    #     #     # (fill_ratio_any will do the BGR→HSV conversion internally each call)
    #     #     fill_ratio = hsv_filter.fill_ratio_any(
    #     #         img,         # pretend this is the fresh screenshot
    #     #         roi,
    #     #         band,
    #     #         is_hsv=False,
    #     #         do_morph=False
    #     #     )
    #     # t1 = time.perf_counter()

    #     # avg_ms = (t1 - t0) * 1000.0 / 1000
    #     # print(f"Avg per call (BGR→HSV each time): {avg_ms:.5f} ms")

    #     # with hsv
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #     t0 = time.perf_counter()
    #     for _ in range(1000):
    #         fill_ratio = hsv_filter.fill_ratio_any(
    #             hsv,            # use precomputed HSV
    #             roi,
    #             band,
    #             is_hsv=True,    # tell it not to re-convert each time
    #             do_morph=False  # keep fast path
    #         )
    #     t1 = time.perf_counter()

    #     avg_ms = (t1 - t0) * 1000 / 1000
    #     print(f"Avg per call: {avg_ms:.5f} ms")

    # hsv_filter.display_img(vis, scale=0.5)
