from processing.hsv_filter import HSVFilter
import cv2

# -------- Example usage --------
if __name__ == "__main__":
    from utils.helpers import resource_path

    IMG_PATH = resource_path("tooling/example_imgs/epoch.jpeg")
    img = cv2.imread(str(IMG_PATH), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    FILTERS = {
        "red":   [(340, 24, 16), (20, 100, 100)],
        "green": [(70,  24, 39), (170, 100, 100)],
        "blue":  [(160, 20, 27), (240, 100, 100)],
    }

    ROI = "rect"  # 'rect' or 'poly'

    hsv_filter = HSVFilter()

    # add in our custom filters
    hsv_filter.create_filters(name="Health", hsv_range=FILTERS["red"])
    hsv_filter.create_filters(name="G", hsv_range=FILTERS["green"])
    hsv_filter.create_filters(name="Mana", hsv_range=FILTERS["blue"])

    # Crop for detection (bottom-left 100% width, 15% height)
    cropped_img, (x0, y0, w, h) = hsv_filter.crop_region(img, {"anchor": "bl", "w": 1.0, "h": 0.15})

    # Convert HSV ONCE for the crop; then build masks from prebuilt bands
    crop_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    mask_r = hsv_filter.create_mask_from_hsv(crop_hsv, hsv_filter.bands['Health'])
    mask_b = hsv_filter.create_mask_from_hsv(crop_hsv, hsv_filter.bands['Mana'])

    if ROI == "rect":
        rois_r = hsv_filter.find_roi(mask_r, min_area=10000, max_area=18000)
        rois_b = hsv_filter.find_roi(mask_b, min_area=10000, max_area=18000)
        
        abs_rois = hsv_filter.offset_rois(rois_r + rois_b, (x0, y0))

        vis = hsv_filter.draw_roi(img, abs_rois, color=(0, 200, 0), thickness=2, copy=True)

        import time
        roi  = abs_rois[0]
        band = hsv_filter.bands['Health']
        t0 = time.perf_counter()

        # benchmarking the fill ratio (1000 cycles)
        for _ in range(1000):
            fill_ratio = hsv_filter.fill_ratio_any(
                hsv,
                roi,
                band,
                is_hsv=True,
                do_morph=False
            )
        print(fill_ratio)
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) * 1000 / 1000
        print(f"Avg per call: {avg_ms:.5f} ms")
        
    else:  # ROI == "poly"
        polys_r = hsv_filter.find_roi_polygons(mask_r, min_area=10000, max_area=18000, epsilon_frac=0.001)
        polys_b = hsv_filter.find_roi_polygons(mask_b, min_area=10000, max_area=18000, epsilon_frac=0.001)
        
        abs_rois = hsv_filter.offset_polygons(polys_r + polys_b, (x0, y0))
        abs_polys_rt = hsv_filter.simplify_polys(abs_rois, epsilon_frac=0.01)
        
        vis = hsv_filter.draw_polygons(img, abs_rois, color=(0, 200, 0), thickness=2, copy=True)

        import time
        roi  = abs_polys_rt[0]
        band = hsv_filter.bands['Health']

        # no hsv
        # t0 = time.perf_counter()
        # for _ in range(1000):
        #     # Simulates: new BGR frame arrives, we call fill_ratio_any on it.
        #     # (fill_ratio_any will do the BGR→HSV conversion internally each call)
        #     fill_ratio = hsv_filter.fill_ratio_any(
        #         img,         # pretend this is the fresh screenshot
        #         roi,
        #         band,
        #         is_hsv=False,
        #         do_morph=False
        #     )
        # t1 = time.perf_counter()

        # avg_ms = (t1 - t0) * 1000.0 / 1000
        # print(f"Avg per call (BGR→HSV each time): {avg_ms:.5f} ms")

        # with hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        t0 = time.perf_counter()
        for _ in range(1000):
            fill_ratio = hsv_filter.fill_ratio_any(
                hsv,            # use precomputed HSV
                roi,
                band,
                is_hsv=True,    # tell it not to re-convert each time
                do_morph=False  # keep fast path
            )
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) * 1000 / 1000
        print(f"Avg per call: {avg_ms:.5f} ms")

    hsv_filter.display_img(vis, scale=0.5)
