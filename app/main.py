from vision.template_manager import TemplateManager, TMConfig
from vision.capture import ScreenCapture
from vision.digits import DigitReader
from vision.utils import load_gray_template
from overlay.overlay import Overlay
from control.controller import Controller
import numpy as np
from time import perf_counter
import ctypes as C
from pathlib import Path

proj_root = Path(__file__).resolve().parents[1]
tpl_path  = proj_root / "templates" / "notepad_example.png"

def main():
    ov = Overlay(
        target_process="notepad.exe",
        title_substr="notepad",
        client_only=False,
        only_when_foreground=False,
    )

    # Build your templates (or load from disk)
    t1 = load_gray_template(tpl_path, trim_border=True, bg_thresh=0)

    cap = ScreenCapture()
    cfg = TMConfig(
        score_thresh=0.95,
        nms_iou=0.4,
        max_hits_per_template=30,
        max_total_hits=120,
        debug=False,
    )

    tm = TemplateManager(
        templates=[t1],
        colors=[(255, 0, 0)],
        capture=cap,
        hwnd=ov.target_hwnd,
        client_only=ov.client_only,
        config=cfg,
    )

    digit_reader = DigitReader(
        ttf_path=r"C:\Windows\Fonts\consola.ttf",
        pixel_height=23,       # match ROI height to minimize resampling
        blur_ksize=3,
        roi_offset=(0, 0),   # your known offset from the match box
        roi_size=(43, 23),     # exact size
        score_thresh=0.4,      # start lenient; tighten later to ~0.6–0.75
    )
    
    control = Controller(pause_key="SPACE", terminate_key="ESC")
    fps = 200
    dt = 1.0 / fps
    frame = 0
    next_t = perf_counter()

    try:
        while not control.terminated:
            control.poll()
            if not ov.tick():
                break

            if not control.paused:
                mask = tm.update()
                # print("matches this frame:", len(tm.last_matches))
                # for m in tm.last_matches:
                #     print(f" idx={m.tmpl_idx}  xy=({m.x},{m.y})  wh=({m.w}x{m.h})  s={m.score:.3f}")
                ov.update_img(mask)
                
                digit_reader.process_matches(tm.last_matches)
                for (ti, x, y, w, h, text, sc) in digit_reader.last_results:
                    print(f"[tmpl:{ti}] ROI=({x},{y},{w}x{h}) → {text} (avg={sc:.2f})")

                summary = tm.get_last_profile_summary()
                if summary and (frame % 30 == 0):
                    print(summary)

            ov.display_overlay()
            next_t = control.pace(next_t, dt)
            frame += 1
    finally:
        try:
            C.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        cap.close()
        ov.close()
        digit_reader.close()

if __name__ == "__main__":
    main()
