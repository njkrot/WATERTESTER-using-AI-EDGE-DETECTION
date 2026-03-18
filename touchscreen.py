# touchscreen gui - replaces the old 16x2 lcd
import sys
import os
import time
import json
import logging
import signal
import threading
import types

import tkinter as tk
import tkinter.font as tkfont

log = logging.getLogger("touchscreen")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

FULLSCREEN = False
WINDOW_W, WINDOW_H = 480, 320

USE_CAMERA = False
try:
    from picamera2 import Picamera2
    if hasattr(Picamera2, 'create_preview_configuration') or hasattr(Picamera2, 'capture_array'):
        USE_CAMERA = True
except ImportError:
    pass

from pump_control import PumpController, SOAK_SECONDS

# stub out pi cam libs so imports dont crash on pc
for _mod in ['picamera2', 'picamera2.devices', 'picamera2.devices.imx500',
             'picamera2.devices.imx500.postprocess']:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
        if _mod == 'picamera2.devices.imx500':
            sys.modules[_mod].NetworkIntrinsics = type('NI', (), {})
            sys.modules[_mod].postprocess_nanodet_detection = lambda *a, **k: None
        elif _mod == 'picamera2.devices':
            sys.modules[_mod].IMX500 = type('IMX500', (), {})
        elif _mod == 'picamera2':
            sys.modules[_mod].Picamera2 = type('Picamera2', (), {})

if 'gpiozero' not in sys.modules:
    sys.modules['gpiozero'] = types.ModuleType('gpiozero')
    sys.modules['gpiozero'].Button = type('Button', (), {})
    sys.modules['gpiozero'].LED = type('LED', (), {})
    sys.modules['gpiozero'].OutputDevice = type('OutputDevice', (), {
        '__init__': lambda self, *a, **k: None,
        'on': lambda self: None, 'off': lambda self: None,
        'close': lambda self: None,
    })

for _mod in ['RPLCD', 'RPLCD.i2c']:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
        if _mod == 'RPLCD.i2c':
            sys.modules[_mod].CharLCD = type('CharLCD', (), {})

import cv2
import numpy as np
from PROGRAM import (analyze_test_strip, compute_water_score,
                     filter_outlier_frames, average_pad_readings,
                     pad_roi_from_layout, robust_patch_lab,
                     PAD_LAYOUT, MIN_GOOD_FRAMES, MAX_CAPTURE_ATTEMPTS,
                     OUTLIER_LAB_THRESHOLD)
from StripEdgeRefinement import refine_strip_edges
from pump_control import (FILL_SECONDS, DRAIN_SECONDS,
                          FILTER_SECONDS, DISPENSE_TIMEOUT)

STRIP_ROI = (400, 100, 480, 520)
FRAME_W, FRAME_H = 1280, 720
CAPTURE_INTERVAL = 0.4
CALIBRATION_FILE = "calibration_data.json"


class WaterFilterApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Water Filter System")
        if FULLSCREEN:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry("{}x{}".format(WINDOW_W, WINDOW_H))
            self.root.resizable(False, False)

        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.pump = PumpController()
        self.picam2 = None  # set later if cam exists
        self._shutdown = threading.Event()
        self._worker_thread = None
        self._score = None
        self._warnings = []

        self._build_ui()

    def _build_ui(self):
        bg = "#1a1a2e"
        # font sizes might look different on the actual pi screen
        self._fonts = {
            "title": tkfont.Font(family="Helvetica", size=16, weight="bold"),
            "status": tkfont.Font(family="Helvetica", size=11),
            "score": tkfont.Font(family="Helvetica", size=28, weight="bold"),
            "btn": tkfont.Font(family="Helvetica", size=13, weight="bold"),
            "warn": tkfont.Font(family="Courier", size=9),
        }

        top = tk.Frame(self.root, bg=bg)
        top.pack(fill="x", padx=10, pady=(8,2))

        self._title_lbl = tk.Label(top, text="Water Filter System",
            font=self._fonts["title"], fg="#e0e0e0", bg=bg)
        self._title_lbl.pack(side="left")

        self._score_lbl = tk.Label(top, text="--", font=self._fonts["score"],
            fg="#33ff33", bg=bg)
        self._score_lbl.pack(side="right", padx=(0,5))

        # wraplength might need tweaking on 3.5" display
        self._status_lbl = tk.Label(self.root, text="Ready. Press START to begin.",
            font=self._fonts["status"], fg="#aaaaaa", bg=bg,
            wraplength=WINDOW_W-20, justify="left")
        self._status_lbl.pack(fill="x", padx=10, pady=(2,4))

        self._warn_frame = tk.Frame(self.root, bg=bg)
        self._warn_frame.pack(fill="both", expand=True, padx=10, pady=(0,4))

        self._warn_text = tk.Text(self._warn_frame, font=self._fonts["warn"],
            fg="#ff6666", bg="#0f0f1a", relief="flat",
            height=5, wrap="word", state="disabled",
            highlightthickness=0)
        self._warn_text.pack(fill="both", expand=True)

        btnf = tk.Frame(self.root, bg=bg)
        btnf.pack(fill="x", padx=10, pady=(0,10))

        self._start_btn = tk.Button(btnf, text="START", font=self._fonts["btn"],
            bg="#2d6a4f", fg="white", activebackground="#40916c",
            activeforeground="white", relief="flat",
            height=2, command=self._on_start)
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0,5))

        self._dispense_btn = tk.Button(btnf, text="DISPENSE", font=self._fonts["btn"],
            bg="#444444", fg="#888888", activebackground="#444444",
            activeforeground="#888888", relief="flat",
            height=2, state="disabled", command=self._on_dispense)
        self._dispense_btn.pack(side="right", fill="x", expand=True, padx=(5,0))

        calf = tk.Frame(self.root, bg=bg)
        calf.pack(fill="x", padx=10, pady=(0,6))

        self._cal_btn = tk.Button(calf, text="CALIBRATE",
            font=tkfont.Font(family="Helvetica", size=9),
            bg="#4a3728", fg="#ccaa88", activebackground="#6b4f3a",
            activeforeground="white", relief="flat",
            command=self._on_calibrate)
        self._cal_btn.pack(fill="x")

    def _set_status(self, text, color="#aaaaaa"):
        self.root.after(0, lambda: self._status_lbl.configure(text=text, fg=color))

    def _set_score(self, score):
        # color thresholds might need adjusting on pi display
        if score is None:
            txt = "--"
            col = "#33ff33"
        else:
            txt = "{}/100".format(score)
            if score >= 70:
                col = "#33ff33"
            elif score >= 40:
                col = "#ffaa00"
            else:
                col = "#ff4444"
        self.root.after(0, lambda: self._score_lbl.configure(text=txt, fg=col))

    def _set_warnings(self, warns):
        def _upd():
            self._warn_text.configure(state="normal")
            self._warn_text.delete("1.0", "end")
            if warns:
                for w in warns:
                    self._warn_text.insert("end", "! {}\n".format(w))
            else:
                self._warn_text.insert("end", "No issues detected.")
            self._warn_text.configure(state="disabled")
        self.root.after(0, _upd)

    def _enable_start(self, enabled=True):
        def _upd():
            if enabled:
                self._start_btn.configure(state="normal", bg="#2d6a4f", fg="white")
            else:
                self._start_btn.configure(state="disabled", bg="#444444", fg="#888888")
        self.root.after(0, _upd)

    def _enable_dispense(self, enabled=True):
        def _upd():
            if enabled:
                self._dispense_btn.configure(state="normal", bg="#1d3557", fg="white",
                    activebackground="#457b9d", activeforeground="white")
            else:
                self._dispense_btn.configure(state="disabled", bg="#444444", fg="#888888")
        self.root.after(0, _upd)

    def _enable_calibrate(self, enabled=True):
        def _upd():
            if enabled:
                self._cal_btn.configure(state="normal", bg="#4a3728", fg="#ccaa88")
            else:
                self._cal_btn.configure(state="disabled", bg="#333333", fg="#666666")
        self.root.after(0, _upd)

    def _lock_all_buttons(self):
        self._enable_start(False)
        self._enable_dispense(False)
        self._enable_calibrate(False)

    def _on_calibrate(self):
        self._lock_all_buttons()
        self._set_score(None)
        self._set_warnings([])
        t = threading.Thread(target=self._run_calibration, daemon=True)
        t.start()

    def _capture_strip_crop(self):
        if USE_CAMERA:
            self._init_camera()
            try:
                fr = self.picam2.capture_array("main")
            except Exception as e:
                log.warning("Capture failed: %s", e)
                return None
            if fr is None:
                return None
            bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        else:
            bgr = self._make_fake_strip_frame()

        rx,ry,rw,rh = STRIP_ROI
        fh,fw = bgr.shape[:2]
        x0 = max(0, min(rx, fw-1))
        y0 = max(0, min(ry, fh-1))
        x1 = max(x0+1, min(rx+rw, fw))
        y1 = max(y0+1, min(ry+rh, fh))
        crop = bgr[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        if crop.shape[1] > crop.shape[0]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        ref = refine_strip_edges(crop)
        if ref is not None and ref.size != 0:
            crop = ref
        return crop

    def _make_fake_strip_frame(self):
        """fake strip for testing"""
        # colors probably wont match real strip exactly
        fr = np.full((FRAME_H, FRAME_W, 3), (40,40,40), dtype=np.uint8)
        rx,ry,rw,rh = STRIP_ROI
        s = np.full((rh, rw, 3), (220,220,220), dtype=np.uint8)
        cols = [
            (210,190,200),(220,220,230),(180,195,215),
            (220,220,230),(220,220,230),(195,185,210),
            (160,170,200),(175,180,210),(220,220,230),
            (220,220,230),(140,150,195),(220,220,230),
            (220,220,230),(180,200,210),(180,200,210),
            (140,160,180),
        ]
        n = len(PAD_LAYOUT)
        ph = rh // n
        for i,c in enumerate(cols):
            y0 = i*ph
            y1 = y0+ph
            s[y0:y1, :] = c
        fr[ry:ry+rh, rx:rx+rw] = s
        return fr

    def _run_calibration(self):
        """cal mode"""
        try:
            self._set_status("Calibrating... capturing strip...", "#ccaa88")
            time.sleep(0.5)

            ncap = 3
            all_labs = {sp["name"]: [] for sp in PAD_LAYOUT}

            for c in range(ncap):
                self._set_status("Calibrating... capture {}/{}".format(c+1, ncap), "#ccaa88")
                crop = self._capture_strip_crop()
                if crop is None:
                    continue
                for sp in PAD_LAYOUT:
                    p, _ = pad_roi_from_layout(crop, sp)
                    lab = robust_patch_lab(p)
                    if lab is not None:
                        all_labs[sp["name"]].append(lab)
                time.sleep(0.3)

            res = {}
            lines = []
            for sp in PAD_LAYOUT:
                labs = all_labs[sp["name"]]
                if labs:
                    ml = round(float(np.median([l[0] for l in labs])), 1)
                    ma = round(float(np.median([l[1] for l in labs])), 1)
                    mb = round(float(np.median([l[2] for l in labs])), 1)
                    res[sp["name"]] = [ml, ma, mb]
                    lines.append("{:12s}  LAB = ({}, {}, {})".format(
                        sp["name"], ml, ma, mb))
                else:
                    res[sp["name"]] = None
                    lines.append("{:12s}  LAB = FAILED".format(sp["name"]))

            cal_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pads": res,
            }

            existing = []
            if os.path.exists(CALIBRATION_FILE):
                try:
                    with open(CALIBRATION_FILE, "r") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            existing.append(cal_entry)
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(existing, f, indent=2)

            log.info("Calibration results saved to %s", CALIBRATION_FILE)
            for ln in lines:
                log.info("  %s", ln)

            def _show():
                self._warn_text.configure(state="normal")
                self._warn_text.delete("1.0", "end")
                self._warn_text.insert("end", "CALIBRATION LAB VALUES\n")
                self._warn_text.insert("end", "-"*35 + "\n")
                for ln in lines:
                    self._warn_text.insert("end", ln + "\n")
                self._warn_text.insert("end", "\nSaved to {}\n".format(CALIBRATION_FILE))
                self._warn_text.configure(state="disabled")
            self.root.after(0, _show)

            self._set_status("Calibration done. {} pads captured.".format(
                sum(1 for v in res.values() if v is not None)), "#ccaa88")

        except Exception as e:
            log.error("Calibration error: %s", e)
            self._set_status("Calibration error: {}".format(e), "#ff4444")
        finally:
            self._enable_start(True)
            self._enable_calibrate(True)

    def _on_start(self):
        self._lock_all_buttons()
        self._set_score(None)
        self._set_warnings([])
        self._worker_thread = threading.Thread(target=self._run_full_cycle, daemon=True)
        self._worker_thread.start()

    def _on_dispense(self):
        if not self.pump.can_dispense:
            return
        self._lock_all_buttons()
        t = threading.Thread(target=self._run_dispense, daemon=True)
        t.start()

    def _run_dispense(self):
        def cb(rem):
            self._set_status("Dispensing... {:.0f}s left".format(rem), "#66ccff")
        self.pump.dispense(status_cb=cb)
        self._set_status("Dispense complete.", "#aaaaaa")
        self._enable_start(True)
        self._enable_dispense(True)
        self._enable_calibrate(True)

    #NEED?!
    def _init_camera(self):
        if USE_CAMERA and self.picam2 is None:
            self.picam2 = Picamera2()
            cfg = self.picam2.create_preview_configuration(
                main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
                controls={"FrameRate": 30},
                buffer_count=4,
            )
            self.picam2.start(cfg, show_preview=False)
            time.sleep(0.5)

    def _capture_and_analyze(self):
        if not USE_CAMERA:
            return None
        try:
            fr = self.picam2.capture_array("main")
        except Exception as e:
            log.warning("Capture failed: %s", e)
            return None
        if fr is None:
            return None

        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        rx,ry,rw,rh = STRIP_ROI
        fh,fw = bgr.shape[:2]
        x0 = max(0, min(rx, fw-1))
        y0 = max(0, min(ry, fh-1))
        x1 = max(x0+1, min(rx+rw, fw))
        y1 = max(y0+1, min(ry+rh, fh))
        crop = bgr[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return None
        if crop.shape[1] > crop.shape[0]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        ref = refine_strip_edges(crop)
        if ref is not None and ref.size != 0:
            crop = ref

        res = analyze_test_strip(crop)
        if "error" in res:
            return None
        return res

    def _read_strip_adaptive(self):
        """#CRIT - keep reading til enough good frames"""
        self._init_camera()
        rdgs = []
        att = 0

        while att < MAX_CAPTURE_ATTEMPTS:
            if self._shutdown.is_set():
                return None

            r = self._capture_and_analyze()
            if r is not None:
                rdgs.append(r)

            att += 1
            self._set_status("Reading strip... frame {}, {} good so far".format(
                att, len(rdgs)), "#ffaa00")

            if len(rdgs) >= MIN_GOOD_FRAMES:
                good, bad = filter_outlier_frames(rdgs, OUTLIER_LAB_THRESHOLD)
                if len(good) > bad and len(good) >= MIN_GOOD_FRAMES:
                    log.info("Got %d good frames, %d outliers.", len(good), bad)
                    return average_pad_readings(good)

            time.sleep(CAPTURE_INTERVAL)

        if not rdgs:
            return None

        if att >= MAX_CAPTURE_ATTEMPTS:
            self._set_status("Frame limit reached!", "#ff4444")
            log.error("Frame limit exceeded.")

        good, _ = filter_outlier_frames(rdgs, OUTLIER_LAB_THRESHOLD)
        return average_pad_readings(good) if good else None

    def _run_full_cycle(self):
        """DNT DEL - main cycle"""
        try:
            self._set_status("Filling test container...", "#66ccff")
            def fill_cb(rem):
                self._set_status("Filling test container... {:.0f}s".format(rem), "#66ccff")
            if not self.pump.fill_test_container(fill_cb):
                self._set_status("Aborted.", "#ff4444")
                self._enable_start(True)
                self._enable_calibrate(True)
                return

            self._set_status("Soaking strip...", "#66ccff")
            def soak_cb(rem):
                self._set_status("Soaking strip... {:.0f}s".format(rem), "#66ccff")
            if not self.pump.soak_strip(soak_cb):
                self._set_status("Aborted.", "#ff4444")
                self._enable_start(True)
                self._enable_calibrate(True)
                return

            self._set_status("Draining test water...", "#66ccff")
            def drain_cb(rem):
                self._set_status("Draining... {:.0f}s".format(rem), "#66ccff")
            if not self.pump.drain_test_container(drain_cb):
                self._set_status("Aborted.", "#ff4444")
                self._enable_start(True)
                self._enable_calibrate(True)
                return

            self._set_status("Reading strip...", "#ffaa00")
            if USE_CAMERA:
                results = self._read_strip_adaptive()
            else:
                self._set_status("No camera - simulating test results...", "#ffaa00")
                time.sleep(1.0)
                results = self._simulate_results()

            if results is None:
                self._set_status("Strip read failed!", "#ff4444")
                self._enable_start(True)
                self._enable_calibrate(True)
                return

            #CRIT
            sc, warns = compute_water_score(results)
            self._score = sc
            self._warnings = warns
            self._set_score(sc)
            self._set_warnings(warns)

            self.pump.mark_test_completed()

            if sc >= 70:
                self._set_status("Test complete. Score: {}/100. Filtering...".format(sc), "#33ff33")
            elif sc >= 40:
                self._set_status("Test complete. Score: {}/100. Warnings present. Filtering...".format(sc), "#ffaa00")
            else:
                self._set_status("Test complete. Score: {}/100. LOW QUALITY! Filtering...".format(sc), "#ff4444")

            log.info("Test done. Score: %d/100, %d warnings.", sc, len(warns))
            for w in warns:
                log.warning("  %s", w)

            def filter_cb(rem):
                self._set_status("Filtering water... {:.0f}s left".format(rem), "#66ccff")
            if not self.pump.run_filter(filter_cb):
                self._set_status("Filter aborted.", "#ff4444")
                self._enable_start(True)
                self._enable_calibrate(True)
                return

            self._set_status("Filtering complete. Press DISPENSE for clean water.", "#33ff33")
            self._enable_start(True)
            self._enable_dispense(True)
            self._enable_calibrate(True)

        except Exception as e:
            log.error("Cycle error: %s", e)
            self._set_status("Error: {}".format(e), "#ff4444")
            self._enable_start(True)
            self._enable_calibrate(True)

    def _simulate_results(self):
        """fake data for pc testing"""
        from PROGRAM import nearest_calibrated_label, CALIBRATION_LAB
        out = {"pads": []}
        # these are just made up values, dont rely on them
        flabs = [
            (230,128,135),(235,128,128),(220,138,142),
            (238,128,128),(228,128,126),(235,128,130),
            (215,142,148),(238,128,128),(238,128,128),
            (238,128,128),(238,128,128),(238,128,128),
            (238,128,128),(220,128,122),(222,128,122),
            (198,132,148),
        ]
        for i,sp in enumerate(PAD_LAYOUT):
            lab = flabs[i]
            m = nearest_calibrated_label(sp["name"], lab)
            out["pads"].append({
                "name": sp["name"],
                "lab": list(lab),
                "estimate": m["label"],
                "distance": m["distance"],
            })
        return out

    def _on_close(self):
        self._shutdown.set()
        self.pump.close()
        if self.picam2 and USE_CAMERA:
            try:
                self.picam2.stop()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        log.info("Touchscreen GUI started (%dx%d, fullscreen=%s).",
                 WINDOW_W, WINDOW_H, FULLSCREEN)
        self.root.mainloop()


def main():
    app = WaterFilterApp()
    signal.signal(signal.SIGINT, lambda *_: app._on_close())
    try:
        signal.signal(signal.SIGTERM, lambda *_: app._on_close())
    except OSError:
        pass
    app.run()


if __name__ == "__main__":
    main()
