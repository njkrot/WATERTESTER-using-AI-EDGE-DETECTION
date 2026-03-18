# grid reader - no AI, just fixed crop position
import time
import json
import logging
import signal
import sys
import threading

import cv2
import numpy as np

from StripEdgeRefinement import refine_strip_edges
from display import Display

USE_GPIO = True
try:
    from gpiozero import Button, LED
except ImportError:
    USE_GPIO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("grid_reader")

# TODO tune this per device
STRIP_ROI = (400, 100, 480, 520)

REACTION_WAIT_SECONDS = 45.0  # works for now
CAPTURE_INTERVAL = 0.4
MIN_GOOD_FRAMES = 3
MAX_CAPTURE_ATTEMPTS = 20
OUTLIER_LAB_THRESHOLD = 25.0  # TODO tune this
ENABLE_EDGE_REFINEMENT = True

CARTRIDGE_SWITCH_PIN = 17
STATUS_LED_PIN = 27
USER_NAME = "User"
FRAME_W, FRAME_H = 1280, 720

#DNT DEL - pad layout from main program
PAD_LAYOUT = [
    {"name": "Hardness",   "y0": 0.020, "y1": 0.070, "x0": 0.20, "x1": 0.80},
    {"name": "FreeCl",     "y0": 0.075, "y1": 0.125, "x0": 0.20, "x1": 0.80},
    {"name": "Iron",       "y0": 0.130, "y1": 0.180, "x0": 0.20, "x1": 0.80},
    {"name": "Mercury",    "y0": 0.185, "y1": 0.235, "x0": 0.20, "x1": 0.80},
    {"name": "TotalCl",    "y0": 0.240, "y1": 0.290, "x0": 0.20, "x1": 0.80},
    {"name": "Copper",     "y0": 0.295, "y1": 0.345, "x0": 0.20, "x1": 0.80},
    {"name": "Lead",       "y0": 0.350, "y1": 0.400, "x0": 0.20, "x1": 0.80},
    {"name": "Zinc",       "y0": 0.405, "y1": 0.455, "x0": 0.20, "x1": 0.80},
    {"name": "Manganese",  "y0": 0.460, "y1": 0.510, "x0": 0.20, "x1": 0.80},
    {"name": "QAC",        "y0": 0.515, "y1": 0.565, "x0": 0.20, "x1": 0.80},
    {"name": "Fluoride",   "y0": 0.570, "y1": 0.620, "x0": 0.20, "x1": 0.80},
    {"name": "NaCl",       "y0": 0.625, "y1": 0.675, "x0": 0.20, "x1": 0.80},
    {"name": "H2S",        "y0": 0.680, "y1": 0.730, "x0": 0.20, "x1": 0.80},
    {"name": "Alkalinity", "y0": 0.735, "y1": 0.785, "x0": 0.20, "x1": 0.80},
    {"name": "Carbonate",  "y0": 0.790, "y1": 0.840, "x0": 0.20, "x1": 0.80},
    {"name": "pH",         "y0": 0.845, "y1": 0.900, "x0": 0.20, "x1": 0.80},
]

CALIBRATION_LAB = {
    "Hardness":   [("0",   (240, 128, 128)), ("25",  (220, 130, 140)), ("50",  (200, 135, 150)),
                   ("120", (180, 140, 160)), ("250", (160, 145, 170)), ("425", (140, 150, 175))],
    "FreeCl":     [("0",   (240, 128, 128)), ("0.5", (230, 128, 125)), ("1",   (220, 128, 120)),
                   ("3",   (200, 130, 115)), ("5",   (180, 132, 110)), ("10",  (160, 135, 105))],
    "Iron":       [("0",   (240, 128, 128)), ("5",   (215, 140, 145)), ("10",  (195, 150, 155)),
                   ("25",  (170, 155, 160)), ("100", (140, 160, 165))],
    "Mercury":    [("0",   (240, 128, 128)), ("0.002", (225, 130, 130)), ("0.005", (210, 135, 135)),
                   ("0.01", (195, 140, 140)), ("0.02", (175, 145, 145))],
    "TotalCl":    [("0",   (240, 128, 128)), ("0.5", (225, 128, 125)), ("1",   (210, 128, 120)),
                   ("3",   (190, 130, 115)), ("5",   (170, 132, 110)), ("10",  (150, 135, 105))],
    "Copper":     [("0",   (240, 128, 128)), ("0.5", (220, 132, 135)), ("1",   (200, 138, 140)),
                   ("3",   (180, 145, 150)), ("10",  (155, 150, 155))],
    "Lead":       [("0",   (240, 128, 128)), ("20",  (210, 145, 150)), ("50",  (185, 155, 158)),
                   ("100", (160, 160, 162)), ("200", (140, 165, 165)), ("500", (120, 170, 168))],
    "Zinc":       [("0",   (240, 128, 128)), ("2",   (220, 140, 142)), ("5",   (195, 150, 150)),
                   ("10",  (175, 155, 155)), ("25",  (155, 160, 160)), ("50",  (135, 165, 162)),
                   ("100", (120, 168, 165))],
    "Manganese":  [("0",   (240, 128, 128)), ("0.1", (225, 132, 130)), ("0.5", (210, 140, 135)),
                   ("1",   (195, 148, 140)), ("2",   (175, 155, 145)), ("5",   (155, 160, 150)),
                   ("10",  (135, 165, 155))],
    "QAC":        [("0",   (240, 128, 128)), ("5",   (220, 130, 125)), ("10",  (200, 135, 120)),
                   ("20",  (180, 140, 115)), ("40",  (155, 148, 110))],
    "Fluoride":   [("0",   (240, 128, 128)), ("10",  (195, 145, 140)), ("25",  (170, 150, 148)),
                   ("50",  (150, 155, 155)), ("100", (130, 160, 160))],
    "NaCl":       [("0",   (240, 128, 128)), ("50",  (215, 135, 140)), ("75",  (195, 142, 148)),
                   ("100", (180, 148, 155)), ("150", (165, 152, 160)), ("250", (145, 158, 165)),
                   ("500", (125, 162, 170))],
    "H2S":        [("0",   (240, 128, 128)), ("0.5", (225, 130, 130)), ("1",   (210, 135, 135)),
                   ("2",   (195, 140, 140)), ("5",   (175, 148, 148))],
    "Alkalinity": [("0",   (240, 128, 128)), ("40",  (225, 128, 120)), ("80",  (210, 125, 115)),
                   ("120", (195, 122, 110)), ("180", (175, 118, 105)), ("240", (155, 115, 100))],
    "Carbonate":  [("0",   (240, 128, 128)), ("40",  (225, 128, 120)), ("80",  (210, 125, 115)),
                   ("120", (195, 122, 110)), ("180", (175, 118, 105)), ("240", (155, 115, 100))],
    "pH":         [("6.0", (210, 145, 135)), ("6.4", (205, 140, 140)), ("6.8", (198, 132, 148)),
                   ("7.2", (190, 125, 155)), ("7.6", (180, 118, 162)), ("8.2", (168, 112, 170)),
                   ("9.0", (150, 105, 180))],
}


def grid_crop(frame_bgr, roi):
    rx, ry, rw, rh = roi
    fh, fw = frame_bgr.shape[:2]
    x0 = max(0, min(rx, fw - 1))
    y0 = max(0, min(ry, fh - 1))
    x1 = max(x0 + 1, min(rx + rw, fw))
    y1 = max(y0 + 1, min(ry + rh, fh))
    crop = frame_bgr[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None
    if crop.shape[1] > crop.shape[0]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop

def pad_roi_from_layout(strip_img, spec):
    h, w = strip_img.shape[:2]
    x0 = max(0, min(int(spec["x0"] * w), w - 1))
    x1 = max(x0 + 1, min(int(spec["x1"] * w), w))
    y0 = max(0, min(int(spec["y0"] * h), h - 1))
    y1 = max(y0 + 1, min(int(spec["y1"] * h), h))
    return strip_img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def robust_patch_lab(patch_bgr):
    """DNT DEL"""
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    blur = cv2.GaussianBlur(patch_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    #CRIT HOW - glare mask threshold
    mask = (v > 30) & (v < 245)
    if np.count_nonzero(mask) < 0.2 * mask.size:
        mask = np.ones_like(mask, dtype=bool)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    px = lab[mask]
    if px.size == 0:
        return None
    med = np.median(px, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


#NEED?! - closest calibration match by euclidean dist in LAB
def nearest_calibrated_label(analyte_name, lab_triplet):
    entries = CALIBRATION_LAB.get(analyte_name, [])
    if not entries or lab_triplet is None:
        return {"label": "unknown", "distance": None}
    best_label, best_dist = None, None
    v = np.array(lab_triplet, dtype=np.float32)
    for lbl, ref in entries:
        d = float(np.linalg.norm(v - np.array(ref, dtype=np.float32)))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_label = lbl
    return {"label": best_label, "distance": round(best_dist, 2) if best_dist is not None else None}


def analyze_strip(strip_img):
    result = {"pads": []}
    if strip_img is None or strip_img.size == 0:
        result["error"] = "empty_strip_crop"
        return result
    for spec in PAD_LAYOUT:
        patch, roi = pad_roi_from_layout(strip_img, spec)
        lab = robust_patch_lab(patch)
        match = nearest_calibrated_label(spec["name"], lab)
        result["pads"].append({
            "name": spec["name"],
            "roi": {"x0": roi[0], "y0": roi[1], "x1": roi[2], "y1": roi[3]},
            "lab": None if lab is None else [round(lab[0], 1), round(lab[1], 1), round(lab[2], 1)],
            "estimate": match["label"],
            "distance": match["distance"],
        })
    return result


DANGER_WEIGHTS = {
    "Lead": 3.0, "Mercury": 3.0,
    "Copper": 2.0, "Iron": 2.0, "Manganese": 2.0,
    "FreeCl": 1.5, "TotalCl": 1.5, "H2S": 1.5, "Fluoride": 1.5,
    "Hardness": 1.0, "Zinc": 1.0, "QAC": 1.0, "NaCl": 1.0,
    "Alkalinity": 1.0, "Carbonate": 1.0, "pH": 1.0,
}

SAFE_THRESHOLDS = {
    "Lead": 0, "Mercury": 0, "Copper": 0, "Iron": 0,
    "Manganese": 0, "FreeCl": 0, "TotalCl": 0, "H2S": 0,
    "Fluoride": 0, "Zinc": 0, "QAC": 0, "NaCl": 0,
    "Hardness": 120, "Alkalinity": 180, "Carbonate": 180, "pH": None,
}


def frame_deviation(reading, median_labs):
    dists = []
    for i, pad in enumerate(reading.get("pads", [])):
        lab = pad.get("lab")
        med = median_labs[i]
        if lab is None or med is None:
            continue
        d = float(np.linalg.norm(np.array(lab) - np.array(med)))
        dists.append(d)
    return float(np.mean(dists)) if dists else 0.0

def compute_median_labs(readings):
    n = len(PAD_LAYOUT)
    medians = [None] * n
    for i in range(n):
        labs = []
        for r in readings:
            pads = r.get("pads", [])
            if i < len(pads) and pads[i].get("lab") is not None:
                labs.append(pads[i]["lab"])
        if labs:
            medians[i] = [
                float(np.median([l[0] for l in labs])),
                float(np.median([l[1] for l in labs])),
                float(np.median([l[2] for l in labs])),
            ]
    return medians

#CRIT HOW - outlier rejection by LAB distance from median
def filter_outlier_frames(readings, threshold=OUTLIER_LAB_THRESHOLD):
    if len(readings) <= 1:
        return readings, 0
    median_labs = compute_median_labs(readings)
    deviations = [(i, frame_deviation(r, median_labs)) for i, r in enumerate(readings)]
    good = []
    n_bad = 0
    for i, dev in deviations:
        if dev > threshold:
            n_bad += 1
            log.info("Frame %d outlier (dev %.1f), discarding.", i, dev)
        else:
            good.append(readings[i])
    if not good:
        good = readings
        n_bad = 0
    return good, n_bad

def average_pad_readings(readings):
    """DNT DEL"""
    if len(readings) == 1:
        return readings[0]
    result = {"pads": [], "frame_count": len(readings)}
    for i, spec in enumerate(PAD_LAYOUT):
        labs = []
        for r in readings:
            if i < len(r["pads"]) and r["pads"][i]["lab"] is not None:
                labs.append(r["pads"][i]["lab"])
        if labs:
            avg_lab = [
                round(float(np.median([l[0] for l in labs])), 1),
                round(float(np.median([l[1] for l in labs])), 1),
                round(float(np.median([l[2] for l in labs])), 1),
            ]
            match = nearest_calibrated_label(spec["name"], tuple(avg_lab))
        else:
            avg_lab = None
            match = {"label": "unknown", "distance": None}
        result["pads"].append({
            "name": spec["name"],
            "lab": avg_lab,
            "estimate": match["label"],
            "distance": match["distance"],
            "sample_count": len(labs),
        })
    return result


# works for now - scoring is kinda rough
def compute_water_score(results):
    if not results or "pads" not in results:
        return 0, []
    pads = results["pads"]
    if not pads:
        return 0, []
    total_weight = 0.0
    earned = 0.0
    warnings = []
    for p in pads:
        name = p.get("name", "")
        est = p.get("estimate", "").strip()
        weight = DANGER_WEIGHTS.get(name, 1.0)
        total_weight += weight
        if est in ("unknown", ""):
            continue
        try:
            val = float(est)
        except ValueError:
            continue
        safe_thresh = SAFE_THRESHOLDS.get(name)
        if name == "pH":
            if 6.5 <= val <= 8.5:
                earned += weight
            else:
                warnings.append((weight, name, val, "pH out of safe range"))
        elif safe_thresh is not None:
            if val <= safe_thresh:
                earned += weight
            else:
                warnings.append((weight, name, val, "{} high ({} ppm)".format(name, est)))
        else:
            if val <= 0:
                earned += weight
            else:
                warnings.append((weight, name, val, "{} detected ({} ppm)".format(name, est)))
    score = int(round(100.0 * earned / total_weight)) if total_weight > 0 else 0
    warnings.sort(key=lambda w: -w[0])
    return score, [w[3] for w in warnings]


def draw_grid_overlay(frame_bgr, roi):
    """cal overlay"""
    vis = frame_bgr.copy()
    rx, ry, rw, rh = roi
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(vis, "STRIP ROI", (rx, ry - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    crop = grid_crop(frame_bgr, roi)
    if crop is None:
        return vis
    ch, cw = crop.shape[:2]
    for spec in PAD_LAYOUT:
        py0 = int(spec["y0"] * ch)
        py1 = int(spec["y1"] * ch)
        px0 = int(spec["x0"] * cw)
        px1 = int(spec["x1"] * cw)
        #CRIT HOW - vertical vs horizontal crop coord mapping
        if crop.shape[0] >= crop.shape[1]:
            fy0 = ry + py0
            fy1 = ry + py1
            fx0 = rx + px0
            fx1 = rx + px1
        else:
            fy0 = ry + px0
            fy1 = ry + px1
            fx0 = rx + py0
            fx1 = rx + py1
        cv2.rectangle(vis, (fx0, fy0), (fx1, fy1), (255, 255, 0), 1)
        cv2.putText(vis, spec["name"], (fx0, fy0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    return vis


class CartridgeTrigger:
    def __init__(self, shutdown_event):
        self._shutdown = shutdown_event
        self._gpio_btn = None
        self._gpio_led = None
        if USE_GPIO:
            self._gpio_btn = Button(CARTRIDGE_SWITCH_PIN, pull_up=True, bounce_time=0.1)
            try:
                self._gpio_led = LED(STATUS_LED_PIN)
            except Exception:
                log.warning("LED pin %d not available.", STATUS_LED_PIN)
        else:
            log.info("No GPIO - using keyboard trigger (press Enter).")

    def is_inserted(self):
        if USE_GPIO:
            return self._gpio_btn.is_pressed
        return getattr(self, "_kb_inserted", False)

    def wait_for_insertion(self):
        log.info("Waiting for cartridge...")
        self.led_off()
        if USE_GPIO:
            return self._gpio_wait(lambda: self.is_inserted())
        return self._kb_wait_insert()

    def wait_for_removal(self):
        log.info("Remove cartridge to reset.")
        if USE_GPIO:
            return self._gpio_wait(lambda: not self.is_inserted())
        return self._kb_wait_remove()

    def led_off(self):
        if self._gpio_led:
            self._gpio_led.off()
    def led_on(self):
        if self._gpio_led:
            self._gpio_led.on()
    def led_blink(self, on_time=0.5, off_time=0.5):
        if self._gpio_led:
            self._gpio_led.blink(on_time=on_time, off_time=off_time)

    def _gpio_wait(self, condition_fn):
        while not self._shutdown.is_set():
            if condition_fn():
                time.sleep(0.05)
                if condition_fn():
                    return True
            time.sleep(0.1)
        return False

    def _kb_wait_insert(self):
        try:
            input(">> Press ENTER to simulate cartridge insertion... ")
        except (EOFError, KeyboardInterrupt):
            return False
        self._kb_inserted = True
        return not self._shutdown.is_set()

    def _kb_wait_remove(self):
        try:
            input(">> Press ENTER to simulate cartridge removal... ")
        except (EOFError, KeyboardInterrupt):
            return False
        self._kb_inserted = False
        return not self._shutdown.is_set()


class GridReader:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.trigger = CartridgeTrigger(self.shutdown_event)
        self.display = Display(username=USER_NAME)
        self.picam2 = None
        self._cam_config = None
        self._cam_running = False

    def setup(self):
        from picamera2 import Picamera2
        log.info("Starting camera (grid mode, no AI model)...")
        self.picam2 = Picamera2()
        self._cam_config = self.picam2.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            controls={"FrameRate": 30},
            buffer_count=4,
        )
        log.info("Camera ready. Strip ROI: %s", STRIP_ROI)

    def _start_camera(self):
        if not self._cam_running:
            self.picam2.start(self._cam_config, show_preview=False)
            self._cam_running = True
            time.sleep(0.5)  # works for now
    def _stop_camera(self):
        if self._cam_running:
            self.picam2.stop()
            self._cam_running = False
    def shutdown(self):
        self.shutdown_event.set()

    def _capture_frame(self):
        try:
            frame = self.picam2.capture_array("main")
        except Exception as e:
            log.warning("Frame capture failed: %s", e)
            return None
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _grab_strip(self, frame_bgr):
        crop = grid_crop(frame_bgr, STRIP_ROI)
        if crop is None:
            return None
        if ENABLE_EDGE_REFINEMENT:
            refined = refine_strip_edges(crop)
            if refined is not None and refined.size != 0:
                crop = refined
        return crop

    def _consensus_read(self):
        """keep grabbing til good > bad"""
        all_readings = []
        att = 0
        #NEED?! - loop til enough good frames or we give up
        while att < MAX_CAPTURE_ATTEMPTS:
            if self.shutdown_event.is_set():
                break
            if USE_GPIO and not self.trigger.is_inserted():
                log.warning("Cartridge removed during capture.")
                break

            frame = self._capture_frame()
            if frame is not None:
                strip = self._grab_strip(frame)
                if strip is not None:
                    res = analyze_strip(strip)
                    if "error" not in res:
                        all_readings.append(res)

            att += 1

            if len(all_readings) >= MIN_GOOD_FRAMES:
                good, bad = filter_outlier_frames(all_readings)
                if len(good) > bad and len(good) >= MIN_GOOD_FRAMES:
                    log.info("Got %d good frames, %d outliers (of %d attempts).",
                             len(good), bad, att)
                    return average_pad_readings(good)

            time.sleep(CAPTURE_INTERVAL)

        if not all_readings:
            return None

        if att >= MAX_CAPTURE_ATTEMPTS:
            log.error("Frame limit exceeded (%d attempts).", MAX_CAPTURE_ATTEMPTS)
            self.display.show_error("Frame limit!")

        good, _ = filter_outlier_frames(all_readings)
        return average_pad_readings(good)

    def _run_read_cycle(self):
        self.trigger.led_blink(on_time=0.5, off_time=0.5)
        log.info("Cartridge in. Waiting %.0fs for reaction...", REACTION_WAIT_SECONDS)
        self._start_camera()

        deadline = time.time() + REACTION_WAIT_SECONDS
        while time.time() < deadline:
            if self.shutdown_event.is_set():
                return None
            if USE_GPIO and not self.trigger.is_inserted():
                log.warning("Cartridge pulled out early.")
                self.display.show_error("Removed early!")
                return {"error": "cartridge_removed_early"}
            rem = deadline - time.time()
            self.display.show_filtering(rem)
            time.sleep(min(0.5, max(0.01, rem)))

        self.trigger.led_blink(on_time=0.1, off_time=0.1)
        self.display.show_reading()
        log.info("Reading strip from grid ROI...")

        results = self._consensus_read()
        if results is None:
            log.error("Failed to get readings from grid.")
            self.display.show_error("Read failed")
            return {"error": "read_failed"}

        self.trigger.led_on()
        return {
            "timestamp": time.time(),
            "mode": "grid",
            "strip_roi": list(STRIP_ROI),
            "reaction_wait_seconds": REACTION_WAIT_SECONDS,
            "capture_frames": results.get("frame_count", 1),
            "results": results,
        }

    def run(self):
        self.setup()
        log.info("Grid reader ready.")
        self.display.show_hello()

        try:
            while not self.shutdown_event.is_set():
                if not self.trigger.wait_for_insertion():
                    break

                payload = self._run_read_cycle()

                if payload is not None:
                    print(json.dumps(payload, indent=2))
                    if "error" in payload:
                        log.error("Read error: %s", payload["error"])
                    else:
                        pads = payload.get("results", {}).get("pads", [])
                        score, warnings = compute_water_score(payload.get("results", {}))
                        payload["score"] = score
                        payload["warnings"] = warnings
                        log.info("Read complete - %d pads, score %d/100.", len(pads), score)
                        for w in warnings:
                            log.warning("  %s", w)
                        self.display.show_results(pads, score="{}/100".format(score))

                self._stop_camera()

                if self.trigger.is_inserted():
                    self.display.show_remove()
                    self.trigger.wait_for_removal()

                self.trigger.led_off()
                self.display.show_score()
                log.info("Ready.\n")
        finally:
            self._stop_camera()
            self.trigger.led_off()
            self.display.show_shutting_down()
            self.display.close()
            log.info("Shut down complete.")

    def calibrate(self):
        """#CRIT - cal mode"""
        from picamera2 import Picamera2
        log.info("Calibration mode - showing live grid overlay.")
        log.info("Adjust STRIP_ROI in GRID_READER.py until the green box")
        log.info("lines up with the strip and yellow boxes hit each pad.")
        log.info("Press 'q' to quit, 's' to save a snapshot.\n")

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
            controls={"FrameRate": 30},
        )
        self.picam2.start(config, show_preview=False)
        time.sleep(1.0)

        try:
            while True:
                frame = self.picam2.capture_array("main")
                if frame is None:
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                vis = draw_grid_overlay(bgr, STRIP_ROI)

                cv2.imshow("Grid Calibration (q=quit, s=save)", vis)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    cv2.imwrite("grid_calibration.jpg", vis)
                    log.info("Saved grid_calibration.jpg")
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()


def main():
    calibrate_mode = "--calibrate" in sys.argv or "-c" in sys.argv

    if calibrate_mode:
        reader = GridReader()
        reader.calibrate()
    else:
        reader = GridReader()
        signal.signal(signal.SIGINT, lambda *_: reader.shutdown())
        try:
            signal.signal(signal.SIGTERM, lambda *_: reader.shutdown())
        except OSError:
            pass
        reader.run()


if __name__ == "__main__":
    main()
