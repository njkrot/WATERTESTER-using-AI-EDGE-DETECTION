import os
import time
import json
import logging
import signal
import sys, threading

import cv2
import numpy as np

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
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
log = logging.getLogger("strip_reader")

MODEL_PATH = "/home/pi/filterprogram/best_imx_model/best.rpk"
LABELS_PATH = "/home/pi/filterprogram/best_imx_model/labels.txt"
TARGET_CLASS_NAME = "strip"
MIN_CONFIDENCE = 0.25

REACTION_WAIT_SECONDS = 45.0
CAPTURE_INTERVAL = 0.4
MIN_GOOD_FRAMES = 3
MAX_CAPTURE_ATTEMPTS = 20
OUTLIER_LAB_THRESHOLD = 25.0
MAX_DETECT_ATTEMPTS = 10
ENABLE_EDGE_REFINEMENT = True
SHOW_WINDOW = False

CARTRIDGE_SWITCH_PIN = 17
# bcm 19 = status led / buzzer per wiring (dont use 27 here - thats main pump relay)
STATUS_LED_PIN = 19
USER_NAME = "User"

# pad positions (normalized)
# basically like a struct for each one
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

# these are just estimates lol, gotta calibrate later
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

CALIBRATION_FILE = "calibration_data.json"


def load_calibration_from_file():
    """loads cal file"""
    if not os.path.exists(CALIBRATION_FILE):
        return False

    try:
        with open(CALIBRATION_FILE, "r") as f:
            entries = json.load(f)
    except Exception as e:
        log.warning("Could not load calibration file: %s", e)
        return False

    if not entries:
        return False

    latest = entries[-1]
    baseline = latest.get("pads", {})
    if not baseline:
        return False

    adj = 0
    for pad_name, measured_lab in baseline.items():
        if measured_lab is None or pad_name not in CALIBRATION_LAB:
            continue

        cal_entries = CALIBRATION_LAB[pad_name]
        if not cal_entries:
            continue

        ref_zero = cal_entries[0][1]
        # pretty sure this offset math is right
        off = (
            measured_lab[0] - ref_zero[0],
            measured_lab[1] - ref_zero[1],
            measured_lab[2] - ref_zero[2],
        )

        new_entries = []
        for label, lab in cal_entries:
            new_entries.append((label, (
                lab[0] + off[0],
                lab[1] + off[1],
                lab[2] + off[2],
            )))
        CALIBRATION_LAB[pad_name] = new_entries
        adj += 1

    log.info("Loaded calibration from %s (%s). Adjusted %d pads.",
             CALIBRATION_FILE, latest.get("timestamp", "?"), adj)
    return True


load_calibration_from_file()


def load_labels(path):
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        log.warning("Labels file not found: %s", path)
        return []
def safe_intrinsics(imx500):
    intr = imx500.network_intrinsics
    if not intr:
        intr = NetworkIntrinsics()
        intr.task = "object detection"
    intr.update_with_defaults()
    return intr


def parse_detections(imx500, picam2, intrinsics, metadata,
                     threshold, iou=0.65, max_detections=10):
    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return []

    input_w, input_h = imx500.get_input_size()

    if getattr(intrinsics, "postprocess", None) == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=outputs[0], conf=threshold,
            iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
        if getattr(intrinsics, "bbox_normalization", False):
            boxes = boxes / float(input_h)
        if getattr(intrinsics, "bbox_order", "yx") == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    dets = []
    for box, score, category in zip(boxes, scores, classes):
        if float(score) < threshold:
            continue
        x, y, w, h = imx500.convert_inference_coords(box, metadata, picam2)
        dets.append({
            "class_id": int(category),
            "score": float(score),
            "box": (int(x), int(y), int(w), int(h))
        })
    return dets

def choose_strip_detection(detections, labels, target_class_name):
    best = None
    for det in detections:
        cid = det["class_id"]
        cname = labels[cid] if 0 <= cid < len(labels) else str(cid)
        if cname != target_class_name:
            continue
        if best is None or det["score"] > best["score"]:
            best = det
    return best
def clamp_box(x, y, w, h, frame_w, frame_h):
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h
def crop_strip(frame_bgr, box):
    h_frame, w_frame = frame_bgr.shape[:2]
    x, y, w, h = clamp_box(box[0], box[1], box[2], box[3], w_frame, h_frame)
    crop = frame_bgr[y:y+h, x:x+w].copy()
    if crop.size == 0:
        return None
    # rotate if wider than tall
    if crop.shape[1] > crop.shape[0]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def pad_roi_from_layout(strip_img, spec):
    h, w = strip_img.shape[:2]
    x0 = max(0, min(int(spec["x0"] * w), w - 1))
    x1 = max(x0+1, min(int(spec["x1"] * w), w))
    y0 = max(0, min(int(spec["y0"] * h), h - 1))
    y1 = max(y0+1, min(int(spec["y1"] * h), h))
    return strip_img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

def robust_patch_lab(patch_bgr):
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    blur = cv2.GaussianBlur(patch_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    mask = (val > 30) & (val < 245)  # filter out glare and shadows
    if np.count_nonzero(mask) < 0.2 * mask.size:
        mask = np.ones_like(mask, dtype=bool)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    px = lab[mask]
    if px.size == 0:
        return None
    med = np.median(px, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))

def nearest_calibrated_label(analyte_name, lab_triplet):
    entries = CALIBRATION_LAB.get(analyte_name, [])
    if not entries or lab_triplet is None:
        return {"label": "unknown", "distance": None}
    bl, bd = None, None  # best label, best dist
    v = np.array(lab_triplet, dtype=np.float32)
    for label, ref in entries:
        d = float(np.linalg.norm(v - np.array(ref, dtype=np.float32)))
        if bd is None or d < bd:
            bd = d
            bl = label
    return {"label": bl, "distance": round(bd, 2) if bd is not None else None}


def analyze_test_strip(strip_img):
    res = {"pads": []}
    if strip_img is None or strip_img.size == 0:
        res["error"] = "empty_strip_crop"
        return res

    for spec in PAD_LAYOUT:
        patch, roi = pad_roi_from_layout(strip_img, spec)
        lab = robust_patch_lab(patch)
        match = nearest_calibrated_label(spec["name"], lab)
        res["pads"].append({
            "name": spec["name"],
            "roi": {"x0": roi[0], "y0": roi[1], "x1": roi[2], "y1": roi[3]},
            "lab": None if lab is None else [round(lab[0], 1), round(lab[1], 1), round(lab[2], 1)],
            "estimate": match["label"],
            "distance": match["distance"]
        })
    return res

# basically like a struct - weight per contaminant
DANGER_WEIGHTS = {
    "Lead":       3.0,
    "Mercury":    3.0,
    "Copper":     2.0,
    "Iron":       2.0,
    "Manganese":  2.0,
    "FreeCl":     1.5,
    "TotalCl":    1.5,
    "H2S":        1.5,
    "Fluoride":   1.5,
    "Hardness":   1.0,
    "Zinc":       1.0,
    "QAC":        1.0,
    "NaCl":       1.0,
    "Alkalinity": 1.0,
    "Carbonate":  1.0,
    "pH":         1.0,
}

# TODO: might need to tweak this
SAFE_THRESHOLDS = {
    "Lead":       0,    "Mercury":    0,
    "Copper":     0,    "Iron":       0,
    "Manganese":  0,    "FreeCl":     0,
    "TotalCl":    0,    "H2S":        0,
    "Fluoride":   0,    "Zinc":       0,
    "QAC":        0,    "NaCl":       0,
    "Hardness":   120,
    "Alkalinity": 180,  "Carbonate":  180,
    "pH":         None,
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


def filter_outlier_frames(readings, threshold=OUTLIER_LAB_THRESHOLD):
    """toss bad frames"""
    if len(readings) <= 1:
        return readings, 0

    median_labs = compute_median_labs(readings)
    deviations = [(i, frame_deviation(r, median_labs)) for i, r in enumerate(readings)]

    good = []
    bad_count = 0
    for i, dev in deviations:
        if dev > threshold:
            bad_count += 1
            log.info("Frame %d is outlier (deviation %.1f > %.1f), discarding.", i, dev, threshold)
        else:
            good.append(readings[i])

    if not good:
        good = readings
        bad_count = 0
        log.warning("All frames were outliers, using all of them anyway.")

    return good, bad_count


def average_pad_readings(readings):
    if len(readings) == 1:
        return readings[0]

    res = {"pads": [], "frame_count": len(readings)}
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

        res["pads"].append({
            "name": spec["name"],
            "lab": avg_lab,
            "estimate": match["label"],
            "distance": match["distance"],
            "sample_count": len(labs)
        })
    return res


def compute_water_score(results):
    """score calc"""
    if not results or "pads" not in results:
        return 0, []
    pads = results["pads"]
    if not pads:
        return 0, []

    tot_w = 0.0
    earned = 0.0
    warnings = []

    for p in pads:
        name = p.get("name", "")
        est = p.get("estimate", "").strip()
        w = DANGER_WEIGHTS.get(name, 1.0)
        tot_w += w

        if est in ("unknown", ""):
            continue
        try:
            val = float(est)
        except ValueError:
            continue

        safe_thresh = SAFE_THRESHOLDS.get(name)

        # idk if this is the best way but it works
        if name == "pH":
            if 6.5 <= val <= 8.5:
                earned += w
            else:
                warnings.append((w, name, val, "pH out of safe range"))
        elif safe_thresh is not None:
            if val <= safe_thresh:
                earned += w
            else:
                warnings.append((w, name, val, "{} high ({} ppm)".format(name, est)))
        else:
            if val <= 0:
                earned += w
            else:
                warnings.append((w, name, val, "{} detected ({} ppm)".format(name, est)))

    score = int(round(100.0 * earned / tot_w)) if tot_w > 0 else 0
    warnings.sort(key=lambda w: -w[0])
    warning_msgs = [w[3] for w in warnings]
    return score, warning_msgs


class CartridgeTrigger:
    """cartridge switch handler"""

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


class StripReader:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.trigger = CartridgeTrigger(self.shutdown_event)
        self.display = Display(username=USER_NAME)
        self.imx500 = None
        self.picam2 = None
        self.intrinsics = None
        self.labels = None
        self._cam_config = None
        self._cam_running = False

    def setup(self):
        log.info("Loading model and labels...")
        self.labels = load_labels(LABELS_PATH)
        self.imx500 = IMX500(MODEL_PATH)
        self.intrinsics = safe_intrinsics(self.imx500)
        self.picam2 = Picamera2(self.imx500.camera_num)

        fps = getattr(self.intrinsics, "inference_rate", 15)
        self._cam_config = self.picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            controls={"FrameRate": fps},
            buffer_count=6
        )
        self.imx500.show_network_fw_progress_bar()
        log.info("Camera and model ready.")

    def _start_camera(self):
        if not self._cam_running:
            self.picam2.start(self._cam_config, show_preview=False)
            self._cam_running = True
            time.sleep(1.0)

    def _stop_camera(self):
        if self._cam_running:
            self.picam2.stop()
            self._cam_running = False

    def shutdown(self):
        self.shutdown_event.set()

    def _capture_frame(self):
        try:
            req = self.picam2.capture_request()
            try:
                frame = req.make_array("main")
                metadata = req.get_metadata()
            finally:
                req.release()
        except Exception as e:
            log.warning("Frame capture failed: %s", e)
            return None, None
        if frame is None:
            return None, None
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr, metadata

    def _detect_and_crop(self, frame_bgr, metadata):
        dets = parse_detections(
            imx500=self.imx500, picam2=self.picam2,
            intrinsics=self.intrinsics, metadata=metadata,
            threshold=MIN_CONFIDENCE
        )
        strip_det = choose_strip_detection(dets, self.labels, TARGET_CLASS_NAME)
        if strip_det is None:
            return None, None

        crop = crop_strip(frame_bgr, strip_det["box"])
        if crop is None or crop.size == 0:
            return strip_det, None

        if ENABLE_EDGE_REFINEMENT:
            refined = refine_strip_edges(crop)
            if refined is not None and refined.size != 0:
                crop = refined
        return strip_det, crop

    def _consensus_read(self):
        """keep grabbing frames til we got enough"""
        all_readings = []
        best_det = None
        best_score = 0.0
        attempts = 0

        while attempts < MAX_CAPTURE_ATTEMPTS:
            if self.shutdown_event.is_set():
                break
            if USE_GPIO and not self.trigger.is_inserted():
                log.warning("Cartridge removed during capture.")
                break

            frame_bgr, metadata = self._capture_frame()
            if frame_bgr is None:
                attempts += 1
                time.sleep(CAPTURE_INTERVAL)
                continue

            strip_det, crop = self._detect_and_crop(frame_bgr, metadata)
            if strip_det is not None and crop is not None:
                res = analyze_test_strip(crop)
                if "error" not in res:
                    all_readings.append(res)
                    if strip_det["score"] > best_score:
                        best_score = strip_det["score"]
                        best_det = strip_det

            attempts += 1

            # TODO: might need to tweak this threshold
            if len(all_readings) >= MIN_GOOD_FRAMES:
                good, bad = filter_outlier_frames(all_readings)
                if len(good) > bad and len(good) >= MIN_GOOD_FRAMES:
                    log.info("Got %d good frames, %d outliers discarded (of %d attempts).",
                             len(good), bad, attempts)
                    return best_det, average_pad_readings(good)

            time.sleep(CAPTURE_INTERVAL)

        if not all_readings:
            return None, None

        if attempts >= MAX_CAPTURE_ATTEMPTS:
            log.error("Frame limit exceeded (%d attempts). Using best available.", MAX_CAPTURE_ATTEMPTS)
            self.display.show_error("Frame limit!")

        good, bad = filter_outlier_frames(all_readings)
        return best_det, average_pad_readings(good)

    def _run_read_cycle(self):
        self.trigger.led_blink(on_time=0.5, off_time=0.5)
        log.info("Cartridge in. Waiting %.0fs for reaction...", REACTION_WAIT_SECONDS)

        self._start_camera()

        deadline = time.time() + REACTION_WAIT_SECONDS
        while time.time() < deadline:
            if self.shutdown_event.is_set():
                return None
            if USE_GPIO and not self.trigger.is_inserted():
                log.warning("Cartridge pulled out early - aborting.")
                self.display.show_error("Removed early!")
                return {"error": "cartridge_removed_early"}

            remaining = deadline - time.time()
            self.display.show_filtering(remaining)

            if SHOW_WINDOW:
                frame_bgr, _ = self._capture_frame()
                if frame_bgr is not None:
                    elapsed = REACTION_WAIT_SECONDS - remaining
                    msg = "Filtering: {:.0f}s / {:.0f}s".format(elapsed, REACTION_WAIT_SECONDS)
                    cv2.putText(frame_bgr, msg, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Test Strip Reader", frame_bgr)
                    cv2.waitKey(1)
            time.sleep(min(0.5, max(0.01, remaining)))

        self.trigger.led_blink(on_time=0.1, off_time=0.1)
        self.display.show_reading()
        log.info("Reading strip...")

        strip_found = False
        for attempt in range(MAX_DETECT_ATTEMPTS):
            if self.shutdown_event.is_set():
                return None
            if USE_GPIO and not self.trigger.is_inserted():
                log.warning("Cartridge removed before read.")
                self.display.show_error("Removed early!")
                return {"error": "cartridge_removed_early"}

            frame_bgr, metadata = self._capture_frame()
            if frame_bgr is None:
                time.sleep(0.2)
                continue
            det, crop = self._detect_and_crop(frame_bgr, metadata)
            if det is not None and crop is not None:
                strip_found = True
                break
            time.sleep(0.3)

        if not strip_found:
            log.error("No strip detected after %d attempts.", MAX_DETECT_ATTEMPTS)
            self.display.show_error("No strip found")
            return {"error": "strip_not_detected"}

        best_det, results = self._consensus_read()
        if results is None:
            log.error("Failed to get readings.")
            self.display.show_error("Read failed")
            return {"error": "read_failed"}

        self.trigger.led_on()
        return {
            "timestamp": time.time(),
            "reaction_wait_seconds": REACTION_WAIT_SECONDS,
            "capture_frames": results.get("frame_count", 1),
            "strip_detection": {
                "score": round(best_det["score"], 3),
                "box": list(best_det["box"])
            },
            "results": results
        }

    def run(self):
        self.setup()
        log.info("Strip reader ready.")
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
            if SHOW_WINDOW:
                cv2.destroyAllWindows()
            self.display.close()
            log.info("Shut down complete.")


def main():
    reader = StripReader()
    signal.signal(signal.SIGINT, lambda *_: reader.shutdown())
    try:
        signal.signal(signal.SIGTERM, lambda *_: reader.shutdown())
    except OSError:
        pass
    reader.run()

if __name__ == "__main__":
    main()
