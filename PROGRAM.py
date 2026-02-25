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

MODEL_PATH = "/home/pi/models/test_strip_yolo.rpk"
LABELS_PATH = "/home/pi/models/labels.txt"
# "object" is the whole-strip class in the Roboflow dataset
TARGET_CLASS_NAME = "object"
MIN_CONFIDENCE = 0.35

REACTION_WAIT_SECONDS = 45.0
CAPTURE_WINDOW_SECONDS = 3.0
CAPTURE_FRAME_COUNT = 5
MAX_DETECT_ATTEMPTS = 10
ENABLE_EDGE_REFINEMENT = True
SHOW_WINDOW = False

CARTRIDGE_SWITCH_PIN = 17
STATUS_LED_PIN = 27
USER_NAME = "User"

# pad layout - normalized coords, vertical strip
PAD_LAYOUT = [
    {"name": "LEU", "y0": 0.08, "y1": 0.16, "x0": 0.15, "x1": 0.85},
    {"name": "NIT", "y0": 0.18, "y1": 0.26, "x0": 0.15, "x1": 0.85},
    {"name": "URO", "y0": 0.28, "y1": 0.36, "x0": 0.15, "x1": 0.85},
    {"name": "PRO", "y0": 0.38, "y1": 0.46, "x0": 0.15, "x1": 0.85},
    {"name": "pH", "y0": 0.48, "y1": 0.56, "x0": 0.15, "x1": 0.85},
    {"name": "BLO", "y0": 0.58, "y1": 0.66, "x0": 0.15, "x1": 0.85},
    {"name": "SG",  "y0": 0.68, "y1": 0.76, "x0": 0.15, "x1": 0.85},
    {"name": "KET", "y0": 0.78, "y1": 0.86, "x0": 0.15, "x1": 0.85},
    {"name": "BIL", "y0": 0.88, "y1": 0.94, "x0": 0.15, "x1": 0.85},
]

# calibration values (placeholder - swap with real measured LAB later)
CALIBRATION_LAB = {
    "LEU": [("neg", (235, 128, 128)), ("trace", (220, 140, 128)), ("pos", (205, 155, 135))],
    "NIT": [("neg", (235, 128, 128)), ("pos", (195, 160, 145))],
    "URO": [("normal", (220, 132, 140)), ("high", (185, 148, 160))],
    "PRO": [("neg", (235, 128, 128)), ("trace", (210, 145, 135)), ("pos", (175, 160, 140))],
    "pH": [("5.0", (210, 145, 135)), ("6.0", (210, 132, 145)), ("7.0", (205, 120, 155)), ("8.0", (195, 115, 165))],
    "BLO": [("neg", (235, 128, 128)), ("trace", (210, 130, 120)), ("pos", (180, 125, 115))],
    "SG": [("1.005", (210, 130, 150)), ("1.015", (205, 135, 140)), ("1.025", (195, 145, 130)), ("1.030", (185, 150, 125))],
    "KET": [("neg", (235, 128, 128)), ("small", (215, 145, 145)), ("mod", (190, 155, 160)), ("large", (165, 165, 175))],
    "BIL": [("neg", (235, 128, 128)), ("small", (210, 140, 120)), ("pos", (180, 150, 110))],
}


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

    detections = []
    for box, score, category in zip(boxes, scores, classes):
        if float(score) < threshold:
            continue
        x, y, w, h = imx500.convert_inference_coords(box, metadata, picam2)
        detections.append({
            "class_id": int(category),
            "score": float(score),
            "box": (int(x), int(y), int(w), int(h))
        })
    return detections

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
    """median LAB from a patch, masks out glare"""
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    blur = cv2.GaussianBlur(patch_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    mask = (val > 30) & (val < 245)
    if np.count_nonzero(mask) < 0.2 * mask.size:
        mask = np.ones_like(mask, dtype=bool)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    pixels = lab[mask]
    if pixels.size == 0:
        return None
    med = np.median(pixels, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))

def nearest_calibrated_label(analyte_name, lab_triplet):
    entries = CALIBRATION_LAB.get(analyte_name, [])
    if not entries or lab_triplet is None:
        return {"label": "unknown", "distance": None}
    best_label, best_dist = None, None
    v = np.array(lab_triplet, dtype=np.float32)
    for label, ref in entries:
        dist = float(np.linalg.norm(v - np.array(ref, dtype=np.float32)))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
    return {"label": best_label, "distance": round(best_dist, 2) if best_dist is not None else None}


def analyze_test_strip(strip_img):
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
            "distance": match["distance"]
        })
    return result

def compute_water_score(results):
    """quick overall score from pad results"""
    if not results or "pads" not in results:
        return "--"
    pads = results["pads"]
    if not pads:
        return "--"
    good = sum(1 for p in pads
               if p.get("estimate", "").lower() in ("neg", "normal", "negative"))
    return "{}/{}".format(good, len(pads))


def average_pad_readings(readings):
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
            "sample_count": len(labs)
        })
    return result


class CartridgeTrigger:
    """GPIO or keyboard fallback for detecting cartridge."""

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
            request = self.picam2.capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()
        except Exception as e:
            log.warning("Frame capture failed: %s", e)
            return None, None
        if frame is None:
            return None, None
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr, metadata

    def _detect_and_crop(self, frame_bgr, metadata):
        detections = parse_detections(
            imx500=self.imx500, picam2=self.picam2,
            intrinsics=self.intrinsics, metadata=metadata,
            threshold=MIN_CONFIDENCE
        )
        strip_det = choose_strip_detection(detections, self.labels, TARGET_CLASS_NAME)
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
        """grab multiple frames, average the readings"""
        interval = CAPTURE_WINDOW_SECONDS / max(1, CAPTURE_FRAME_COUNT)
        readings = []
        best_det = None
        best_score = 0.0

        for i in range(CAPTURE_FRAME_COUNT):
            if self.shutdown_event.is_set():
                break
            if USE_GPIO and not self.trigger.is_inserted():
                log.warning("Cartridge removed during capture.")
                break

            frame_bgr, metadata = self._capture_frame()
            if frame_bgr is None:
                time.sleep(interval)
                continue

            strip_det, crop = self._detect_and_crop(frame_bgr, metadata)
            if strip_det is not None and crop is not None:
                res = analyze_test_strip(crop)
                if "error" not in res:
                    readings.append(res)
                    if strip_det["score"] > best_score:
                        best_score = strip_det["score"]
                        best_det = strip_det

            if i < CAPTURE_FRAME_COUNT - 1:
                time.sleep(interval)

        if not readings:
            return None, None
        return best_det, average_pad_readings(readings)

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

        # try to find the strip
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
                        score = compute_water_score(payload.get("results", {}))
                        log.info("Read complete - %d pads, score %s.", len(pads), score)
                        self.display.show_results(pads, score=score)

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
