"""
verifies the image pipeline and color analysis
without needing Pi hardware. Also demos the LCD display via tkinter.

Run:  python test_offline.py
"""
import sys
import time
import numpy as np
import cv2

# stub out Pi-only libs so PROGRAM.py can import on PC
import types

for mod_name in [
    "picamera2", "picamera2.devices", "picamera2.devices.imx500",
    "picamera2.devices.imx500.postprocess",
    "gpiozero",
]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

picam2_imx500 = sys.modules["picamera2.devices.imx500"]
picam2_imx500.NetworkIntrinsics = type("NetworkIntrinsics", (), {})
picam2_imx500.postprocess_nanodet_detection = lambda *a, **k: None

picam2_devices = sys.modules["picamera2.devices"]
picam2_devices.IMX500 = type("IMX500", (), {})

picam2_mod = sys.modules["picamera2"]
picam2_mod.Picamera2 = type("Picamera2", (), {})

gpiozero_mod = sys.modules["gpiozero"]
gpiozero_mod.Button = type("Button", (), {})
gpiozero_mod.LED = type("LED", (), {})

for mod_name in ["RPLCD", "RPLCD.i2c"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["RPLCD.i2c"].CharLCD = type("CharLCD", (), {})

from PROGRAM import (
    crop_strip, pad_roi_from_layout, robust_patch_lab,
    nearest_calibrated_label, analyze_test_strip, average_pad_readings,
    clamp_box, compute_water_score, PAD_LAYOUT, CALIBRATION_LAB,
)
from StripEdgeRefinement import refine_strip_edges


def make_synthetic_strip(width=60, height=300):
    """fake strip image with 9 colored pads"""
    strip = np.full((height, width, 3), (220, 220, 220), dtype=np.uint8)
    colors_bgr = [
        (200, 200, 230),
        (200, 200, 230),
        (140, 170, 200),
        (200, 200, 230),
        (180, 160, 140),
        (200, 200, 230),
        (170, 170, 190),
        (200, 200, 230),
        (200, 200, 230),
    ]
    for i, spec in enumerate(PAD_LAYOUT):
        y0 = int(spec["y0"] * height)
        y1 = int(spec["y1"] * height)
        x0 = int(spec["x0"] * width)
        x1 = int(spec["x1"] * width)
        strip[y0:y1, x0:x1] = colors_bgr[i]
    return strip


def run_tests():
    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}  {detail}")

    print("=" * 60)
    print("OFFLINE TESTS")
    print("=" * 60)

    print("\n[clamp_box]")
    x, y, w, h = clamp_box(-5, -5, 100, 100, 50, 50)
    check("negative coords clamped", x == 0 and y == 0)
    check("width clamped to frame", w <= 50)

    print("\n[crop_strip]")
    horizontal = np.zeros((30, 200, 3), dtype=np.uint8)
    crop = crop_strip(horizontal, (0, 0, 200, 30))
    check("horizontal rotated to vertical", crop.shape[0] > crop.shape[1])

    vertical = np.zeros((200, 30, 3), dtype=np.uint8)
    crop_v = crop_strip(vertical, (0, 0, 30, 200))
    check("vertical stays vertical", crop_v.shape[0] > crop_v.shape[1])

    print("\n[pad_roi_from_layout]")
    strip = make_synthetic_strip()
    for i, spec in enumerate(PAD_LAYOUT):
        patch, roi = pad_roi_from_layout(strip, spec)
        check(f"pad {spec['name']} extracted", patch.size > 0)

    print("\n[robust_patch_lab]")
    red_patch = np.full((20, 20, 3), (0, 0, 200), dtype=np.uint8)
    lab = robust_patch_lab(red_patch)
    check("returns 3-tuple", lab is not None and len(lab) == 3)
    check("L channel in range", 0 <= lab[0] <= 255)

    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    check("empty patch returns None", robust_patch_lab(empty) is None)

    print("\n[nearest_calibrated_label]")
    exact_neg = (235, 128, 128)
    result = nearest_calibrated_label("LEU", exact_neg)
    check("exact match -> neg", result["label"] == "neg")
    check("distance ~= 0", result["distance"] < 1.0)

    result_none = nearest_calibrated_label("LEU", None)
    check("None triplet -> unknown", result_none["label"] == "unknown")

    result_bad = nearest_calibrated_label("NONEXISTENT", (100, 100, 100))
    check("unknown analyte -> unknown", result_bad["label"] == "unknown")

    print("\n[analyze_test_strip]")
    strip = make_synthetic_strip()
    results = analyze_test_strip(strip)
    check("no error key", "error" not in results)
    check("9 pads returned", len(results["pads"]) == 9)
    for pad in results["pads"]:
        check(f"pad {pad['name']} has estimate",
              pad["estimate"] is not None and pad["estimate"] != "")

    null_result = analyze_test_strip(None)
    check("None input -> error", "error" in null_result)

    print("\n[average_pad_readings]")
    r1 = analyze_test_strip(make_synthetic_strip(60, 300))
    r2 = analyze_test_strip(make_synthetic_strip(60, 300))
    avg = average_pad_readings([r1, r2])
    check("averaged has 9 pads", len(avg["pads"]) == 9)
    check("frame_count == 2", avg.get("frame_count") == 2)
    for pad in avg["pads"]:
        check(f"avg pad {pad['name']} has sample_count",
              "sample_count" in pad and pad["sample_count"] >= 1)

    single = average_pad_readings([r1])
    check("single read returned as-is", single is r1)

    print("\n[compute_water_score]")
    score = compute_water_score(results)
    check("returns string", isinstance(score, str))
    check("format is X/Y", "/" in score)
    check("empty returns --", compute_water_score({}) == "--")
    check("None returns --", compute_water_score(None) == "--")

    print("\n[refine_strip_edges]")
    strip = make_synthetic_strip(60, 300)
    refined = refine_strip_edges(strip)
    check("returns non-None", refined is not None)
    check("returns non-empty", refined.size > 0)
    check("still vertical", refined.shape[0] >= refined.shape[1])

    refined_none = refine_strip_edges(None)
    check("None input -> None", refined_none is None)

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    refined_empty = refine_strip_edges(empty_img)
    check("empty input -> passthrough", refined_empty is not None)

    refined_dbg, debug = refine_strip_edges(strip, return_debug=True)
    check("debug mode returns tuple", isinstance(debug, dict))

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"RESULTS:  {passed}/{total} passed,  {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print("Some tests failed - check above.")
    print("=" * 60)
    return failed


def run_display_demo():
    """pops up a tkinter LCD window and cycles through the screens"""
    print("\n" + "=" * 60)
    print("LCD DISPLAY DEMO")
    print("=" * 60 + "\n")

    from display import Display

    disp = Display(username="Student")
    time.sleep(0.5)

    def step(label, delay=2.0):
        print("  > {}".format(label))
        time.sleep(delay)

    disp.show_hello()
    step("Hello screen")

    for secs in [45, 30, 15, 5, 0]:
        disp.show_filtering(secs)
        step("Filtering... {}s left".format(secs), delay=1.0)

    disp.show_score()
    step("Water score (no data)")

    disp.write("Cartridge in!", "Starting test...")
    step("Cartridge detected", delay=1.5)

    for remaining in [45, 30, 15, 5, 1]:
        disp.show_filtering(remaining)
        step("Filtering... {}s".format(remaining), delay=0.8)

    disp.show_reading()
    step("Reading strip")

    fake_pads = [{"name": n} for n in
                 ["LEU", "NIT", "URO", "PRO", "pH", "BLO", "SG", "KET", "BIL"]]
    disp.show_results(fake_pads, score="7/9")
    step("Results", delay=3.0)

    disp.show_remove()
    step("Remove")

    disp.show_score()
    step("Showing last score", delay=2.5)

    disp.show_hello()
    step("Hello with score", delay=3.0)

    disp.show_shutting_down()
    step("Shutting down", delay=1.5)

    disp.close()
    print("\n  Demo complete!\n")


if __name__ == "__main__":
    failed = run_tests()
    if failed == 0:
        run_display_demo()
    sys.exit(failed)

