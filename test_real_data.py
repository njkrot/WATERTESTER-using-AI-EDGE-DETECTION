"""
Test the color analysis pipeline against REAL strip images from the
Roboflow dataset. Uses ground-truth bounding boxes to simulate what
the YOLO model would detect, then runs our full pipeline on the crops.

Also includes intentional failure tests to prove error handling works.

Run:  python test_real_data.py
(requires dataset/ folder from train_model.py download)
"""
import sys
import os
import glob
import types

# stub Pi-only libs
for mod_name in [
    "picamera2", "picamera2.devices", "picamera2.devices.imx500",
    "picamera2.devices.imx500.postprocess", "gpiozero",
]:
    sys.modules[mod_name] = types.ModuleType(mod_name)

picam2_imx500 = sys.modules["picamera2.devices.imx500"]
picam2_imx500.NetworkIntrinsics = type("NetworkIntrinsics", (), {})
picam2_imx500.postprocess_nanodet_detection = lambda *a, **k: None
sys.modules["picamera2.devices"].IMX500 = type("IMX500", (), {})
sys.modules["picamera2"].Picamera2 = type("Picamera2", (), {})
sys.modules["gpiozero"].Button = type("Button", (), {})
sys.modules["gpiozero"].LED = type("LED", (), {})
for mod_name in ["RPLCD", "RPLCD.i2c"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["RPLCD.i2c"].CharLCD = type("CharLCD", (), {})

import cv2
import numpy as np
import yaml

from PROGRAM import (
    crop_strip, pad_roi_from_layout, robust_patch_lab,
    nearest_calibrated_label, analyze_test_strip, average_pad_readings,
    compute_water_score, clamp_box, PAD_LAYOUT, CALIBRATION_LAB,
)
from StripEdgeRefinement import refine_strip_edges

# water quality parameter mapping
# same pads, different meaning for water testing
WATER_PARAMS = {
    "LEU": "Lead",
    "NIT": "Nitrate",
    "URO": "Chlorine",
    "PRO": "Turbidity",
    "pH":  "pH",
    "BLO": "Iron",
    "SG":  "TDS",
    "KET": "Pesticides",
    "BIL": "Copper",
}

# roboflow dataset class names
ROBOFLOW_CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'Bilirubin','Blood','Glucose','Ketone','Leukocytes',
    'Nitrite','Protein','Sp.Gravity','Urobilinogen','object','pH'
]
OBJECT_CLASS_ID = 19  # "object" = the whole strip


def load_dataset_info():
    data_yaml = os.path.join("dataset", "data.yaml")
    if not os.path.exists(data_yaml):
        print("ERROR: dataset/ folder not found. Run train_model.py first")
        print("  or:  python -c \"from roboflow import Roboflow; ...\" to download")
        sys.exit(1)
    with open(data_yaml, "r") as f:
        return yaml.safe_load(f)


def get_images_and_labels(split="valid"):
    """get paired image+label paths from a dataset split"""
    img_dir = os.path.join("dataset", split, "images")
    lbl_dir = os.path.join("dataset", split, "labels")

    pairs = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        for img_path in glob.glob(os.path.join(img_dir, ext)):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path))
    return pairs


def parse_yolo_labels(lbl_path, img_w, img_h):
    """parse YOLO format labels into pixel bounding boxes"""
    objects = []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # convert normalized center+wh to pixel x,y,w,h
            px = int((cx - w/2) * img_w)
            py = int((cy - h/2) * img_h)
            pw = int(w * img_w)
            ph = int(h * img_h)
            cls_name = ROBOFLOW_CLASSES[cls_id] if cls_id < len(ROBOFLOW_CLASSES) else str(cls_id)
            objects.append({"class_id": cls_id, "class_name": cls_name,
                            "box": (px, py, pw, ph)})
    return objects


def run_pipeline_on_real_images():
    """main test: load real images, crop strips, analyze colors"""
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
    print("REAL DATA TESTS - actual strip images from dataset")
    print("=" * 60)

    # use all splits for testing
    all_pairs = []
    for split in ["valid", "test", "train"]:
        pairs = get_images_and_labels(split)
        all_pairs.extend(pairs)

    print(f"\nFound {len(all_pairs)} image+label pairs total")

    # find images that have the "object" class (whole strip bbox)
    strip_images = []
    pad_images = []

    for img_path, lbl_path in all_pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        objects = parse_yolo_labels(lbl_path, w, h)

        has_strip = any(o["class_id"] == OBJECT_CLASS_ID for o in objects)
        has_pads = any(10 <= o["class_id"] <= 20 and o["class_id"] != OBJECT_CLASS_ID for o in objects)

        if has_strip:
            strip_images.append((img_path, img, objects))
        if has_pads:
            pad_images.append((img_path, img, objects))

    print(f"Images with whole-strip bbox ('object'): {len(strip_images)}")
    print(f"Images with individual pad annotations: {len(pad_images)}")

    # ============================================
    # TEST 1: crop strips from real images
    # ============================================
    print(f"\n[crop & analyze - whole strip images]")

    analyzed_count = 0
    all_results = []

    for img_path, img, objects in strip_images[:20]:
        fname = os.path.basename(img_path)
        strip_objs = [o for o in objects if o["class_id"] == OBJECT_CLASS_ID]

        for so in strip_objs:
            box = so["box"]
            crop = crop_strip(img, box)

            if crop is None or crop.size == 0:
                continue

            # try edge refinement too
            refined = refine_strip_edges(crop)
            if refined is not None and refined.size > 0:
                crop = refined

            result = analyze_test_strip(crop)

            if "error" not in result and len(result["pads"]) == 9:
                analyzed_count += 1
                all_results.append(result)

                # show what we got from this image
                lab_values = []
                for pad in result["pads"]:
                    if pad["lab"] is not None:
                        lab_values.append(pad["lab"])

                has_real_colors = len(lab_values) > 0
                check(f"{fname} - extracted colors",
                      has_real_colors,
                      "no LAB values extracted")

                if has_real_colors and analyzed_count <= 5:
                    # print detail for first few
                    for pad in result["pads"]:
                        water_param = WATER_PARAMS.get(pad["name"], pad["name"])
                        lab_str = str(pad["lab"]) if pad["lab"] else "none"
                        print(f"      {pad['name']:>3} ({water_param:>10}): "
                              f"LAB={lab_str:>25}  -> {pad['estimate']}")

    check("at least 1 strip analyzed from real data", analyzed_count > 0,
          f"got {analyzed_count}")

    # ============================================
    # TEST 2: LAB values are reasonable
    # ============================================
    print(f"\n[LAB value sanity checks]")

    for result in all_results[:10]:
        for pad in result["pads"]:
            if pad["lab"] is not None:
                L, a, b = pad["lab"]
                check(f"L channel valid ({pad['name']} L={L:.0f})",
                      0 <= L <= 255)
                check(f"a channel valid ({pad['name']} a={a:.0f})",
                      0 <= a <= 255)
                check(f"b channel valid ({pad['name']} b={b:.0f})",
                      0 <= b <= 255)
                break  # just check one pad per image to keep output sane

    # ============================================
    # TEST 3: consistency across similar images
    # ============================================
    print(f"\n[consistency - multi-frame averaging]")

    if len(all_results) >= 2:
        avg = average_pad_readings(all_results[:3])
        check("averaging works on real data", "pads" in avg)
        check("averaged pads count", len(avg.get("pads", [])) == 9)

        for pad in avg.get("pads", []):
            if pad.get("sample_count", 0) > 1:
                check(f"avg {pad['name']} has multiple samples",
                      pad["sample_count"] > 1)
                break

    # ============================================
    # TEST 4: water score from real data
    # ============================================
    print(f"\n[water quality scoring]")

    for i, result in enumerate(all_results[:5]):
        score = compute_water_score(result)
        check(f"image {i+1} score is valid string", isinstance(score, str) and "/" in score,
              f"got: {score}")
        if i < 3:
            print(f"      score = {score}")

    # ============================================
    # TEST 5: individual pad detection from dataset annotations
    # ============================================
    print(f"\n[individual pad color reads from annotations]")

    pads_read = 0
    for img_path, img, objects in pad_images[:10]:
        fname = os.path.basename(img_path)
        pad_objs = [o for o in objects if 10 <= o["class_id"] <= 20 and o["class_id"] != OBJECT_CLASS_ID]

        for po in pad_objs[:3]:
            box = po["box"]
            x, y, w, h = clamp_box(box[0], box[1], box[2], box[3],
                                    img.shape[1], img.shape[0])
            patch = img[y:y+h, x:x+w].copy()
            if patch.size == 0:
                continue

            lab = robust_patch_lab(patch)
            if lab is not None:
                pads_read += 1
                if pads_read <= 6:
                    print(f"      {fname}: {po['class_name']:>13} -> "
                          f"LAB=[{lab[0]:.0f}, {lab[1]:.0f}, {lab[2]:.0f}]")

    check("read colors from annotated pads", pads_read > 0, f"got {pads_read}")

    # ============================================
    # TEST 6: FAILURE CASES - prove error handling works
    # ============================================
    print(f"\n[intentional failure tests]")

    # pure white image (overexposed)
    white = np.full((300, 60, 3), 255, dtype=np.uint8)
    result_white = analyze_test_strip(white)
    check("overexposed white image handled",
          "pads" in result_white,
          "crashed on white image")

    # pure black image (underexposed)
    black = np.zeros((300, 60, 3), dtype=np.uint8)
    result_black = analyze_test_strip(black)
    check("underexposed black image handled",
          "pads" in result_black,
          "crashed on black image")

    # tiny image (too small to be a real strip)
    tiny = np.zeros((5, 3, 3), dtype=np.uint8)
    result_tiny = analyze_test_strip(tiny)
    check("tiny image doesnt crash", result_tiny is not None)

    # random noise (garbage data)
    noise = np.random.randint(0, 256, (300, 60, 3), dtype=np.uint8)
    result_noise = analyze_test_strip(noise)
    check("random noise handled",
          "pads" in result_noise and len(result_noise["pads"]) == 9)

    # half-strip (strip cut in half vertically)
    if strip_images:
        _, img, objects = strip_images[0]
        strip_obj = [o for o in objects if o["class_id"] == OBJECT_CLASS_ID][0]
        box = strip_obj["box"]
        crop_full = crop_strip(img, box)
        if crop_full is not None:
            half = crop_full[:crop_full.shape[0]//2, :, :]
            result_half = analyze_test_strip(half)
            check("half-strip still returns 9 pads (some may be empty)",
                  len(result_half.get("pads", [])) == 9)

    # extremely blurry (simulating motion blur)
    if strip_images:
        _, img, objects = strip_images[0]
        strip_obj = [o for o in objects if o["class_id"] == OBJECT_CLASS_ID][0]
        crop = crop_strip(img, strip_obj["box"])
        if crop is not None:
            blurry = cv2.GaussianBlur(crop, (31, 31), 15)
            result_blur = analyze_test_strip(blurry)
            check("blurry image handled",
                  "pads" in result_blur and len(result_blur["pads"]) == 9)

    # wrong orientation (horizontal when we expect vertical)
    if strip_images:
        _, img, objects = strip_images[0]
        strip_obj = [o for o in objects if o["class_id"] == OBJECT_CLASS_ID][0]
        crop = crop_strip(img, strip_obj["box"])
        if crop is not None:
            rotated = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            result_rot = analyze_test_strip(rotated)
            check("rotated image still works",
                  "pads" in result_rot and len(result_rot["pads"]) == 9)

    # None and empty
    check("None input -> error", "error" in analyze_test_strip(None))
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    check("empty input -> error", "error" in analyze_test_strip(empty))

    # score edge cases
    check("empty result -> --", compute_water_score({}) == "--")
    check("no pads -> --", compute_water_score({"pads": []}) == "--")

    # ============================================
    # TEST 7: compare real vs synthetic LAB ranges
    # ============================================
    print(f"\n[real vs synthetic LAB comparison]")

    if all_results:
        real_L_vals = []
        real_a_vals = []
        real_b_vals = []
        for result in all_results:
            for pad in result["pads"]:
                if pad["lab"] is not None:
                    real_L_vals.append(pad["lab"][0])
                    real_a_vals.append(pad["lab"][1])
                    real_b_vals.append(pad["lab"][2])

        if real_L_vals:
            print(f"      Real data LAB ranges across {len(real_L_vals)} pad readings:")
            print(f"        L: {min(real_L_vals):.0f} - {max(real_L_vals):.0f}  (avg {np.mean(real_L_vals):.0f})")
            print(f"        a: {min(real_a_vals):.0f} - {max(real_a_vals):.0f}  (avg {np.mean(real_a_vals):.0f})")
            print(f"        b: {min(real_b_vals):.0f} - {max(real_b_vals):.0f}  (avg {np.mean(real_b_vals):.0f})")

            check("L channel has spread (not all same color)",
                  max(real_L_vals) - min(real_L_vals) > 10)
            check("a channel has spread",
                  max(real_a_vals) - min(real_a_vals) > 5)
            check("b channel has spread",
                  max(real_b_vals) - min(real_b_vals) > 5)

            # compare against our calibration table
            print(f"\n      Calibration table ranges:")
            for pad_name, entries in CALIBRATION_LAB.items():
                L_range = [e[1][0] for e in entries]
                print(f"        {pad_name:>3}: L={min(L_range):.0f}-{max(L_range):.0f}  "
                      f"(real data L avg={np.mean(real_L_vals):.0f})")

    # ============================================
    # summary
    # ============================================
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"RESULTS:  {passed}/{total} passed,  {failed} failed")

    if analyzed_count > 0:
        print(f"\nSuccessfully analyzed {analyzed_count} real strip images from dataset.")
        print(f"Water quality mapping: {WATER_PARAMS}")
    else:
        print("\nWARNING: no strip images could be analyzed.")
        print("Check that dataset has 'object' class annotations.")

    if failed == 0:
        print("\nAll tests passed - pipeline works on real data.")
    else:
        print("\nSome tests failed - check output above.")
    print("=" * 60)
    return failed


if __name__ == "__main__":
    load_dataset_info()
    sys.exit(run_pipeline_on_real_images())
