"""test pipeline on actual strip photos from water_crops/"""
import sys
import os
import glob
import types

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
sys.modules["gpiozero"].OutputDevice = type("OutputDevice", (), {
    "__init__": lambda self, *a, **k: None,
    "on": lambda self: None, "off": lambda self: None,
    "close": lambda self: None,
})
for mod_name in ["RPLCD", "RPLCD.i2c"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["RPLCD.i2c"].CharLCD = type("CharLCD", (), {})

import cv2
import numpy as np

from PROGRAM import (
    crop_strip, pad_roi_from_layout, robust_patch_lab,
    nearest_calibrated_label, analyze_test_strip, average_pad_readings,
    compute_water_score, clamp_box, filter_outlier_frames,
    PAD_LAYOUT, CALIBRATION_LAB, DANGER_WEIGHTS,
)
from StripEdgeRefinement import refine_strip_edges

NUM_PADS = 16
DATASET_DIR = "water_crops"


def get_crop_images(split="valid"):
    """grab imgs from split folder"""
    idir = os.path.join(DATASET_DIR, split, "images")
    if not os.path.isdir(idir):
        return []
    out = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        out += glob.glob(os.path.join(idir, ext))
    return sorted(out)


def run_pipeline_on_real_images():
    ok = 0
    nope = 0

    def chk(name, condition, detail=""):  # pass/fail
        nonlocal ok, nope
        if condition:
            ok += 1
            print(f"  PASS  {name}")
        else:
            nope += 1
            print(f"  FAIL  {name}  {detail}")

    print("=" * 60)
    print("REAL DATA TESTS - actual water strip images")
    print("=" * 60)

    aimgs = get_crop_images("valid") + get_crop_images("train")
    print(f"\nFound {len(aimgs)} crop images")

    if not aimgs:
        print("No images found. Run prepare_water_dataset.py first.")
        return 1

    # run pipeline on crops
    print(f"\n[crop & analyze - water strip images]")
    acnt = 0
    allres = []

    for ip in aimgs[:20]:
        fn = os.path.basename(ip)
        im = cv2.imread(ip)
        if im is None:
            continue

        # rotate horizontal->vertical like real pipeline does
        if im.shape[1] > im.shape[0]:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

        ref = refine_strip_edges(im)
        if ref is not None and ref.size > 0:
            im = ref

        r = analyze_test_strip(im)

        if "error" not in r and len(r["pads"]) == NUM_PADS:
            acnt += 1
            allres.append(r)

            lv = [p["lab"] for p in r["pads"] if p["lab"] is not None]
            chk(f"{fn} - extracted colors", len(lv) > 0)

            if acnt <= 3:
                for p in r["pads"]:
                    ls = str(p["lab"]) if p["lab"] else "none"
                    print(f"      {p['name']:>12}: LAB={ls:>25}  -> {p['estimate']}")

    chk("at least 1 strip analyzed", acnt > 0, f"got {acnt}")

    # LAB sanity  #CRIT
    print(f"\n[LAB value sanity checks]")
    for r in allres[:10]:
        for p in r["pads"]:
            if p["lab"] is not None:
                L, a, b = p["lab"]
                chk(f"L valid ({p['name']} L={L:.0f})", 0 <= L <= 255)
                chk(f"a valid ({p['name']} a={a:.0f})", 0 <= a <= 255)
                chk(f"b valid ({p['name']} b={b:.0f})", 0 <= b <= 255)
                break

    # multi-frame avg
    print(f"\n[consistency - multi-frame averaging]")
    if len(allres) >= 2:
        avg = average_pad_readings(allres[:3])
        chk("averaging works on real data", "pads" in avg)
        chk("averaged pads count", len(avg.get("pads", [])) == NUM_PADS)

        for p in avg.get("pads", []):
            if p.get("sample_count", 0) > 1:
                chk(f"avg {p['name']} has multiple samples", p["sample_count"] > 1)
                break

    # outlier detection  # works for now
    print(f"\n[outlier detection on real data]")
    if len(allres) >= 3:
        good, bad = filter_outlier_frames(allres[:5])
        chk("outlier filter runs on real data", len(good) >= 1)
        chk("good + bad = total", len(good) + bad == min(5, len(allres)))

    # water score
    print(f"\n[water quality scoring]")  #CRIT
    for i, r in enumerate(allres[:5]):
        sc, wrn = compute_water_score(r)
        chk(f"image {i+1} score 0-100", isinstance(sc, int) and 0 <= sc <= 100,
              f"got: {sc}")
        if i < 3:
            ws = ", ".join(wrn[:3]) if wrn else "none"
            print(f"      score = {sc}/100, warnings: {ws}")

    # danger weights
    print(f"\n[danger weight checks]")
    chk("Lead is critical (3x)", DANGER_WEIGHTS.get("Lead") == 3.0)
    chk("Mercury is critical (3x)", DANGER_WEIGHTS.get("Mercury") == 3.0)
    chk("Iron is high (2x)", DANGER_WEIGHTS.get("Iron") == 2.0)
    chk("pH is standard (1x)", DANGER_WEIGHTS.get("pH") == 1.0)

    # edge cases / failure handling
    print(f"\n[failure handling]")

    wh = np.full((500, 60, 3), 255, dtype=np.uint8)
    rw = analyze_test_strip(wh)
    chk("overexposed white handled", "pads" in rw)

    bl = np.zeros((500, 60, 3), dtype=np.uint8)
    rb = analyze_test_strip(bl)
    chk("underexposed black handled", "pads" in rb)

    tn = np.zeros((5, 3, 3), dtype=np.uint8)
    rt = analyze_test_strip(tn)
    chk("tiny image doesnt crash", rt is not None)

    ns = np.random.randint(0, 256, (500, 60, 3), dtype=np.uint8)
    rn = analyze_test_strip(ns)
    chk("random noise handled", "pads" in rn and len(rn["pads"]) == NUM_PADS)

    chk("None input -> error", "error" in analyze_test_strip(None))
    emp = np.zeros((0, 0, 3), dtype=np.uint8)
    chk("empty input -> error", "error" in analyze_test_strip(emp))

    se, _ = compute_water_score({})
    chk("empty result -> 0", se == 0)
    snp, _ = compute_water_score({"pads": []})
    chk("no pads -> 0", snp == 0)

    # LAB range sanity  """DNT DEL"""
    print(f"\n[real vs calibration LAB ranges]")
    if allres:
        rL = []
        ra = []
        rb_ = []  # rb is taken above lol
        for r in allres:
            for p in r["pads"]:
                if p["lab"] is not None:
                    rL.append(p["lab"][0])
                    ra.append(p["lab"][1])
                    rb_.append(p["lab"][2])

        if rL:
            print(f"      Real LAB across {len(rL)} readings:")
            print(f"        L: {min(rL):.0f} - {max(rL):.0f}")
            print(f"        a: {min(ra):.0f} - {max(ra):.0f}")
            print(f"        b: {min(rb_):.0f} - {max(rb_):.0f}")

            chk("L channel has spread", max(rL) - min(rL) > 10)
            chk("a channel has spread", max(ra) - min(ra) > 3)
            chk("b channel has spread", max(rb_) - min(rb_) > 3)

    # summary
    print("\n" + "=" * 60)
    tot = ok + nope
    print(f"RESULTS:  {ok}/{tot} passed,  {nope} failed")
    if acnt > 0:
        print(f"\nAnalyzed {acnt} real strip images.")
    if nope == 0:
        print("All tests passed.")
    else:
        print("Some tests failed.")
    print("=" * 60)
    return nope


if __name__ == "__main__":
    sys.exit(run_pipeline_on_real_images())
