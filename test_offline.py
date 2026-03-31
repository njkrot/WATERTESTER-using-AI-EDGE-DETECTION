"""offline tests - no Pi needed, just run on PC"""
import sys
import time
import numpy as np
import cv2

# fake out Pi libs so imports dont explode on windows
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

class _FakeDI:
    def __init__(self, *a, **k):
        pass
    def close(self):
        pass
    @property
    def value(self):
        return 1


gpiozero_mod = sys.modules["gpiozero"]
gpiozero_mod.Button = type("Button", (), {})
gpiozero_mod.LED = type("LED", (), {})
gpiozero_mod.OutputDevice = type("OutputDevice", (), {
    "__init__": lambda self, *a, **k: None,
    "on": lambda self: None, "off": lambda self: None,
    "close": lambda self: None,
})
gpiozero_mod.DigitalInputDevice = _FakeDI

for mod_name in ["RPLCD", "RPLCD.i2c"]:
    sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules["RPLCD.i2c"].CharLCD = type("CharLCD", (), {})

from PROGRAM import (
    crop_strip, pad_roi_from_layout, robust_patch_lab,
    nearest_calibrated_label, analyze_test_strip, average_pad_readings,
    clamp_box, compute_water_score, filter_outlier_frames,
    compute_median_labs, frame_deviation,
    PAD_LAYOUT, CALIBRATION_LAB, DANGER_WEIGHTS,
)
from StripEdgeRefinement import refine_strip_edges
from pump_control import PumpController
import pump_control


def make_synthetic_strip(width=60, height=500):
    """fake strip w/ 16 pads"""
    s = np.full((height, width, 3), (220, 220, 220), dtype=np.uint8)
    cbgr = [  # colors i eyeballed from a real strip photo
        (200, 190, 210),
        (200, 200, 230),
        (180, 195, 215),
        (200, 200, 230),
        (200, 200, 230),
        (195, 185, 210),
        (160, 170, 200),
        (175, 180, 210),
        (200, 200, 230),
        (200, 200, 230),
        (140, 150, 195),
        (200, 200, 230),
        (200, 200, 230),
        (180, 190, 200),
        (180, 190, 200),
        (180, 160, 140),
    ]
    for i, sp in enumerate(PAD_LAYOUT):
        y0 = int(sp["y0"] * height)
        y1 = int(sp["y1"] * height)
        x0 = int(sp["x0"] * width)
        x1 = int(sp["x1"] * width)
        s[y0:y1, x0:x1] = cbgr[i]
    return s


def run_tests():
    ok = 0
    nope = 0

    def chk(name, condition, detail=""):  # quick pass/fail printer
        nonlocal ok, nope
        if condition:
            ok += 1
            print(f"  PASS  {name}")
        else:
            nope += 1
            print(f"  FAIL  {name}  {detail}")

    print("=" * 60)
    print("OFFLINE TESTS")
    print("=" * 60)

    print("\n[clamp_box]")
    x, y, w, h = clamp_box(-5, -5, 100, 100, 50, 50)
    chk("negative coords clamped", x == 0 and y == 0)
    chk("width clamped to frame", w <= 50)

    print("\n[crop_strip]")  #CRIT - rotation logic
    hz = np.zeros((30, 200, 3), dtype=np.uint8)
    cr = crop_strip(hz, (0, 0, 200, 30))
    chk("horizontal rotated to vertical", cr.shape[0] > cr.shape[1])

    vt = np.zeros((200, 30, 3), dtype=np.uint8)
    cr_v = crop_strip(vt, (0, 0, 30, 200))
    chk("vertical stays vertical", cr_v.shape[0] > cr_v.shape[1])

    print("\n[pad_roi_from_layout]")
    strip = make_synthetic_strip()
    for i, sp in enumerate(PAD_LAYOUT):
        patch, roi = pad_roi_from_layout(strip, sp)
        chk(f"pad {sp['name']} extracted", patch.size > 0)

    print("\n[robust_patch_lab]")
    rp = np.full((20, 20, 3), (0, 0, 200), dtype=np.uint8)
    lab = robust_patch_lab(rp)
    chk("returns 3-tuple", lab is not None and len(lab) == 3)
    chk("L channel in range", 0 <= lab[0] <= 255)

    emp = np.zeros((0, 0, 3), dtype=np.uint8)
    chk("empty patch returns None", robust_patch_lab(emp) is None)

    print("\n[nearest_calibrated_label]")
    ez = (240, 128, 128)
    res = nearest_calibrated_label("Hardness", ez)
    chk("exact match -> 0", res["label"] == "0")
    chk("distance ~= 0", res["distance"] < 5.0)

    res_n = nearest_calibrated_label("Hardness", None)
    chk("None triplet -> unknown", res_n["label"] == "unknown")

    res_bad = nearest_calibrated_label("NONEXISTENT", (100, 100, 100))
    chk("unknown analyte -> unknown", res_bad["label"] == "unknown")

    print("\n[analyze_test_strip]")  #CRIT - main pipeline
    strip = make_synthetic_strip()
    res = analyze_test_strip(strip)
    chk("no error key", "error" not in res)
    chk("16 pads returned", len(res["pads"]) == 16)
    for p in res["pads"]:
        chk(f"pad {p['name']} has estimate",
              p["estimate"] is not None and p["estimate"] != "")

    nr = analyze_test_strip(None)
    chk("None input -> error", "error" in nr)

    print("\n[average_pad_readings]")
    r1 = analyze_test_strip(make_synthetic_strip(60, 500))
    r2 = analyze_test_strip(make_synthetic_strip(60, 500))
    avg = average_pad_readings([r1, r2])
    chk("averaged has 16 pads", len(avg["pads"]) == 16)
    chk("frame_count == 2", avg.get("frame_count") == 2)
    for p in avg["pads"]:
        chk(f"avg pad {p['name']} has sample_count",
              "sample_count" in p and p["sample_count"] >= 1)

    single = average_pad_readings([r1])
    chk("single read returned as-is", single is r1)

    print("\n[compute_water_score]")  #CRIT - scoring
    sc, wrn = compute_water_score(res)
    chk("returns int score", isinstance(sc, int))
    chk("score 0-100", 0 <= sc <= 100)
    chk("warnings is list", isinstance(wrn, list))
    sc_e, w_e = compute_water_score({})
    chk("empty returns 0", sc_e == 0)
    sc_n, w_n = compute_water_score(None)
    chk("None returns 0", sc_n == 0)

    print("\n[danger_weights]")
    chk("Lead weight 3x", DANGER_WEIGHTS["Lead"] == 3.0)
    chk("Mercury weight 3x", DANGER_WEIGHTS["Mercury"] == 3.0)
    chk("Iron weight 2x", DANGER_WEIGHTS["Iron"] == 2.0)
    chk("pH weight 1x", DANGER_WEIGHTS["pH"] == 1.0)

    print("\n[outlier_detection]")  #CRIT
    rg1 = analyze_test_strip(make_synthetic_strip(60, 500))
    rg2 = analyze_test_strip(make_synthetic_strip(60, 500))
    rg3 = analyze_test_strip(make_synthetic_strip(60, 500))
    bstrip = np.full((500, 60, 3), (50, 50, 200), dtype=np.uint8)
    rb = analyze_test_strip(bstrip)
    all_r = [rg1, rg2, rb, rg3]
    good, bcnt = filter_outlier_frames(all_r)
    chk("outlier detected", bcnt >= 1)
    chk("good frames kept", len(good) >= 2)
    chk("total = good + bad", len(good) + bcnt == len(all_r))

    mlabs = compute_median_labs([rg1, rg2, rg3])
    chk("median has 16 entries", len(mlabs) == 16)
    dg = frame_deviation(rg1, mlabs)
    db = frame_deviation(rb, mlabs)
    chk("bad frame has higher deviation", db > dg)

    print("\n[refine_strip_edges]")
    strip = make_synthetic_strip(60, 300)
    ref = refine_strip_edges(strip)
    chk("returns non-None", ref is not None)
    chk("returns non-empty", ref.size > 0)
    chk("still vertical", ref.shape[0] >= ref.shape[1])

    ref_n = refine_strip_edges(None)
    chk("None input -> None", ref_n is None)

    eimg = np.zeros((0, 0, 3), dtype=np.uint8)
    ref_e = refine_strip_edges(eimg)
    chk("empty input -> passthrough", ref_e is not None)

    ref_dbg, dbg = refine_strip_edges(strip, return_debug=True)
    chk("debug mode returns tuple", isinstance(dbg, dict))

    print("\n[pump_controller]")  #CRIT - hardware sequence
    # save originals so we can speed up for test
    of = pump_control.FILL_SECONDS
    os_ = pump_control.SOAK_SECONDS   # os is taken lol
    od = pump_control.DRAIN_SECONDS
    ofl = pump_control.FILTER_SECONDS
    odp = pump_control.DISPENSE_TIMEOUT
    pump_control.FILL_SECONDS = 0.1
    pump_control.SOAK_SECONDS = 0.1
    pump_control.DRAIN_SECONDS = 0.1
    pump_control.FILTER_SECONDS = 0.1
    pump_control.DISPENSE_TIMEOUT = 0.1

    pc = PumpController()
    st = pc.get_status()
    chk("initial state all off", not st["main_pump"] and not st["dispense_pump"])
    chk("dump starts closed", st["dump_valve"] == "closed")
    chk("aux valve starts closed", st.get("aux_valve") == "closed")
    chk("test not completed initially", not st["test_completed"])
    chk("cannot dispense before test", not pc.can_dispense)

    rv = pc.fill_test_container()
    chk("fill completes", rv)
    rv = pc.soak_strip()
    chk("soak completes", rv)
    rv = pc.drain_test_container()
    chk("drain completes", rv)

    blocked = pc.run_filter()
    chk("filter blocked before test complete", not blocked)

    pc.mark_test_completed()
    chk("test marked complete", pc.test_completed)
    chk("can dispense after test", pc.can_dispense)

    rv = pc.run_filter()
    chk("filter runs after test complete", rv)
    chk("filtering_done flag set", pc.filtering_done)

    rv = pc.dispense()
    chk("dispense works after test", rv)

    fin = pc.get_status()
    chk("pumps off after sequence", not fin["main_pump"] and not fin["dispense_pump"])
    pc.close()

    # restore """DNT DEL"""
    pump_control.FILL_SECONDS = of
    pump_control.SOAK_SECONDS = os_
    pump_control.DRAIN_SECONDS = od
    pump_control.FILTER_SECONDS = ofl
    pump_control.DISPENSE_TIMEOUT = odp

    print("\n" + "=" * 60)
    tot = ok + nope
    print(f"RESULTS:  {ok}/{tot} passed,  {nope} failed")
    if nope == 0:
        print("All tests passed.")
    else:
        print("Some tests failed - check above.")
    print("=" * 60)
    return nope


def run_touchscreen_demo():
    # opens the GUI so you can click around  # works for now
    print("\n" + "=" * 60)
    print("TOUCHSCREEN GUI DEMO")
    print("=" * 60)
    print("  The touchscreen window will open.")
    print("  Press START to run a simulated test cycle.")
    print("  Press DISPENSE after the test completes.")
    print("  Close the window when done.\n")

    from display import WaterFilterApp
    app = WaterFilterApp()
    app.run()


if __name__ == "__main__":
    f = run_tests()
    if f == 0:
        run_touchscreen_demo()
    sys.exit(f)
