"""test trained model on images - pass a file/folder or uses val set"""
import os
import sys
import glob


def main():
    try:
        from ultralytics import YOLO
        import cv2
        import yaml
    except ImportError:
        print("Missing packages. Run:")
        print("  pip install ultralytics pyyaml opencv-python")
        sys.exit(1)

    # find weights - checks a few known spots
    weights = os.environ.get("WEIGHTS", "")
    if not weights:
        for candidate in [
            os.path.join("runs", "detect", "water_strip_v2", "weights", "best.pt"),
            os.path.join("runs", "detect", "strip_detector_v2", "weights", "best.pt"),
            os.path.join("runs", "detect", "strip_detector", "weights", "best.pt"),
        ]:
            if os.path.exists(candidate):
                weights = candidate
                break

    if not weights or not os.path.exists(weights):
        # brute force search lol
        for root, dirs, files in os.walk("runs"):
            if "best.pt" in files:
                weights = os.path.join(root, "best.pt")
                break

    if not weights or not os.path.exists(weights):
        print("No trained model found.")
        print("Run train_model.py first, or set WEIGHTS=path/to/best.pt")
        sys.exit(1)

    print(f"Using model: {weights}")
    mdl = YOLO(weights)

    # figure out what imgs to run on
    imgs = []
    if len(sys.argv) > 1:
        tgt = sys.argv[1]
        if os.path.isfile(tgt):
            imgs = [tgt]
        elif os.path.isdir(tgt):
            for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
                imgs += glob.glob(os.path.join(tgt, ext))
        else:
            print(f"Not a file or folder: {tgt}")
            sys.exit(1)
    else:
        # fallback to dataset val set
        dy = os.path.join("dataset", "data.yaml")
        if os.path.exists(dy):
            with open(dy, "r") as f:
                cfg = yaml.safe_load(f)
            vp = cfg.get("valid", cfg.get("val", ""))
            if vp and not os.path.isabs(vp):
                vp = os.path.join("dataset", vp)
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                imgs += glob.glob(os.path.join(vp, ext))

    if not imgs:
        print("No images found to test on.")
        print("Pass an image path or folder as argument,")
        print("or make sure dataset/ folder exists from training.")
        sys.exit(1)

    print(f"Testing on {len(imgs)} images...\n")

    odir = "test_predictions"
    os.makedirs(odir, exist_ok=True)

    td = 0
    wd = 0  # imgs with detections

    for ip in imgs:
        res = mdl.predict(ip, imgsz=640, conf=0.25, verbose=False)

        for r in res:
            bxs = r.boxes
            n = len(bxs)
            td += n
            fn = os.path.basename(ip)

            if n > 0:
                wd += 1
                for b in bxs:
                    cid = int(b.cls[0])
                    cf = float(b.conf[0])
                    nm = r.names.get(cid, str(cid))
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    print(f"  {fn}: '{nm}' conf={cf:.2f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            else:
                print(f"  {fn}: no detections")

            ann = r.plot()
            cv2.imwrite(os.path.join(odir, fn), ann)

    print(f"\n{'=' * 50}")
    print(f"Images tested:   {len(imgs)}")
    print(f"With detections: {wd}")
    print(f"Total boxes:     {td}")
    print(f"Annotated saved: {odir}/")
    print(f"{'=' * 50}")

    if td == 0:
        print("\nNothing detected. Model might need more training.")
    elif wd < len(imgs) * 0.5:
        print("\nLess than half the images had detections - might need tuning.")
    else:
        print("\nLooks good! Check the annotated images in test_predictions/")


if __name__ == "__main__":
    main()
