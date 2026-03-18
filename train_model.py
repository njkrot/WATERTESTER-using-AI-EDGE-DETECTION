"""train YOLO on strip dataset -> export for IMX500"""
import os
import sys
import glob


def check_deps():
    try:
        from roboflow import Roboflow
        from ultralytics import YOLO
        import yaml
        return Roboflow, YOLO, yaml
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Run:  pip install roboflow ultralytics pyyaml")
        sys.exit(1)


def download_dataset(Roboflow):
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        api_key = input("Enter your Roboflow API key: ").strip()
    if not api_key:
        print("No API key.")
        sys.exit(1)

    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("urine-test-strips-qq9jx").project("urine-test-ikfjh")
    project.version(1).download("yolov8", location="./dataset")

    data_yaml = os.path.join("dataset", "data.yaml")
    if not os.path.exists(data_yaml):
        print("ERROR: data.yaml not found after download")
        sys.exit(1)
    return data_yaml


def validate_dataset(data_yaml, yaml_mod):
    """quick check"""
    print("\n" + "=" * 50)
    print("VALIDATING DATASET")
    print("=" * 50)

    with open(data_yaml, "r") as f:
        cfg = yaml_mod.safe_load(f)

    cnames = cfg.get("names", {})
    nc = cfg.get("nc", 0)
    print(f"Classes ({nc}): {cnames}")

    # splits gotta have images or whats the point
    for split in ["train", "valid", "test"]:
        split_path = cfg.get(split, "")
        if not split_path:
            print(f"  {split}: not defined (ok if test)")
            continue

        if not os.path.isabs(split_path):
            split_path = os.path.join("dataset", split_path)

        imgs = glob.glob(os.path.join(split_path, "*.jpg"))
        imgs += glob.glob(os.path.join(split_path, "*.png"))
        imgs += glob.glob(os.path.join(split_path, "*.jpeg"))

        lbl_dir = split_path.replace("images", "labels")
        labels = glob.glob(os.path.join(lbl_dir, "*.txt"))

        print(f"  {split}: {len(imgs)} images, {len(labels)} label files")

        if len(imgs) == 0:
            print(f"  WARNING: no images found in {split_path}")

        # peek at labels to see whats annotated
        if labels and split == "train":
            cc = {}
            for lf in labels:
                with open(lf, "r") as f:
                    for ln in f:
                        pts = ln.strip().split()
                        if pts:
                            cid = int(pts[0])
                            cn = cnames.get(cid, str(cid))
                            cc[cn] = cc.get(cn, 0) + 1

            print(f"\n  Annotations in train set:")
            for nm, cnt in sorted(cc.items(), key=lambda x: -x[1]):
                print(f"    {nm}: {cnt} boxes")

    # "object" = whole strip class
    has_object = any(v == "object" for v in cnames.values())
    if has_object:
        print("\n  'object' class found - this is the whole strip detection class.")
    else:
        print("\n  NOTE: no 'object' class found.")
        print("  You may need to update TARGET_CLASS_NAME in PROGRAM.py")
        print("  to match one of the class names above.")

    print()
    return cfg


def train(YOLO, data_yaml):
    print("=" * 50)
    print("TRAINING YOLO11s")
    print("=" * 50)

    import torch
    dev = 0 if torch.cuda.is_available() else "cpu"
    if dev == 0:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, training on CPU (this will be slow)")

    mdl = YOLO("yolo11s.pt")
    #CRIT - actual training call, dont mess with these params
    results = mdl.train(
        data=data_yaml,
        epochs=150,
        imgsz=640,
        batch=32,
        device=dev,
        name="strip_detector",
        patience=25,
        workers=4,
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )
    return mdl, results


def find_best_weights():
    for candidate in [
        os.path.join("runs", "detect", "water_strip_v2", "weights", "best.pt"),
        os.path.join("runs", "detect", "strip_detector", "weights", "best.pt"),
        os.path.join("runs", "detect", "strip_detector_v2", "weights", "best.pt"),
    ]:
        if os.path.exists(candidate):
            return candidate
    for root, dirs, files in os.walk("runs"):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    return None


def test_model(YOLO, best_pt, data_yaml, yaml_mod):
    """run model on val imgs"""
    print("\n" + "=" * 50)
    print("TESTING MODEL ON VALIDATION SET")
    print("=" * 50)

    mdl = YOLO(best_pt)

    # official val pass
    metrics = mdl.val(data=data_yaml, imgsz=640)
    print(f"\nValidation results:")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    if metrics.box.map50 < 0.1:
        print("\n  WARNING: mAP50 is very low - model may not be detecting much.")
        print("  The dataset might be too small or need more epochs.")

    # inference on val imgs, save annotated  # works for now
    with open(data_yaml, "r") as f:
        cfg = yaml_mod.safe_load(f)

    val_path = cfg.get("valid", cfg.get("val", ""))
    if val_path and not os.path.isabs(val_path):
        val_path = os.path.join("dataset", val_path)

    val_imgs = glob.glob(os.path.join(val_path, "*.jpg"))
    val_imgs += glob.glob(os.path.join(val_path, "*.png"))

    if not val_imgs:
        print("No validation images found to run inference on.")
        return mdl

    # predict on up to 10 imgs
    sample = val_imgs[:10]
    print(f"\nRunning inference on {len(sample)} sample images...")

    out_dir = "test_predictions"
    os.makedirs(out_dir, exist_ok=True)

    td = 0
    for ip in sample:
        res = mdl.predict(ip, imgsz=640, conf=0.25, save=False, verbose=False)
        for r in res:
            bxs = r.boxes
            n = len(bxs)
            td += n

            ann = r.plot()
            fn = os.path.basename(ip)
            sp = os.path.join(out_dir, fn)
            import cv2
            cv2.imwrite(sp, ann)

            if n > 0:
                cn = r.names
                for b in bxs:
                    cid = int(b.cls[0])
                    cf = float(b.conf[0])
                    nm = cn.get(cid, str(cid))
                    print(f"  {fn}: detected '{nm}' ({cf:.2f})")
            else:
                print(f"  {fn}: nothing detected")

    print(f"\nTotal: {td} detections across {len(sample)} images")
    print(f"Annotated images saved to {out_dir}/")

    if td == 0:
        print("\nWARNING: model didnt detect anything on the validation set.")
        print("May need more training data or more epochs.")

    return mdl


def export_imx(YOLO, best_pt):
    print("\n" + "=" * 50)
    print("EXPORTING TO IMX500")
    print("=" * 50)

    m = YOLO(best_pt)
    m.export(format="imx", imgsz=640)  # dumps to *_imx_model/

    print("\nDone! Check the *_imx_model/ folder for:")
    print("  - packerOut.zip  (copy to Pi)")
    print("  - labels.txt     (copy to Pi)")
    print()
    print("On the Pi, unpack packerOut.zip for the .rpk file,")
    print("then update MODEL_PATH and LABELS_PATH in PROGRAM.py.")


def main():
    Roboflow, YOLO, yaml_mod = check_deps()

    # local dataset first if it exists
    wy = os.path.join("water_crops", "data.yaml")
    if os.path.exists(wy):
        print("Found local water strip dataset at water_crops/")
        print("To use the Roboflow dataset instead, rename/remove water_crops/")
        data_yaml = wy
    else:
        data_yaml = download_dataset(Roboflow)

    validate_dataset(data_yaml, yaml_mod)

    #CRIT - training
    mdl, results = train(YOLO, data_yaml)

    bp = find_best_weights()
    if not bp:
        print("Couldnt find best.pt - check runs/ folder")
        sys.exit(1)
    print(f"\nBest weights: {bp}")

    test_model(YOLO, bp, data_yaml, yaml_mod)

    export_imx(YOLO, bp)

    print("\n" + "=" * 50)
    print("ALL DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
