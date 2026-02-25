"""
Train a YOLO model on the Roboflow urine test strip dataset
and export it for the IMX500 camera.

Steps:
  1. pip install roboflow ultralytics
  2. Get a free API key from https://app.roboflow.com/settings/api
  3. Run: python train_model.py

The exported model goes into *_imx_model/ which contains
the packerOut.zip and labels.txt you need for the Pi.
"""
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
    """check dataset looks right before training"""
    print("\n" + "=" * 50)
    print("VALIDATING DATASET")
    print("=" * 50)

    with open(data_yaml, "r") as f:
        cfg = yaml_mod.safe_load(f)

    class_names = cfg.get("names", {})
    num_classes = cfg.get("nc", 0)
    print(f"Classes ({num_classes}): {class_names}")

    # check the splits exist and have images
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

        # check for labels too
        label_dir = split_path.replace("images", "labels")
        labels = glob.glob(os.path.join(label_dir, "*.txt"))

        print(f"  {split}: {len(imgs)} images, {len(labels)} label files")

        if len(imgs) == 0:
            print(f"  WARNING: no images found in {split_path}")

        # peek at a couple label files to see what classes are actually annotated
        if labels and split == "train":
            class_counts = {}
            for lf in labels:
                with open(lf, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            cname = class_names.get(cid, str(cid))
                            class_counts[cname] = class_counts.get(cname, 0) + 1

            print(f"\n  Annotations in train set:")
            for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"    {name}: {count} boxes")

    # check if "object" class exists (thats the whole strip)
    has_object = any(v == "object" for v in class_names.values())
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
    print("TRAINING YOLO11n")
    print("=" * 50)

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=320,
        batch=16,
        name="strip_detector",
        patience=10,
    )
    return model, results


def find_best_weights():
    best_pt = os.path.join("runs", "detect", "strip_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        return best_pt
    for root, dirs, files in os.walk("runs"):
        if "best.pt" in files:
            return os.path.join(root, "best.pt")
    return None


def test_model(YOLO, best_pt, data_yaml, yaml_mod):
    """run the trained model on validation images and show what it detects"""
    print("\n" + "=" * 50)
    print("TESTING MODEL ON VALIDATION SET")
    print("=" * 50)

    model = YOLO(best_pt)

    # run official validation
    metrics = model.val(data=data_yaml, imgsz=320)
    print(f"\nValidation results:")
    print(f"  mAP50:     {metrics.box.map50:.3f}")
    print(f"  mAP50-95:  {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall:    {metrics.box.mr:.3f}")

    if metrics.box.map50 < 0.1:
        print("\n  WARNING: mAP50 is very low - model may not be detecting much.")
        print("  The dataset might be too small or need more epochs.")

    # run inference on some validation images and save annotated results
    with open(data_yaml, "r") as f:
        cfg = yaml_mod.safe_load(f)

    val_path = cfg.get("valid", cfg.get("val", ""))
    if val_path and not os.path.isabs(val_path):
        val_path = os.path.join("dataset", val_path)

    val_imgs = glob.glob(os.path.join(val_path, "*.jpg"))
    val_imgs += glob.glob(os.path.join(val_path, "*.png"))

    if not val_imgs:
        print("No validation images found to run inference on.")
        return model

    # predict on up to 10 images and save the annotated outputs
    sample = val_imgs[:10]
    print(f"\nRunning inference on {len(sample)} sample images...")

    out_dir = "test_predictions"
    os.makedirs(out_dir, exist_ok=True)

    total_detections = 0
    for img_path in sample:
        results = model.predict(img_path, imgsz=320, conf=0.25, save=False, verbose=False)
        for r in results:
            boxes = r.boxes
            n = len(boxes)
            total_detections += n

            # save annotated image
            annotated = r.plot()
            fname = os.path.basename(img_path)
            save_path = os.path.join(out_dir, fname)
            import cv2
            cv2.imwrite(save_path, annotated)

            # print what was found
            if n > 0:
                class_names = r.names
                for b in boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    name = class_names.get(cls_id, str(cls_id))
                    print(f"  {fname}: detected '{name}' ({conf:.2f})")
            else:
                print(f"  {fname}: nothing detected")

    print(f"\nTotal: {total_detections} detections across {len(sample)} images")
    print(f"Annotated images saved to {out_dir}/")

    if total_detections == 0:
        print("\nWARNING: model didnt detect anything on the validation set.")
        print("May need more training data or more epochs.")

    return model


def export_imx(YOLO, best_pt):
    print("\n" + "=" * 50)
    print("EXPORTING TO IMX500")
    print("=" * 50)

    model = YOLO(best_pt)
    model.export(format="imx", imgsz=320)

    print("\nDone! Check the *_imx_model/ folder for:")
    print("  - packerOut.zip  (copy to Pi)")
    print("  - labels.txt     (copy to Pi)")
    print()
    print("On the Pi, unpack packerOut.zip for the .rpk file,")
    print("then update MODEL_PATH and LABELS_PATH in PROGRAM.py.")


def main():
    Roboflow, YOLO, yaml_mod = check_deps()

    # step 1: download
    data_yaml = download_dataset(Roboflow)

    # step 2: validate dataset
    validate_dataset(data_yaml, yaml_mod)

    # step 3: train
    model, results = train(YOLO, data_yaml)

    # step 4: find weights
    best_pt = find_best_weights()
    if not best_pt:
        print("Couldnt find best.pt - check runs/ folder")
        sys.exit(1)
    print(f"\nBest weights: {best_pt}")

    # step 5: test on validation images
    test_model(YOLO, best_pt, data_yaml, yaml_mod)

    # step 6: export
    export_imx(YOLO, best_pt)

    print("\n" + "=" * 50)
    print("ALL DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
