"""
Test a trained YOLO model against images to verify it detects strips.
Can test against the dataset validation set or your own images.

Usage:
  python test_model.py                          # uses dataset val set
  python test_model.py path/to/image.jpg        # test single image
  python test_model.py path/to/folder/          # test all images in folder

Looks for best.pt in runs/detect/ by default.
Set WEIGHTS env var to use a different path.
"""
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

    # find model weights
    weights = os.environ.get("WEIGHTS", "")
    if not weights:
        for candidate in [
            os.path.join("runs", "detect", "strip_detector_v2", "weights", "best.pt"),
            os.path.join("runs", "detect", "strip_detector", "weights", "best.pt"),
        ]:
            if os.path.exists(candidate):
                weights = candidate
                break

    if not weights or not os.path.exists(weights):
        # search for any best.pt
        for root, dirs, files in os.walk("runs"):
            if "best.pt" in files:
                weights = os.path.join(root, "best.pt")
                break

    if not weights or not os.path.exists(weights):
        print("No trained model found.")
        print("Run train_model.py first, or set WEIGHTS=path/to/best.pt")
        sys.exit(1)

    print(f"Using model: {weights}")
    model = YOLO(weights)

    # figure out what images to test on
    images = []
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target):
            images = [target]
        elif os.path.isdir(target):
            for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
                images += glob.glob(os.path.join(target, ext))
        else:
            print(f"Not a file or folder: {target}")
            sys.exit(1)
    else:
        # try to use the dataset validation set
        data_yaml = os.path.join("dataset", "data.yaml")
        if os.path.exists(data_yaml):
            with open(data_yaml, "r") as f:
                cfg = yaml.safe_load(f)
            val_path = cfg.get("valid", cfg.get("val", ""))
            if val_path and not os.path.isabs(val_path):
                val_path = os.path.join("dataset", val_path)
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                images += glob.glob(os.path.join(val_path, ext))

    if not images:
        print("No images found to test on.")
        print("Pass an image path or folder as argument,")
        print("or make sure dataset/ folder exists from training.")
        sys.exit(1)

    print(f"Testing on {len(images)} images...\n")

    out_dir = "test_predictions"
    os.makedirs(out_dir, exist_ok=True)

    total_dets = 0
    images_with_dets = 0

    for img_path in images:
        results = model.predict(img_path, imgsz=640, conf=0.25, verbose=False)

        for r in results:
            boxes = r.boxes
            n = len(boxes)
            total_dets += n
            fname = os.path.basename(img_path)

            if n > 0:
                images_with_dets += 1
                for b in boxes:
                    cls_id = int(b.cls[0])
                    conf = float(b.conf[0])
                    name = r.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    print(f"  {fname}: '{name}' conf={conf:.2f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            else:
                print(f"  {fname}: no detections")

            # save annotated image
            annotated = r.plot()
            cv2.imwrite(os.path.join(out_dir, fname), annotated)

    print(f"\n{'=' * 50}")
    print(f"Images tested:   {len(images)}")
    print(f"With detections: {images_with_dets}")
    print(f"Total boxes:     {total_dets}")
    print(f"Annotated saved: {out_dir}/")
    print(f"{'=' * 50}")

    if total_dets == 0:
        print("\nNothing detected. Model might need more training.")
    elif images_with_dets < len(images) * 0.5:
        print("\nLess than half the images had detections - might need tuning.")
    else:
        print("\nLooks good! Check the annotated images in test_predictions/")


if __name__ == "__main__":
    main()
