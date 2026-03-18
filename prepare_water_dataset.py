"""auto-annotate strip imgs -> YOLO dataset"""
import cv2
import numpy as np
import os
import shutil
import random


SRC_DIR = "water_strips"
OUT_DIR = "water_dataset"
VAL_RATIO = 0.2
PAD_PX = 15
MIN_STRIP_H = 80
MAX_STRIP_H = 220
MIN_ASPECT = 12
MAX_ASPECT = 40
MIN_WIDTH_RATIO = 0.35


def find_strips(img_path):
    im = cv2.imread(img_path)
    if im is None:
        return [], None
    h, w = im.shape[:2]
    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #NEED?! - brightness profiling to find strip rows
    prof = np.mean(gr, axis=1)
    bg = np.percentile(prof, 25)
    thr = bg + 35  # works for now

    bright = prof > thr
    rows = []
    st = None
    for i in range(len(bright)):  #NEED?! - run detection
        if bright[i] and st is None:
            st = i
        elif not bright[i] and st is not None:
            rows.append((st, i))
            st = None
    if st is not None:
        rows.append((st, len(bright)))

    strips = []
    for (y0, y1) in rows:
        sh = y1 - y0
        if sh < MIN_STRIP_H or sh > MAX_STRIP_H:
            continue

        rslice = gr[y0:y1, :]
        cpro = np.mean(rslice, axis=0)
        bcols = cpro > (bg + 25)
        xs = np.where(bcols)[0]
        if len(xs) < 20:
            continue

        x0, x1p = int(xs[0]), int(xs[-1])
        bw = x1p - x0
        bh = y1 - y0
        asp = bw / max(1, bh)

        if asp < MIN_ASPECT or asp > MAX_ASPECT:
            continue
        if bw < w * MIN_WIDTH_RATIO:
            continue

        # pad the box a bit  """DNT DEL"""
        px0 = max(0, x0 - PAD_PX)
        py0 = max(0, y0 - PAD_PX)
        px1 = min(w, x1p + PAD_PX)
        py1 = min(h, y1 + PAD_PX)

        strips.append((px0, py0, px1 - px0, py1 - py0))

    return strips, (w, h)


def box_to_yolo(x, y, bw, bh, img_w, img_h):
    cx = (x + bw / 2.0) / img_w
    cy = (y + bh / 2.0) / img_h
    nw = bw / img_w
    nh = bh / img_h
    return cx, cy, nw, nh


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"Source folder '{SRC_DIR}' not found.")
        print("Run the HEIC conversion first.")
        return

    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    fls = sorted([f for f in os.listdir(SRC_DIR) if f.lower().endswith('.jpg')])
    random.seed(42)
    random.shuffle(fls)

    vc = max(1, int(len(fls) * VAL_RATIO))
    vset = set(fls[:vc])

    tstrips = 0
    skp = 0

    for f in sorted(fls):
        ip = os.path.join(SRC_DIR, f)
        strips, dims = find_strips(ip)

        if not strips:
            print(f"  SKIP {f}: no strips found")
            skp += 1
            continue

        spl = "valid" if f in vset else "train"
        iw, ih = dims

        shutil.copy2(ip, os.path.join(OUT_DIR, spl, "images", f))

        ln = os.path.splitext(f)[0] + ".txt"
        lp = os.path.join(OUT_DIR, spl, "labels", ln)
        with open(lp, "w") as lf:
            for (x, y, bw, bh) in strips:
                cx, cy, nw, nh = box_to_yolo(x, y, bw, bh, iw, ih)
                lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        tstrips += len(strips)
        print(f"  {spl:5s} {f}: {len(strips)} strips")

    # write data.yaml for YOLO
    dyaml = os.path.join(OUT_DIR, "data.yaml")
    ap = os.path.abspath(OUT_DIR).replace("\\", "/")
    with open(dyaml, "w") as f:
        f.write(f"path: {ap}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: strip\n")

    ti = len(os.listdir(os.path.join(OUT_DIR, "train", "images")))
    vi = len(os.listdir(os.path.join(OUT_DIR, "valid", "images")))

    print(f"\nDataset ready in '{OUT_DIR}/'")
    print(f"  Train: {ti} images")
    print(f"  Valid: {vi} images")
    print(f"  Total strip annotations: {tstrips}")
    print(f"  Skipped images: {skp}")
    print(f"  data.yaml: {dyaml}")


if __name__ == "__main__":
    main()
