import cv2
import numpy as np


def _order_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _warp_min_area_rect(image, contour, pad_px=4):
    """warp rect"""
    rect = cv2.minAreaRect(contour)
    (cx, cy),(rw, rh), angle = rect

    if rw < 2 or rh < 2:
        return None

    box = cv2.boxPoints(rect)
    box = _order_points_clockwise(box)

    ow = max(1, int(round(rw)))
    oh = max(1, int(round(rh)))

    dst = np.array([
        [0, 0],[ow - 1, 0],
        [ow - 1, oh - 1],[0, oh - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (ow, oh))

    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # chop border noise
    if pad_px > 0 and warped.shape[0] > 2*pad_px and warped.shape[1] > 2*pad_px:
        warped = warped[pad_px:-pad_px, pad_px:-pad_px].copy()

    return warped


def _score_strip_contour(contour, img_shape):
    """contour score"""
    h, w = img_shape[:2]
    img_a = float(h * w)

    ar = cv2.contourArea(contour)
    if ar < 0.01 * img_a:
        return None

    rect = cv2.minAreaRect(contour)
    (cx, cy),(rw, rh), angle = rect

    if rw < 2 or rh < 2:
        return None

    lng = max(rw, rh)
    sht = max(1.0, min(rw, rh))
    aspect = lng / sht

    if aspect < 2.0:
        return None

    rect_a = max(1.0, rw * rh)
    fill = ar / rect_a

    cdist = np.hypot(cx - w/2.0, cy - h/2.0)
    cdist_n = cdist / max(1.0, np.hypot(w/2.0, h/2.0))

    a_ratio = ar / img_a

    #CRIT HOW -- weighted combo, found by trial-and-error
    score = (
        2.2 * a_ratio
        + 0.8 * min(aspect, 12.0) / 12.0
        + 0.9 * np.clip(fill, 0.0, 1.2)
        - 0.8 * cdist_n
    )  #NEED?! maybe tune weights later
    return score

def _best_contour_from_mask(mask, img_shape):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    bscore = None
    for c in cnts:
        sc = _score_strip_contour(c, img_shape)
        if sc is None:
            continue
        if bscore is None or sc > bscore:
            best = c
            bscore = sc
    return best, bscore


def refine_strip_edges(strip_img_bgr, return_debug=False):
    """DNT DEL - main refinement"""
    debug = {}

    if strip_img_bgr is None or strip_img_bgr.size == 0:
        return (strip_img_bgr, debug) if return_debug else strip_img_bgr

    img = strip_img_bgr.copy()

    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    if h < 20 or w < 8:
        return (img, debug) if return_debug else img

    bl = cv2.GaussianBlur(img, (5, 5), 0)
    gr = cv2.cvtColor(bl, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gr_eq = clahe.apply(gr)

    _, th_bin = cv2.threshold(gr_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gr_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 17))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def clean_mask(m):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kv, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kh, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
        return m

    masks = {
        "otsu_bin": clean_mask(th_bin),
        "otsu_inv": clean_mask(th_inv),
    }

    edges = cv2.Canny(gr_eq, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kv, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kh, iterations=1)
    masks["edges"] = edges

    if return_debug:
        debug["gray_eq"] = gr_eq
        debug["masks"] = masks

    best_contour = None
    best_score = None
    best_mask_name = None

    for name, m in masks.items():
        c, sc = _best_contour_from_mask(m, img.shape)
        if c is not None and (best_score is None or sc > best_score):
            best_contour = c
            best_score = sc
            best_mask_name = name

    if return_debug:
        debug["best_mask_name"] = best_mask_name
        debug["best_score"] = best_score

    if best_contour is None:
        return (img, debug) if return_debug else img

    refined = _warp_min_area_rect(img, best_contour, pad_px=4)
    if refined is None or refined.size == 0:
        return (img, debug) if return_debug else img

    if return_debug:
        vis = img.copy()
        cv2.drawContours(vis, [best_contour], -1, (0,255,0), 2)
        debug["contour_overlay"] = vis
        debug["refined_shape"] = refined.shape

    return (refined, debug) if return_debug else refined
