import cv2
import numpy as np


def _order_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _warp_min_area_rect(image, contour, pad_px=4):
    """warp minAreaRect into a straight rectangle"""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect

    if rw < 2 or rh < 2:
        return None

    box = cv2.boxPoints(rect)
    box = _order_points_clockwise(box)

    out_w = max(1, int(round(rw)))
    out_h = max(1, int(round(rh)))

    dst = np.array([
        [0, 0], [out_w - 1, 0],
        [out_w - 1, out_h - 1], [0, out_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))

    if warped.shape[1] > warped.shape[0]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    # trim noisy border pixels
    if pad_px > 0 and warped.shape[0] > 2*pad_px and warped.shape[1] > 2*pad_px:
        warped = warped[pad_px:-pad_px, pad_px:-pad_px].copy()

    return warped


def _score_strip_contour(contour, img_shape):
    """score how strip-like a contour is"""
    h, w = img_shape[:2]
    img_area = float(h * w)

    area = cv2.contourArea(contour)
    if area < 0.01 * img_area:
        return None

    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect

    if rw < 2 or rh < 2:
        return None

    long_side = max(rw, rh)
    short_side = max(1.0, min(rw, rh))
    aspect = long_side / short_side

    if aspect < 2.0:
        return None

    rect_area = max(1.0, rw * rh)
    fill_ratio = area / rect_area

    center_dist = np.hypot(cx - w/2.0, cy - h/2.0)
    center_dist_norm = center_dist / max(1.0, np.hypot(w/2.0, h/2.0))

    area_ratio = area / img_area

    score = (
        2.2 * area_ratio
        + 0.8 * min(aspect, 12.0) / 12.0
        + 0.9 * np.clip(fill_ratio, 0.0, 1.2)
        - 0.8 * center_dist_norm
    )
    return score

def _best_contour_from_mask(mask, img_shape):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = None
    for c in contours:
        score = _score_strip_contour(c, img_shape)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best = c
            best_score = score
    return best, best_score


def refine_strip_edges(strip_img_bgr, return_debug=False):
    """tighten the crop around the strip and warp it straight"""
    debug = {}

    if strip_img_bgr is None or strip_img_bgr.size == 0:
        return (strip_img_bgr, debug) if return_debug else strip_img_bgr

    img = strip_img_bgr.copy()

    if img.shape[1] > img.shape[0]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    if h < 20 or w < 8:
        return (img, debug) if return_debug else img

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    _, th_bin = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 17))
    k_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def clean_mask(m):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close_v, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close_h, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
        return m

    masks = {
        "otsu_bin": clean_mask(th_bin),
        "otsu_inv": clean_mask(th_inv),
    }

    edges = cv2.Canny(gray_eq, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close_v, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_close_h, iterations=1)
    masks["edges"] = edges

    if return_debug:
        debug["gray_eq"] = gray_eq
        debug["masks"] = masks

    best_contour = None
    best_score = None
    best_mask_name = None

    for name, m in masks.items():
        c, score = _best_contour_from_mask(m, img.shape)
        if c is not None and (best_score is None or score > best_score):
            best_contour = c
            best_score = score
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
        dbg_vis = img.copy()
        cv2.drawContours(dbg_vis, [best_contour], -1, (0, 255, 0), 2)
        debug["contour_overlay"] = dbg_vis
        debug["refined_shape"] = refined.shape

    return (refined, debug) if return_debug else refined
