import os
from typing import Tuple, List

import cv2
import numpy as np

from lab01_utils import (
    get_logger,
    log_io,
    read_image_bgr,
    save_image,
    ensure_color_bgr,
    stack_side_by_side,
    save_histogram_plot_gray,
    save_histogram_plot_bgr,
    list_input_images,
    make_output_path,
    get_stem,
)


# -----------------------------
# Part A — OpenCV implementations
# -----------------------------
def opencv_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    logger = get_logger()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    log_io(logger, "opencv_grayscale", img_bgr, gray, {})
    return gray


def opencv_threshold(gray: np.ndarray, threshold_value: int) -> np.ndarray:
    logger = get_logger()
    _, bin_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    log_io(logger, "opencv_threshold", gray, bin_img, {"t": threshold_value})
    return bin_img


def opencv_resize(img: np.ndarray, fx: float, fy: float, method: str) -> np.ndarray:
    logger = get_logger()
    if method == "nearest":
        interp = cv2.INTER_NEAREST
    elif method == "linear":
        interp = cv2.INTER_LINEAR
    elif method == "area":
        interp = cv2.INTER_AREA
    else:
        raise ValueError(f"unknown method: {method}")
    out = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=interp)
    log_io(logger, "opencv_resize", img, out, {"fx": fx, "fy": fy, "method": method})
    return out


def opencv_flip(img: np.ndarray, code: int) -> np.ndarray:
    logger = get_logger()
    out = cv2.flip(img, code)
    log_io(logger, "opencv_flip", img, out, {"code": code})
    return out


def _rotation_bounds(w: int, h: int, angle_deg: float) -> Tuple[int, int, float, float]:
    # Compute new width/height for expanded canvas and translation
    angle = np.deg2rad(angle_deg)
    cos_a = abs(np.cos(angle))
    sin_a = abs(np.sin(angle))
    new_w = int(np.ceil(w * cos_a + h * sin_a))
    new_h = int(np.ceil(w * sin_a + h * cos_a))
    tx = (new_w - w) / 2.0
    ty = (new_h - h) / 2.0
    return new_w, new_h, tx, ty


def opencv_rotate(img: np.ndarray, angle_deg: float, expand: bool = False, method: str = "linear") -> np.ndarray:
    logger = get_logger()
    (h, w) = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    if method == "nearest":
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_LINEAR

    if not expand:
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        log_io(logger, "opencv_rotate_fixed", img, out, {"deg": angle_deg, "expand": expand, "method": method})
        return out

    new_w, new_h, tx, ty = _rotation_bounds(w, h, angle_deg)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # Shift to the center of the expanded canvas
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(img, M, (new_w, new_h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    log_io(logger, "opencv_rotate_expand", img, out, {"deg": angle_deg, "expand": expand, "method": method})
    return out


def opencv_rotate_multiples(img: np.ndarray, deg: int) -> np.ndarray:
    logger = get_logger()
    d = deg % 360
    if d == 90:
        out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif d == 180:
        out = cv2.rotate(img, cv2.ROTATE_180)
    elif d == 270:
        out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif d == 0:
        out = img.copy()
    else:
        raise ValueError("deg must be a multiple of 90 for opencv_rotate_multiples")
    log_io(logger, "opencv_rotate_multiples", img, out, {"deg": deg})
    return out


def opencv_hist_gray(gray: np.ndarray) -> np.ndarray:
    logger = get_logger()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    log_io(logger, "opencv_hist_gray", gray, None, {})
    return hist


def opencv_hist_bgr(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = get_logger()
    chans = cv2.split(img_bgr)
    colors = ("b", "g", "r")
    hists: List[np.ndarray] = []
    for chan, c in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256]).flatten()
        hists.append(hist)
    log_io(logger, "opencv_hist_bgr", img_bgr, None, {})
    return hists[0], hists[1], hists[2]


def run_part_a(input_dir: str, out_partA: str, out_figures: str) -> None:
    logger = get_logger()
    img_paths = list_input_images(input_dir)
    logger.info(f"Part A(OpenCV) 처리 대상 이미지: {len(img_paths)}장")
    for path in img_paths:
        stem = get_stem(path)
        img = read_image_bgr(path)

        # Grayscale
        gray = opencv_grayscale(img)
        save_image(os.path.join(out_partA, f"{stem}__gray.png"), gray)
        side = stack_side_by_side([img, ensure_color_bgr(gray)])
        save_image(os.path.join(out_partA, f"{stem}__gray_side.png"), side)

        # Thresholds
        for t in [64, 128, 200]:
            bin_img = opencv_threshold(gray, t)
            save_image(os.path.join(out_partA, f"{stem}__th_{t}.png"), bin_img)

        # Resize integer scales and arbitrary
        for fx, fy in [(2.0, 2.0), (3.0, 3.0), (1.5, 1.5), (0.75, 0.75)]:
            for method in (["nearest", "linear"] + (["area"] if fx < 1.0 or fy < 1.0 else [])):
                r = opencv_resize(img, fx, fy, method)
                save_image(os.path.join(out_partA, f"{stem}__resize_{fx}x{fy}_{method}.png"), r)

        # Flip
        for code in [0, 1, -1]:
            f = opencv_flip(img, code)
            save_image(os.path.join(out_partA, f"{stem}__flip_{code}.png"), f)

        # Rotations
        for d in [90, 180, 270]:
            rr = opencv_rotate_multiples(img, d)
            save_image(os.path.join(out_partA, f"{stem}__rot{d}.png"), rr)
        for deg in [33.0, -30.0]:
            rf = opencv_rotate(img, deg, expand=False, method="linear")
            re = opencv_rotate(img, deg, expand=True, method="linear")
            save_image(os.path.join(out_partA, f"{stem}__rot{int(deg)}_fixed.png"), rf)
            save_image(os.path.join(out_partA, f"{stem}__rot{int(deg)}_expand.png"), re)

        # Histograms
        h_gray = opencv_hist_gray(gray)
        save_histogram_plot_gray(h_gray, os.path.join(out_figures, f"{stem}__hist_gray.png"), f"Histogram Gray - {stem}")
        hb, hg, hr = opencv_hist_bgr(img)
        save_histogram_plot_bgr(hb, hg, hr, os.path.join(out_figures, f"{stem}__hist_bgr.png"), f"Histogram BGR - {stem}")


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(base, "input")
    out_partA = os.path.join(base, "output", "partA")
    out_figures = os.path.join(base, "output", "figures")
    os.makedirs(out_partA, exist_ok=True)
    os.makedirs(out_figures, exist_ok=True)
    run_part_a(input_dir, out_partA, out_figures)


