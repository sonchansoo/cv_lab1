import os
from typing import Tuple, List

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
    get_stem,
)


# -----------------------------
# Part B — NumPy-only implementations (no OpenCV processing fns)
# Allowed: cv2.imread/cv2.imwrite in utils only for I/O
# -----------------------------
def numpy_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    logger = get_logger()
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    log_io(logger, "numpy_grayscale", img_bgr, gray, {})
    return gray


def numpy_threshold(gray: np.ndarray, threshold_value: int) -> np.ndarray:
    logger = get_logger()
    out = np.where(gray >= threshold_value, 255, 0).astype(np.uint8)
    log_io(logger, "numpy_threshold", gray, out, {"t": threshold_value})
    return out


def _create_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    y = np.arange(h)
    x = np.arange(w)
    return np.meshgrid(x, y)  # x_grid, y_grid with shape (h, w)


def numpy_resize_nearest(img: np.ndarray, fx: float, fy: float) -> np.ndarray:
    logger = get_logger()
    in_h, in_w = img.shape[:2]
    out_h = max(1, int(round(in_h * fy)))
    out_w = max(1, int(round(in_w * fx)))

    x_out = np.arange(out_w)
    y_out = np.arange(out_h)
    x_grid, y_grid = np.meshgrid(x_out, y_out)

    # inverse mapping
    x_src = np.clip(np.round(x_grid / fx), 0, in_w - 1).astype(np.int64)
    y_src = np.clip(np.round(y_grid / fy), 0, in_h - 1).astype(np.int64)

    if img.ndim == 2:
        out = img[y_src, x_src]
    else:
        out = img[y_src, x_src, :]
    out = out.astype(img.dtype)
    log_io(logger, "numpy_resize_nearest", img, out, {"fx": fx, "fy": fy})
    return out


def numpy_resize_bilinear(img: np.ndarray, fx: float, fy: float) -> np.ndarray:
    logger = get_logger()
    in_h, in_w = img.shape[:2]
    out_h = max(1, int(round(in_h * fy)))
    out_w = max(1, int(round(in_w * fx)))

    x_out = np.arange(out_w)
    y_out = np.arange(out_h)
    x_grid, y_grid = np.meshgrid(x_out, y_out)

    x_src = (x_grid + 0.5) / fx - 0.5
    y_src = (y_grid + 0.5) / fy - 0.5

    x0 = np.floor(x_src).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y_src).astype(np.int64)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, in_w - 1)
    x1 = np.clip(x1, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    y1 = np.clip(y1, 0, in_h - 1)

    wx = (x_src - x0).astype(np.float32)
    wy = (y_src - y0).astype(np.float32)

    def gather(im: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        if im.ndim == 2:
            return im[yy, xx]
        return im[yy, xx, :]

    Ia = gather(img, y0, x0)
    Ib = gather(img, y0, x1)
    Ic = gather(img, y1, x0)
    Id = gather(img, y1, x1)

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    if img.ndim == 2:
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    else:
        wa = wa[..., None]
        wb = wb[..., None]
        wc = wc[..., None]
        wd = wd[..., None]
        out = wa * Ia.astype(np.float32) + wb * Ib.astype(np.float32) + wc * Ic.astype(np.float32) + wd * Id.astype(np.float32)

    out = np.clip(np.round(out), 0, 255).astype(img.dtype)
    log_io(logger, "numpy_resize_bilinear", img, out, {"fx": fx, "fy": fy})
    return out


def numpy_flip(img: np.ndarray, axis: int) -> np.ndarray:
    logger = get_logger()
    if axis == 0:
        out = img[::-1, ...]
    elif axis == 1:
        out = img[:, ::-1, ...]
    elif axis == -1:
        out = img[::-1, ::-1, ...]
    else:
        raise ValueError("axis must be 0, 1, or -1")
    log_io(logger, "numpy_flip", img, out, {"axis": axis})
    return out


def numpy_rotate_multiples(img: np.ndarray, deg: int) -> np.ndarray:
    logger = get_logger()
    d = deg % 360
    if d == 0:
        out = img.copy()
    elif d == 90:
        out = np.flipud(np.transpose(img, (1, 0, 2)) if img.ndim == 3 else img.T)
    elif d == 180:
        out = img[::-1, ::-1, ...]
    elif d == 270:
        out = np.transpose(np.flipud(img), (1, 0, 2)) if img.ndim == 3 else np.flipud(img).T
    else:
        raise ValueError("deg must be multiple of 90")
    log_io(logger, "numpy_rotate_multiples", img, out, {"deg": deg})
    return out


def _rotation_matrix(cx: float, cy: float, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    M = np.array([[cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy],
                  [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx]], dtype=np.float32)
    return M


def _rotation_bounds(w: int, h: int, angle_deg: float) -> Tuple[int, int, float, float]:
    angle = np.deg2rad(angle_deg)
    cos_a = abs(np.cos(angle))
    sin_a = abs(np.sin(angle))
    new_w = int(np.ceil(w * cos_a + h * sin_a))
    new_h = int(np.ceil(w * sin_a + h * cos_a))
    tx = (new_w - w) / 2.0
    ty = (new_h - h) / 2.0
    return new_w, new_h, tx, ty


def _warp_affine_numpy(img: np.ndarray, M: np.ndarray, out_w: int, out_h: int, method: str = "nearest") -> np.ndarray:
    # inverse mapping: for each dst (x, y) find src (u, v)
    y_out = np.arange(out_h)
    x_out = np.arange(out_w)
    x_grid, y_grid = np.meshgrid(x_out, y_out)
    ones = np.ones_like(x_grid, dtype=np.float32)
    dst = np.stack([x_grid.astype(np.float32), y_grid.astype(np.float32), ones], axis=-1)  # (H, W, 3)
    Minv = np.zeros((3, 3), dtype=np.float32)
    Minv[:2, :] = M
    Minv[2, 2] = 1.0
    Minv = np.linalg.inv(Minv)
    src = dst @ Minv.T  # (H, W, 3)
    u = src[..., 0]
    v = src[..., 1]

    in_h, in_w = img.shape[:2]
    if method == "nearest":
        xs = np.clip(np.round(u), 0, in_w - 1).astype(np.int64)
        ys = np.clip(np.round(v), 0, in_h - 1).astype(np.int64)
        if img.ndim == 2:
            out = img[ys, xs]
        else:
            out = img[ys, xs, :]
        return out.astype(img.dtype)

    # bilinear
    x0 = np.floor(u).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(v).astype(np.int64)
    y1 = y0 + 1
    x0 = np.clip(x0, 0, in_w - 1)
    x1 = np.clip(x1, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    y1 = np.clip(y1, 0, in_h - 1)
    wx = (u - x0).astype(np.float32)
    wy = (v - y0).astype(np.float32)

    def gather(im: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        if im.ndim == 2:
            return im[yy, xx]
        return im[yy, xx, :]

    Ia = gather(img, y0, x0).astype(np.float32)
    Ib = gather(img, y0, x1).astype(np.float32)
    Ic = gather(img, y1, x0).astype(np.float32)
    Id = gather(img, y1, x1).astype(np.float32)

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    if img.ndim == 2:
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    else:
        wa = wa[..., None]
        wb = wb[..., None]
        wc = wc[..., None]
        wd = wd[..., None]
        out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    out = np.clip(np.round(out), 0, 255).astype(img.dtype)
    return out


def numpy_rotate(img: np.ndarray, angle_deg: float, expand: bool = False, method: str = "nearest") -> np.ndarray:
    logger = get_logger()
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = _rotation_matrix(cx, cy, angle_deg)  # forward mapping matrix
    if not expand:
        out = _warp_affine_numpy(img, M, w, h, method=method)
        log_io(logger, "numpy_rotate_fixed", img, out, {"deg": angle_deg, "expand": expand, "method": method})
        return out
    new_w, new_h, tx, ty = _rotation_bounds(w, h, angle_deg)
    # translate to center of expanded canvas
    M2 = M.copy()
    M2[0, 2] += tx
    M2[1, 2] += ty
    out = _warp_affine_numpy(img, M2, new_w, new_h, method=method)
    log_io(logger, "numpy_rotate_expand", img, out, {"deg": angle_deg, "expand": expand, "method": method})
    return out


def numpy_hist_gray(gray: np.ndarray) -> np.ndarray:
    logger = get_logger()
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.int64)
    log_io(logger, "numpy_hist_gray", gray, None, {})
    return hist


def numpy_hist_bgr(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = get_logger()
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]
    hb = np.bincount(b.flatten(), minlength=256).astype(np.int64)
    hg = np.bincount(g.flatten(), minlength=256).astype(np.int64)
    hr = np.bincount(r.flatten(), minlength=256).astype(np.int64)
    log_io(logger, "numpy_hist_bgr", img_bgr, None, {})
    return hb, hg, hr


def run_part_b(input_dir: str, out_partB: str, out_figures: str) -> None:
    logger = get_logger()
    img_paths = list_input_images(input_dir)
    logger.info(f"Part B(NumPy) 처리 대상 이미지: {len(img_paths)}장")
    for path in img_paths:
        stem = get_stem(path)
        img = read_image_bgr(path)

        # Grayscale
        gray = numpy_grayscale(img)
        save_image(os.path.join(out_partB, f"{stem}__gray_np.png"), gray)
        side = stack_side_by_side([img, ensure_color_bgr(gray)])
        save_image(os.path.join(out_partB, f"{stem}__gray_side_np.png"), side)

        # Thresholds
        for t in [64, 128, 200]:
            bin_img = numpy_threshold(gray, t)
            save_image(os.path.join(out_partB, f"{stem}__th_{t}_np.png"), bin_img)

        # Resize integer scales and arbitrary
        for fx, fy in [(2.0, 2.0), (3.0, 3.0), (1.5, 1.5), (0.75, 0.75)]:
            rn = numpy_resize_nearest(img, fx, fy)
            rb = numpy_resize_bilinear(img, fx, fy)
            save_image(os.path.join(out_partB, f"{stem}__resize_{fx}x{fy}_nearest_np.png"), rn)
            save_image(os.path.join(out_partB, f"{stem}__resize_{fx}x{fy}_bilinear_np.png"), rb)

        # Flip
        for axis in [0, 1, -1]:
            f = numpy_flip(img, axis)
            save_image(os.path.join(out_partB, f"{stem}__flip_{axis}_np.png"), f)

        # Rotations
        for d in [90, 180, 270]:
            rr = numpy_rotate_multiples(img, d)
            save_image(os.path.join(out_partB, f"{stem}__rot{d}_np.png"), rr)
        for deg in [33.0, -30.0]:
            rf_n = numpy_rotate(img, deg, expand=False, method="nearest")
            re_n = numpy_rotate(img, deg, expand=True, method="nearest")
            rf_b = numpy_rotate(img, deg, expand=False, method="bilinear")
            re_b = numpy_rotate(img, deg, expand=True, method="bilinear")
            save_image(os.path.join(out_partB, f"{stem}__rot{int(deg)}_fixed_nearest_np.png"), rf_n)
            save_image(os.path.join(out_partB, f"{stem}__rot{int(deg)}_expand_nearest_np.png"), re_n)
            save_image(os.path.join(out_partB, f"{stem}__rot{int(deg)}_fixed_bilinear_np.png"), rf_b)
            save_image(os.path.join(out_partB, f"{stem}__rot{int(deg)}_expand_bilinear_np.png"), re_b)

        # Histograms
        h_gray = numpy_hist_gray(numpy_grayscale(img))
        save_histogram_plot_gray(h_gray, os.path.join(out_figures, f"{stem}__hist_gray_np.png"), f"Histogram Gray (NumPy) - {stem}")
        hb, hg, hr = numpy_hist_bgr(img)
        save_histogram_plot_bgr(hb, hg, hr, os.path.join(out_figures, f"{stem}__hist_bgr_np.png"), f"Histogram BGR (NumPy) - {stem}")


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(base, "input")
    out_partB = os.path.join(base, "output", "partB_numpy")
    out_figures = os.path.join(base, "output", "figures")
    os.makedirs(out_partB, exist_ok=True)
    os.makedirs(out_figures, exist_ok=True)
    run_part_b(input_dir, out_partB, out_figures)


