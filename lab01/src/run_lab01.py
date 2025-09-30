import os
import shutil
from typing import List, Tuple

import cv2
import numpy as np

from lab01_utils import (
    get_logger,
    read_image_bgr,
    save_image,
    list_input_images,
    get_stem,
    diff_image,
    compute_mse,
    compute_psnr,
    write_csv,
    save_histogram_plot_gray,
    save_histogram_plot_bgr,
)
from lab01_opencv import (
    opencv_grayscale,
    opencv_threshold,
    opencv_resize,
    opencv_flip,
    opencv_rotate,
    opencv_rotate_multiples,
    opencv_hist_gray,
    opencv_hist_bgr,
)
from lab01_numpy import (
    numpy_grayscale,
    numpy_threshold,
    numpy_resize_nearest,
    numpy_resize_bilinear,
    numpy_flip,
    numpy_rotate,
    numpy_rotate_multiples,
    numpy_hist_gray,
    numpy_hist_bgr,
)


def compare_and_save(a: np.ndarray, b: np.ndarray, diff_out_path: str) -> Tuple[float, float]:
    d = diff_image(a, b, amplify=4)
    save_image(diff_out_path, d)
    mse = compute_mse(a, b)
    psnr = compute_psnr(a, b)
    return mse, psnr


def run_pipeline() -> None:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(base, "input")
    out_a = os.path.join(base, "output", "partA")
    out_b = os.path.join(base, "output", "partB_numpy")
    out_cmp = os.path.join(base, "output", "comparisons")
    out_fig = os.path.join(base, "output", "figures")
    # Part C (personal images) outputs
    out_c_root = os.path.join(base, "output", "partC")
    out_c_a = os.path.join(out_c_root, "partA")
    out_c_b = os.path.join(out_c_root, "partB_numpy")
    out_c_cmp = os.path.join(out_c_root, "comparisons")
    
    os.makedirs(out_a, exist_ok=True)
    os.makedirs(out_b, exist_ok=True)
    os.makedirs(out_cmp, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)
   
    # Part C dirs
    os.makedirs(out_c_a, exist_ok=True)
    os.makedirs(out_c_b, exist_ok=True)
    os.makedirs(out_c_cmp, exist_ok=True)

    logger = get_logger()
    img_paths = list_input_images(input_dir)
    logger.info(f"전체 처리 이미지: {len(img_paths)}장")

    rows: List[List[object]] = []

    for path in img_paths:
        stem = get_stem(path)
        img = read_image_bgr(path)
        # route outputs depending on whether image is from personal folder
        is_personal = (os.sep + "personal" + os.sep) in path
        oa = out_c_a if is_personal else out_a
        ob = out_c_b if is_personal else out_b
        ocmp = out_c_cmp if is_personal else out_cmp

        # Grayscale
        g_a = opencv_grayscale(img)
        g_b = numpy_grayscale(img)
        save_image(os.path.join(oa, f"{stem}__gray.png"), g_a)
        save_image(os.path.join(ob, f"{stem}__gray_np.png"), g_b)
        mse, psnr = compare_and_save(g_a, g_b, os.path.join(ocmp, f"{stem}__gray_diff.png"))
        rows.append([stem, "grayscale", mse, psnr, bool(np.allclose(g_a, g_b, atol=1))])

        # Thresholds
        for t in [64, 128, 200]:
            th_a = opencv_threshold(g_a, t)
            th_b = numpy_threshold(g_b, t)
            save_image(os.path.join(oa, f"{stem}__th_{t}.png"), th_a)
            save_image(os.path.join(ob, f"{stem}__th_{t}_np.png"), th_b)
            mse, psnr = compare_and_save(th_a, th_b, os.path.join(ocmp, f"{stem}__th_{t}_diff.png"))
            rows.append([stem, f"threshold_{t}", mse, psnr, bool(np.allclose(th_a, th_b))])

        # Resize integer and arbitrary scales
        for fx, fy in [(2.0, 2.0), (3.0, 3.0), (1.5, 1.5), (0.75, 0.75)]:
            a_nn = opencv_resize(img, fx, fy, "nearest")
            b_nn = numpy_resize_nearest(img, fx, fy)
            save_image(os.path.join(oa, f"{stem}__resize_{fx}x{fy}_nearest.png"), a_nn)
            save_image(os.path.join(ob, f"{stem}__resize_{fx}x{fy}_nearest_np.png"), b_nn)
            mse, psnr = compare_and_save(a_nn, b_nn, os.path.join(ocmp, f"{stem}__resize_{fx}x{fy}_nearest_diff.png"))
            rows.append([stem, f"resize_{fx}x{fy}_nearest", mse, psnr, bool(np.allclose(a_nn, b_nn))])

            a_li = opencv_resize(img, fx, fy, "linear")
            b_bl = numpy_resize_bilinear(img, fx, fy)
            save_image(os.path.join(oa, f"{stem}__resize_{fx}x{fy}_linear.png"), a_li)
            save_image(os.path.join(ob, f"{stem}__resize_{fx}x{fy}_bilinear_np.png"), b_bl)
            mse, psnr = compare_and_save(a_li, b_bl, os.path.join(ocmp, f"{stem}__resize_{fx}x{fy}_linear_vs_bilinear_diff.png"))
            rows.append([stem, f"resize_{fx}x{fy}_linear_vs_bilinear", mse, psnr, bool(np.allclose(a_li, b_bl, atol=1))])

            if fx < 1.0 or fy < 1.0:
                a_ar = opencv_resize(img, fx, fy, "area")
                # No numpy AREA counterpart; skip compare
                save_image(os.path.join(oa, f"{stem}__resize_{fx}x{fy}_area.png"), a_ar)

        # Flip
        for axis in [0, 1, -1]:
            code = axis
            f_a = opencv_flip(img, code)
            f_b = numpy_flip(img, axis)
            save_image(os.path.join(oa, f"{stem}__flip_{code}.png"), f_a)
            save_image(os.path.join(ob, f"{stem}__flip_{axis}_np.png"), f_b)
            mse, psnr = compare_and_save(f_a, f_b, os.path.join(ocmp, f"{stem}__flip_{axis}_diff.png"))
            rows.append([stem, f"flip_{axis}", mse, psnr, bool(np.allclose(f_a, f_b))])

        # Rotations multiples
        for d in [90, 180, 270]:
            r_a = opencv_rotate_multiples(img, d)
            r_b = numpy_rotate_multiples(img, d)
            save_image(os.path.join(oa, f"{stem}__rot{d}.png"), r_a)
            save_image(os.path.join(ob, f"{stem}__rot{d}_np.png"), r_b)
            mse, psnr = compare_and_save(r_a, r_b, os.path.join(ocmp, f"{stem}__rot{d}_diff.png"))
            rows.append([stem, f"rot{d}", mse, psnr, bool(np.allclose(r_a, r_b))])

        # Arbitrary rotations (compare same interpolation)
        for deg in [33.0, -30.0]:
            for expand in [False, True]:
                a_lin = opencv_rotate(img, deg, expand=expand, method="linear")
                b_bil = numpy_rotate(img, deg, expand=expand, method="bilinear")
                save_image(os.path.join(oa, f"{stem}__rot{int(deg)}_{'expand' if expand else 'fixed'}_linear.png"), a_lin)
                save_image(os.path.join(ob, f"{stem}__rot{int(deg)}_{'expand' if expand else 'fixed'}_bilinear_np.png"), b_bil)
                # Compare linear vs bilinear (they should be close, not identical)
                # Shapes may differ if expand True; ensure match
                if a_lin.shape == b_bil.shape:
                    mse, psnr = compare_and_save(a_lin, b_bil, os.path.join(ocmp, f"{stem}__rot{int(deg)}_{'expand' if expand else 'fixed'}_diff.png"))
                    rows.append([stem, f"rot{int(deg)}_{'expand' if expand else 'fixed'}", mse, psnr, bool(np.allclose(a_lin, b_bil, atol=1))])

        # Histograms (save figures)
        g_for_hist = g_a  # gray
        hga = opencv_hist_gray(g_for_hist)
        save_histogram_plot_gray(hga, os.path.join(out_fig, f"{stem}__hist_gray.png"), f"Histogram Gray - {stem}")
        hb, hg, hr = opencv_hist_bgr(img)
        save_histogram_plot_bgr(hb, hg, hr, os.path.join(out_fig, f"{stem}__hist_bgr.png"), f"Histogram BGR - {stem}")

    # Write summary CSV
    write_csv(os.path.join(out_cmp, "metrics_summary.csv"), ["image", "operation", "mse", "psnr", "allclose"], rows)
    write_csv(os.path.join(out_c_cmp, "metrics_summary.csv"), ["image", "operation", "mse", "psnr", "allclose"], rows)


if __name__ == "__main__":
    run_pipeline()


