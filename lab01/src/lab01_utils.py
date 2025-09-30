import os
import csv
import math
import logging
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt


# -----------------------------
# Logging
# -----------------------------
_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("lab01")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers in interactive runs
    if not logger.handlers:
        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "run.log"))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    _LOGGER = logger
    return logger


def log_io(logger: logging.Logger, title: str, img_in: Optional[np.ndarray], img_out: Optional[np.ndarray], extra: Dict[str, object]) -> None:
    parts: List[str] = [f"{title}"]
    if img_in is not None:
        parts.append(f"in_shape={tuple(img_in.shape)} in_dtype={img_in.dtype}")
    if img_out is not None:
        parts.append(f"out_shape={tuple(img_out.shape)} out_dtype={img_out.dtype}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(" | ".join(parts))


# -----------------------------
# I/O helpers
# -----------------------------
def read_image_bgr(path: str) -> np.ndarray:
    logger = get_logger()
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    log_io(logger, "read_image", None, img, {"path": os.path.abspath(path)})
    return img


def save_image(path: str, img: np.ndarray) -> None:
    logger = get_logger()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f"이미지를 저장할 수 없습니다: {path}")
    log_io(logger, "save_image", img, None, {"path": os.path.abspath(path)})


def ensure_color_bgr(img: np.ndarray) -> np.ndarray:
    # Avoid cv2.cvtColor to keep Part B free of OpenCV processing fns
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    return img


def stack_side_by_side(images: List[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("이미지 리스트가 비어 있습니다")
    normed: List[np.ndarray] = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        normed.append(im)
    # Height alignment by padding if necessary
    heights = [im.shape[0] for im in normed]
    max_h = max(heights)
    padded: List[np.ndarray] = []
    for im in normed:
        if im.shape[0] == max_h:
            padded.append(im)
        else:
            pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=im.dtype)
            padded.append(np.vstack([im, pad]))
    return np.hstack(padded)


# -----------------------------
# Metrics
# -----------------------------
def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"형상이 다릅니다: {a.shape} vs {b.shape}")
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    diff = a_f - b_f
    return float(np.mean(diff * diff))


def compute_psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    mse = compute_mse(a, b)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def diff_image(a: np.ndarray, b: np.ndarray, amplify: int = 1) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError(f"형상이 다릅니다: {a.shape} vs {b.shape}")
    da = a.astype(np.int16)
    db = b.astype(np.int16)
    d = np.abs(da - db).astype(np.uint8)
    if amplify > 1:
        d = np.clip(d.astype(np.int32) * amplify, 0, 255).astype(np.uint8)
    return d


def write_csv(path: str, headers: List[str], rows: List[List[object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)


# -----------------------------
# Plot helpers
# -----------------------------
def save_histogram_plot_gray(hist: np.ndarray, out_path: str, title: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    plt.plot(hist, color="k")
    plt.xlim([0, 255])
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_histogram_plot_bgr(h_b: np.ndarray, h_g: np.ndarray, h_r: np.ndarray, out_path: str, title: str) -> None:
    fig = plt.figure(figsize=(6, 4))
    plt.plot(h_b, color="b", label="B")
    plt.plot(h_g, color="g", label="G")
    plt.plot(h_r, color="r", label="R")
    plt.legend(loc="upper right")
    plt.xlim([0, 255])
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Dataset helpers
# -----------------------------
def list_input_images(base_dir: str) -> List[str]:
    paths: List[str] = []
    for sub in ["samples", "personal"]:
        dir_path = os.path.join(base_dir, sub)
        if not os.path.isdir(dir_path):
            continue
        for name in sorted(os.listdir(dir_path)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                paths.append(os.path.join(dir_path, name))
    return paths


def make_output_path(base_out_dir: str, stem: str, suffix: str) -> str:
    filename = f"{stem}__{suffix}.png"
    return os.path.join(base_out_dir, filename)


def get_stem(path: str) -> str:
    name = os.path.basename(path)
    stem, _ = os.path.splitext(name)
    return stem



