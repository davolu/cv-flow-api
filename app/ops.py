"""
CV-Flow — OpenCV operation registry (real cv2).

Each operation is a plain function `op(img, params) -> ndarray` where `img` is a
BGR (3-channel) or single-channel uint8 numpy array and `params` is the dict sent
from the frontend block. Every function returns a NEW ndarray and never mutates
its input.

The OPS dict at the bottom maps an op name (matching the frontend block `type`)
to its function, so adding a new op is a one-line append — nothing else changes.

────────────────────────────────────────────────────────────────────────────
PHASE 2 NOTE
Phase 2 adds deep-learning ops here (object detection / instance & semantic
segmentation / pose estimation). They keep the SAME signature
`op(img, params) -> ndarray`; their bodies will run a model (ultralytics / torch /
onnxruntime) and draw results onto the image, then register in OPS exactly like
the classic cv2 ops below.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  param helpers                                                              #
# --------------------------------------------------------------------------- #
def _num(p: dict, k: str, d: float = 0) -> float:
    v = p.get(k, d)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(d)


def _int(p: dict, k: str, d: int = 0) -> int:
    return int(round(_num(p, k, d)))


def _str(p: dict, k: str, d: str = "") -> str:
    v = p.get(k, d)
    return v if isinstance(v, str) else d


def _odd(v: float, min_v: int = 1) -> int:
    """Nearest odd integer >= min_v (kernel sizes must be odd)."""
    n = int(round(v))
    if n % 2 == 0:
        n += 1
    return max(min_v, n)


# --------------------------------------------------------------------------- #
#  channel coercion (mirrors the frontend mat-utils helpers)                  #
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# --------------------------------------------------------------------------- #
#  Haar cascade (loaded once, shared) — bundled with opencv-python-headless   #
# --------------------------------------------------------------------------- #
_FACE_CASCADE: "cv2.CascadeClassifier | None" = None


def _face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


# --------------------------------------------------------------------------- #
#  operations                                                                 #
# --------------------------------------------------------------------------- #
def op_source(img, p):
    # The source op is a pass-through; the engine feeds it the decoded image.
    return img.copy()


def op_grayscale(img, p):
    return cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2GRAY)


def op_gaussian_blur(img, p):
    k = _odd(_num(p, "ksize", 5))
    return cv2.GaussianBlur(img, (k, k), _num(p, "sigma", 0))


def op_median_blur(img, p):
    return cv2.medianBlur(img, _odd(_num(p, "ksize", 5), 3))


def op_canny(img, p):
    gray = _to_gray(img)
    return cv2.Canny(gray, _num(p, "t1", 50), _num(p, "t2", 150))


def op_threshold(img, p):
    gray = _to_gray(img)
    t = _str(p, "ttype", "BINARY")
    maxval = _num(p, "maxval", 255)
    if t == "OTSU":
        _, dst = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        flag = getattr(cv2, f"THRESH_{t}", cv2.THRESH_BINARY)
        _, dst = cv2.threshold(gray, _num(p, "thresh", 127), maxval, flag)
    return dst


def op_adaptive_threshold(img, p):
    gray = _to_gray(img)
    method = (
        cv2.ADAPTIVE_THRESH_MEAN_C
        if _str(p, "method", "GAUSSIAN") == "MEAN"
        else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    return cv2.adaptiveThreshold(
        gray,
        _num(p, "maxval", 255),
        method,
        cv2.THRESH_BINARY,
        _odd(_num(p, "blockSize", 11), 3),
        _num(p, "C", 2),
    )


def op_resize(img, p):
    interp = getattr(cv2, _str(p, "interp", "INTER_LINEAR"), cv2.INTER_LINEAR)
    if _str(p, "mode", "scale") == "scale":
        s = _num(p, "scale", 0.5)
        return cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    return cv2.resize(
        img, (_int(p, "width", 320), _int(p, "height", 240)), interpolation=interp
    )


def op_rotate_flip(img, p):
    out = img.copy()
    rot = _str(p, "rotate", "0")
    if rot == "90":
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rot == "180":
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot == "270":
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flip = _str(p, "flip", "none")
    if flip == "h":
        out = cv2.flip(out, 1)
    elif flip == "v":
        out = cv2.flip(out, 0)
    elif flip == "both":
        out = cv2.flip(out, -1)
    return out


def op_cvt_color(img, p):
    """Convert color space and present the result as a viewable (false-color) BGR image."""
    bgr = _to_bgr(img)
    space = _str(p, "space", "HSV")
    code = {
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "YCrCb": cv2.COLOR_BGR2YCrCb,
    }.get(space, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(bgr, code)


def op_brightness_contrast(img, p):
    return cv2.convertScaleAbs(img, alpha=_num(p, "alpha", 1), beta=_num(p, "beta", 0))


def op_morphology(img, p):
    k = _odd(_num(p, "ksize", 5))
    shape = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }.get(_str(p, "shape", "rect"), cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (k, k))
    it = _int(p, "iterations", 1)
    op = _str(p, "op", "erode")
    if op == "erode":
        return cv2.erode(img, kernel, iterations=it)
    if op == "dilate":
        return cv2.dilate(img, kernel, iterations=it)
    mop = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }.get(op, cv2.MORPH_OPEN)
    return cv2.morphologyEx(img, mop, kernel, iterations=it)


def op_contours(img, p):
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, _num(p, "thresh", 127), 255, cv2.THRESH_BINARY)
    contours, _h = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    out = _to_bgr(img).copy()
    min_area = _num(p, "minArea", 50)
    keep = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(out, keep, -1, (94, 63, 244), _int(p, "thickness", 2))  # rose, BGR
    return out


def op_hist_equalize(img, p):
    if _str(p, "mode", "gray") == "gray":
        return cv2.equalizeHist(_to_gray(img))
    ycc = cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def op_sobel(img, p):
    gray = _to_gray(img)
    k = _odd(_num(p, "ksize", 3))
    d = _str(p, "direction", "both")
    if d == "x":
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k))
    if d == "y":
        return cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k))
    gx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k))
    gy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k))
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)


def op_laplacian(img, p):
    gray = _to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=_odd(_num(p, "ksize", 3)))
    return cv2.convertScaleAbs(lap)


def op_invert(img, p):
    return cv2.bitwise_not(img)


def op_face_detect(img, p):
    out = _to_bgr(img).copy()
    cascade = _face_cascade()
    if cascade.empty():
        return out  # cascade unavailable — return image unchanged
    gray = _to_gray(img)
    ms = _int(p, "minSize", 40)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=_num(p, "scaleFactor", 1.1),
        minNeighbors=_int(p, "minNeighbors", 4),
        minSize=(ms, ms),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), (153, 72, 236), 3)  # pink, BGR
    return out


# --------------------------------------------------------------------------- #
#  registry: op name -> function                                              #
# --------------------------------------------------------------------------- #
OPS = {
    "source": op_source,
    "grayscale": op_grayscale,
    "gaussian_blur": op_gaussian_blur,
    "median_blur": op_median_blur,
    "canny": op_canny,
    "threshold": op_threshold,
    "adaptive_threshold": op_adaptive_threshold,
    "resize": op_resize,
    "rotate_flip": op_rotate_flip,
    "cvt_color": op_cvt_color,
    "brightness_contrast": op_brightness_contrast,
    "morphology": op_morphology,
    "contours": op_contours,
    "hist_equalize": op_hist_equalize,
    "sobel": op_sobel,
    "laplacian": op_laplacian,
    "invert": op_invert,
    "face_detect": op_face_detect,
}
