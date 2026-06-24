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

import os

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


def _bool(p: dict, k: str, d: bool = False) -> bool:
    v = p.get(k, d)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


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


# =========================================================================== #
#  Filtering / effects                                                        #
# =========================================================================== #
def op_bilateral_filter(img, p):
    return cv2.bilateralFilter(
        _to_bgr(img),
        _int(p, "d", 9),
        _num(p, "sigmaColor", 75),
        _num(p, "sigmaSpace", 75),
    )


def op_box_filter(img, p):
    k = _odd(_num(p, "ksize", 5))
    return cv2.boxFilter(img, -1, (k, k), normalize=_bool(p, "normalize", True))


def op_custom_kernel_convolution(img, p):
    """3x3 convolution with a user-editable kernel (params k00..k22)."""
    vals = [_num(p, f"k{r}{c}", 1.0 if (r == 1 and c == 1) else 0.0) for r in range(3) for c in range(3)]
    kernel = np.array(vals, dtype=np.float32).reshape(3, 3)
    if _bool(p, "normalize", False):
        s = float(kernel.sum())
        if abs(s) > 1e-6:
            kernel = kernel / s
    return cv2.filter2D(img, -1, kernel)


def op_sharpen(img, p):
    a = _num(p, "amount", 1.0)
    kernel = np.array([[0, -a, 0], [-a, 1 + 4 * a, -a], [0, -a, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def op_emboss(img, p):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    dst = cv2.filter2D(_to_bgr(img).astype(np.float32), cv2.CV_32F, kernel)
    return np.clip(dst + _num(p, "offset", 128), 0, 255).astype(np.uint8)


def op_unsharp_mask(img, p):
    k = _odd(_num(p, "ksize", 5))
    blurred = cv2.GaussianBlur(img, (k, k), _num(p, "sigma", 1.0))
    a = _num(p, "amount", 1.0)
    return cv2.addWeighted(img, 1 + a, blurred, -a, 0)


def op_pencil_sketch(img, p):
    gray, color = cv2.pencilSketch(
        _to_bgr(img),
        sigma_s=_num(p, "sigma_s", 60),
        sigma_r=_num(p, "sigma_r", 0.07),
        shade_factor=_num(p, "shade_factor", 0.02),
    )
    return color if _str(p, "mode", "color") == "color" else gray


def op_cartoonize(img, p):
    bgr = _to_bgr(img)
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        _odd(_num(p, "blockSize", 9), 3), _num(p, "C", 2),
    )
    color = cv2.bilateralFilter(bgr, _int(p, "d", 9), 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


def op_stylization(img, p):
    return cv2.stylization(
        _to_bgr(img), sigma_s=_num(p, "sigma_s", 60), sigma_r=_num(p, "sigma_r", 0.45)
    )


def op_inpaint_telea(img, p):
    """Inpaint bright marks/specular spots: build a mask by thresholding bright
    pixels, then reconstruct with the Telea algorithm (single-image demo)."""
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, _num(p, "thresh", 240), 255, cv2.THRESH_BINARY)
    d = _int(p, "dilate", 2)
    if d > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=d)
    return cv2.inpaint(bgr, mask, _num(p, "radius", 3), cv2.INPAINT_TELEA)


# =========================================================================== #
#  Edges / gradients                                                          #
# =========================================================================== #
def op_scharr(img, p):
    gray = _to_gray(img)
    d = _str(p, "direction", "both")
    if d == "x":
        return cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_16S, 1, 0))
    if d == "y":
        return cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_16S, 0, 1))
    gx = cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_16S, 1, 0))
    gy = cv2.convertScaleAbs(cv2.Scharr(gray, cv2.CV_16S, 0, 1))
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)


def op_gradient_magnitude(img, p):
    gray = _to_gray(img).astype(np.float32)
    k = _odd(_num(p, "ksize", 3))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def op_distance_transform(img, p):
    gray = _to_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_size = _int(p, "maskSize", 5)
    if mask_size not in (3, 5):
        mask_size = 5
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, mask_size)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap = _str(p, "colormap", "none")
    if cmap != "none":
        return cv2.applyColorMap(dist, getattr(cv2, f"COLORMAP_{cmap}", cv2.COLORMAP_JET))
    return dist


# =========================================================================== #
#  Threshold / segmentation                                                   #
# =========================================================================== #
def op_threshold_triangle(img, p):
    gray = _to_gray(img)
    _, dst = cv2.threshold(
        gray, 0, _num(p, "maxval", 255), cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    return dst


def op_hsv_in_range(img, p):
    bgr = _to_bgr(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([_int(p, "hLow", 0), _int(p, "sLow", 0), _int(p, "vLow", 0)])
    hi = np.array([_int(p, "hHigh", 179), _int(p, "sHigh", 255), _int(p, "vHigh", 255)])
    mask = cv2.inRange(hsv, lo, hi)
    if _str(p, "output", "mask") == "mask":
        return mask
    return cv2.bitwise_and(bgr, bgr, mask=mask)


def op_kmeans_color_quantize(img, p):
    bgr = _to_bgr(img)
    k = max(2, _int(p, "k", 8))
    Z = bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(bgr.shape)


def op_watershed(img, p):
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, _num(p, "fgRatio", 0.5) * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr, markers)
    out = bgr.copy()
    out[markers == -1] = (0, 0, 255)  # boundaries in red, BGR
    return out


def op_grabcut_rect(img, p):
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]
    m = min(0.45, max(0.0, _num(p, "margin", 0.1)))
    rect = (int(w * m), int(h * m), int(w * (1 - 2 * m)), int(h * (1 - 2 * m)))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, _int(p, "iters", 3), cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    return bgr * fg[:, :, None]


# =========================================================================== #
#  Features / detection                                                       #
# =========================================================================== #
def op_harris_corners(img, p):
    bgr = _to_bgr(img)
    gray = np.float32(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    dst = cv2.cornerHarris(gray, _int(p, "blockSize", 2), _odd(_num(p, "ksize", 3)), _num(p, "k", 0.04))
    dst = cv2.dilate(dst, None)
    out = bgr.copy()
    out[dst > _num(p, "thresh", 0.01) * dst.max()] = (0, 0, 255)
    return out


def op_shi_tomasi(img, p):
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, _int(p, "maxCorners", 100), _num(p, "quality", 0.01), _num(p, "minDist", 10)
    )
    out = bgr.copy()
    if corners is not None:
        for c in corners.astype(int):
            x, y = c.ravel()
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
    return out


def op_fast_corners(img, p):
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(
        threshold=_int(p, "threshold", 25), nonmaxSuppression=_bool(p, "nonmax", True)
    )
    kps = fast.detect(gray, None)
    return cv2.drawKeypoints(bgr, kps, None, color=(255, 0, 0))


def op_orb_keypoints(img, p):
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=_int(p, "nfeatures", 500))
    kps = orb.detect(gray, None)
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if _bool(p, "rich", True) else 0
    return cv2.drawKeypoints(bgr, kps, None, color=(0, 255, 0), flags=flags)


def op_sift_keypoints(img, p):
    """SIFT keypoints (requires the opencv-contrib build)."""
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=_int(p, "nfeatures", 0))
    kps = sift.detect(gray, None)
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if _bool(p, "rich", True) else 0
    return cv2.drawKeypoints(bgr, kps, None, color=(0, 255, 0), flags=flags)


def op_hough_lines(img, p):
    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, _num(p, "t1", 50), _num(p, "t2", 150))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, _int(p, "threshold", 80),
        minLineLength=_num(p, "minLineLength", 30), maxLineGap=_num(p, "maxLineGap", 10),
    )
    out = bgr.copy()
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return out


def op_hough_circles(img, p):
    bgr = _to_bgr(img)
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, _num(p, "dp", 1.2), _num(p, "minDist", 30),
        param1=_num(p, "param1", 100), param2=_num(p, "param2", 40),
        minRadius=_int(p, "minRadius", 0), maxRadius=_int(p, "maxRadius", 0),
    )
    out = bgr.copy()
    if circles is not None:
        for c in np.uint16(np.around(circles))[0]:
            cv2.circle(out, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(out, (c[0], c[1]), 2, (0, 0, 255), 3)
    return out


# =========================================================================== #
#  Color / photo                                                              #
# =========================================================================== #
def op_gamma_correction(img, p):
    g = max(0.01, _num(p, "gamma", 1.0))
    inv = 1.0 / g
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(_to_bgr(img), table)


def op_apply_colormap(img, p):
    gray = _to_gray(img)
    cmap = getattr(cv2, f"COLORMAP_{_str(p, 'cmap', 'JET')}", cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, cmap)


def op_clahe(img, p):
    g = _int(p, "grid", 8)
    clahe = cv2.createCLAHE(clipLimit=_num(p, "clip", 2.0), tileGridSize=(g, g))
    if _str(p, "mode", "color") == "gray":
        return clahe.apply(_to_gray(img))
    lab = cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def op_channel_split(img, p):
    bgr = _to_bgr(img)
    idx = {"B": 0, "G": 1, "R": 2}.get(_str(p, "channel", "R"), 2)
    return bgr[:, :, idx]


def op_sepia(img, p):
    bgr = _to_bgr(img).astype(np.float32)
    # sepia matrix expressed for OpenCV's BGR channel ordering
    kernel = np.array(
        [[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]]
    )
    return np.clip(cv2.transform(bgr, kernel), 0, 255).astype(np.uint8)


def op_white_balance(img, p):
    """Simple gray-world white balance."""
    bgr = _to_bgr(img).astype(np.float32)
    avg = bgr.reshape(-1, 3).mean(axis=0)
    scale = float(avg.mean()) / (avg + 1e-6)
    return np.clip(bgr * scale, 0, 255).astype(np.uint8)


# =========================================================================== #
#  Geometry / transform                                                       #
# =========================================================================== #
def op_perspective_warp(img, p):
    """4-point perspective skew demo (trapezoid warp)."""
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]
    a = min(0.45, max(0.0, _num(p, "amount", 0.25)))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * a, 0], [w * (1 - a), 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (w, h))


def op_affine_rotate_scale(img, p):
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), _num(p, "angle", 30), _num(p, "scale", 1.0))
    return cv2.warpAffine(bgr, M, (w, h))


def op_pyramid_down(img, p):
    out = img
    for _ in range(max(1, _int(p, "levels", 1))):
        out = cv2.pyrDown(out)
    return out


def op_pyramid_up(img, p):
    out = img
    for _ in range(max(1, _int(p, "levels", 1))):
        out = cv2.pyrUp(out)
    return out


# =========================================================================== #
#  Frequency                                                                  #
# =========================================================================== #
def op_fourier_magnitude(img, p):
    gray = _to_gray(img).astype(np.float32)
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
    mag = np.log1p(mag)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# =========================================================================== #
#  Misc / detect                                                              #
# =========================================================================== #
def op_qr_detect(img, p):
    bgr = _to_bgr(img)
    det = cv2.QRCodeDetector()
    data, pts, _ = det.detectAndDecode(bgr)
    out = bgr.copy()
    if pts is not None:
        poly = pts.astype(int).reshape(-1, 2)
        cv2.polylines(out, [poly], True, (0, 255, 0), 3)
        if data:
            cv2.putText(
                out, data[:48], (10, max(20, out.shape[0] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )
    return out


# --- super-resolution (ESPCN x2) --------------------------------------------
# The ESPCN_x2.pb model is tiny (~100 KB). To keep the free-tier deploy safe we
# do NOT download at startup: the model is fetched lazily to /tmp on first use
# and cached. If the contrib dnn_superres module or the download is unavailable,
# we fall back to a plain bicubic 2x upscale so the op never hard-fails.
_SR_MODEL = None
_ESPCN_X2_URL = "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb"


def _espcn_x2():
    global _SR_MODEL
    if _SR_MODEL is not None:
        return _SR_MODEL
    try:
        import urllib.request

        path = "/tmp/ESPCN_x2.pb"
        if not os.path.exists(path):
            urllib.request.urlretrieve(_ESPCN_X2_URL, path)
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(path)
        sr.setModel("espcn", 2)
        _SR_MODEL = sr
    except Exception:  # noqa: BLE001 — any failure -> bicubic fallback
        _SR_MODEL = False
    return _SR_MODEL


def op_super_resolution(img, p):
    bgr = _to_bgr(img)
    sr = _espcn_x2()
    if sr:
        try:
            return sr.upsample(bgr)
        except Exception:  # noqa: BLE001
            pass
    return cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


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
    # --- Filtering / effects ---
    "bilateral_filter": op_bilateral_filter,
    "box_filter": op_box_filter,
    "custom_kernel_convolution": op_custom_kernel_convolution,
    "sharpen": op_sharpen,
    "emboss": op_emboss,
    "unsharp_mask": op_unsharp_mask,
    "pencil_sketch": op_pencil_sketch,
    "cartoonize": op_cartoonize,
    "stylization": op_stylization,
    "inpaint_telea": op_inpaint_telea,
    # --- Edges / gradients ---
    "scharr": op_scharr,
    "gradient_magnitude": op_gradient_magnitude,
    "distance_transform": op_distance_transform,
    # --- Threshold / segmentation ---
    "threshold_triangle": op_threshold_triangle,
    "hsv_in_range": op_hsv_in_range,
    "kmeans_color_quantize": op_kmeans_color_quantize,
    "watershed": op_watershed,
    "grabcut_rect": op_grabcut_rect,
    # --- Features / detection ---
    "harris_corners": op_harris_corners,
    "shi_tomasi": op_shi_tomasi,
    "fast_corners": op_fast_corners,
    "orb_keypoints": op_orb_keypoints,
    "sift_keypoints": op_sift_keypoints,
    "hough_lines": op_hough_lines,
    "hough_circles": op_hough_circles,
    # --- Color / photo ---
    "gamma_correction": op_gamma_correction,
    "apply_colormap": op_apply_colormap,
    "clahe": op_clahe,
    "channel_split": op_channel_split,
    "sepia": op_sepia,
    "white_balance": op_white_balance,
    # --- Geometry / transform ---
    "perspective_warp": op_perspective_warp,
    "affine_rotate_scale": op_affine_rotate_scale,
    "pyramid_down": op_pyramid_down,
    "pyramid_up": op_pyramid_up,
    # --- Frequency ---
    "fourier_magnitude": op_fourier_magnitude,
    # --- Misc / detect ---
    "qr_detect": op_qr_detect,
    "super_resolution": op_super_resolution,
}
