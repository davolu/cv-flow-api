"""
CV-Flow — Deep-Learning operation registry (CPU, OpenCV DNN module).

These ops run real model inference WITHOUT PyTorch / ultralytics — everything
goes through OpenCV's `cv2.dnn` (ONNX / Caffe / Torch models) plus
`cv2.dnn_superres`, so the only heavy dependency stays `opencv-contrib-python`.

Design rules (shared by every op here):
  • Models are NOT bundled. Each op lazily downloads its model file(s) on first
    use into a local cache dir (``models/`` next to the repo, over/ridable via
    the ``CV_FLOW_MODELS_DIR`` env var; falls back to ``/tmp`` if read-only) and
    keeps the loaded ``cv2.dnn_Net`` in memory for subsequent calls.
  • If a download or load fails the op raises a CLEAR ``ModelError`` describing
    what went wrong. The /run engine turns that into a per-node error message —
    it never crashes the whole request.
  • Models are kept SMALL (nano/mobile variants) so the lighter ops fit Render's
    free 512 MB tier. The heavier ones (object detection, segmentation, style)
    are flagged in the README as possibly needing a paid tier.

Each op keeps the same signature as the classic cv2 ops: ``op(img, params) -> ndarray``.
CPU inference is slow (often seconds per image) — that is expected.
"""

from __future__ import annotations

import os
import ssl
import threading
import urllib.request

import cv2
import numpy as np


def _ssl_context() -> "ssl.SSLContext | None":
    """A verifying SSL context backed by certifi's CA bundle when available.

    Linux/Docker images ship system CA certs, but some Python builds (notably
    python.org macOS) don't — using certifi keeps downloads working everywhere
    while staying fully certificate-verified. Returns None to fall back to
    urllib's default handling if certifi isn't installed.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
#  small param helpers (kept local so this module has no import cycle on ops)  #
# --------------------------------------------------------------------------- #
def _num(p: dict, k: str, d: float = 0) -> float:
    try:
        return float(p.get(k, d))
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


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# --------------------------------------------------------------------------- #
#  model cache + lazy download                                                #
# --------------------------------------------------------------------------- #
class ModelError(RuntimeError):
    """Raised when a model cannot be downloaded or loaded — surfaced per-node."""


_DOWNLOAD_LOCK = threading.Lock()
_NET_CACHE: dict = {}
_LABEL_CACHE: dict = {}


def _models_dir() -> str:
    """First writable dir among the configured cache dir, repo ./models, /tmp."""
    candidates = [
        os.getenv("CV_FLOW_MODELS_DIR"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
        "/tmp/cv-flow-models",
    ]
    for d in candidates:
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            test = os.path.join(d, ".write-test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            return d
        except OSError:
            continue
    return "/tmp"


def _download(url: str, filename: str, min_bytes: int = 1024) -> str:
    """Download ``url`` to the cache as ``filename`` (cached). Returns the path.

    Raises ModelError on any network/IO failure or an implausibly small file
    (LFS-pointer / error-page guard).
    """
    path = os.path.join(_models_dir(), filename)
    if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
        return path
    with _DOWNLOAD_LOCK:
        if os.path.exists(path) and os.path.getsize(path) >= min_bytes:
            return path
        tmp = path + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cv-flow/1.0"})
            ctx = _ssl_context()
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    out.write(chunk)
            size = os.path.getsize(tmp)
            if size < min_bytes:
                raise ModelError(
                    f"Downloaded model '{filename}' looks truncated ({size} bytes) — "
                    f"the host may have moved it. URL: {url}"
                )
            os.replace(tmp, path)
        except ModelError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        except Exception as exc:  # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            raise ModelError(
                f"Could not download model '{filename}'. CPU model downloads can be "
                f"slow on first use; retry, or check connectivity. ({exc})"
            ) from exc
    return path


def _labels(url: str, filename: str) -> list:
    if filename in _LABEL_CACHE:
        return _LABEL_CACHE[filename]
    path = _download(url, filename, min_bytes=64)
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    _LABEL_CACHE[filename] = names
    return names


# --------------------------------------------------------------------------- #
#  drawing helpers                                                            #
# --------------------------------------------------------------------------- #
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_label(img, text, x, y, color=(40, 200, 60), scale=0.5, thick=1):
    """Draw text with a filled background box for legibility (BGR)."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), _FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)


def _palette(n: int) -> np.ndarray:
    """Deterministic distinct BGR colors for class indices."""
    rng = np.arange(n, dtype=np.uint8).reshape(-1, 1)
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = (rng[:, 0] * 179 // max(1, n)).astype(np.uint8)
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)


# =========================================================================== #
#  1. object detection — YOLOv4-tiny (Darknet) via cv2.dnn                    #
# --------------------------------------------------------------------------- #
#  YOLOv4-tiny loads natively in cv2.dnn (readNetFromDarknet) and covers the   #
#  full COCO-80 set — unlike most YOLOv5/v8 ONNX exports, which use ops        #
#  (Expand/Slice) that OpenCV 4.10's ONNX importer can't parse.               #
# =========================================================================== #
_YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
_YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
_COCO_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"


def _yolo_net():
    if "yolo" not in _NET_CACHE:
        cfg = _download(_YOLO_CFG_URL, "yolov4-tiny.cfg", min_bytes=1000)
        weights = _download(_YOLO_WEIGHTS_URL, "yolov4-tiny.weights", min_bytes=10_000_000)
        try:
            _NET_CACHE["yolo"] = cv2.dnn.readNetFromDarknet(cfg, weights)
        except cv2.error as exc:
            raise ModelError(f"Failed to load YOLOv4-tiny: {exc}") from exc
    return _NET_CACHE["yolo"]


def op_object_detection(img, p):
    """YOLOv4-tiny (COCO-80) detection drawn as boxes + labels + scores."""
    bgr = _to_bgr(img)
    net = _yolo_net()
    names = _labels(_COCO_URL, "coco.names")
    conf_th = _num(p, "conf", 0.4)
    nms_th = _num(p, "nms", 0.45)

    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confs, class_ids = [], [], []
    for out in outs:
        for row in out:
            scores = row[5:]
            cid = int(np.argmax(scores))
            score = float(scores[cid])
            if score < conf_th:
                continue
            cx, cy, bw, bh = row[0] * w, row[1] * h, row[2] * w, row[3] * h
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            confs.append(score)
            class_ids.append(cid)

    out_img = bgr.copy()
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_th, nms_th) if boxes else []
    colors = _palette(len(names))
    count = 0
    for i in np.array(idxs).flatten():
        x, y, bw, bh = boxes[i]
        cid = class_ids[i]
        color = tuple(int(c) for c in colors[cid % len(colors)])
        cv2.rectangle(out_img, (x, y), (x + bw, y + bh), color, 2)
        name = names[cid] if cid < len(names) else str(cid)
        _draw_label(out_img, f"{name} {confs[i]:.0%}", x, y, color)
        count += 1
    _draw_label(out_img, f"{count} objects @ conf>={conf_th:.2f}", 8, 22, (60, 60, 60), 0.5)
    return out_img


# =========================================================================== #
#  2. face detection — OpenCV res10 SSD (Caffe) via cv2.dnn                   #
# =========================================================================== #
_FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def _face_net():
    if "face_dnn" not in _NET_CACHE:
        proto = _download(_FACE_PROTO_URL, "res10_deploy.prototxt", min_bytes=2000)
        model = _download(_FACE_MODEL_URL, "res10_ssd.caffemodel", min_bytes=5_000_000)
        try:
            _NET_CACHE["face_dnn"] = cv2.dnn.readNetFromCaffe(proto, model)
        except cv2.error as exc:
            raise ModelError(f"Failed to load res10 SSD face model: {exc}") from exc
    return _NET_CACHE["face_dnn"]


def op_face_detection_dnn(img, p):
    """res10 SSD face detector (more robust than Haar) — draws boxes + scores."""
    bgr = _to_bgr(img)
    net = _face_net()
    conf_th = _num(p, "conf", 0.5)
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    det = net.forward()  # (1,1,N,7)
    out = bgr.copy()
    count = 0
    for i in range(det.shape[2]):
        score = float(det[0, 0, i, 2])
        if score < conf_th:
            continue
        x1 = int(det[0, 0, i, 3] * w)
        y1 = int(det[0, 0, i, 4] * h)
        x2 = int(det[0, 0, i, 5] * w)
        y2 = int(det[0, 0, i, 6] * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (153, 72, 236), 2)
        _draw_label(out, f"face {score:.0%}", x1, y1, (153, 72, 236))
        count += 1
    _draw_label(out, f"{count} face(s)", 8, 22, (60, 60, 60), 0.5)
    return out


# =========================================================================== #
#  3. image classification — SqueezeNet 1.1 ONNX (ImageNet-1000)             #
# =========================================================================== #
_SQUEEZE_URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
_IMAGENET_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dnn/classification_classes_ILSVRC2012.txt"


def _squeeze_net():
    if "squeeze" not in _NET_CACHE:
        path = _download(_SQUEEZE_URL, "squeezenet1.1-7.onnx", min_bytes=1_000_000)
        try:
            _NET_CACHE["squeeze"] = cv2.dnn.readNetFromONNX(path)
        except cv2.error as exc:
            raise ModelError(f"Failed to load SqueezeNet ONNX: {exc}") from exc
    return _NET_CACHE["squeeze"]


def op_image_classification(img, p):
    """SqueezeNet1.1 top-k ImageNet labels overlaid on the image."""
    bgr = _to_bgr(img)
    net = _squeeze_net()
    names = _labels(_IMAGENET_URL, "imagenet_classes.txt")
    topk = max(1, _int(p, "topk", 5))

    blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (224, 224), swapRB=True, crop=True)
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    for c in range(3):
        blob[0, c] = (blob[0, c] - mean[c]) / std[c]
    net.setInput(blob)
    out = net.forward().flatten()
    out = np.exp(out - out.max())
    probs = out / out.sum()
    top = probs.argsort()[::-1][:topk]

    canvas = bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24 + 22 * len(top)), (30, 30, 30), -1)
    cv2.putText(canvas, "Top predictions:", (8, 18), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for r, idx in enumerate(top):
        name = names[idx] if idx < len(names) else f"class {idx}"
        cv2.putText(
            canvas, f"{r + 1}. {name}  {probs[idx]:.1%}", (12, 40 + r * 22),
            _FONT, 0.5, (120, 255, 160), 1, cv2.LINE_AA,
        )
    return canvas


# =========================================================================== #
#  4. semantic segmentation — PPHumanSeg (OpenCV Zoo) via cv2.dnn             #
# --------------------------------------------------------------------------- #
#  PPHumanSeg is a small (~6 MB) person/background segmenter shipped in the    #
#  OpenCV Model Zoo, so it loads cleanly in cv2.dnn (most ENet/DeepLab/FCN     #
#  exports either don't parse in OpenCV 4.10 or are far too heavy for the      #
#  512 MB free tier). Output is a 2-class (background, person) mask.           #
# =========================================================================== #
_PPHUMANSEG_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx"


def _seg_net():
    if "pphumanseg" not in _NET_CACHE:
        path = _download(_PPHUMANSEG_URL, "pphumanseg_2023mar.onnx", min_bytes=1_000_000)
        try:
            _NET_CACHE["pphumanseg"] = cv2.dnn.readNetFromONNX(path)
        except cv2.error as exc:
            raise ModelError(f"Failed to load PPHumanSeg ONNX: {exc}") from exc
    return _NET_CACHE["pphumanseg"]


def op_semantic_segmentation(img, p):
    """PPHumanSeg person/background segmentation; colorized mask or blend."""
    bgr = _to_bgr(img)
    net = _seg_net()
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (192, 192), swapRB=True, crop=False)
    blob = (blob - 0.5) / 0.5  # PPHumanSeg expects inputs scaled to [-1, 1]
    net.setInput(blob)
    out = net.forward()  # (1, 2, 192, 192) -> [bg, person]
    cls = out[0].argmax(0).astype(np.uint8)  # 0 bg, 1 person
    cls = cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)

    color = np.array([60, 220, 80], np.uint8)  # person highlight (BGR)
    mask = np.zeros_like(bgr)
    mask[cls == 1] = color

    if _str(p, "mode", "blend") == "mask":
        return mask
    alpha = min(1.0, max(0.0, _num(p, "alpha", 0.5)))
    out_img = bgr.copy()
    person = cls == 1
    out_img[person] = (
        bgr[person] * (1 - alpha) + color * alpha
    ).astype(np.uint8)
    return out_img


# =========================================================================== #
#  5. style transfer — fast-neural-style Torch (.t7) via cv2.dnn             #
# =========================================================================== #
_STYLE_BASE = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/"
_STYLES = {
    "candy": "instance_norm/candy.t7",
    "mosaic": "instance_norm/mosaic.t7",
    "udnie": "instance_norm/udnie.t7",
    "the_scream": "instance_norm/the_scream.t7",
    "feathers": "instance_norm/feathers.t7",
    "starry_night": "eccv16/starry_night.t7",
    "the_wave": "eccv16/the_wave.t7",
    "la_muse": "eccv16/la_muse.t7",
    "composition_vii": "eccv16/composition_vii.t7",
}


def _style_net(style: str):
    key = f"style:{style}"
    if key not in _NET_CACHE:
        rel = _STYLES.get(style, _STYLES["candy"])
        path = _download(_STYLE_BASE + rel, f"style_{style}.t7", min_bytes=1_000_000)
        try:
            _NET_CACHE[key] = cv2.dnn.readNetFromTorch(path)
        except cv2.error as exc:
            raise ModelError(f"Failed to load style model '{style}': {exc}") from exc
    return _NET_CACHE[key]


def op_style_transfer(img, p):
    """fast-neural-style (Johnson et al.) artistic stylization, CPU."""
    bgr = _to_bgr(img)
    style = _str(p, "style", "candy")
    if style not in _STYLES:
        style = "candy"
    net = _style_net(style)

    # Cap the working resolution so CPU stays within time/memory budget.
    h, w = bgr.shape[:2]
    max_side = _int(p, "maxSide", 512)
    scale = min(1.0, max_side / max(h, w))
    work = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else bgr
    wh, ww = work.shape[:2]

    blob = cv2.dnn.blobFromImage(
        work, 1.0, (ww, wh), (103.939, 116.779, 123.68), swapRB=False, crop=False
    )
    net.setInput(blob)
    out = net.forward().reshape(3, wh, ww)
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out = out.clip(0, 255).transpose(1, 2, 0).astype(np.uint8)
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


# =========================================================================== #
#  6. super-resolution — cv2.dnn_superres (ESPCN / FSRCNN, x2/x3/x4)         #
# =========================================================================== #
_SR_URLS = {
    ("espcn", 2): "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
    ("espcn", 3): "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
    ("espcn", 4): "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
    ("fsrcnn", 2): "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb",
    ("fsrcnn", 3): "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
    ("fsrcnn", 4): "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb",
}


def _sr_model(model: str, scale: int):
    key = f"sr:{model}:{scale}"
    if key not in _NET_CACHE:
        url = _SR_URLS.get((model, scale))
        if url is None:
            raise ModelError(f"No super-resolution model for {model} x{scale}")
        if not hasattr(cv2, "dnn_superres"):
            raise ModelError(
                "cv2.dnn_superres is unavailable — needs opencv-contrib-python."
            )
        path = _download(url, f"{model.upper()}_x{scale}.pb", min_bytes=8_000)
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(path)
            sr.setModel(model, scale)
            _NET_CACHE[key] = sr
        except cv2.error as exc:
            raise ModelError(f"Failed to load {model} x{scale}: {exc}") from exc
    return _NET_CACHE[key]


# cv2.dnn_superres runs the network at full OUTPUT resolution, so RAM scales
# with output pixels (~200 MB per output megapixel on top of the ~120 MB cv2
# baseline). Cap projected output (env-tunable) so large inputs can't OOM
# Render's 512 MB free tier — the input is pre-shrunk to fit the budget.
# Default ~1.0 Mpx output keeps peak RSS ~300-350 MB (safe on free tier); raise
# it (and use a paid 2 GB tier) for genuinely large super-resolution outputs.
_SR_MAX_OUTPUT_PX = int(os.getenv("CV_FLOW_SR_MAX_OUTPUT_PX", str(1_000_000)))


def op_super_resolution_dnn(img, p):
    """Learned upscaling via ESPCN/FSRCNN (light, CPU-friendly)."""
    bgr = _to_bgr(img)
    model = _str(p, "model", "espcn").lower()
    if model not in ("espcn", "fsrcnn"):
        model = "espcn"
    scale = _int(p, "scale", 2)
    if scale not in (2, 3, 4):
        scale = 2

    # Pre-shrink the input if the upscaled output would blow the pixel budget.
    h, w = bgr.shape[:2]
    out_px = (h * scale) * (w * scale)
    if out_px > _SR_MAX_OUTPUT_PX:
        shrink = (_SR_MAX_OUTPUT_PX / out_px) ** 0.5
        bgr = cv2.resize(bgr, (max(1, int(w * shrink)), max(1, int(h * shrink))))

    sr = _sr_model(model, scale)
    return sr.upsample(bgr)


# =========================================================================== #
#  7. OCR — Tesseract via pytesseract (text boxes + recognized text)         #
# =========================================================================== #
def op_ocr_text(img, p):
    """Detect + recognize text with Tesseract; draw word boxes and a summary."""
    try:
        import pytesseract  # noqa: WPS433 (optional, only present with the tesseract binary)
    except ImportError as exc:
        raise ModelError(
            "pytesseract is not installed. Add 'pytesseract' to requirements and "
            "the 'tesseract-ocr' apt package (see Dockerfile)."
        ) from exc

    bgr = _to_bgr(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    except Exception as exc:  # noqa: BLE001 — tesseract binary missing/broken
        raise ModelError(
            f"Tesseract OCR failed — is the 'tesseract-ocr' binary installed? ({exc})"
        ) from exc

    out = bgr.copy()
    min_conf = _num(p, "minConf", 40)
    words = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not txt or conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        cv2.rectangle(out, (x, y), (x + w, y + h), (40, 200, 60), 2)
        words.append(txt)
    summary = " ".join(words) if words else "(no text found)"
    _draw_label(out, summary[:60], 8, out.shape[0] - 8, (40, 200, 60), 0.5)
    return out


# --------------------------------------------------------------------------- #
#  registry merged into the main OPS dict by app/ops.py                       #
# --------------------------------------------------------------------------- #
DNN_OPS = {
    "object_detection": op_object_detection,
    "face_detection_dnn": op_face_detection_dnn,
    "image_classification": op_image_classification,
    "semantic_segmentation": op_semantic_segmentation,
    "style_transfer": op_style_transfer,
    "super_resolution_dnn": op_super_resolution_dnn,
    "ocr_text": op_ocr_text,
}
