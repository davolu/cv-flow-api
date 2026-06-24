# CV-Flow API

Real **OpenCV (`cv2`)** execution backend for the [CV-Flow](https://github.com/davolu/cv-flow)
no-code computer-vision builder. It replaces the browser-side OpenCV.js engine:
the frontend POSTs a source image plus an ordered pipeline of operations, and this
FastAPI service runs each op with **real OpenCV in Python** and returns a preview
image for every node (so the canvas shows a live thumbnail per step) plus the final
output.

Built with **FastAPI + opencv-python-headless + NumPy**. Deploy target: **Render**.

---

## Endpoints

### `GET /health`
```json
{ "status": "ok", "service": "cv-flow-api", "opencv": "4.10.0", "ops": 18 }
```

### `POST /run`
Request body:
```json
{
  "image": "data:image/png;base64,iVBOR...",   // data URL or raw base64 (PNG/JPEG)
  "pipeline": [
    { "id": "source",        "op": "source",        "params": {} },
    { "id": "grayscale-1",   "op": "grayscale",     "params": {} },
    { "id": "canny-2",       "op": "canny",         "params": { "t1": 50, "t2": 150 } }
  ]
}
```
The `pipeline` is applied **in order**; each step's input is the previous step's
output (the `source` step is fed the decoded image).

Response:
```json
{
  "nodes":   { "source": "data:image/png;base64,...", "grayscale-1": "data:image/png;base64,...", "canny-2": "data:image/png;base64,..." },
  "final":   "data:image/png;base64,...",
  "dims":    { "source": [720, 540], "grayscale-1": [720, 540] },
  "errors":  {},
  "timings": { "grayscale-1": 1.2, "canny-2": 3.8 }
}
```
- `nodes[id]` is the base64 PNG preview for that node (`null` if that op errored).
- `errors[id]` carries a clear per-op message — a bad op/param never 500s the whole
  request; only that node is marked failed and its downstream steps report
  "No input from previous step".

Interactive docs: **`/docs`** (Swagger UI).

---

## Operations (18)

| op | description |
|----|-------------|
| `source` | pass-through (the decoded input image) |
| `grayscale` | BGR → grayscale |
| `gaussian_blur` | Gaussian smoothing (`ksize`, `sigma`) |
| `median_blur` | median filter (`ksize`) |
| `canny` | Canny edges (`t1`, `t2`) |
| `threshold` | global threshold incl. Otsu (`ttype`, `thresh`, `maxval`) |
| `adaptive_threshold` | per-region threshold (`method`, `blockSize`, `C`, `maxval`) |
| `resize` | scale or exact dimensions (`mode`, `scale`/`width`/`height`, `interp`) |
| `rotate_flip` | 90° rotate steps + mirror (`rotate`, `flip`) |
| `cvt_color` | color-space convert HSV/LAB/YCrCb, shown as false color (`space`) |
| `brightness_contrast` | linear `alpha`·in + `beta` |
| `morphology` | erode/dilate/open/close/gradient/tophat/blackhat (`op`, `shape`, `ksize`, `iterations`) |
| `contours` | threshold → find & draw external contours (`thresh`, `thickness`, `minArea`) |
| `hist_equalize` | histogram equalization, grayscale or color/YCrCb (`mode`) |
| `sobel` | Sobel gradient edges (`direction`, `ksize`) |
| `laplacian` | Laplacian edges (`ksize`) |
| `invert` | bitwise NOT |
| `face_detect` | Haar-cascade frontal-face boxes (`scaleFactor`, `minNeighbors`, `minSize`) |

Ops live in [`app/ops.py`](app/ops.py) as a simple `OPS` registry mapping
`op name -> function(img, params)`. Adding an op is a one-line append. (The table
above lists the original classic-CV ops; ~45 more classic ops — filters, edges,
features, geometry, etc. — are also registered. The deep-learning ops are below.)

---

## AI / Deep-Learning ops (CPU, `cv2.dnn`)

These run **real model inference with no PyTorch / ultralytics** — everything goes
through OpenCV's DNN module (`cv2.dnn` with ONNX / Caffe / Darknet / Torch models)
plus `cv2.dnn_superres`. They live in [`app/dnn_ops.py`](app/dnn_ops.py) and are
merged into the same `OPS` registry.

**How models are handled.** Nothing heavy is committed. Each op **lazily downloads
its model on first use** into a cache dir (`CV_FLOW_MODELS_DIR`, default `./models`,
falling back to `/tmp`), keeps the loaded net in memory, and reuses both afterward.
If a download or load fails the op returns a **clear error message** (surfaced as
that node's `errors[id]`) — it never crashes the request. **CPU inference is slow
(seconds per image); the first call to each op also pays a one-time model download.**

| op | model | source | size | params |
|----|-------|--------|------|--------|
| `object_detection` | YOLOv4-tiny (Darknet, COCO-80) | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) cfg + `darknet_yolo_v4_pre` weights | ~24 MB | `conf`, `nms` |
| `face_detection_dnn` | res10 SSD (Caffe) | [opencv/opencv_3rdparty](https://github.com/opencv/opencv_3rdparty) + opencv `deploy.prototxt` | ~10 MB | `conf` |
| `image_classification` | SqueezeNet 1.1 (ONNX, ImageNet-1000) | [onnx/models](https://github.com/onnx/models) zoo | ~5 MB | `topk` |
| `semantic_segmentation` | PPHumanSeg (ONNX, person/bg) | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | ~6 MB | `mode`, `alpha` |
| `style_transfer` | fast-neural-style (Torch `.t7`) | [Johnson fast-neural-style](https://cs.stanford.edu/people/jcjohns/fast-neural-style/) | ~15 MB / style | `style`, `maxSide` |
| `super_resolution_dnn` | ESPCN / FSRCNN (`dnn_superres` `.pb`) | [TF-ESPCN](https://github.com/fannymonori/TF-ESPCN) / [FSRCNN_Tensorflow](https://github.com/Saafke/FSRCNN_Tensorflow) | <0.1 MB | `model`, `scale` (×2/×3/×4) |
| `ocr_text` | Tesseract (via `pytesseract`) | system `tesseract-ocr` binary | n/a | `minConf` |

> ⚠️ **YOLOv5/v8 ONNX exports don't load in OpenCV 4.10** (unsupported `Expand`/
> `Slice` ONNX ops), which is why detection uses **YOLOv4-tiny (Darknet)** and
> segmentation uses **PPHumanSeg** from the OpenCV Zoo — both load natively in
> `cv2.dnn`. All seven ops were verified end-to-end on CPU.

### Memory / Render tier

Measured peak RSS (single op, one model loaded; includes the cv2 baseline ~100 MB):

| op | peak RSS | fits 512 MB free tier? |
|----|----------|------------------------|
| `ocr_text` | ~60 MB | ✅ |
| `image_classification` | ~95 MB | ✅ |
| `style_transfer` | ~125 MB | ✅ (slow: ~30–60 s/image) |
| `face_detection_dnn` | ~125 MB | ✅ |
| `semantic_segmentation` | ~135 MB | ✅ |
| `object_detection` | ~290 MB | ✅ (single op) |
| `super_resolution_dnn` | **~800 MB – 1.2 GB on large inputs** | ⚠️ see below |

- **Lighter ops fit the free 512 MB tier individually.** `dnn_superres` runs the
  network at full **output** resolution, so RAM grows with output pixels — a 1 MP
  input at ×3 can peak well over 1 GB. The op **pre-shrinks inputs** so projected
  output stays under `CV_FLOW_SR_MAX_OUTPUT_PX` (**default 1.0 M px**, ~300–350 MB
  peak — safe on free tier). For true large-image super-resolution, raise that env
  var **and** move to a **paid Starter / 2 GB** plan (the uncapped op peaked
  ~0.8–1.2 GB in testing and will OOM the 512 MB free tier).
- **Chaining several DL ops in one pipeline loads several models at once** and adds
  up — a multi-model pipeline (e.g. detection → segmentation → style) can exceed
  512 MB even though each op fits alone. For heavy use, prefer **Render Starter
  (2 GB)** or larger.
- Models cache to disk after first download, so only RAM (not re-download) is the
  recurring cost.

### `/models` and the model cache

Model files are downloaded at runtime to **`CV_FLOW_MODELS_DIR`** (default
`./models`, pre-created and writable in the Docker image; `/tmp` fallback). The
directory is **git-ignored** — weights are never committed. To pre-warm, just call
each op once after deploy.

---

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# health check:
curl http://localhost:8000/health
```

(Or with Docker: `docker build -t cv-flow-api . && docker run -p 8000:8000 cv-flow-api`.)

---

## Deploy to Render

This repo ships a **`render.yaml` Blueprint** and a **`Dockerfile`** — either works.

### Recommended: Blueprint (uses `render.yaml`)
1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. In the [Render dashboard](https://dashboard.render.com): **New + → Blueprint**.
3. **Connect** this repository. Render detects and reads `render.yaml`.
4. Confirm — it provisions a **Web Service** named `cv-flow-api`:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Health check: `/health`
5. **Create / Deploy**. First build takes a few minutes.

### Alternative: manual Web Service
**New + → Web Service → connect repo →** Runtime **Python 3**, Build
`pip install -r requirements.txt`, Start
`uvicorn main:app --host 0.0.0.0 --port $PORT`. (Or choose **Docker** to use the
included `Dockerfile`.)

### Environment variables
| var | required? | purpose |
|-----|-----------|---------|
| `PORT` | auto | provided by Render; the start command reads it |
| `PYTHON_VERSION` | set by blueprint (`3.11.9`) | pins the runtime |
| `FRONTEND_ORIGIN` | optional | lock CORS to your frontend origin (e.g. `https://cv-flow.vercel.app`); comma-separate multiple. Defaults to `*` |

**No required env vars** beyond what the blueprint sets — it deploys as-is.

### Resulting base URL
Render gives the service a URL like:

```
https://cv-flow-api.onrender.com
```

Set that (no trailing slash) as **`NEXT_PUBLIC_API_URL`** in the CV-Flow frontend
(Vercel → Project → Settings → Environment Variables), then redeploy the frontend.
The frontend calls `${NEXT_PUBLIC_API_URL}/run`.

> Note: Render's free tier sleeps on inactivity, so the first request after idle
> can take ~30–60s to wake the service.
