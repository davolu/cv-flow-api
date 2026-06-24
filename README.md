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
`op name -> function(img, params)`. Adding an op is a one-line append.

> **Phase 2** adds deep-learning ops here (object detection, instance/semantic
> segmentation, pose estimation) with the *same* signature — see the note in
> `app/ops.py`.

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
