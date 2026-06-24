"""
CV-Flow API — FastAPI + real OpenCV (cv2) execution engine.

Replaces the browser-side OpenCV.js engine: the frontend POSTs a source image
plus an ordered pipeline of ops, and this service decodes the image into a numpy
ndarray, applies each op in order with real cv2, and returns a base64 PNG preview
for every node (so the canvas shows a thumbnail per step) plus the final output.

Endpoints:
  GET  /health  -> {"status": "ok", ...}
  POST /run     -> run a pipeline, return per-node + final previews
"""

from __future__ import annotations

import base64
import binascii
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ops import OPS

app = FastAPI(
    title="CV-Flow API",
    version="1.0.0",
    description="Real-OpenCV (cv2) execution backend for the CV-Flow no-code builder.",
)

# --------------------------------------------------------------------------- #
#  CORS — allow the frontend origin(s). FRONTEND_ORIGIN may be a single origin #
#  or a comma-separated list. Defaults to "*" for now.                         #
# --------------------------------------------------------------------------- #
_origin_env = os.getenv("FRONTEND_ORIGIN", "*").strip()
_origins = ["*"] if _origin_env in ("", "*") else [o.strip() for o in _origin_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # must stay False while allow_origins can be "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  request / response models                                                  #
# --------------------------------------------------------------------------- #
class PipelineStep(BaseModel):
    id: str
    op: str
    params: Dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    image: str  # data URL ("data:image/png;base64,...") or raw base64
    pipeline: List[PipelineStep] = Field(default_factory=list)


class RunResponse(BaseModel):
    nodes: Dict[str, Optional[str]]   # nodeId -> base64 PNG data URL (None if errored)
    final: Optional[str] = None       # base64 PNG data URL of the final step
    dims: Dict[str, List[int]] = Field(default_factory=dict)   # nodeId -> [w, h]
    errors: Dict[str, str] = Field(default_factory=dict)       # nodeId -> message
    timings: Dict[str, float] = Field(default_factory=dict)    # nodeId -> ms


# --------------------------------------------------------------------------- #
#  image (de)serialization                                                    #
# --------------------------------------------------------------------------- #
def decode_image(data: str) -> np.ndarray:
    """Decode a data URL or raw base64 string into a BGR uint8 ndarray."""
    if not data:
        raise ValueError("No image provided")
    if data.startswith("data:"):
        # strip the "data:image/...;base64," prefix
        _, _, data = data.partition(",")
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # force 3-channel BGR
    if img is None:
        raise ValueError("Could not decode image (unsupported or corrupt format)")
    return img


def encode_png(img: np.ndarray) -> str:
    """Encode a uint8 ndarray (1- or 3-channel) to a base64 PNG data URL."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode PNG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
#  routes                                                                      #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "cv-flow-api", "opencv": cv2.__version__, "ops": len(OPS)}


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "cv-flow-api", "docs": "/docs", "health": "/health", "ops": sorted(OPS)}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest) -> RunResponse:
    """
    Apply the ordered pipeline to the source image and return a preview for every
    node plus the final output. Per-op failures are reported per-node (the bad
    node gets an `errors[id]` message) and never 500 the whole request.
    """
    try:
        source = decode_image(req.image)
    except ValueError as exc:
        # A bad source image is the one thing that fails the whole request,
        # but we still return a structured 200 so the UI can show the message.
        return RunResponse(nodes={}, final=None, errors={"__source__": str(exc)})

    nodes: Dict[str, Optional[str]] = {}
    dims: Dict[str, List[int]] = {}
    errors: Dict[str, str] = {}
    timings: Dict[str, float] = {}

    current: Optional[np.ndarray] = None  # output of the previous step
    final_id: Optional[str] = None

    for step in req.pipeline:
        op_fn = OPS.get(step.op)
        try:
            if step.op == "source":
                out = source.copy()
            else:
                if op_fn is None:
                    raise ValueError(f"Unknown op: {step.op}")
                if current is None:
                    raise ValueError("No input from previous step")
                t0 = time.perf_counter()
                out = op_fn(current, step.params)
                timings[step.id] = round((time.perf_counter() - t0) * 1000, 2)

            if out is None or not isinstance(out, np.ndarray) or out.size == 0:
                raise ValueError("Operation produced no output")

            nodes[step.id] = encode_png(out)
            h, w = out.shape[:2]
            dims[step.id] = [int(w), int(h)]
            current = out
            final_id = step.id
        except Exception as exc:  # noqa: BLE001 — report any op error per-node
            nodes[step.id] = None
            errors[step.id] = str(exc) or "Processing error"
            current = None  # downstream steps lose their input -> reported clearly

    return RunResponse(
        nodes=nodes,
        final=nodes.get(final_id) if final_id else None,
        dims=dims,
        errors=errors,
        timings=timings,
    )
