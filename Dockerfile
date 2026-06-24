FROM python:3.11-slim

# opencv-contrib-python-headless ships without GUI/GL deps, but cv2 still links
# against libglib's gthread at import time — install just that one small system lib.
# (the contrib build unlocks SIFT/ORB feature detectors and the dnn_superres module.)
# tesseract-ocr is the system binary behind the `ocr_text` DL op (pytesseract wraps it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Deep-learning ops download small models on first use and cache them here.
# Pre-create the dir so it stays writable at runtime (override via CV_FLOW_MODELS_DIR).
ENV CV_FLOW_MODELS_DIR=/app/models
RUN mkdir -p /app/models

ENV PORT=8000
EXPOSE 8000

# $PORT is provided by Render at runtime; default to 8000 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
