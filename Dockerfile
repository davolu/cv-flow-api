FROM python:3.11-slim

# opencv-contrib-python-headless ships without GUI/GL deps, but cv2 still links
# against libglib's gthread at import time — install just that one small system lib.
# (the contrib build unlocks SIFT/ORB feature detectors and the dnn_superres module.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# $PORT is provided by Render at runtime; default to 8000 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
