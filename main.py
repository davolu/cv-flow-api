"""Root entry point so `uvicorn main:app` works (re-exports the app package)."""

from app.main import app  # noqa: F401
