"""FastAPI application — serves the dashboard UI and REST API."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

from .api import router as api_router

app = FastAPI(title="ModelForge", version="0.1.0", docs_url="/docs", redoc_url="/redoc")
app.include_router(api_router)

_UI = Path(__file__).parent / "ui" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_UI)
