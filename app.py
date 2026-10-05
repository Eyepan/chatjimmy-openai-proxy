"""Vercel FastAPI entrypoint for the src-layout chatjimmy package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatjimmy.server import app

__all__ = ["app"]
