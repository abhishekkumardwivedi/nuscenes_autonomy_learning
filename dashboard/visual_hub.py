from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading
import time

from PIL import Image, ImageDraw, ImageFont


class VisualHub:
    """Shared latest visual used by HTTP fallback and the optional WebRTC track."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image_path: Path | None = None
        self._stage = 0
        self._title = "Autonomy Learning Dashboard"
        self._status = "idle"
        self._updated = time.time()

    def set_status(self, stage: int, title: str, status: str) -> None:
        with self._lock:
            self._stage = stage
            self._title = title
            self._status = status
            self._updated = time.time()

    def set_image(self, path: Path, stage: int, title: str) -> None:
        if not path.exists():
            return
        with self._lock:
            self._image_path = path
            self._stage = stage
            self._title = title
            self._status = "visual updated"
            self._updated = time.time()

    def snapshot(self) -> tuple[Path | None, int, str, str, float]:
        with self._lock:
            return self._image_path, self._stage, self._title, self._status, self._updated

    def image_rgb(self, size: tuple[int, int] = (1280, 720)) -> Image.Image:
        path, stage, title, status, _ = self.snapshot()
        canvas = Image.new("RGB", size, (15, 23, 42))
        draw = ImageDraw.Draw(canvas)
        if path and path.exists():
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((size[0] - 40, size[1] - 90))
                x = (size[0] - image.width) // 2
                y = 55 + (size[1] - 90 - image.height) // 2
                canvas.paste(image, (x, y))
            except Exception:
                pass
        draw.rectangle((0, 0, size[0], 48), fill=(10, 16, 30))
        draw.text((18, 14), f"Stage {stage:02d} — {title}", fill=(226, 232, 240))
        draw.text((size[0] - 250, 14), status, fill=(148, 163, 184))
        if path is None:
            draw.text((40, 110), "No stage visual yet. Run a stage or open a saved artifact.", fill=(148, 163, 184))
        return canvas

    def jpeg_bytes(self, quality: int = 88) -> bytes:
        image = self.image_rgb()
        bio = BytesIO()
        image.save(bio, format="JPEG", quality=quality)
        return bio.getvalue()
