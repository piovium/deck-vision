from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .assets import AssetStore
from .errors import DeckVisionError
from .recognize import recognize_deck

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

BAD_UPLOAD_ERRORS = {"empty_upload", "image_not_found", "image_read_failed", "upload_read_failed"}
ASSET_ERRORS = {"asset_cache_invalid", "asset_fetch_failed", "asset_cache_not_ready"}


@dataclass(frozen=True)
class ServerSettings:
    endpoint: str | None
    cache_dir: str | Path | None
    max_upload_bytes: int


class RecognitionResponse(BaseModel):
    characters: list[int]
    cards: list[int]
    code: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ok: bool


class UploadHTTPError(Exception):
    def __init__(self, status_code: int, error: DeckVisionError):
        super().__init__(error.message)
        self.status_code = status_code
        self.error = error


def create_app(
    *,
    endpoint: str | None = None,
    cache_dir: str | Path | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    settings = ServerSettings(
        endpoint=endpoint if endpoint is not None else os.environ.get("ASSETS_API_ENDPOINT"),
        cache_dir=cache_dir if cache_dir is not None else os.environ.get("DECK_VISION_CACHE_DIR"),
        max_upload_bytes=max_upload_bytes
        if max_upload_bytes is not None
        else read_max_upload_bytes(),
    )
    app = FastAPI(
        title="deck-vision",
        version="0.1.0",
        description="Recognize Genius Invokation TCG deck sharing images.",
    )
    app.state.deck_vision_settings = settings
    app.state.asset_lock = threading.Lock()

    @app.exception_handler(DeckVisionError)
    async def handle_deck_vision_error(_request: object, exc: DeckVisionError) -> JSONResponse:
        return error_response(exc, status_for_error(exc))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: object, _exc: Exception) -> JSONResponse:
        return error_response(
            DeckVisionError("internal_server_error", "Unexpected server error."),
            500,
        )

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz", response_model=None)
    async def readyz() -> dict[str, Any] | JSONResponse:
        store = make_store(settings)
        info = store.info()
        if cache_ready(store):
            return {"ok": True, **info}
        return error_response(
            DeckVisionError(
                "asset_cache_not_ready",
                "Asset metadata and template cache are not ready.",
                {"ok": False, **info},
            ),
            503,
        )

    @app.get("/v1/assets")
    async def assets_info() -> dict[str, Any]:
        return make_store(settings).info()

    @app.post("/v1/assets/refresh")
    async def assets_refresh() -> dict[str, Any]:
        with app.state.asset_lock:
            store = make_store(settings)
            cards = store.refresh()
            return {"ok": True, "cards": len(cards), **store.info()}

    @app.post(
        "/v1/recognize",
        response_model=RecognitionResponse,
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def recognize(image: UploadFile = File(...)) -> Any:
        temp_path: Path | None = None
        try:
            try:
                temp_path = await save_upload_to_temp(image, settings.max_upload_bytes)
            except UploadHTTPError as exc:
                return error_response(exc.error, exc.status_code)

            ensure_assets_ready(settings, app.state.asset_lock)
            output = recognize_deck(
                temp_path,
                endpoint=settings.endpoint,
                cache_dir=settings.cache_dir,
            )
            return output.to_dict()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            await image.close()

    return app


def read_max_upload_bytes() -> int:
    raw = os.environ.get("DECK_VISION_MAX_UPLOAD_BYTES")
    if raw is None:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES
    return value if value > 0 else DEFAULT_MAX_UPLOAD_BYTES


def make_store(settings: ServerSettings) -> AssetStore:
    return AssetStore(endpoint=settings.endpoint, cache_dir=settings.cache_dir)


def cache_ready(store: AssetStore) -> bool:
    return store.metadata_path.exists() and store.templates_path.exists()


def ensure_assets_ready(settings: ServerSettings, lock: threading.Lock) -> None:
    store = make_store(settings)
    if cache_ready(store):
        return
    with lock:
        store = make_store(settings)
        if not cache_ready(store):
            store.load_template_index()


async def save_upload_to_temp(upload: UploadFile, max_upload_bytes: int) -> Path:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="deck-vision-",
            suffix=safe_upload_suffix(upload.filename),
            delete=False,
        ) as temp:
            temp_path = Path(temp.name)
            total = 0
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                if total + len(chunk) > max_upload_bytes:
                    raise UploadHTTPError(
                        413,
                        DeckVisionError(
                            "upload_too_large",
                            "Uploaded image exceeds the configured size limit.",
                            {"max_bytes": max_upload_bytes},
                        ),
                    )
                total += len(chunk)
                temp.write(chunk)

        if total == 0:
            raise UploadHTTPError(
                400,
                DeckVisionError("empty_upload", "Uploaded image is empty."),
            )
        return temp_path
    except UploadHTTPError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise UploadHTTPError(
            400,
            DeckVisionError(
                "upload_read_failed",
                "Uploaded image could not be read.",
                {"cause": repr(exc)},
            ),
        ) from exc


def safe_upload_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return suffix
    return ".img"


def status_for_error(exc: DeckVisionError) -> int:
    if exc.code == "upload_too_large":
        return 413
    if exc.code in BAD_UPLOAD_ERRORS:
        return 400
    if exc.code in ASSET_ERRORS:
        return 503
    return 422


def error_response(error: DeckVisionError, status_code: int) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error.to_dict())


app = create_app()
