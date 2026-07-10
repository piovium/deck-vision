from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import deck_vision.server as server_module
from deck_vision.errors import DeckVisionError
from deck_vision.recognize import Output


def test_health_readiness_and_asset_info_use_cache_status(tmp_path: Path) -> None:
    app = server_module.create_app(endpoint="https://example.test/api", cache_dir=tmp_path)
    client = TestClient(app)

    assert client.get("/healthz").json() == {"ok": True}

    ready = client.get("/readyz")
    assert ready.status_code == 503
    assert ready.json()["error"] == "asset_cache_not_ready"
    assert ready.json()["details"]["metadata_cached"] is False
    assert ready.json()["details"]["templates_cached"] is False

    assets = client.get("/v1/assets")
    assert assets.status_code == 200
    assert assets.json()["endpoint"] == "https://example.test/api"


def test_asset_refresh_endpoint_returns_card_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeStore:
        def __init__(self, *, endpoint: str | None = None, cache_dir: str | Path | None = None):
            self.endpoint = endpoint
            self.cache_dir = cache_dir

        def refresh(self) -> list[object]:
            return [object(), object()]

        def info(self) -> dict[str, object]:
            return {
                "endpoint": self.endpoint,
                "cache_dir": str(self.cache_dir),
                "metadata_cached": True,
                "templates_cached": True,
                "images_cached": 2,
                "characters": 1,
                "actions": 1,
            }

    monkeypatch.setattr(server_module, "AssetStore", FakeStore)
    app = server_module.create_app(endpoint="https://example.test/api", cache_dir=tmp_path)
    client = TestClient(app)

    response = client.post("/v1/assets/refresh")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["cards"] == 2


def test_recognize_upload_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_recognize_deck(
        image_path: str | Path,
        *,
        endpoint: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Output:
        assert Path(image_path).read_bytes() == b"image-bytes"
        assert endpoint == "https://example.test/api"
        assert cache_dir == tmp_path
        return Output(characters=[1, 2, 3], cards=list(range(4, 34)), code="share-code")

    monkeypatch.setattr(server_module, "ensure_assets_ready", lambda _settings, _lock: None)
    monkeypatch.setattr(server_module, "recognize_deck", fake_recognize_deck)
    app = server_module.create_app(endpoint="https://example.test/api", cache_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/recognize",
        files={"image": ("deck.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "characters": [1, 2, 3],
        "cards": list(range(4, 34)),
        "code": "share-code",
    }


def test_recognize_rejects_empty_upload(tmp_path: Path) -> None:
    app = server_module.create_app(endpoint="https://example.test/api", cache_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/recognize",
        files={"image": ("deck.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "empty_upload"


def test_recognize_rejects_oversized_upload(tmp_path: Path) -> None:
    app = server_module.create_app(
        endpoint="https://example.test/api",
        cache_dir=tmp_path,
        max_upload_bytes=3,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/recognize",
        files={"image": ("deck.png", b"1234", "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["error"] == "upload_too_large"


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (DeckVisionError("image_read_failed", "bad image"), 400),
        (DeckVisionError("wrong_card_counts", "wrong counts"), 422),
        (DeckVisionError("asset_fetch_failed", "fetch failed"), 503),
    ],
)
def test_recognize_maps_deck_vision_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: DeckVisionError,
    status_code: int,
) -> None:
    def fake_recognize_deck(
        _image_path: str | Path,
        *,
        endpoint: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> Output:
        raise error

    monkeypatch.setattr(server_module, "ensure_assets_ready", lambda _settings, _lock: None)
    monkeypatch.setattr(server_module, "recognize_deck", fake_recognize_deck)
    app = server_module.create_app(endpoint="https://example.test/api", cache_dir=tmp_path)
    client = TestClient(app)

    response = client.post(
        "/v1/recognize",
        files={"image": ("deck.png", b"image-bytes", "image/png")},
    )

    assert response.status_code == status_code
    assert response.json()["error"] == error.code
