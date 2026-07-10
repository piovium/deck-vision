from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import cv2
import httpx
import numpy as np
from PIL import Image
from platformdirs import user_cache_path

from .errors import DeckVisionError

CardKind = Literal["character", "action"]
DEFAULT_ENDPOINT = "https://static-data.piovium.org/api/v4"
TEMPLATE_SIZE = (70, 120)
INNER_TEMPLATE_SIZE = (56, 96)


@dataclass(frozen=True)
class CardAsset:
    id: int
    share_id: int
    kind: CardKind
    name: str
    card_face: str
    image_path: str


@dataclass
class TemplateIndex:
    cards: list[CardAsset]
    full: np.ndarray
    inner: np.ndarray
    gray: np.ndarray
    inner_gray: np.ndarray


def default_cache_dir() -> Path:
    return user_cache_path("deck-vision", "deck-vision")


def resolve_endpoint(endpoint: str | None = None) -> str:
    return (endpoint or os.environ.get("ASSETS_API_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


def endpoint_cache_key(endpoint: str) -> str:
    return hashlib.sha1(endpoint.encode("utf-8")).hexdigest()[:12]


class AssetStore:
    def __init__(self, *, endpoint: str | None = None, cache_dir: str | Path | None = None):
        self.endpoint = resolve_endpoint(endpoint)
        base_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.cache_dir = base_dir / endpoint_cache_key(self.endpoint)
        self.images_dir = self.cache_dir / "images"
        self.metadata_path = self.cache_dir / "cards.json"
        self.templates_path = self.cache_dir / "templates.npz"

    def info(self) -> dict[str, Any]:
        cards = self.load_cards(require_present=False)
        character_count = sum(1 for card in cards if card.kind == "character")
        action_count = sum(1 for card in cards if card.kind == "action")
        image_count = len(list(self.images_dir.glob("*.png"))) if self.images_dir.exists() else 0
        return {
            "endpoint": self.endpoint,
            "cache_dir": str(self.cache_dir),
            "metadata_cached": self.metadata_path.exists(),
            "templates_cached": self.templates_path.exists(),
            "images_cached": image_count,
            "characters": character_count,
            "actions": action_count,
        }

    def refresh(self) -> list[CardAsset]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0, connect=15.0), follow_redirects=True) as client:
                character_payload = self._get_json(client, f"{self.endpoint}/data/beta/CHS/characters")
                action_payload = self._get_json(client, f"{self.endpoint}/data/beta/CHS/action_cards")
                cards = [
                    *self._parse_cards(character_payload, "character"),
                    *self._parse_cards(action_payload, "action"),
                ]
                for card in cards:
                    if not Path(card.image_path).exists():
                        self._download_card_face(client, card.card_face, Path(card.image_path))
        except DeckVisionError:
            raise
        except Exception as exc:
            raise DeckVisionError(
                "asset_fetch_failed",
                "Failed to fetch or cache deck assets.",
                {"endpoint": self.endpoint, "cause": repr(exc)},
            ) from exc

        self._write_json_atomic(self.metadata_path, [asdict(card) for card in cards])
        self.build_template_index(cards, refresh=True)
        return cards

    def load_cards(self, *, require_present: bool = True) -> list[CardAsset]:
        if not self.metadata_path.exists():
            if require_present:
                return self.refresh()
            return []
        try:
            raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            return [CardAsset(**item) for item in raw]
        except Exception as exc:
            if require_present:
                raise DeckVisionError(
                    "asset_cache_invalid",
                    "Cached asset metadata is unreadable; run `deck-vision assets refresh`.",
                    {"path": str(self.metadata_path), "cause": repr(exc)},
                ) from exc
            return []

    def load_template_index(self) -> TemplateIndex:
        cards = self.load_cards()
        if not self.templates_path.exists():
            return self.build_template_index(cards)

        try:
            npz = np.load(self.templates_path, allow_pickle=False)
            ids = npz["ids"].astype(np.int64).tolist()
            if ids != [card.id for card in cards]:
                return self.build_template_index(cards, refresh=True)
            return TemplateIndex(
                cards=cards,
                full=npz["full"].astype(np.float32),
                inner=npz["inner"].astype(np.float32),
                gray=npz["gray"].astype(np.float32),
                inner_gray=npz["inner_gray"].astype(np.float32),
            )
        except Exception:
            return self.build_template_index(cards, refresh=True)

    def build_template_index(self, cards: list[CardAsset], *, refresh: bool = False) -> TemplateIndex:
        if refresh and self.templates_path.exists():
            self.templates_path.unlink()
        full: list[np.ndarray] = []
        inner: list[np.ndarray] = []
        gray: list[np.ndarray] = []
        inner_gray: list[np.ndarray] = []

        for card in cards:
            image = cv2.imread(card.image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise DeckVisionError(
                    "asset_cache_invalid",
                    "A cached card-face image is unreadable; refresh assets.",
                    {"card_face": card.card_face, "path": card.image_path},
                )
            normalized = preprocess_card_image(image)
            full.append(resize_float(normalized, TEMPLATE_SIZE))
            inner_crop = center_crop(normalized, x_margin=0.08, y_margin=0.08)
            inner.append(resize_float(inner_crop, INNER_TEMPLATE_SIZE))
            gray.append(to_gray_float(resize_float(normalized, TEMPLATE_SIZE)))
            inner_gray.append(to_gray_float(resize_float(inner_crop, INNER_TEMPLATE_SIZE)))

        index = TemplateIndex(
            cards=cards,
            full=np.stack(full, axis=0),
            inner=np.stack(inner, axis=0),
            gray=np.stack(gray, axis=0),
            inner_gray=np.stack(inner_gray, axis=0),
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.templates_path,
            ids=np.array([card.id for card in cards], dtype=np.int64),
            full=index.full.astype(np.float32),
            inner=index.inner.astype(np.float32),
            gray=index.gray.astype(np.float32),
            inner_gray=index.inner_gray.astype(np.float32),
        )
        return index

    def _get_json(self, client: httpx.Client, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = client.get(url)
                response.raise_for_status()
                payload = response.json()
                if payload.get("success") is not True or not isinstance(payload.get("data"), list):
                    raise ValueError("unexpected asset payload shape")
                return payload
            except Exception as exc:
                last_error = exc
        raise DeckVisionError(
            "asset_fetch_failed",
            "Failed to fetch card metadata.",
            {"url": url, "cause": repr(last_error)},
        )

    def _parse_cards(self, payload: dict[str, Any], kind: CardKind) -> list[CardAsset]:
        cards: list[CardAsset] = []
        for item in payload.get("data", []):
            share_id = item.get("shareId")
            card_face = item.get("cardFace")
            card_id = item.get("id")
            if share_id is None or card_face is None or card_id is None:
                continue
            image_path = self.images_dir / f"{safe_filename(str(card_face))}.png"
            cards.append(
                CardAsset(
                    id=int(card_id),
                    share_id=int(share_id),
                    kind=kind,
                    name=str(item.get("name") or item.get("englishName") or card_id),
                    card_face=str(card_face),
                    image_path=str(image_path),
                )
            )
        return cards

    def _download_card_face(self, client: httpx.Client, card_face: str, image_path: Path) -> None:
        url = f"{self.endpoint}/image/raw/{card_face}"
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = client.get(url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    suffix=".png", dir=str(image_path.parent), delete=False
                ) as temp:
                    temp_path = Path(temp.name)
                try:
                    image.save(temp_path, format="PNG")
                    temp_path.replace(image_path)
                finally:
                    if temp_path.exists() and temp_path != image_path:
                        temp_path.unlink()
                return
            except Exception as exc:
                last_error = exc
        raise DeckVisionError(
            "asset_fetch_failed",
            "Failed to download a card-face image.",
            {"url": url, "card_face": card_face, "cause": repr(last_error)},
        )

    def _write_json_atomic(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as temp:
            json.dump(payload, temp, ensure_ascii=False, indent=2)
            temp_path = Path(temp.name)
        temp_path.replace(path)


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def preprocess_card_image(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    return image_bgr


def resize_float(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def center_crop(image: np.ndarray, *, x_margin: float, y_margin: float) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = int(width * x_margin)
    x1 = int(width * (1.0 - x_margin))
    y0 = int(height * y_margin)
    y1 = int(height * (1.0 - y_margin))
    return image[y0:y1, x0:x1]


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 255.0
    return image.astype(np.float32)
