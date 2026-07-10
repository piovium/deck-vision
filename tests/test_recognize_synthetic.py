from pathlib import Path

import cv2
import numpy as np
import pytest

from deck_vision.assets import CardAsset, TemplateIndex, resize_float, to_gray_float
from deck_vision.errors import DeckVisionError
from deck_vision.recognize import CARD_ASPECT, recognize_cards


def make_card(color: tuple[int, int, int], label: str) -> np.ndarray:
    image = np.full((120, 70, 3), color, dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (67, 117), (20, 20, 20), 2)
    cv2.putText(image, label, (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def build_index(cards: list[tuple[int, int, str, tuple[int, int, int], str]]) -> TemplateIndex:
    assets: list[CardAsset] = []
    full: list[np.ndarray] = []
    inner: list[np.ndarray] = []
    gray: list[np.ndarray] = []
    inner_gray: list[np.ndarray] = []
    for card_id, share_id, kind, color, label in cards:
        image = make_card(color, label)
        assets.append(
            CardAsset(
                id=card_id,
                share_id=share_id,
                kind=kind,  # type: ignore[arg-type]
                name=label,
                card_face=label,
                image_path=f"{label}.png",
            )
        )
        full_img = resize_float(image, (70, 120))
        inner_img = resize_float(image[10:110, 6:64], (56, 96))
        full.append(full_img)
        inner.append(inner_img)
        gray.append(to_gray_float(full_img))
        inner_gray.append(to_gray_float(inner_img))
    return TemplateIndex(
        cards=assets,
        full=np.stack(full),
        inner=np.stack(inner),
        gray=np.stack(gray),
        inner_gray=np.stack(inner_gray),
    )


def test_synthetic_grid_detects_duplicate_cards() -> None:
    specs: list[tuple[int, int, str, tuple[int, int, int], str]] = [
        (100 + i, 1 + i, "character" if i < 3 else "action", ((30 + i * 5) % 255, (80 + i * 7) % 255, (150 + i * 11) % 255), f"C{i}")
        for i in range(33)
    ]
    index = build_index(specs)
    canvas = np.full((900, 620, 3), 245, dtype=np.uint8)
    positions: list[tuple[int, int, int]] = []
    for i in range(3):
        positions.append((120 + i * 120, 30, i))
    for i in range(30):
        row = i // 6
        col = i % 6
        positions.append((30 + col * 92, 190 + row * 130, 3 + i))
    for x, y, idx in positions:
        card = make_card(specs[idx][3], specs[idx][4])
        canvas[y : y + 120, x : x + 70] = card

    matches = recognize_cards(canvas, index)

    assert len(matches) == 33
    assert [match.card.id for match in matches[:3]] == [100, 101, 102]
    assert [match.card.kind for match in matches[3:]] == ["action"] * 30


def test_not_enough_cards_error() -> None:
    index = build_index([(1, 1, "character", (20, 50, 90), "A")])
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)

    with pytest.raises(DeckVisionError) as exc:
        recognize_cards(canvas, index)

    assert exc.value.code == "not_enough_cards"
