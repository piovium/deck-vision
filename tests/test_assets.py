from pathlib import Path

from deck_vision.assets import AssetStore


def test_parse_cards_keeps_deck_usable_entries(tmp_path: Path) -> None:
    store = AssetStore(endpoint="https://example.test/api", cache_dir=tmp_path)
    payload = {
        "success": True,
        "data": [
            {"id": 1, "shareId": 10, "name": "A", "cardFace": "Face_A"},
            {"id": 2, "name": "B", "cardFace": "Face_B"},
            {"id": 3, "shareId": 12, "name": "C"},
        ],
    }

    cards = store._parse_cards(payload, "character")

    assert [card.id for card in cards] == [1]
    assert cards[0].share_id == 10
    assert cards[0].kind == "character"
    assert cards[0].image_path.endswith("Face_A.png")
