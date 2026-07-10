from pathlib import Path

import pytest

from deck_vision.recognize import recognize_deck


@pytest.mark.integration
@pytest.mark.parametrize("image_path", sorted(Path("examples").glob("deck_img_*.*")))
def test_example_code_matches_fixture(image_path: Path) -> None:
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        pytest.skip("not an image")
    expected_path = image_path.with_name(f"{image_path.stem}_code.txt")
    if not expected_path.exists():
        pytest.skip("missing expected code")

    output = recognize_deck(image_path)

    assert output.code == expected_path.read_text(encoding="utf-8").strip()
