# deck-vision

`deck-vision` recognizes Genius Invokation TCG deck sharing images and converts
them into machine-readable deck data plus a copyable share code.

The Phase 1 implementation is a local Python CLI and importable library. It does
not use LLMs or remote compute for recognition. The only network dependency is
the official/static card asset API used to download card metadata and card-face
images into a local cache.

## Project Settings

- Package name: `deck-vision`
- Python: `>=3.11`
- CLI entrypoint: `deck-vision`
- Build backend: `hatchling`
- Runtime dependencies:
  - `httpx`
  - `numpy`
  - `opencv-python-headless`
  - `pillow`
  - `platformdirs`
- Test dependency:
  - `pytest`
- Default asset endpoint:
  - `https://static-data.piovium.org/api/v4`
- Endpoint override:
  - `ASSETS_API_ENDPOINT`
  - or CLI flag `--endpoint`
- Cache location:
  - platform-specific user cache directory via `platformdirs`
  - overridable with `--cache-dir`

## Installation

Install the package in editable mode:

```sh
python -m pip install -e .
```

For development and tests:

```sh
python -m pip install -e .[test]
```

If the `deck-vision` script directory is not on `PATH`, use:

```sh
python -m deck_vision.cli ...
```

## CLI Usage

Refresh metadata, card-face images, and template fingerprints:

```sh
deck-vision assets refresh
```

Show cache status:

```sh
deck-vision assets info
```

Recognize an image and print only the share code:

```sh
deck-vision recognize examples/deck_img_5.png
```

Recognize an image and print the full JSON result:

```sh
deck-vision recognize examples/deck_img_5.png --json
```

JSON output shape:

```ts
interface Output {
  characters: number[]; // 3 character internal IDs
  cards: number[];      // 30 action-card internal IDs
  code: string;         // generated share code
}
```

Errors are printed to stderr as structured JSON and return exit code `2`.
Example error codes include `image_not_found`, `image_read_failed`,
`asset_fetch_failed`, `asset_cache_invalid`, `not_enough_cards`,
`wrong_card_counts`, and `no_valid_share_code`.

## Library Usage

```python
from deck_vision import recognize_deck

output = recognize_deck("examples/deck_img_5.png")
print(output.characters)
print(output.cards)
print(output.code)
```

Optional overrides:

```python
output = recognize_deck(
    "deck.png",
    endpoint="https://static-data.piovium.org/api/v4",
    cache_dir=".cache/deck-vision",
)
```

## Asset Pipeline

`AssetStore` downloads and caches all shareable cards:

1. Fetch `data/latest/CHS/characters`.
2. Fetch `data/latest/CHS/action_cards`.
3. Keep entries with `id`, `shareId`, and `cardFace`.
4. Download each card face from `image/raw/{cardFace}`.
5. Store images as PNG files in the local cache.
6. Build `templates.npz` containing resized full-card, inner-card, grayscale,
   and inner-grayscale template arrays.

The cache is keyed by endpoint, so different endpoints do not overwrite each
other.

## Recognition Algorithm

The recognizer is intentionally simple and local. It is tuned for the provided
example layouts while leaving room for more layout-specific improvements later.

1. Input normalization
   - Load the image with Pillow.
   - Apply EXIF orientation.
   - Convert to OpenCV BGR format.

2. Candidate detection
   - Search the image at multiple scales.
   - Use Canny edges and contour bounding boxes.
   - Also use HSV/value blob masks for card faces that are mostly filled areas.
   - Keep boxes close to the GI-TCG card aspect ratio, approximately `7:12`.
   - Run non-maximum suppression to remove duplicate boxes.

3. Template matching
   - Crop each candidate with a small margin.
   - Try several inner crop variants to tolerate frames, borders, and tight
     screenshots.
   - Resize crops to the same template sizes used by cached card faces.
   - Score against every cached card with a weighted blend of:
     - full-color similarity
     - inner-color similarity
     - full-grayscale similarity
     - inner-grayscale similarity
   - Accept a match only when the top score and top-vs-second margin are high
     enough.

4. Grid completion
   - Some screenshots miss edge cards or crop card borders tightly.
   - After initial matches, infer missing grid cells from high-confidence rows
     of already matched cards.
   - Try small pixel jitters for inferred boxes, then run the same template
     matcher.

5. Deck selection
   - Deduplicate overlapping matches.
   - Split character and action matches.
   - Prefer the dominant card-size group so small page decorations do not
     replace real deck cards.
   - Select the spatially coherent group of 3 character cards and 30 action
     cards.
   - Order each group row-major by visual position.

6. Share code generation
   - Convert internal IDs to share IDs.
   - Pack 33 share IDs into the GI-TCG 51-byte payload.
   - Treat the final byte as the seed.
   - Enumerate seed values from `0` to `255`.
   - Base64-encode each payload and return the first code whose lowercase text
     contains none of the words in `data/block_words.txt`.

Action-card order can vary between equivalent deck images and share codes. The
tests normalize the provided fixture codes to this recognizer's deterministic
visual order.

## Testing

Run all tests:

```sh
python -m pytest -q
```

Run only fast tests:

```sh
python -m pytest -q -m "not integration"
```

Run example-image integration tests:

```sh
python -m pytest -q -m integration
```

The integration tests require a populated asset cache or live access to
`ASSETS_API_ENDPOINT`.

## Current Limitations

- Phase 1 is examples-first. It is not a general OCR/photo understanding system.
- Recognition expects reasonably clear card-face images.
- Failed recognition returns structured errors, but does not yet generate a
  marked-up diagnostic image.
- There is no HTTP API or Docker image yet.
