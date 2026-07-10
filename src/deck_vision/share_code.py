from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable, Sequence

from .errors import DeckVisionError


def load_block_words(path: Path | None = None) -> list[str]:
    if path is None:
        candidates = [
            Path(__file__).resolve().parents[2] / "data" / "block_words.txt",
            Path.cwd() / "data" / "block_words.txt",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        return []
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def encode_share_ids(share_ids: Sequence[int], seed: int) -> bytes:
    if len(share_ids) != 33:
        raise ValueError(f"expected 33 share IDs, got {len(share_ids)}")
    if not 0 <= seed <= 255:
        raise ValueError(f"seed must be in range 0..255, got {seed}")

    padded = [int(value) for value in share_ids] + [0]
    reordered: list[int] = []
    for i in range(17):
        left = padded[i * 2]
        right = padded[i * 2 + 1]
        if not 0 <= left <= 0xFFF or not 0 <= right <= 0xFFF:
            raise ValueError("share IDs must fit in 12 bits")
        reordered.extend(
            [
                left >> 4,
                ((left & 0xF) << 4) + (right >> 8),
                right & 0xFF,
            ]
        )

    original: list[int] = []
    for i in range(25):
        original.append((reordered[i] + seed) & 0xFF)
        original.append((reordered[i + 25] + seed) & 0xFF)
    return bytes([*original, seed])


def decode_share_code(code: str) -> tuple[list[int], int]:
    payload = base64.b64decode(code)
    if len(payload) != 51:
        raise ValueError(f"expected a 51-byte share-code payload, got {len(payload)}")

    seed = payload[-1]
    reordered = [0] * 51
    for i in range(25):
        reordered[i] = (payload[i * 2] - seed) & 0xFF
        reordered[i + 25] = (payload[i * 2 + 1] - seed) & 0xFF

    share_ids: list[int] = []
    for i in range(16):
        a = reordered[i * 3]
        b = reordered[i * 3 + 1]
        c = reordered[i * 3 + 2]
        share_ids.append((a << 4) + (b >> 4))
        share_ids.append(((b & 0xF) << 8) + c)

    a = reordered[48]
    b = reordered[49]
    share_ids.append((a << 4) + (b >> 4))
    return share_ids, seed


def generate_share_code(
    share_ids: Sequence[int],
    *,
    block_words: Iterable[str] | None = None,
) -> str:
    blocked = [word.lower() for word in (block_words if block_words is not None else load_block_words())]
    for seed in range(256):
        code = base64.b64encode(encode_share_ids(share_ids, seed)).decode("ascii")
        lowered = code.lower()
        if not any(word and word in lowered for word in blocked):
            return code
    raise DeckVisionError(
        "no_valid_share_code",
        "No seed produced a share code that avoided the block-word list.",
        {"share_id_count": len(share_ids), "seed_count": 256},
    )
