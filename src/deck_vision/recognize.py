from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .assets import (
    INNER_TEMPLATE_SIZE,
    TEMPLATE_SIZE,
    AssetStore,
    CardAsset,
    TemplateIndex,
    center_crop,
    resize_float,
    to_gray_float,
)
from .errors import DeckVisionError
from .share_code import generate_share_code

CARD_ASPECT = 7.0 / 12.0


@dataclass(frozen=True)
class Output:
    characters: list[int]
    cards: list[int]
    code: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CardCandidate:
    x: int
    y: int
    w: int
    h: int
    source: str

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass(frozen=True)
class MatchResult:
    candidate: CardCandidate
    card: CardAsset
    score: float
    margin: float
    second_score: float
    kind: str


def recognize_deck(
    image_path: str | Path,
    *,
    endpoint: str | None = None,
    cache_dir: str | Path | None = None,
) -> Output:
    image = load_image_bgr(Path(image_path))
    store = AssetStore(endpoint=endpoint, cache_dir=cache_dir)
    index = store.load_template_index()
    matches = recognize_cards(image, index)

    characters = [match for match in matches if match.card.kind == "character"]
    actions = [match for match in matches if match.card.kind == "action"]
    if len(characters) != 3 or len(actions) != 30:
        raise DeckVisionError(
            "wrong_card_counts",
            "The image did not resolve to exactly 3 character cards and 30 action cards.",
            {"characters": len(characters), "actions": len(actions), "matches": len(matches)},
        )

    ordered = order_matches(characters) + order_matches(actions)
    share_ids = [match.card.share_id for match in ordered]
    code = generate_share_code(share_ids)
    return Output(
        characters=[match.card.id for match in ordered[:3]],
        cards=[match.card.id for match in ordered[3:]],
        code=code,
    )


def load_image_bgr(path: Path) -> np.ndarray:
    if not path.exists():
        raise DeckVisionError("image_not_found", "Input image does not exist.", {"path": str(path)})
    try:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:
        raise DeckVisionError(
            "image_read_failed",
            "Input image could not be read as an image.",
            {"path": str(path), "cause": repr(exc)},
        ) from exc
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def recognize_cards(image: np.ndarray, index: TemplateIndex) -> list[MatchResult]:
    candidates = detect_card_candidates(image)
    if not candidates:
        raise DeckVisionError(
            "not_enough_cards",
            "No card-like rectangles were detected.",
            {"detected": len(candidates)},
        )

    raw_matches: list[MatchResult] = []
    for candidate in candidates:
        match = match_candidate(image, candidate, index)
        if match is not None:
            raw_matches.append(match)
    raw_matches = expand_grid_matches(image, raw_matches, index)

    if len(raw_matches) < 33:
        raise DeckVisionError(
            "not_enough_cards",
            "Fewer than 33 detected rectangles matched known cards confidently.",
            {"detected": len(candidates), "matched": len(raw_matches)},
        )

    return choose_deck_matches(raw_matches)


def expand_grid_matches(
    image: np.ndarray,
    raw_matches: list[MatchResult],
    index: TemplateIndex,
) -> list[MatchResult]:
    inferred: list[CardCandidate] = []
    for kind, expected_count in (("character", 3), ("action", 30)):
        matches = [
            match
            for match in raw_matches
            if match.card.kind == kind and match.score >= 0.72 and match.margin >= 0.012
        ]
        inferred.extend(infer_grid_cells_from_matches(matches, expected_count=expected_count))

    additions: list[MatchResult] = []
    for candidate in non_max_suppression(inferred, iou_threshold=0.45):
        if any(iou(candidate, match.candidate) > 0.35 for match in raw_matches):
            continue
        match = match_candidate_with_jitter(image, candidate, index)
        if match is not None:
            additions.append(match)
    return [*raw_matches, *additions]


def match_candidate_with_jitter(
    image: np.ndarray,
    candidate: CardCandidate,
    index: TemplateIndex,
) -> MatchResult | None:
    best = match_candidate(image, candidate, index)
    height, width = image.shape[:2]
    radius = max(1, int(round(min(candidate.w, candidate.h) * 0.02)))
    for dx in (-radius, 0, radius):
        for dy in (-radius, 0, radius):
            if dx == 0 and dy == 0:
                continue
            shifted = CardCandidate(
                x=min(max(0, candidate.x + dx), max(0, width - candidate.w)),
                y=min(max(0, candidate.y + dy), max(0, height - candidate.h)),
                w=candidate.w,
                h=candidate.h,
                source=candidate.source,
            )
            match = match_candidate(image, shifted, index)
            if match is not None and (best is None or match.score > best.score):
                best = match
    return best


def infer_grid_cells_from_matches(
    matches: list[MatchResult],
    *,
    expected_count: int,
) -> list[CardCandidate]:
    if len(matches) < 2:
        return []

    inferred: list[CardCandidate] = []
    for group in cluster_match_size(matches):
        if len(group) < 2:
            continue
        median_h = float(np.median([match.candidate.h for match in group]))
        median_w = float(np.median([match.candidate.w for match in group]))
        rows = cluster_match_rows(group, threshold=max(8.0, median_h * 0.45))
        useful_rows = [row for row in rows if len(row) >= 2]
        if not useful_rows:
            continue

        global_xs = cluster_axis_values(
            [match.candidate.x for row in useful_rows for match in row],
            threshold=max(6.0, median_w * 0.45),
        )
        if len(global_xs) < 2:
            continue

        if expected_count == 3:
            target_cols = min(3, max(len(row) for row in useful_rows))
            target_rows = 1
        else:
            target_cols = max(len(row) for row in useful_rows)
            target_rows = int(np.ceil(expected_count / max(1, target_cols)))
            target_cols = min(
                len(global_xs),
                max(target_cols, int(np.ceil(expected_count / max(1, len(useful_rows))))),
            )

        if target_cols < 2:
            continue
        selected_xs = choose_axis_window(global_xs, target_cols)
        row_y_values = [
            float(np.median([match.candidate.y for match in row]))
            for row in choose_row_window(useful_rows, min(target_rows, len(useful_rows)))
        ]
        selected_ys = choose_axis_positions(row_y_values, target_rows)

        for row_y in selected_ys:
            for x in selected_xs:
                inferred.append(
                    CardCandidate(
                        x=int(round(x)),
                        y=max(0, int(round(row_y))),
                        w=int(round(median_w)),
                        h=int(round(median_h)),
                        source="grid",
                    )
                )
    return inferred


def cluster_match_size(matches: list[MatchResult]) -> list[list[MatchResult]]:
    groups: list[list[MatchResult]] = []
    for match in sorted(matches, key=lambda m: m.candidate.h):
        for group in groups:
            median_h = float(np.median([item.candidate.h for item in group]))
            if abs(match.candidate.h - median_h) <= max(10.0, median_h * 0.25):
                group.append(match)
                break
        else:
            groups.append([match])
    return groups


def cluster_match_rows(matches: list[MatchResult], *, threshold: float) -> list[list[MatchResult]]:
    rows: list[list[MatchResult]] = []
    for match in sorted(matches, key=lambda m: m.candidate.y):
        for row in rows:
            row_y = float(np.median([item.candidate.y for item in row]))
            if abs(match.candidate.y - row_y) <= threshold:
                row.append(match)
                break
        else:
            rows.append([match])
    return rows


def cluster_axis_values(values: list[int], *, threshold: float) -> list[float]:
    clusters: list[list[int]] = []
    for value in sorted(values):
        for cluster in clusters:
            if abs(value - float(np.median(cluster))) <= threshold:
                cluster.append(value)
                break
        else:
            clusters.append([value])
    return [float(np.median(cluster)) for cluster in clusters]


def choose_axis_window(values: list[float], count: int) -> list[float]:
    values = sorted(values)
    if len(values) <= count:
        return values
    best = values[:count]
    best_cost = float("inf")
    for start in range(0, len(values) - count + 1):
        window = values[start : start + count]
        diffs = np.diff(window)
        cost = 0.0 if len(diffs) == 0 else float(np.std(diffs) + (max(diffs) - min(diffs)) * 0.25)
        if cost < best_cost:
            best = window
            best_cost = cost
    return best


def choose_axis_positions(values: list[float], count: int) -> list[float]:
    values = sorted(values)
    if not values or count <= 0:
        return []
    if len(values) >= count:
        return choose_axis_window(values, count)
    if len(values) == 1:
        return values

    diffs = [
        diff
        for diff in np.diff(values)
        if diff > 1
    ]
    if not diffs:
        return values
    step = float(np.median(diffs))
    positions = list(values)
    while len(positions) < count:
        before = positions[0] - step
        after = positions[-1] + step
        # Deck screenshots are usually cropped from the top downward, so prefer
        # extending after the existing grid unless doing so would go negative on
        # the other side for tiny crops.
        positions.append(after)
        if len(positions) >= count:
            break
        if before >= 0:
            positions.insert(0, before)
    return sorted(positions[:count])


def choose_row_window(rows: list[list[MatchResult]], count: int) -> list[list[MatchResult]]:
    rows = sorted(rows, key=lambda row: float(np.median([match.candidate.y for match in row])))
    if len(rows) <= count:
        return rows
    best = rows[:count]
    best_score = -float("inf")
    for start in range(0, len(rows) - count + 1):
        window = rows[start : start + count]
        score = sum(len(row) for row in window) + sum(match.score for row in window for match in row)
        if score > best_score:
            best = window
            best_score = score
    return best


def detect_card_candidates(image: np.ndarray) -> list[CardCandidate]:
    height, width = image.shape[:2]
    min_h = max(42, int(height * 0.035))
    max_h = int(height * 0.55)
    candidates: list[CardCandidate] = []

    for scale in (1.0, 0.75, 0.5):
        scaled = image
        if scale != 1.0:
            scaled = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        candidates.extend(_contour_candidates(scaled, 1.0 / scale, min_h, max_h))

    candidates.extend(_background_blob_candidates(image, min_h, max_h))
    return non_max_suppression(candidates, iou_threshold=0.45)


def _contour_candidates(
    image: np.ndarray, scale_back: float, min_h: int, max_h: int
) -> list[CardCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates: list[CardCandidate] = []

    for blur in (3, 5):
        blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
        for low, high in ((40, 120), (70, 180), (100, 240)):
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                candidate = _make_candidate(x, y, w, h, scale_back, "contour")
                if candidate is not None and _is_card_shape(candidate, min_h, max_h):
                    candidates.append(candidate)
    return candidates


def _background_blob_candidates(image: np.ndarray, min_h: int, max_h: int) -> list[CardCandidate]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    masks = [
        cv2.inRange(saturation, 35, 255),
        cv2.inRange(value, 0, 235),
    ]
    candidates: list[CardCandidate] = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            candidate = _make_candidate(x, y, w, h, 1.0, "blob")
            if candidate is not None and _is_card_shape(candidate, min_h, max_h):
                candidates.append(candidate)
    return candidates


def _make_candidate(
    x: int, y: int, w: int, h: int, scale_back: float, source: str
) -> CardCandidate | None:
    x = int(round(x * scale_back))
    y = int(round(y * scale_back))
    w = int(round(w * scale_back))
    h = int(round(h * scale_back))
    if w <= 0 or h <= 0:
        return None
    return CardCandidate(x=x, y=y, w=w, h=h, source=source)


def _is_card_shape(candidate: CardCandidate, min_h: int, max_h: int) -> bool:
    aspect = candidate.w / candidate.h
    if candidate.h < min_h or candidate.h > max_h:
        return False
    if candidate.w < 24:
        return False
    return 0.42 <= aspect <= 0.78


def non_max_suppression(
    candidates: list[CardCandidate], *, iou_threshold: float
) -> list[CardCandidate]:
    ordered = sorted(candidates, key=candidate_priority, reverse=True)
    kept: list[CardCandidate] = []
    for candidate in ordered:
        if any(iou(candidate, other) > iou_threshold or center_inside(candidate, other) for other in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda c: (c.y, c.x))


def candidate_priority(candidate: CardCandidate) -> tuple[float, float, int]:
    aspect = candidate.w / candidate.h
    source_bonus = 1.0 if candidate.source == "grid" else 0.0
    return (source_bonus, -abs(aspect - CARD_ASPECT), candidate.area)


def iou(a: CardCandidate, b: CardCandidate) -> float:
    ax0, ay0, ax1, ay1 = a.as_xyxy()
    bx0, by0, bx1, by1 = b.as_xyxy()
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / float(a.area + b.area - inter)


def center_inside(a: CardCandidate, b: CardCandidate) -> bool:
    ax0, ay0, ax1, ay1 = a.as_xyxy()
    bx0, by0, bx1, by1 = b.as_xyxy()
    return bx0 <= a.cx <= bx1 and by0 <= a.cy <= by1 and b.area >= a.area * 1.15


def match_candidate(image: np.ndarray, candidate: CardCandidate, index: TemplateIndex) -> MatchResult | None:
    crop = crop_candidate(image, candidate)
    if crop.size == 0:
        return None
    prepared = prepare_crop_variants(crop)
    best: MatchResult | None = None
    for full, inner in prepared:
        scores = score_templates(full, inner, index)
        order = np.argsort(scores)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        score = float(scores[top_idx])
        second = float(scores[second_idx])
        margin = score - second
        if score < 0.58 or margin < 0.015:
            continue
        result = MatchResult(
            candidate=candidate,
            card=index.cards[top_idx],
            score=score,
            margin=margin,
            second_score=second,
            kind=index.cards[top_idx].kind,
        )
        if best is None or result.score > best.score:
            best = result
    return best


def crop_candidate(image: np.ndarray, candidate: CardCandidate) -> np.ndarray:
    height, width = image.shape[:2]
    pad_x = int(candidate.w * 0.03)
    pad_y = int(candidate.h * 0.03)
    x0 = max(0, candidate.x - pad_x)
    y0 = max(0, candidate.y - pad_y)
    x1 = min(width, candidate.x + candidate.w + pad_x)
    y1 = min(height, candidate.y + candidate.h + pad_y)
    return image[y0:y1, x0:x1]


def prepare_crop_variants(crop: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    variants: list[np.ndarray] = [crop]
    height, width = crop.shape[:2]
    for margin in (0.02, 0.05, 0.08):
        x0 = int(width * margin)
        x1 = int(width * (1.0 - margin))
        y0 = int(height * margin)
        y1 = int(height * (1.0 - margin))
        if x1 > x0 and y1 > y0:
            variants.append(crop[y0:y1, x0:x1])

    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for variant in variants:
        full = resize_float(variant, TEMPLATE_SIZE)
        inner = resize_float(center_crop(variant, x_margin=0.08, y_margin=0.08), INNER_TEMPLATE_SIZE)
        prepared.append((full, inner))
    return prepared


def score_templates(full: np.ndarray, inner: np.ndarray, index: TemplateIndex) -> np.ndarray:
    full_gray = to_gray_float(full)
    inner_gray = to_gray_float(inner)
    color_score = 1.0 - np.mean(np.abs(index.full - full), axis=(1, 2, 3))
    inner_color_score = 1.0 - np.mean(np.abs(index.inner - inner), axis=(1, 2, 3))
    gray_score = 1.0 - np.mean(np.abs(index.gray - full_gray), axis=(1, 2))
    inner_gray_score = 1.0 - np.mean(np.abs(index.inner_gray - inner_gray), axis=(1, 2))
    return (
        color_score * 0.25
        + inner_color_score * 0.45
        + gray_score * 0.10
        + inner_gray_score * 0.20
    )


def choose_deck_matches(raw_matches: list[MatchResult]) -> list[MatchResult]:
    deduped = dedupe_matches(raw_matches)
    characters = [match for match in deduped if match.card.kind == "character"]
    actions = [match for match in deduped if match.card.kind == "action"]
    characters = select_dominant_size_group(characters, 3)
    actions = select_dominant_size_group(actions, 30)
    characters = select_spatial_group(characters, 3)
    actions = select_spatial_group(actions, 30)
    if len(characters) < 3 or len(actions) < 30:
        raise DeckVisionError(
            "not_enough_cards",
            "Could not find a complete 3-character and 30-action deck after filtering matches.",
            {"characters": len(characters), "actions": len(actions), "raw_matches": len(raw_matches)},
        )
    return order_matches(characters[:3]) + order_matches(actions[:30])


def select_dominant_size_group(matches: list[MatchResult], count: int) -> list[MatchResult]:
    if len(matches) <= count:
        return matches
    groups = [group for group in cluster_match_size(matches) if len(group) >= count]
    if not groups:
        return matches
    return max(groups, key=lambda group: (len(group), sum(match.score for match in group)))


def dedupe_matches(matches: list[MatchResult]) -> list[MatchResult]:
    ordered = sorted(matches, key=lambda m: m.score, reverse=True)
    kept: list[MatchResult] = []
    for match in ordered:
        if any(iou(match.candidate, other.candidate) > 0.35 for other in kept):
            continue
        kept.append(match)
    return kept


def select_spatial_group(matches: list[MatchResult], count: int) -> list[MatchResult]:
    if len(matches) <= count:
        return matches
    ordered = sorted(matches, key=lambda m: m.score, reverse=True)
    seeds = ordered[: min(len(ordered), count + 12)]
    best: list[MatchResult] = ordered[:count]
    best_cost = float("inf")
    for seed in seeds:
        group = sorted(
            matches,
            key=lambda m: (abs(m.candidate.cy - seed.candidate.cy) * 0.15 + abs(m.candidate.cx - seed.candidate.cx) * 0.03 - m.score),
        )[:count]
        if len(group) != count:
            continue
        xs = np.array([m.candidate.cx for m in group], dtype=np.float32)
        ys = np.array([m.candidate.cy for m in group], dtype=np.float32)
        score_penalty = sum(1.0 - m.score for m in group)
        cost = float(np.std(xs) * 0.05 + np.std(ys) * 0.08 + score_penalty)
        if cost < best_cost:
            best = group
            best_cost = cost
    return sorted(best, key=lambda m: m.score, reverse=True)


def order_matches(matches: list[MatchResult]) -> list[MatchResult]:
    if not matches:
        return []
    sorted_by_y = sorted(matches, key=lambda m: m.candidate.cy)
    median_h = float(np.median([m.candidate.h for m in matches]))
    row_threshold = max(12.0, median_h * 0.55)
    rows: list[list[MatchResult]] = []
    for match in sorted_by_y:
        placed = False
        for row in rows:
            row_y = float(np.mean([m.candidate.cy for m in row]))
            if abs(match.candidate.cy - row_y) <= row_threshold:
                row.append(match)
                placed = True
                break
        if not placed:
            rows.append([match])
    rows.sort(key=lambda row: float(np.mean([m.candidate.cy for m in row])))
    ordered: list[MatchResult] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda m: m.candidate.cx))
    return ordered
