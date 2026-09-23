"""Making repeated symbols agree with each other.

Comparing a scanned glyph with an installed font is unreliable: the typeface in
an old figure is usually not installed, and at label sizes the shape differences
between a candidate and its confusable twin are smaller than the difference
between two typefaces. Comparing a scanned glyph with *another scanned glyph
from the same figure* has no such problem, so agreement between repeated symbols
is evidence the pixels really support.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .components import Component


@dataclass
class Slot:
    """One laid-out character and the single ink component it covers."""

    label: int
    run: object
    character: str
    component: Component


def _normalise(mask: np.ndarray, size: int = 96) -> np.ndarray | None:
    ys, xs = np.nonzero(mask > 127)
    if xs.size == 0:
        return None
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA) > 127


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    a, b = _normalise(first), _normalise(second)
    if a is None or b is None:
        return 0.0
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def congruence(first: np.ndarray, second: np.ndarray, tolerance: int = 3) -> float:
    """Shape agreement that tolerates a pixel or two of registration error.

    Plain overlap collapses for thin outlines: two rings traced from the same
    drawing score 0.83, because shifting a ring by a pixel moves nearly all of
    its ink. Allowing a small slack, and taking the worse of the two coverages
    so that a blob cannot satisfy a ring, holds up for outlines and solids
    alike.
    """
    a, b = _normalise(first), _normalise(second)
    if a is None or b is None:
        return 0.0
    solid_a = (a * np.uint8(255)).astype(np.uint8)
    solid_b = (b * np.uint8(255)).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    near_a = cv2.dilate(solid_a, kernel)
    near_b = cv2.dilate(solid_b, kernel)

    count_a = max(1, int(np.count_nonzero(solid_a)))
    count_b = max(1, int(np.count_nonzero(solid_b)))
    recall = np.count_nonzero(cv2.bitwise_and(solid_a, near_b)) / count_a
    precision = np.count_nonzero(cv2.bitwise_and(solid_b, near_a)) / count_b
    return float(min(recall, precision))


def cluster(slots: list[Slot], *, threshold: float = 0.72, aspect_tolerance: float = 0.3) -> list[list[int]]:
    """Complete-link grouping, so every member resembles every other member.

    Single-link would chain a delta to a zero through whatever lies between
    them; complete-link keeps a cluster to symbols that all agree.
    """
    count = len(slots)
    if count < 2:
        return [[index] for index in range(count)]

    scores = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            first, second = slots[i].component, slots[j].component
            ratio = first.aspect / max(1e-6, second.aspect)
            if not (1.0 - aspect_tolerance <= ratio <= 1.0 + aspect_tolerance):
                continue
            scores[i, j] = scores[j, i] = similarity(first.mask, second.mask)

    groups = [[index] for index in range(count)]
    while True:
        best_value, best_pair = threshold, None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                weakest = min(scores[a, b] for a in groups[i] for b in groups[j])
                if weakest >= best_value:
                    best_value, best_pair = weakest, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        groups[i] = groups[i] + groups[j]
        del groups[j]
    return groups


def reconcile(
    slots: list[Slot], *, minimum_members: int = 4, majority: float = 0.6, minimum_support: int = 3
) -> list[tuple[int, str]]:
    """Force every member of a shape cluster to the reading most of them got.

    Returns one (label, description) pair per rewritten glyph.
    """
    changes: list[tuple[int, str]] = []
    for group in cluster(slots):
        if len(group) < minimum_members:
            continue
        characters = [slots[index].character for index in group]
        counts = Counter(characters)
        winner, hits = counts.most_common(1)[0]
        support = hits / len(group)
        if support < majority or hits < minimum_support or len(counts) == 1:
            continue

        for index in group:
            slot = slots[index]
            if slot.character == winner:
                continue
            changes.append(
                (
                    slot.label,
                    f"{slot.character!r} agreed as {winner!r} with "
                    f"{hits} of {len(group)} matching symbols",
                )
            )
            slot.run.text = winner
            slot.character = winner
    return changes
