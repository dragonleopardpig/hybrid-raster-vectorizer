"""Recovering the figure's own alphabet, by solving for every glyph at once.

Deciding one scanned glyph against installed fonts does not work: the typeface
in an old figure is rarely installed, so the score is dominated by how the
candidate font draws in general rather than by which letter was drawn. Asking
the same question of every glyph simultaneously cancels much of that, because a
wrong typeface is wrong for all of them equally. Two constraints do the work:
one font has to explain the whole alphabet, and two shapes that differ cannot be
the same letter.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .consensus import Slot, cluster, similarity
from .fonts import FontFace
from .refine import CONFUSIONS, FontSet, _shape_of, build_font_set


@dataclass
class Cluster:
    """One distinct shape in the figure, and everything the OCR called it."""

    slots: list[Slot]
    exemplar: np.ndarray = field(repr=False)
    proposals: Counter = field(default_factory=Counter)

    @property
    def majority(self) -> str:
        return self.proposals.most_common(1)[0][0]

    @property
    def support(self) -> float:
        return self.proposals.most_common(1)[0][1] / max(1, sum(self.proposals.values()))


def build_clusters(slots: list[Slot], *, threshold: float = 0.72) -> list[Cluster]:
    clusters: list[Cluster] = []
    for group in cluster(slots, threshold=threshold):
        members = [slots[index] for index in group]
        if not members:
            continue
        if len(members) == 1:
            exemplar = members[0].component.mask
        else:
            scores = [
                sum(
                    similarity(one.component.mask, other.component.mask)
                    for other in members
                    if other is not one
                )
                for one in members
            ]
            exemplar = members[int(np.argmax(scores))].component.mask
        clusters.append(
            Cluster(
                slots=members,
                exemplar=exemplar,
                proposals=Counter(member.character for member in members),
            )
        )
    return clusters


def candidate_alphabet(clusters: list[Cluster]) -> list[str]:
    """Every letter the readings propose, plus the letters they confuse with."""
    letters: set[str] = set()
    for entry in clusters:
        for character in entry.proposals:
            letters.add(character)
            for confusion in CONFUSIONS:
                if character in confusion:
                    letters.update(confusion)
    return sorted(letters)


@dataclass
class Solution:
    family: str
    bold: bool
    assignment: dict[int, str]
    score: float
    agreement: float
    changes: list[tuple[int, str]] = field(default_factory=list)


def _score_matrix(clusters: list[Cluster], letters: list[str], fonts: FontSet) -> np.ndarray:
    matrix = np.zeros((len(clusters), len(letters)))
    for column, letter in enumerate(letters):
        shapes = [
            shape
            for shape in (
                _shape_of(str(fonts.upright.path), letter),
                _shape_of(str(fonts.italic.path), letter),
            )
            if shape is not None
        ]
        if not shapes:
            continue
        for row, entry in enumerate(clusters):
            matrix[row, column] = max(similarity(entry.exemplar, shape) for shape in shapes)
    return matrix


def solve(
    clusters: list[Cluster],
    families: list[str],
    *,
    faces: list[FontFace] | None = None,
    bold: bool = True,
) -> Solution | None:
    """Choose the family and the letter for each shape that agree best overall."""
    if not clusters or not families:
        return None
    letters = candidate_alphabet(clusters)
    if len(letters) < len(clusters):
        letters = letters + [""] * (len(clusters) - len(letters))

    best: Solution | None = None
    for family in families:
        fonts = build_font_set(family, bold=bold, faces=faces)
        if fonts is None:
            continue
        matrix = _score_matrix(clusters, letters, fonts)
        rows, columns = linear_sum_assignment(-matrix)
        assignment = {int(r): letters[int(c)] for r, c in zip(rows, columns) if letters[int(c)]}
        total = float(matrix[rows, columns].sum())
        matched = sum(
            1 for index, letter in assignment.items() if letter == clusters[index].majority
        )
        candidate = Solution(
            family=family,
            bold=bold,
            assignment=assignment,
            score=total,
            agreement=matched / max(1, len(assignment)),
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def apply(solution: Solution, clusters: list[Cluster], *, margin: float = 0.0) -> list[tuple[int, str]]:
    """Rewrite readings the joint solution disagrees with."""
    changes: list[tuple[int, str]] = []
    for index, letter in solution.assignment.items():
        entry = clusters[index]
        if not letter or letter == entry.majority:
            continue
        for slot in entry.slots:
            if slot.character == letter:
                continue
            changes.append(
                (slot.label, f"{slot.character!r} solved as {letter!r} with the figure's alphabet")
            )
            slot.run.text = letter
            slot.character = letter
    return changes
