"""Trace skeleton branches through junctions without changing their connectivity.

The ink is thinned to a one-pixel skeleton and read as a graph: pixels with
three or more neighbours are junctions, pixels with one neighbour are ends,
and the runs of two-neighbour pixels between them are branches. Each branch
is then paired, at every node, with the branch that continues it, and the
chains of paired branches are the strokes.

Three corrections exist because the raw graph misreads common figures:

* Junction merging. Thinning does not reduce a crossing of two lines to one
  junction pixel unless they cross at a right angle: two 5 px lines crossing
  at 60/45/30/20 degrees leave two degree-3 pixels joined by a bridge of
  6.4/10.4/20.2/34.7 px (roughly pen / sin(angle / 2)). Read as two
  junctions, each bridge end pairs with one arm per side and the other arm
  is orphaned, so ``strokes`` returned 3 cut, kinked strokes for one crossing
  at pens 3, 5 and 8 alike, and 255 strokes for a 15 x 15 grid of 3 px lines
  at +/-30 degrees whose orthogonal 20 x 20 counterpart correctly gave 40.
  Two junction groups are therefore merged, with the bridge pixels, when the
  bridge is short against the pen (at most ``shortest / 2``; production
  passes 3 x pen as ``shortest``) or when the two arms at each end continue
  the two arms at the other end straight through the bridge (anti-parallel
  far directions, cross-chords through the bridge midpoint that stay on
  ink), which is a crossing and never a grid cell edge. The merged centre is
  the mean of the merged pixels, which lands on the true crossing point, and
  an arm's first pen beyond the merged cluster, where thinning bends it into
  the old junction pixel, is skipped so the stroke runs straight.

* Loop-first pairing. Pairing by tangent alignment alone opens closed
  outlines whenever a third branch meets the outline where the outline's own
  tangent is within 120 degrees of the newcomer: a 3 px rectangle with a
  45-degree tail from one corner came back as one open 875-point stroke, and
  a 3 px circle with a 3 px line tangent to it came back as an open arc
  while the line ran straight through the 52 px run the two share. A branch
  that starts and ends at the same node is a loop and is closed first,
  regardless of alignment. Two branches that join the same pair of nodes and
  continue each other at both nodes (a smooth outline that shares a run with
  another stroke) are likewise closed first; a leftover branch whose
  continuation was taken by such a closure continues through a copy of the
  shared run, so the circle closes and the tangent line still comes back
  whole.

* Geometric lengths throughout. A stroke, a branch or a bare cycle is kept
  when its path length reaches ``shortest``, never by counting its pixels: a
  24-pixel diamond ring is 33.9 px of path and was dropped at
  ``shortest = 30`` while an open stroke of the same path length was kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

Pixel = tuple[int, int]
Incident = tuple[int, int]

ALIGNED = -0.5
"""Dot product below which two departures from a node continue each other."""

ANTIPARALLEL = -0.8
"""Dot product below which two arms are the two halves of one straight stroke."""


@dataclass
class _Branch:
    """A run of skeleton pixels from one node centre to another."""

    start: int
    end: int
    points: np.ndarray
    pixels: list[Pixel]

    @property
    def length(self) -> float:
        return _path_length(self.points)


def _path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _skeleton_graph(mask: np.ndarray) -> tuple[set[Pixel], dict[Pixel, list[Pixel]]]:
    """Return the skeleton pixels and their neighbours, dropping diagonal steps
    that an orthogonal step already covers so a staircase is a chain."""
    pixels = {(int(row), int(column)) for row, column in np.argwhere(skeletonize(mask > 0))}
    adjacent: dict[Pixel, list[Pixel]] = {}
    for row, column in sorted(pixels):
        neighbours = []
        for row_step in (-1, 0, 1):
            for column_step in (-1, 0, 1):
                candidate = (row + row_step, column + column_step)
                if candidate == (row, column) or candidate not in pixels:
                    continue
                if row_step and column_step and (
                    (row + row_step, column) in pixels or (row, column + column_step) in pixels
                ):
                    continue
                neighbours.append(candidate)
        adjacent[row, column] = neighbours
    return pixels, adjacent


def _junction_groups(pixels: set[Pixel], adjacent: dict[Pixel, list[Pixel]]) -> tuple[list[list[Pixel]], int]:
    """Group adjacent junction pixels, then append every end pixel as its own
    node; the count says how many leading groups are junctions."""
    junctions = {pixel for pixel in pixels if len(adjacent[pixel]) > 2}
    groups: list[list[Pixel]] = []
    while junctions:
        pending = [min(junctions)]
        junctions.remove(pending[0])
        group = []
        while pending:
            pixel = pending.pop()
            group.append(pixel)
            for neighbour in adjacent[pixel]:
                if neighbour in junctions:
                    junctions.remove(neighbour)
                    pending.append(neighbour)
        groups.append(group)
    junction_count = len(groups)
    groups.extend([pixel] for pixel in sorted(pixels) if len(adjacent[pixel]) <= 1)
    return groups, junction_count


def _edge(first: Pixel, second: Pixel) -> tuple[Pixel, Pixel]:
    return tuple(sorted((first, second)))  # type: ignore[return-value]


def _trimmed(pixels: list[Pixel], centre: np.ndarray, radius: float, from_end: bool) -> list[Pixel]:
    """Drop the pixels at one end of a run that lie within ``radius`` of the
    node centre it leaves from, leaving the rest in order."""
    ordered = pixels[::-1] if from_end else pixels
    kept = 0
    while kept < len(ordered) and float(np.linalg.norm(np.asarray(ordered[kept], dtype=float) - centre)) <= radius:
        kept += 1
    remaining = ordered[kept:]
    return remaining[::-1] if from_end else remaining


def _walk_branches(
    groups: list[list[Pixel]],
    membership: dict[Pixel, int],
    centres: list[np.ndarray],
    radii: list[float],
    adjacent: dict[Pixel, list[Pixel]],
    shortest: float,
) -> tuple[list[_Branch], set[tuple[Pixel, Pixel]]]:
    """Follow every run of free pixels from one node to the next.

    A branch keeps its whole pixel run (a bridge's pixels join the merged
    group) but its polyline skips the pixels within ``radii`` of either node
    centre: thinning bends an arm into the junction pixel over the first pen
    beyond the junction cluster (a 20-degree crossing of 8 px lines put the
    diagonal 4 px off its chord there and under 1 px everywhere else), so
    the polyline runs straight from the centre to the first undistorted
    pixel instead.

    A short run that ends at a free end and starts at a junction is a thinning
    spur, not a stroke, and is dropped.
    """
    visited: set[tuple[Pixel, Pixel]] = set()
    branches: list[_Branch] = []
    for node, group in enumerate(groups):
        for start in sorted(group):
            for neighbour in adjacent[start]:
                if membership.get(neighbour) == node or _edge(start, neighbour) in visited:
                    continue
                pixels: list[Pixel] = []
                previous, current = start, neighbour
                visited.add(_edge(previous, current))
                while current not in membership:
                    pixels.append(current)
                    following = next(candidate for candidate in adjacent[current] if candidate != previous)
                    visited.add(_edge(current, following))
                    previous, current = current, following
                target = membership[current]
                undistorted = _trimmed(pixels, centres[node], radii[node], from_end=False)
                undistorted = _trimmed(undistorted, centres[target], radii[target], from_end=True)
                points = np.asarray([centres[node], *undistorted, centres[target]], dtype=float)
                branch = _Branch(node, target, points, pixels)
                terminal = len(adjacent[start]) == 1 or len(adjacent[current]) == 1
                if terminal and branch.length < shortest and (len(groups[node]) > 1 or len(adjacent[start]) > 2
                                                              or len(groups[target]) > 1 or len(adjacent[current]) > 2):
                    continue
                branches.append(branch)
    return branches, visited


def _trim_radii(groups: list[list[Pixel]], junction_count: int, centres: list[np.ndarray], shortest: float) -> list[float]:
    """How far from each junction centre an arm is still thinning's, not the
    line's: the group's own extent plus one pen (``shortest`` is 3 pens).
    End nodes trim nothing."""
    radii = []
    for node, group in enumerate(groups):
        if node >= junction_count:
            radii.append(0.0)
            continue
        extent = float(np.max(np.linalg.norm(np.asarray(group, dtype=float) - centres[node], axis=1)))
        radii.append(extent + shortest / 3)
    return radii


def _heading(points: np.ndarray, shortest: float) -> np.ndarray:
    """Unit direction a branch leaves its first point in, read over a few
    pixels so a staircase does not decide it."""
    distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    reach = min(len(points) - 1, int(np.searchsorted(distances, max(5.0, shortest / 2))) + 1)
    tangent = points[reach] - points[0]
    return tangent / max(1e-9, float(np.linalg.norm(tangent)))


def _oriented(branches: list[_Branch], incident: Incident) -> np.ndarray:
    index, endpoint = incident
    return branches[index].points[::(-1 if endpoint else 1)]


def _incidents(branches: list[_Branch], node_count: int) -> dict[int, list[Incident]]:
    incidents: dict[int, list[Incident]] = {node: [] for node in range(node_count)}
    for index, branch in enumerate(branches):
        incidents[branch.start].append((index, 0))
        incidents[branch.end].append((index, 1))
    return incidents


def _point_at_distance(points: np.ndarray, distance: float) -> np.ndarray:
    """The point reached after ``distance`` along the polyline, or its last
    point when the polyline is shorter."""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    travelled = np.concatenate(([0.0], np.cumsum(steps)))
    if distance >= travelled[-1]:
        return points[-1]
    index = int(np.searchsorted(travelled, distance, side='right'))
    fraction = (distance - travelled[index - 1]) / max(1e-9, float(steps[index - 1]))
    return points[index - 1] + fraction * (points[index] - points[index - 1])


def _chord_in_ink(ink: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    """Does the straight segment between two skeleton points stay on ink?

    ``ink`` is the mask grown by one pixel, because the skeleton of a 3 px
    line wobbles by a pixel around the true centre and a chord between two
    of its points may graze the edge pixel of the line it belongs to.
    """
    count = int(np.ceil(float(np.linalg.norm(second - first)))) + 1
    samples = np.rint(np.linspace(first, second, count)).astype(int)
    inside = ((samples[:, 0] >= 0) & (samples[:, 0] < ink.shape[0])
              & (samples[:, 1] >= 0) & (samples[:, 1] < ink.shape[1]))
    return bool(inside.all() and ink[samples[:, 0], samples[:, 1]].all())


def _is_split_crossing(
    branches: list[_Branch],
    incidents: dict[int, list[Incident]],
    centres: list[np.ndarray],
    ink: np.ndarray,
    bridge: int,
    shortest: float,
) -> bool:
    """Do the two outer arms at each end of this bridge pair up into two
    straight strokes that run through it?

    Thinning leaves a crossing of two lines as two degree-3 pixels whose
    bridge grows as roughly pen / sin(angle / 2): for 5 px lines it measures
    6.4 px at 60 degrees, 10.4 at 45, 20.2 at 30 and 34.7 at 20, so length
    alone cannot tell an acute crossing from a genuine segment between two
    junctions (a 40 px cell edge of a grid). The arms can. Each arm is read
    at a far point past the junction distortion, ``shortest`` beyond the
    bridge's own length, because its first pixels bend into the junction
    pixel (3 px off the true line at 20 degrees) and a heading taken there
    is tilted. A start arm continues an end arm when the two far directions
    are anti-parallel, the chord between the far points passes within half
    of ``shortest`` of the bridge's midpoint, and that chord lies on ink the
    whole way: at a grid cell edge the two perpendicular arms are parallel,
    and at a circle cut by a chord the cross-chord runs through the
    background, so neither merges. Both cross-pairings must continue for
    the bridge to be one crossing.
    """
    branch = branches[bridge]
    reach = branch.length + shortest
    arms: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for node in (branch.start, branch.end):
        others = [incident for incident in incidents[node] if incident[0] != bridge]
        if len(others) != 2:
            return False
        arms[node] = []
        for incident in others:
            far = _point_at_distance(_oriented(branches, incident), reach)
            direction = far - centres[node]
            arms[node].append((far, direction / max(1e-9, float(np.linalg.norm(direction)))))
    midpoint = _point_at_distance(branch.points, branch.length / 2)

    def continues(near: tuple[np.ndarray, np.ndarray], far: tuple[np.ndarray, np.ndarray]) -> bool:
        if float(np.dot(near[1], far[1])) > ANTIPARALLEL:
            return False
        chord = far[0] - near[0]
        toward_midpoint = midpoint - near[0]
        offset = (abs(float(chord[0] * toward_midpoint[1] - chord[1] * toward_midpoint[0]))
                  / max(1e-9, float(np.linalg.norm(chord))))
        return offset <= shortest / 2 and _chord_in_ink(ink, near[0], far[0])

    first, second = arms[branch.start], arms[branch.end]
    return any(continues(first[0], second[one]) and continues(first[1], second[1 - one]) for one in (0, 1))


def _merge_candidates(
    branches: list[_Branch],
    junction_count: int,
    centres: list[np.ndarray],
    ink: np.ndarray,
    shortest: float,
) -> list[int]:
    """Indices of bridges whose two junction groups are one crossing.

    A bridge no longer than half of ``shortest`` (1.5 pens in production)
    cannot be a segment between two separate junctions and merges on its
    own; any longer bridge must pass the arm test. Short bridges merge in a
    pass of their own so that, in a dense grid, every crossing is one
    four-arm node before its cell edges are examined.
    """
    incidents = _incidents(branches, len(centres))
    between_junctions = [index for index, branch in enumerate(branches)
                         if branch.start != branch.end and branch.start < junction_count
                         and branch.end < junction_count]
    short = [index for index in between_junctions if branches[index].length <= shortest / 2]
    if short:
        return short
    return [index for index in between_junctions
            if _is_split_crossing(branches, incidents, centres, ink, index, shortest)]


def _merged_groups(
    groups: list[list[Pixel]], junction_count: int, branches: list[_Branch], bridges: list[int]
) -> tuple[list[list[Pixel]], int]:
    """Union the junction groups each bridge joins, folding the bridge pixels in."""
    parent = list(range(junction_count))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    extra: dict[int, list[Pixel]] = {}
    for index in bridges:
        branch = branches[index]
        first, second = root(branch.start), root(branch.end)
        if first != second:
            parent[max(first, second)] = min(first, second)
        extra.setdefault(root(branch.start), []).extend(branch.pixels)
    merged: dict[int, list[Pixel]] = {}
    for node in range(junction_count):
        merged.setdefault(root(node), []).extend(groups[node])
    for node, pixels in extra.items():
        merged[root(node)].extend(pixels)
    junction_groups = sorted(merged.values(), key=min)
    return junction_groups + groups[junction_count:], len(junction_groups)


def _pair_incidents(branches: list[_Branch], node_count: int, shortest: float) -> dict[Incident, Incident]:
    """Decide, at every node, which branch continues which.

    Loops close first, then the remaining incidents pair by alignment, and a
    leftover whose continuation a closure took runs through a copy of it.
    ``branches`` grows by those copies.
    """
    incidents = _incidents(branches, node_count)
    headings = {incident: _heading(_oriented(branches, incident), shortest)
                for members in incidents.values() for incident in members}

    def alignment(first: Incident, second: Incident) -> float:
        return float(np.dot(headings[first], headings[second]))

    paired: dict[Incident, Incident] = {}

    def pair(first: Incident, second: Incident) -> None:
        paired[first] = second
        paired[second] = first

    lens_branches: set[int] = set()
    by_nodes: dict[frozenset[int], list[int]] = {}
    for index, branch in enumerate(branches):
        if branch.start == branch.end:
            pair((index, 0), (index, 1))
        else:
            by_nodes.setdefault(frozenset((branch.start, branch.end)), []).append(index)
    lens_candidates = []
    for members in by_nodes.values():
        for first, second in combinations(members, 2):
            at_start = alignment((first, 0), (second, 0 if branches[second].start == branches[first].start else 1))
            at_end = alignment((first, 1), (second, 1 if branches[second].end == branches[first].end else 0))
            if at_start <= ALIGNED and at_end <= ALIGNED:
                lens_candidates.append((at_start + at_end, first, second))
    for _score, first, second in sorted(lens_candidates):
        if any((index, endpoint) in paired for index in (first, second) for endpoint in (0, 1)):
            continue
        for node in (branches[first].start, branches[first].end):
            pair(next(incident for incident in incidents[node] if incident[0] == first),
                 next(incident for incident in incidents[node] if incident[0] == second))
        lens_branches.update((first, second))

    for members in incidents.values():
        candidates = []
        for position, first in enumerate(members):
            for second in members[position + 1:]:
                score = alignment(first, second)
                if score <= ALIGNED:
                    candidates.append((score, first, second))
        for _score, first, second in sorted(candidates):
            if first not in paired and second not in paired:
                pair(first, second)

    requests: dict[int, dict[int, tuple[float, Incident]]] = {}
    for node, members in incidents.items():
        for leftover in members:
            if leftover in paired:
                continue
            options = [(alignment(leftover, taken), taken) for taken in members
                       if taken[0] in lens_branches and alignment(leftover, taken) <= ALIGNED]
            if not options:
                continue
            score, taken = min(options)
            best = requests.setdefault(taken[0], {}).get(node)
            if best is None or score < best[0]:
                requests[taken[0]][node] = (score, leftover)
    for index, by_node in requests.items():
        shared = branches[index]
        copy = len(branches)
        branches.append(_Branch(shared.start, shared.end, shared.points.copy(), list(shared.pixels)))
        for node, (_score, leftover) in by_node.items():
            pair(leftover, (copy, 0 if node == shared.start else 1))
    return paired


def _assemble(branches: list[_Branch], paired: dict[Incident, Incident], shortest: float) -> list[np.ndarray]:
    """Chain paired branches into strokes, starting from free ends so an open
    stroke begins where it begins; a loop is entered anywhere and returns."""
    result = []
    consumed: set[int] = set()
    starts = [(index, endpoint) for index in range(len(branches)) for endpoint in (0, 1)]
    starts.sort(key=lambda incident: (incident in paired, incident))
    for start in starts:
        if start[0] in consumed:
            continue
        current = start
        parts: list[np.ndarray] = []
        while current[0] not in consumed:
            index, endpoint = current
            consumed.add(index)
            points = _oriented(branches, current)
            parts.append(points if not parts else points[1:])
            following = paired.get((index, 1 - endpoint))
            if following is None:
                break
            current = following
        points = np.vstack(parts)
        if _path_length(points) >= shortest:
            result.append(points)
    return result


def _cycles(
    pixels: set[Pixel],
    adjacent: dict[Pixel, list[Pixel]],
    membership: dict[Pixel, int],
    visited: set[tuple[Pixel, Pixel]],
    shortest: float,
) -> list[np.ndarray]:
    """Closed runs of pixels that touch no node at all, such as a lone circle."""
    result = []
    for start in sorted(pixels):
        for neighbour in adjacent[start]:
            if start in membership or _edge(start, neighbour) in visited:
                continue
            points = [start]
            previous, current = start, neighbour
            visited.add(_edge(previous, current))
            while current != start:
                points.append(current)
                candidates = [candidate for candidate in adjacent[current]
                              if candidate != previous and _edge(current, candidate) not in visited]
                if not candidates:
                    break
                following = candidates[0]
                visited.add(_edge(current, following))
                previous, current = current, following
            if current == start:
                points.append(start)
            array = np.asarray(points, dtype=float)
            if len(array) >= 3 and _path_length(array) >= shortest:
                result.append(array)
    return result


def strokes(mask: np.ndarray, origin: tuple[int, int], shortest: float) -> list[np.ndarray]:
    """Strokes of the ink in ``mask`` as (x, y) point arrays offset by ``origin``.

    ``shortest`` is the shortest stroke worth returning (3 x pen in
    production); it also scales the junction-merge and tangent reach.
    """
    pixels, adjacent = _skeleton_graph(mask)
    ink = binary_dilation(mask > 0, np.ones((3, 3), dtype=bool))
    groups, junction_count = _junction_groups(pixels, adjacent)
    for _pass in range(junction_count + 1):
        membership = {pixel: index for index, group in enumerate(groups) for pixel in group}
        centres = [np.mean(np.asarray(group, dtype=float), axis=0) for group in groups]
        radii = _trim_radii(groups, junction_count, centres, shortest)
        branches, visited = _walk_branches(groups, membership, centres, radii, adjacent, shortest)
        bridges = _merge_candidates(branches, junction_count, centres, ink, shortest)
        if not bridges:
            break
        groups, junction_count = _merged_groups(groups, junction_count, branches, bridges)

    paired = _pair_incidents(branches, len(groups), shortest)
    result = _assemble(branches, paired, shortest)
    result.extend(_cycles(pixels, adjacent, membership, visited, shortest))

    offset = np.asarray(origin, dtype=float)
    return [points[:, ::-1] + offset for points in result]
