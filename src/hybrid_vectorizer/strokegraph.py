"""Trace skeleton branches through junctions without changing their connectivity."""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def strokes(mask: np.ndarray, origin: tuple[int, int], shortest: float) -> list[np.ndarray]:
    pixels = {tuple(position) for position in np.argwhere(skeletonize(mask > 0))}
    adjacent = {}
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

    junctions = {pixel for pixel in pixels if len(adjacent[pixel]) > 2}
    groups = []
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
    groups.extend([pixel] for pixel in sorted(pixels) if len(adjacent[pixel]) <= 1)
    membership = {pixel: index for index, group in enumerate(groups) for pixel in group}
    centres = [np.mean(group, axis=0) for group in groups]
    visited = set()
    branches = []

    def edge(first, second):
        return tuple(sorted((first, second)))

    for node, group in enumerate(groups):
        for start in sorted(group):
            for neighbour in adjacent[start]:
                if membership.get(neighbour) == node or edge(start, neighbour) in visited:
                    continue
                points = [centres[node]]
                previous, current = start, neighbour
                visited.add(edge(previous, current))
                while current not in membership:
                    points.append(current)
                    following = next(candidate for candidate in adjacent[current] if candidate != previous)
                    visited.add(edge(current, following))
                    previous, current = current, following
                target = membership[current]
                points.append(centres[target])
                points = np.asarray(points, dtype=float)
                length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                terminal = len(adjacent[start]) == 1 or len(adjacent[current]) == 1
                if terminal and length < shortest and (len(groups[node]) > 1 or len(adjacent[start]) > 2
                                                       or len(groups[target]) > 1 or len(adjacent[current]) > 2):
                    continue
                branches.append((node, target, points))

    incidents = {node: [] for node in range(len(groups))}
    for index, (start, end, points) in enumerate(branches):
        incidents[start].append((index, 0))
        incidents[end].append((index, 1))

    def direction(incident):
        index, endpoint = incident
        points = branches[index][2][::(-1 if endpoint else 1)]
        distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        reach = min(len(points) - 1, int(np.searchsorted(distances, max(5.0, shortest / 2))) + 1)
        tangent = points[reach] - points[0]
        return tangent / max(1e-9, float(np.linalg.norm(tangent)))

    paired = {}
    for members in incidents.values():
        candidates = []
        for position, first in enumerate(members):
            for second in members[position + 1:]:
                alignment = float(np.dot(direction(first), direction(second)))
                if alignment <= -0.5:
                    candidates.append((alignment, first, second))
        for alignment, first, second in sorted(candidates):
            if first not in paired and second not in paired:
                paired[first] = second
                paired[second] = first

    result = []
    consumed = set()
    starts = [(index, endpoint) for index in range(len(branches)) for endpoint in (0, 1)]
    starts.sort(key=lambda incident: (incident in paired, incident))
    for start in starts:
        if start[0] in consumed:
            continue
        current = start
        parts = []
        while current[0] not in consumed:
            index, endpoint = current
            consumed.add(index)
            points = branches[index][2][::(-1 if endpoint else 1)]
            parts.append(points if not parts else points[1:])
            following = paired.get((index, 1 - endpoint))
            if following is None:
                break
            current = following
        points = np.vstack(parts)
        if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() >= shortest:
            result.append(points)

    for start in sorted(pixels):
        for neighbour in adjacent[start]:
            if start in membership or edge(start, neighbour) in visited:
                continue
            points = [start]
            previous, current = start, neighbour
            visited.add(edge(previous, current))
            while current != start:
                points.append(current)
                candidates = [candidate for candidate in adjacent[current]
                              if candidate != previous and edge(current, candidate) not in visited]
                if not candidates:
                    break
                following = candidates[0]
                visited.add(edge(current, following))
                previous, current = current, following
            if current == start:
                points.append(start)
            if len(points) >= max(3, shortest):
                result.append(np.asarray(points, dtype=float))

    offset = np.asarray(origin, dtype=float)
    return [points[:, ::-1] + offset for points in result]
