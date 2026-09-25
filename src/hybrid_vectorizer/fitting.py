"""Fitting traced pixels with analytic models and cubic Bezier paths."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class Model:
    """An analytic description of a traced curve, kept for the SVG metadata."""

    name: str
    parameters: dict[str, float]
    rms: float
    description: str = ""

    def sample(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Polynomial(Model):
    coefficients: np.ndarray = field(default_factory=lambda: np.zeros(1))
    centre: float = 0.0
    scale: float = 1.0

    def sample(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.coefficients, (np.asarray(x) - self.centre) / self.scale)


@dataclass
class Sinusoid(Model):
    def sample(self, x: np.ndarray) -> np.ndarray:
        p = self.parameters
        omega = 2.0 * np.pi / p["period"]
        return p["offset"] + p["cosine"] * np.cos(omega * x) + p["sine"] * np.sin(omega * x)


def _rms(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> Polynomial | None:
    """Fit on a centred, scaled abscissa.

    Fitting a fifth-degree curve directly against pixel coordinates in the
    thousands raises x to the fifteenth power inside the normal equations, which
    numpy rightly calls poorly conditioned. Moving x to roughly [-1, 1] first
    costs nothing and makes the fit mean what it says.

    A curve of degree d also needs d + 1 abscissae that differ. A stroke
    standing on end offers one, however many points are on it, and no amount of
    scaling makes that fit anything: it is refused rather than fitted badly.
    """
    if x.size <= degree + 1:
        return None
    centre = float(np.mean(x))
    scale = float(np.max(np.abs(x - centre))) or 1.0
    reduced = (x - centre) / scale
    if np.unique(np.round(reduced, 6)).size <= degree + 1:
        return None

    coefficients = np.polyfit(reduced, y, degree)
    residual = np.polyval(coefficients, reduced) - y
    name = {1: "line", 2: "parabola"}.get(degree, f"polynomial-{degree}")
    return Polynomial(
        name=name,
        parameters={"degree": float(degree)},
        rms=_rms(residual),
        description=" + ".join(
            f"{c:.6g}t^{degree - i}" for i, c in enumerate(coefficients)
        )
        + f"  where t = (x - {centre:.6g}) / {scale:.6g}",
        coefficients=coefficients,
        centre=centre,
        scale=scale,
    )


def _sinusoid_residual(x: np.ndarray, y: np.ndarray, period: float) -> tuple[float, np.ndarray]:
    omega = 2.0 * np.pi / period
    design = np.column_stack([np.ones_like(x), np.cos(omega * x), np.sin(omega * x)])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return _rms(design @ solution - y), solution


def fit_sinusoid(
    x: np.ndarray, y: np.ndarray, *, minimum_period: float = 8.0
) -> Sinusoid | None:
    """Recover a periodic trace: the period by spectrum, the rest in closed form."""
    span = float(x.max() - x.min())
    if x.size < 16 or span <= 2.0 * minimum_period:
        return None

    grid = np.linspace(x.min(), x.max(), max(256, int(span)))
    resampled = np.interp(grid, x, y)
    centred = resampled - resampled.mean()
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(centred.size)))
    spectrum[0] = 0.0
    step = grid[1] - grid[0]
    frequencies = np.fft.rfftfreq(centred.size, d=step)

    peak = int(np.argmax(spectrum))
    if frequencies[peak] <= 0:
        return None
    seed = 1.0 / frequencies[peak]

    # The spectrum locates the period to within a bin, a scan to within a step,
    # and a bracketed minimisation to the precision the trace actually supports.
    lower = max(minimum_period, 0.6 * seed)
    upper = 1.6 * seed
    grid = np.linspace(lower, upper, 200)
    scores = np.array([_sinusoid_residual(x, y, period)[0] for period in grid])
    index = int(np.argmin(scores))
    step = grid[1] - grid[0]
    result = minimize_scalar(
        lambda period: _sinusoid_residual(x, y, period)[0],
        bounds=(max(lower, grid[index] - step), min(upper, grid[index] + step)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    period = float(result.x)
    best_rms, solution = _sinusoid_residual(x, y, period)
    offset, cosine, sine = (float(v) for v in solution)
    amplitude = float(np.hypot(cosine, sine))
    phase = float(np.arctan2(-sine, cosine))
    return Sinusoid(
        name="sinusoid",
        parameters={
            "period": float(period),
            "offset": offset,
            "cosine": cosine,
            "sine": sine,
            "amplitude": amplitude,
            "phase": phase,
        },
        rms=float(best_rms),
        description=(
            f"y = {offset:.4g} + {amplitude:.4g}*cos(2*pi*(x - {phase * period / (2 * np.pi):.4g})"
            f"/{period:.4g})"
        ),
    )


def choose_model(x: np.ndarray, y: np.ndarray, tolerance: float) -> Model | None:
    """Best analytic family within tolerance, preferring the simplest that fits."""
    candidates: list[Model] = []
    for degree in (1, 2, 3, 4, 5):
        polynomial = fit_polynomial(x, y, degree)
        if polynomial is not None:
            candidates.append(polynomial)
    sinusoid = fit_sinusoid(x, y)
    if sinusoid is not None:
        candidates.append(sinusoid)

    acceptable = [model for model in candidates if model.rms <= tolerance]
    if not acceptable:
        return None
    complexity = {"line": 0, "parabola": 1, "sinusoid": 2}
    return min(
        acceptable,
        key=lambda model: (complexity.get(model.name, 3 + int(model.parameters.get("degree", 5))), model.rms),
    )


# --- Cubic Bezier fitting (Schneider, "Graphics Gems", 1990) -------------------


def _bezier(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    s = 1.0 - t
    return (
        (s**3)[:, None] * control[0]
        + (3.0 * s**2 * t)[:, None] * control[1]
        + (3.0 * s * t**2)[:, None] * control[2]
        + (t**3)[:, None] * control[3]
    )


def _bezier_derivative(control: np.ndarray, t: float) -> np.ndarray:
    s = 1.0 - t
    return (
        3.0 * s * s * (control[1] - control[0])
        + 6.0 * s * t * (control[2] - control[1])
        + 3.0 * t * t * (control[3] - control[2])
    )


def _chord_parameters(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    if cumulative[-1] <= 0:
        return np.linspace(0.0, 1.0, points.shape[0])
    return cumulative / cumulative[-1]


def _generate(points: np.ndarray, t: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    first, last = points[0], points[-1]
    s = 1.0 - t
    a1 = (3.0 * s * s * t)[:, None] * left
    a2 = (3.0 * s * t * t)[:, None] * right

    c = np.array(
        [[float(np.sum(a1 * a1)), float(np.sum(a1 * a2))],
         [float(np.sum(a1 * a2)), float(np.sum(a2 * a2))]]
    )
    base = points - (
        (s**3)[:, None] * first
        + (3.0 * s * s * t)[:, None] * first
        + (3.0 * s * t * t)[:, None] * last
        + (t**3)[:, None] * last
    )
    rhs = np.array([float(np.sum(a1 * base)), float(np.sum(a2 * base))])

    determinant = c[0, 0] * c[1, 1] - c[0, 1] * c[1, 0]
    chord = float(np.linalg.norm(last - first))
    if abs(determinant) < 1e-12:
        alpha1 = alpha2 = chord / 3.0
    else:
        alpha1 = (rhs[0] * c[1, 1] - rhs[1] * c[0, 1]) / determinant
        alpha2 = (c[0, 0] * rhs[1] - c[0, 1] * rhs[0]) / determinant
        if alpha1 < 1e-6 or alpha2 < 1e-6:
            alpha1 = alpha2 = chord / 3.0
    return np.array([first, first + left * alpha1, last + right * alpha2, last])


def _max_error(points: np.ndarray, control: np.ndarray, t: np.ndarray) -> tuple[float, int]:
    errors = np.linalg.norm(_bezier(control, t) - points, axis=1)
    index = int(np.argmax(errors))
    return float(errors[index]), index


def _reparameterise(points: np.ndarray, control: np.ndarray, t: np.ndarray) -> np.ndarray:
    updated = t.copy()
    for index, parameter in enumerate(t):
        point = _bezier(control, np.array([parameter]))[0]
        first = _bezier_derivative(control, parameter)
        s = 1.0 - parameter
        second = (
            6.0 * s * (control[2] - 2.0 * control[1] + control[0])
            + 6.0 * parameter * (control[3] - 2.0 * control[2] + control[1])
        )
        difference = point - points[index]
        denominator = float(np.dot(first, first) + np.dot(difference, second))
        if abs(denominator) > 1e-12:
            updated[index] = parameter - float(np.dot(difference, first)) / denominator
    return np.clip(updated, 0.0, 1.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.array([1.0, 0.0])


def _fit_recursive(
    points: np.ndarray, left: np.ndarray, right: np.ndarray, tolerance: float, depth: int
) -> list[np.ndarray]:
    if points.shape[0] == 2:
        chord = float(np.linalg.norm(points[1] - points[0])) / 3.0
        return [np.array([points[0], points[0] + left * chord, points[1] + right * chord, points[1]])]

    t = _chord_parameters(points)
    control = _generate(points, t, left, right)
    error, split = _max_error(points, control, t)

    if error < tolerance:
        return [control]

    if error < 4.0 * tolerance and depth < 24:
        for _ in range(16):
            t = _reparameterise(points, control, t)
            control = _generate(points, t, left, right)
            error, split = _max_error(points, control, t)
            if error < tolerance:
                return [control]

    split = int(np.clip(split, 1, points.shape[0] - 2))
    centre = _unit(points[split - 1] - points[split + 1])
    return _fit_recursive(points[: split + 1], left, centre, tolerance, depth + 1) + _fit_recursive(
        points[split:], -centre, right, tolerance, depth + 1
    )


def fit_bezier(points: np.ndarray, tolerance: float) -> list[np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return []
    left = _unit(points[1] - points[0])
    right = _unit(points[-2] - points[-1])
    return _fit_recursive(points, left, right, max(tolerance, 1e-3), 0)


def path_data(segments: list[np.ndarray], precision: int = 2) -> str:
    def number(value: float) -> str:
        return f"{value:.{precision}f}".rstrip("0").rstrip(".") or "0"

    if not segments:
        return ""
    commands = [f"M{number(segments[0][0][0])} {number(segments[0][0][1])}"]
    for control in segments:
        commands.append(
            "C"
            f"{number(control[1][0])} {number(control[1][1])} "
            f"{number(control[2][0])} {number(control[2][1])} "
            f"{number(control[3][0])} {number(control[3][1])}"
        )
    return " ".join(commands)
