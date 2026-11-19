"""Reproducible local rotation angles, independent of site and measurement RNGs."""

from __future__ import annotations

import numpy as np


def validate_theta_range(theta_range: tuple[float, float] | None) -> tuple[float, float] | None:
    if theta_range is None:
        return None
    message = "theta_range must contain two finite angles (min, max) with min <= max"
    try:
        bounds = np.asarray(theta_range, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(message)
    lower, upper = map(float, bounds)
    if not np.isfinite(upper - lower):
        raise ValueError("theta_range width must be finite")
    return lower, upper


def sample_rotation_angles(
    count: int, theta: float, theta_range: tuple[float, float] | None, seed: int,
) -> tuple[float, ...]:
    bounds = validate_theta_range(theta_range)
    if bounds is None:
        return (float(theta),) * count

    rng = np.random.default_rng([seed, 0x414E474C])
    return tuple(float(angle) for angle in rng.uniform(*bounds, size=count))
