"""Exact sequential Born sampling of locally Z-rotated graph states."""

from __future__ import annotations

import numpy as np

from ._compat import strict_zip


class GraphStateSampler:
    def __init__(
        self, adjacency: tuple[int, ...], rotations: tuple[int, ...], theta: float,
        max_frontier: int = 20,
        *, rotation_angles: tuple[float, ...] | None = None,
    ) -> None:
        if rotation_angles is None:
            rotation_angles = (theta,) * len(rotations)
        if len(rotation_angles) != len(rotations):
            raise ValueError("rotation_angles must contain one angle per rotation")
        angles = dict(strict_zip(rotations, rotation_angles))
        self.steps = []
        frontier: tuple[int, ...] = ()
        self.width = 0
        for site, neighbours in enumerate(adjacency):
            future = tuple(q for q in range(site + 1, len(adjacency)) if (neighbours >> q) & 1)
            retained = tuple(q for q in frontier if q != site)
            following = tuple(sorted(set(retained) | set(future)))
            self.width = max(self.width, len(following))
            if len(following) > max_frontier:
                raise ValueError(
                    f"Exact graph sampling needs frontier width {len(following)}; "
                    f"the configured limit is {max_frontier}. Choose a narrower grid."
                )
            index = np.arange(1 << len(retained), dtype=np.int64)
            embedding = np.zeros(len(index), dtype=np.int64)
            for j, q in enumerate(retained):
                embedding |= ((index >> j) & 1) << following.index(q)
            toggle = sum(1 << following.index(q) for q in future)
            position = frontier.index(site) if site in frontier else None
            angle = angles.get(site, 0.0)
            self.steps.append((position, embedding, embedding ^ toggle, len(following), angle))
            frontier = following

    def sample(self, basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        amplitude = np.ones(1, dtype=complex)
        outcomes = np.empty(len(self.steps), dtype=np.int8)
        root_two = np.sqrt(2.0)
        for site, (position, embedding, flipped, width, angle) in enumerate(self.steps):
            if position is None:
                plus = minus = amplitude
            else:
                shaped = amplitude.reshape(-1, 2, 1 << position)
                zero, one = shaped[:, 0].reshape(-1), shaped[:, 1].reshape(-1)
                plus, minus = zero + one, zero - one
            physical_zero = np.zeros(1 << width, dtype=complex)
            physical_one = np.zeros(1 << width, dtype=complex)
            physical_zero[embedding] = plus * (np.exp(1j * angle) / root_two)
            physical_one[flipped] = minus * (np.exp(-1j * angle) / root_two)
            axis = int(basis[site])
            if axis == 2:
                first, second = physical_zero, physical_one
            else:
                transverse = physical_one if axis == 0 else -1j * physical_one
                first = (physical_zero + transverse) / root_two
                second = (physical_zero - transverse) / root_two
            p_first = float(np.vdot(first, first).real)
            p_second = float(np.vdot(second, second).real)
            choose_first = rng.random() < p_first / (p_first + p_second)
            probability = p_first if choose_first else p_second
            amplitude = (first if choose_first else second) / np.sqrt(probability)
            outcomes[site] = 1 if choose_first else -1
        return outcomes
