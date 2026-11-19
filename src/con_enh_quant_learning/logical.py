"""Reference-only Clifford frames and the O(n) focus used for global magic."""

from __future__ import annotations

import numpy as np

from ._compat import dataclass
from .pauli import Pauli


@dataclass(frozen=True, slots=True)
class FocusObservable:
    pauli: Pauli
    mode: int
    axis: int
    sign: int = 1


def magic_focus(state: str, adjacency: tuple[int, ...]) -> tuple[FocusObservable, ...]:
    """Physical images of X_i,Y_i,Z_i, independent of realized rotations."""
    count = len(adjacency)
    result = []
    for site in range(count):
        bit = 1 << site
        if state == "CLUSTER":
            words = (
                (Pauli(bit, adjacency[site]), 1),
                (Pauli(bit, adjacency[site] | bit), 1),
                (Pauli(0, bit), 1),
            )
        elif state == "GHZ" and site == 0:
            all_sites = (1 << count) - 1
            words = ((Pauli(all_sites), 1), (Pauli(all_sites, 1), 1), (Pauli(0, 1), 1))
        elif state == "GHZ":
            words = ((Pauli(0, 1 | bit), 1), (Pauli(bit, 1 | bit), -1), (Pauli(bit), 1))
        else:
            raise ValueError("Magic focus supports CLUSTER and GHZ")
        result.extend(
            FocusObservable(word, site, axis, sign) for axis, (word, sign) in enumerate(words)
        )
    return tuple(result)


def sample_focus_parities(oracle, observables, rng: np.random.Generator):
    """Simulator-only exact joint law for compatible single-mode focus words."""
    values = {}
    outcomes = []
    for item in observables:
        if item.mode in values:
            axis, outcome = values[item.mode]
            if axis != item.axis:
                raise ValueError("Focus observables on one mode must be compatible")
        else:
            mean = item.sign * oracle.expectation(item.pauli)
            outcome = 1 if rng.random() < (1 + mean) / 2 else -1
            values[item.mode] = (item.axis, outcome)
        outcomes.append(item.sign * outcome)
    return np.asarray(outcomes, dtype=np.int8)
