"""Bounded compatible-generator pool shared by both subsystem backends."""

from __future__ import annotations

import numpy as np

from ._compat import bit_count


def generator_cliques(axes: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(axes)
    compatible = [0] * count
    for left in range(count):
        for right in range(count):
            if left != right:
                shared = (axes[left] >= 0) & (axes[right] >= 0)
                if np.all(axes[left][shared] == axes[right][shared]):
                    compatible[left] |= 1 << right
    cliques = [0]
    for index in range(count):
        cliques += [
            clique | (1 << index) for clique in cliques if not (clique & ~compatible[index])
        ]
        if len(cliques) > budget:


            mandatory = [0] + [1 << site for site in range(index + 1)]
            mandatory_set = set(mandatory)
            extra = sorted(
                (clique for clique in cliques if clique not in mandatory_set),
                key=bit_count, reverse=True,
            )
            cliques = mandatory + extra[:max(1, budget - len(mandatory))]
    membership = np.asarray(
        [[(clique >> index) & 1 for index in range(count)] for clique in cliques], dtype=float
    ).reshape(len(cliques), count)
    imposed = np.full((len(cliques), axes.shape[1]), -1, dtype=np.int8)
    for row, clique in enumerate(cliques):
        for index in range(count):
            if (clique >> index) & 1:
                imposed[row] = np.where(axes[index] >= 0, axes[index], imposed[row])
    return membership, imposed
