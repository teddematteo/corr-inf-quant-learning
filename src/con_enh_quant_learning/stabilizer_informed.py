"""Static stabilizer-informed local-Pauli acquisition with exact inclusions."""

from __future__ import annotations

import numpy as np

from .measurement import (
    PilotData,
    ShotDesign,
    _basis_to_pauli,
    _local_digits,
    explored_inclusion,
)
from .pauli import Pauli, axis_at
from .sampled import PauliPool, axis_masks, coverage, parity


def stabilizer_bases(generators: tuple[Pauli, ...], n_qubits: int) -> np.ndarray:
    """Greedily partition reference words into qubit-wise compatible groups."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    groups: list[np.ndarray] = []
    for generator in dict.fromkeys(generators):
        if generator.support.bit_length() > n_qubits:
            raise ValueError("A generator acts outside the measurement register")
        if not generator.support:
            continue
        axes = np.array([
            -1 if axis_at(generator, q) is None else axis_at(generator, q)
            for q in range(n_qubits)
        ], dtype=np.int8)
        for group in groups:
            if np.all((axes < 0) | (group < 0) | (axes == group)):
                group[axes >= 0] = axes[axes >= 0]
                break
        else:
            groups.append(axes)
    if not groups:
        raise ValueError("At least one nonidentity reference generator is required")
    return np.unique(np.where(np.asarray(groups) < 0, 2, groups), axis=0).astype(np.int8)


class StabilizerInformedSelector:
    """q(B) = (1-eta) sum_l w_l 1[B=B_l] + eta 3**(-n)."""

    def __init__(
        self, generators: tuple[Pauli, ...], n_qubits: int, rng: np.random.Generator,
        *, exploration: float = 0.2, subsystem: tuple[int, ...] | None = None,
        pool: PauliPool | None = None, pilot: PilotData | None = None,
    ) -> None:
        if not np.isfinite(exploration) or not 0 < exploration <= 1:
            raise ValueError("exploration must lie in (0, 1]")
        if subsystem is not None and (
            not subsystem or len(set(subsystem)) != len(subsystem)
            or any(q < 0 or q >= n_qubits for q in subsystem)
        ):
            raise ValueError("subsystem must contain distinct valid qubits")
        if pool is not None and pool.subsystem != subsystem:
            raise ValueError("The Pauli pool uses another subsystem")
        self.n_qubits = n_qubits
        self.rng = rng
        self.exploration = float(exploration)
        self.subsystem = subsystem
        mask = (1 << n_qubits) - 1 if subsystem is None else sum(1 << q for q in subsystem)
        self.generators = tuple(dict.fromkeys(g for g in generators if g.support & mask))
        self.bases = stabilizer_bases(self.generators, n_qubits)
        self.weights = np.full(len(self.bases), 1.0 / len(self.bases))
        self.bases.setflags(write=False)
        self.weights.setflags(write=False)
        self._axes = np.array([
            [-1 if axis_at(g, q) is None else axis_at(g, q) for q in range(n_qubits)]
            for g in self.generators
        ], dtype=np.int8)
        self.generator_counts = np.zeros(len(self.generators), dtype=np.int64)
        self.generator_sums = np.zeros(len(self.generators), dtype=np.int64)
        self.shot_count = 0
        self._pool = pool
        self._inclusion = None
        self._anti = None
        if subsystem is not None:
            sites = np.asarray(subsystem, dtype=int)
            size = len(subsystem)
            if pool is None:

                digits = _local_digits(4**size, 4**np.arange(size, dtype=np.int64), 4)
                policy = np.zeros(3**size)
                codes = self.bases[:, sites] @ (3**np.arange(size, dtype=np.int64))
                np.add.at(policy, codes, self.weights)
                uniform = 3.0**(-np.count_nonzero(digits, axis=1))
                mass = _basis_to_pauli(policy, size)
                self._anti = np.zeros((len(digits), len(self.generators)), dtype=bool)
                for column, axes in enumerate(self._axes[:, sites]):
                    for site, axis in enumerate(axes):
                        if axis >= 0:
                            self._anti[:, column] ^= (
                                (digits[:, site] > 0) & (digits[:, site] != axis + 1)
                            )
            else:
                uniform = pool.uniform_inclusion
                mass = self.weights @ coverage(axis_masks(self.bases[:, sites]), pool.masks)
                gx = np.array([
                    sum(((g.x >> q) & 1) << j for j, q in enumerate(subsystem))
                    for g in self.generators
                ], dtype=pool.x.dtype)
                gz = np.array([
                    sum(((g.z >> q) & 1) << j for j, q in enumerate(subsystem))
                    for g in self.generators
                ], dtype=pool.x.dtype)
                self._anti = parity((pool.x[:, None] & gz) ^ (pool.z[:, None] & gx))
            self._inclusion = explored_inclusion(mass, uniform, self.exploration)
            self._inclusion.setflags(write=False)
        if pilot is not None:
            indices = {g: i for i, g in enumerate(self.generators)}
            for word, outcome in zip(pilot.parity_paulis, pilot.parity_outcomes):
                if word in indices:
                    index = indices[word]
                    self.generator_counts[index] += 1
                    self.generator_sums[index] += int(outcome)

    def draw(self) -> ShotDesign:
        if self.rng.random() < self.exploration:
            basis = self.rng.integers(0, 3, size=self.n_qubits, dtype=np.int8)
        else:
            basis = self.bases[self.rng.choice(len(self.weights), p=self.weights)].copy()
        return ShotDesign(
            basis=basis, exploration=self.exploration, local_sites=self.subsystem,
            local_inclusion=self._inclusion, candidate_bases=self.bases,
            candidate_policy=self.weights,
            local_paulis=None if self._pool is None else self._pool.paulis,
        )

    def _covered_generators(self, basis: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.all((self._axes < 0) | (self._axes == basis), axis=1))

    def parity_probes(self, basis: np.ndarray) -> tuple[Pauli, ...]:
        return tuple(self.generators[i] for i in self._covered_generators(basis))

    def observe(self, design: ShotDesign | np.ndarray, parities: np.ndarray) -> None:
        basis = design.basis if isinstance(design, ShotDesign) else design
        covered = self._covered_generators(basis)
        if len(parities) != len(covered):
            raise ValueError("One parity is required for every covered generator")
        self.generator_counts[covered] += 1
        self.generator_sums[covered] += parities
        self.shot_count += 1

    def generator_epsilon(self) -> np.ndarray:
        means = np.divide(
            self.generator_sums, self.generator_counts,
            out=np.zeros(len(self.generators)), where=self.generator_counts > 0,
        )
        return 1.0 - np.abs(means)

    def pauli_bounds(self, margin: bool = True) -> np.ndarray:
        if self._anti is None:
            raise ValueError("Correlator bounds require a subsystem or Pauli pool")
        epsilon = self.generator_epsilon()
        if margin:
            epsilon = np.minimum(1.0, epsilon + 1 / np.sqrt(1 + self.generator_counts))
        result = np.ones(len(self._anti))
        for column, bound in enumerate(epsilon * (2 - epsilon)):
            np.minimum(result, bound, out=result, where=self._anti[:, column])
        return result

    def diagnostics(self) -> dict[str, object]:
        return {
            "acquisition": "static_stabilizer_informed_mixture",
            "reference_generators": len(self.generators),
            "bases": self.bases.tolist(), "weights": self.weights.tolist(),
            "weight_rule": "uniform_over_distinct_QWC_bases",
            "exploration": self.exploration, "policy_updates": 0,
            "exact_generator_means_supplied": False,
        }
