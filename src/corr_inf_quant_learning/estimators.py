from __future__ import annotations

import numpy as np

from .cluster_state import RotatedClusterState
from .measurement import LocalShot, ShotDesign, shared_magic_probe


class IncrementalPurityUStatistic:
    def __init__(self, subsystem: tuple[int, ...]) -> None:
        self.subsystem = tuple(int(qubit) for qubit in subsystem)
        self.subsystem_size = len(self.subsystem)
        self.dimension = 1 << self.subsystem_size
        self._sites = np.asarray(self.subsystem, dtype=int)
        subset_ids = np.arange(self.dimension, dtype=np.int64)
        local_bits = np.arange(self.subsystem_size, dtype=np.int64)
        self._subset_membership = ((subset_ids[:, None] >> local_bits[None, :]) & 1).astype(bool)
        self._uniform_probabilities = np.power(3.0, -self._subset_membership.sum(axis=1))
        self._digit_powers = np.power(4, local_bits, dtype=np.int64)
        pauli_count = 4**self.subsystem_size
        self._feature_sums = np.zeros(pauli_count, dtype=float)
        self._numerators = np.zeros(pauli_count, dtype=float)
        self.shot_count = 0
        self.pair_count = 0
        self.max_weight = 0.0

    def add(self, shot: LocalShot) -> None:
        design = shot.design
        local_basis = design.basis[self._sites].astype(np.int64)
        parities = np.prod(
            np.where(self._subset_membership, shot.outcomes[None, :], 1), axis=1, dtype=np.int64
        ).astype(float)
        codes = self._subset_membership @ ((local_basis + 1) * self._digit_powers)
        if design.local_inclusion is not None:
            if tuple(design.local_sites) != self.subsystem:
                raise ValueError("The local design was built for another subsystem")
            probabilities = design.local_inclusion[codes]
        else:
            probabilities = self._uniform_probabilities
        features = parities / probabilities
        self.max_weight = max(self.max_weight, float(np.max(1.0 / probabilities)))
        self._numerators[codes] += self._feature_sums[codes] * features
        self._feature_sums[codes] += features
        self.pair_count += self.shot_count
        self.shot_count += 1

    def value(self, upper_bounds: np.ndarray | None = None) -> float:
        if not self.pair_count:
            return float("nan")
        squares = self._numerators / self.pair_count
        if upper_bounds is not None:
            squares = np.clip(squares, 0.0, np.asarray(upper_bounds))
        return float(np.sum(squares) / self.dimension)


def magic_block_kernel(
    designs: tuple[ShotDesign, ShotDesign, ShotDesign, ShotDesign],
    oracle: RotatedClusterState,
    rng: np.random.Generator,
) -> float:
    pauli, matching_sites = shared_magic_probe(designs, rng)
    outcomes = tuple(oracle.sample_pauli(pauli, rng) for _ in designs)
    log_magnitude = (matching_sites - oracle.n_qubits) * np.log(2.0)
    for design in designs:
        log_magnitude -= np.log(design.coverage_probability(pauli))
    magnitude = float(np.exp(np.clip(log_magnitude, -745.0, 700.0)))
    return int(np.prod(outcomes)) * magnitude


def fourth_moment_from_kernels(kernels: list[float]) -> float:
    if not kernels:
        return float("nan")
    return float(np.mean(np.asarray(kernels, dtype=float)))


def magic_from_fourth_moment(moment: float, n_qubits: int) -> float:
    if not np.isfinite(moment):
        return float("nan")
    physical_moment = float(np.clip(moment, 2.0 ** (-n_qubits), 1.0))
    return float(-np.log2(physical_moment))
