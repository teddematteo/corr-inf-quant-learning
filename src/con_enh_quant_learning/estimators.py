from __future__ import annotations

from math import comb

import numpy as np

from ._compat import bit_count
from .measurement import LocalShot, ShotDesign, shared_magic_probe
from .states import StateOracle


class IncrementalPauliMomentUStatistic:
    """Exact order-2/4 U-statistic, using distinct-shot elementary products."""

    def __init__(self, subsystem: tuple[int, ...], moment_order: int = 2) -> None:
        if moment_order not in (2, 4):
            raise ValueError("moment_order must be 2 or 4")
        self.moment_order = moment_order
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
        if moment_order == 4:
            self._triples = np.zeros(pauli_count, dtype=float)
            self._fourths = np.zeros(pauli_count, dtype=float)
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
        if self.moment_order == 4:
            self._fourths[codes] += self._triples[codes] * features
            self._triples[codes] += self._numerators[codes] * features
        self._numerators[codes] += self._feature_sums[codes] * features
        self._feature_sums[codes] += features
        self.pair_count += self.shot_count
        self.shot_count += 1

    def value(self, upper_bounds: np.ndarray | None = None) -> float:
        count = comb(self.shot_count, self.moment_order)
        if not count:
            return float("nan")
        numerators = self._numerators if self.moment_order == 2 else self._fourths
        squares = numerators / count
        if upper_bounds is not None:
            squares = np.clip(squares, 0.0, np.asarray(upper_bounds))
        return float(np.sum(squares) / self.dimension)


class IncrementalPurityUStatistic(IncrementalPauliMomentUStatistic):
    """Backward-compatible order-2 specialization."""

    def __init__(self, subsystem: tuple[int, ...]) -> None:
        super().__init__(subsystem, moment_order=2)


class FactorizedUniformPurityUStatistic:
    """Uniform-basis purity U-statistic evaluated through a factorized pair kernel."""

    def __init__(self, subsystem: tuple[int, ...], capacity: int) -> None:
        if not 1 <= len(subsystem) <= 100 or capacity < 1:
            raise ValueError("Use 1 to 100 subsystem sites and a positive shot capacity")
        self.subsystem = subsystem
        self._sites = np.asarray(subsystem, dtype=int)
        dtype = np.uint64 if len(subsystem) <= 63 else object
        self._bases = np.empty((capacity, 3), dtype=dtype)
        self._negative = np.empty(capacity, dtype=dtype)
        self._powers = (
            np.left_shift(np.uint64(1), np.arange(len(subsystem), dtype=np.uint64))
            if len(subsystem) <= 63
            else np.array([1 << q for q in range(len(subsystem))], dtype=object)
        )
        self._bit_counts = np.array([bit_count(i) for i in range(256)], dtype=np.uint8)
        self._positive_factor = 5.0**np.arange(len(subsystem) + 1)
        self._negative_factor = (-4.0)**np.arange(len(subsystem) + 1)
        self._different_factor = 2.0**(-np.arange(len(subsystem) + 1))
        self._numerator = 0.0
        self.shot_count = 0
        self.pair_count = 0
        self.max_weight = 3.0 ** len(subsystem)

    def add(self, shot: LocalShot) -> None:
        if shot.design.local_inclusion is not None or shot.design.candidate_bases is not None:
            raise ValueError("The factorized estimator requires uniform independent local bases")
        if self.shot_count >= len(self._bases):
            raise ValueError("The configured shot capacity has been exceeded")
        basis = shot.design.basis[self._sites]
        masks = np.array(
            [np.sum((basis == a) * self._powers) for a in range(3)], dtype=self._bases.dtype,
        )
        negative = np.sum((shot.outcomes < 0) * self._powers, dtype=self._negative.dtype)
        for start in range(0, self.shot_count, 8192):
            stop = min(start + 8192, self.shot_count)
            previous = self._bases[start:stop]
            matching = ((previous[:, 0] & masks[0]) | (previous[:, 1] & masks[1])
                        | (previous[:, 2] & masks[2]))
            opposite = matching & (self._negative[start:stop] ^ negative)
            if matching.dtype == object:
                matches = np.fromiter((bit_count(int(mask)) for mask in matching), dtype=int)
                negatives = np.fromiter((bit_count(int(mask)) for mask in opposite), dtype=int)
            else:
                matches = self._bit_counts[matching.view(np.uint8).reshape(-1, 8)].sum(axis=1)
                negatives = self._bit_counts[opposite.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            self._numerator += float(np.sum(
                self._positive_factor[matches - negatives] * self._negative_factor[negatives]
                * self._different_factor[len(self.subsystem) - matches]
            ))
        self._bases[self.shot_count] = masks
        self._negative[self.shot_count] = negative
        self.pair_count += self.shot_count
        self.shot_count += 1

    def value(self, upper_bounds: np.ndarray | None = None) -> float:
        if upper_bounds is not None:
            raise ValueError("Per-Pauli clipping cannot be factorized")
        return self._numerator / self.pair_count if self.pair_count else float("nan")


def magic_block_kernel(
    designs: tuple[ShotDesign, ShotDesign, ShotDesign, ShotDesign],
    oracle: StateOracle,
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


def magic_from_fourth_moment(moment: float, n_qubits: int, *, clip: bool = True) -> float:
    """Convert the estimated moment, optionally applying the original physical bounds."""
    if not np.isfinite(moment):
        return float("nan")
    value = float(np.clip(moment, 2.0 ** (-n_qubits), 1.0)) if clip else moment
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.log2(value))
