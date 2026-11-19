"""Importance-sampled Pauli sums and a bounded ConEnh candidate policy."""

from __future__ import annotations

from math import comb

import numpy as np

from ._compat import bit_count, dataclass, strict_zip
from .cliques import generator_cliques
from .compute import ComputeBackend
from .measurement import LocalShot, ShotDesign, softmax_policy_weights
from .pauli import Pauli, axis_at


def parity(values: np.ndarray) -> np.ndarray:
    if np.asarray(values).dtype == object:
        return np.fromiter(
            (bit_count(int(value)) % 2 for value in np.asarray(values).flat), dtype=bool,
        ).reshape(np.shape(values))
    work = np.asarray(values, dtype=np.uint64).copy()
    for shift in (32, 16, 8, 4, 2, 1):
        work ^= work >> np.uint64(shift)
    return (work & np.uint64(1)).astype(bool)


def axis_masks(bases: np.ndarray) -> np.ndarray:
    bases = np.atleast_2d(bases)
    powers = bit_powers(bases.shape[1])
    return np.stack([np.sum((bases == a) * powers, axis=1) for a in range(3)], axis=1)


def bit_powers(size: int) -> np.ndarray:
    """Fast uint64 masks on small registers, arbitrary-width integers beyond 63."""
    if size <= 63:
        return np.left_shift(np.uint64(1), np.arange(size, dtype=np.uint64))
    return np.array([1 << q for q in range(size)], dtype=object)


def coverage(bases: np.ndarray, words: np.ndarray) -> np.ndarray:
    """Rows are bases, columns are Pauli words, three uint64 axis masks each."""
    result = np.ones((len(bases), len(words)), dtype=bool)
    for axis in range(3):
        result &= (words[None, :, axis] & ~bases[:, None, axis]) == 0
    return result


def _insert(row: int, echelon: dict[int, int]) -> bool:
    while row:
        pivot = row.bit_length() - 1
        if pivot not in echelon:
            echelon[pivot] = row
            return True
        row ^= echelon[pivot]
    return False


def _in_span(row: int, echelon: dict[int, int]) -> bool:
    while row:
        pivot = row.bit_length() - 1
        if pivot not in echelon:
            return False
        row ^= echelon[pivot]
    return True


def local_generator_span(
    generators: tuple[Pauli, ...], subsystem: tuple[int, ...], n_qubits: int,
) -> dict[int, int]:
    """Intersect the binary Pauli span with support on A, by elimination."""
    outside = ((1 << n_qubits) - 1) ^ sum(1 << q for q in subsystem)
    outside_mask = outside | (outside << n_qubits)
    eliminated: dict[int, tuple[int, int]] = {}
    local: dict[int, int] = {}
    for generator in generators:
        full = generator.x | (generator.z << n_qubits)
        projected = full & outside_mask
        while projected:
            pivot = projected.bit_length() - 1
            if pivot not in eliminated:
                eliminated[pivot] = projected, full
                break
            other, full_other = eliminated[pivot]
            projected ^= other
            full ^= full_other
        if not projected:
            x = sum(((full >> q) & 1) << j for j, q in enumerate(subsystem))
            z = sum(((full >> (n_qubits + q)) & 1) << j for j, q in enumerate(subsystem))
            _insert(x | (z << len(subsystem)), local)
    return local


@dataclass(slots=True)
class PauliPool:
    subsystem: tuple[int, ...]
    paulis: tuple[Pauli, ...]
    masks: np.ndarray
    x: np.ndarray
    z: np.ndarray
    support: np.ndarray
    population_weights: np.ndarray
    uniform_inclusion: np.ndarray
    sample_count: int
    span_rank: int

    @classmethod
    def sample(
        cls, generators: tuple[Pauli, ...], subsystem: tuple[int, ...], n_qubits: int,
        samples: int, uniform_fraction: float, rng: np.random.Generator,
    ) -> PauliPool:
        size = len(subsystem)
        if size < 1:
            raise ValueError("The sampled backend requires a nonempty register")
        span = local_generator_span(generators, subsystem, n_qubits)
        rank = len(span)
        uniform_fraction = uniform_fraction if rank else 1.0
        basis = tuple(span.values())
        counts: dict[int, int] = {}
        for _ in range(samples):
            use_uniform = rng.random() < uniform_fraction
            code = 0
            while not code:
                if use_uniform:
                    code = sum(int(b) << j for j, b in enumerate(rng.integers(0, 2, 2 * size)))
                else:
                    for row, bit in strict_zip(basis, rng.integers(0, 2, rank)):
                        if bit:
                            code ^= row
            counts[code] = counts.get(code, 0) + 1
        codes = tuple(counts)
        mask = (1 << size) - 1
        dtype = np.uint64 if size <= 63 else object
        x = np.array([code & mask for code in codes], dtype=dtype)
        z = np.array([code >> size for code in codes], dtype=dtype)
        masks = np.stack((x & ~z, x & z, z & ~x), axis=1)
        population_weights = []
        for code in codes:
            probability = uniform_fraction / (4**size - 1)
            if rank and _in_span(code, span):
                probability += (1 - uniform_fraction) / (2**rank - 1)
            population_weights.append(counts[code] / (samples * probability))
        paulis = tuple(
            Pauli(
                sum(((int(px) >> j) & 1) << q for j, q in enumerate(subsystem)),
                sum(((int(pz) >> j) & 1) << q for j, q in enumerate(subsystem)),
            ) for px, pz in strict_zip(x, z)
        )
        support = x | z
        uniform = np.array([3.0 ** (-bit_count(int(word))) for word in support])
        return cls(
            subsystem, paulis, masks, x, z, support, np.array(population_weights),
            uniform, samples, rank,
        )


class SampledPauliMomentUStatistic:
    """Importance-sampled order-2/4 moments from the actual local-shot records."""

    def __init__(self, pool: PauliPool, moment_order: int = 2) -> None:
        if moment_order not in (2, 4):
            raise ValueError("moment_order must be 2 or 4")
        self.moment_order = moment_order
        self.pool = pool
        self.subsystem = pool.subsystem
        self._powers = bit_powers(len(self.subsystem))
        self._sums = np.zeros(len(pool.paulis))
        self._numerators = np.zeros(len(pool.paulis))
        if moment_order == 4:
            self._triples = np.zeros(len(pool.paulis))
            self._fourths = np.zeros(len(pool.paulis))
        self.hits = np.zeros(len(pool.paulis), dtype=np.int64)
        self.shot_count = 0
        self.pair_count = 0
        self.max_weight = 0.0

    def add(self, shot: LocalShot) -> None:
        local = shot.design.basis[np.asarray(self.subsystem)]
        covered = coverage(axis_masks(local), self.pool.masks)[0]
        self.hits += covered
        probabilities = self.pool.uniform_inclusion
        if shot.design.local_inclusion is not None:
            if shot.design.local_paulis is not self.pool.paulis:
                raise ValueError("The shot uses a different sampled Pauli pool")
            probabilities = shot.design.local_inclusion
        negative = np.sum((shot.outcomes < 0) * self._powers, dtype=self.pool.x.dtype)
        signs = 1.0 - 2.0 * parity(self.pool.support[covered] & negative)
        features = signs / probabilities[covered]
        if self.moment_order == 4:
            self._fourths[covered] += self._triples[covered] * features
            self._triples[covered] += self._numerators[covered] * features
        self._numerators[covered] += self._sums[covered] * features
        self._sums[covered] += features
        if np.any(covered):
            self.max_weight = max(self.max_weight, float(np.max(1 / probabilities[covered])))
        self.pair_count += self.shot_count
        self.shot_count += 1

    def value(self, upper_bounds: np.ndarray | None = None) -> float:
        count = comb(self.shot_count, self.moment_order)
        if not count:
            return float("nan")
        numerators = self._numerators if self.moment_order == 2 else self._fourths
        squares = numerators / count
        if upper_bounds is not None:
            squares = np.clip(squares, 0.0, upper_bounds)
        return float((1.0 + self.pool.population_weights @ squares) / 2**len(self.subsystem))


class SampledPurityUStatistic(SampledPauliMomentUStatistic):
    """Backward-compatible order-2 specialization."""

    def __init__(self, pool: PauliPool) -> None:
        super().__init__(pool, moment_order=2)


class SampledSubsystemSelector:
    def __init__(
        self, generators: tuple[Pauli, ...], pool: PauliPool, rng: np.random.Generator,
        *, n_qubits: int, candidates: int, max_cliques: int, update_every: int,
        exploration: float, temperature: float, pauli_floor: float,
        commuting_bias: float, purity_weight: float, generator_weight: float,
        local_regularization: float,
        moment_order: int = 2,
        compute: ComputeBackend | None = None,
    ) -> None:
        if moment_order not in (2, 4):
            raise ValueError("moment_order must be 2 or 4")
        self.compute = compute if compute is not None else ComputeBackend()
        self.moment_order = moment_order
        self.pool = pool
        self.subsystem = pool.subsystem
        self._sites = np.asarray(self.subsystem)
        self.n_qubits = n_qubits
        mask = sum(1 << q for q in self.subsystem)
        self.generators = tuple(g for g in generators if g.support & mask)
        self.rng = rng
        self.candidates = candidates
        self.update_every = update_every
        self.exploration = exploration
        self.temperature = temperature
        self.pauli_floor = pauli_floor
        self.commuting_bias = commuting_bias
        self.purity_weight = purity_weight
        self.generator_weight = generator_weight
        self.local_regularization = local_regularization
        self._axes = np.array([
            [(-1 if axis_at(g, q) is None else axis_at(g, q)) for q in range(n_qubits)]
            for g in self.generators
        ], dtype=np.int8).reshape(len(self.generators), n_qubits)
        self._membership, self._clique_axes = generator_cliques(self._axes, max_cliques)
        self.clique_pool_size = len(self._membership)
        self._local_cliques = self._clique_axes[:, self._sites]
        gx = np.array([
            sum(((g.x >> q) & 1) << j for j, q in enumerate(self.subsystem))
            for g in self.generators
        ], dtype=pool.x.dtype)
        gz = np.array([
            sum(((g.z >> q) & 1) << j for j, q in enumerate(self.subsystem))
            for g in self.generators
        ], dtype=pool.x.dtype)
        self._gx, self._gz = gx, gz
        self._anti = parity((pool.x[:, None] & gz) ^ (pool.z[:, None] & gx))
        self.generator_counts = np.zeros(len(self.generators), dtype=np.int64)
        self.generator_sums = np.zeros(len(self.generators), dtype=np.int64)
        self.pauli_counts = np.zeros(len(pool.paulis), dtype=np.int64)
        self.anti_counts = np.zeros(len(self.generators), dtype=np.int64)
        self.local_counts = np.zeros((len(self.subsystem), 3), dtype=np.int64)
        self.shot_count = 0
        self.policy_updates = 0
        self._last_update = -update_every

    def generator_epsilon(self) -> np.ndarray:
        means = np.divide(
            self.generator_sums, self.generator_counts,
            out=np.zeros(len(self.generators)), where=self.generator_counts > 0,
        )
        return 1.0 - np.abs(means)

    def pauli_bounds(self, margin: bool = True) -> np.ndarray:
        epsilon = self.generator_epsilon()
        if margin:
            epsilon = np.minimum(1, epsilon + 1 / np.sqrt(1 + self.generator_counts))
        result = np.ones(len(self.pool.paulis))
        for column, bound in enumerate(epsilon * (2 - epsilon)):
            np.minimum(result, bound, out=result, where=self._anti[:, column])
        return result

    def _refresh(self) -> None:
        u = 1.0 / np.sqrt(1.0 + self.generator_counts)
        clique_scores = self.compute.matvec(self._membership, u, cache=True)
        bases = self.rng.integers(0, 3, (self.candidates, len(self.subsystem)), dtype=np.int8)

        packed = self.candidates // 2
        mass = 1 + clique_scores
        choices = self.rng.choice(len(mass), size=packed, p=mass / mass.sum())
        imposed = self._local_cliques[choices]
        bases[:packed] = np.where(imposed >= 0, imposed, bases[:packed])


        seeded = max(1, self.candidates // 4)
        word_ids = self.rng.integers(0, len(self.pool.paulis), size=seeded)
        for row, word in enumerate(word_ids, start=packed):
            px, pz = int(self.pool.x[word]), int(self.pool.z[word])
            for j in range(len(self.subsystem)):
                x, z = (px >> j) & 1, (pz >> j) & 1
                if x or z:
                    bases[row, j] = 1 if x and z else (0 if x else 2)

        fits = self.compute.clique_fits(bases, self._local_cliques)
        g = np.max(np.where(fits, clique_scores, -np.inf), axis=1)
        completion = []
        for row, best in strict_zip(fits, g):
            winners = np.flatnonzero(row & np.isclose(clique_scores, best))
            completion.append(self.rng.choice(winners))
        masks = axis_masks(bases)
        covered = coverage(masks, self.pool.masks)
        r = self.pool.population_weights * np.maximum(
            self.pauli_bounds(margin=False) ** (self.moment_order // 2), self.pauli_floor
        ) / (1 + self.pauli_counts)
        s = self.generator_epsilon() / (1 + self.anti_counts)
        rc, ra = float(r.sum()), float(s.sum())
        denominator = self.commuting_bias * rc + ra
        alpha = self.commuting_bias * rc / denominator if denominator > 1e-15 else (
            0.5 if self.commuting_bias else 0.0
        )
        x, z = masks[:, 0] | masks[:, 1], masks[:, 1] | masks[:, 2]
        anti = parity((x[:, None] & self._gz) ^ (z[:, None] & self._gx))
        local_reward = np.mean(
            1 / (1 + self.local_counts[np.arange(len(self.subsystem))[None, :], bases]), axis=1
        )
        cov = (
            alpha * (self.compute.matvec(covered, r) / rc if rc else 0)
            + (1 - alpha) * (self.compute.matvec(anti, s) / ra if ra else 0)
            + self.local_regularization * local_reward
        )
        score = self.purity_weight * cov + self.generator_weight * (g / g.max() if g.max() else 0)
        self.policy = softmax_policy_weights(score, self.temperature)
        self._bases = bases
        self._completion = np.asarray(completion)
        self._inclusion = (
            (1 - self.exploration) * self.compute.matvec(covered, self.policy, transpose=True)
            + self.exploration * self.pool.uniform_inclusion
        )
        self._last_update = self.shot_count
        self.policy_updates += 1

    def draw(self) -> ShotDesign:
        if self.shot_count - self._last_update >= self.update_every:
            self._refresh()
        basis = self.rng.integers(0, 3, self.n_qubits, dtype=np.int8)
        if self.rng.random() >= self.exploration:
            candidate = int(self.rng.choice(len(self.policy), p=self.policy))
            axes = self._clique_axes[self._completion[candidate]]
            basis[axes >= 0] = axes[axes >= 0]
            basis[self._sites] = self._bases[candidate]
        return ShotDesign(
            basis=basis, exploration=self.exploration, local_sites=self.subsystem,
            local_inclusion=self._inclusion, local_paulis=self.pool.paulis,
        )

    def _covered_generators(self, basis: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.all((self._axes < 0) | (self._axes == basis[None, :]), axis=1))

    def parity_probes(self, basis: np.ndarray) -> tuple[Pauli, ...]:
        return tuple(self.generators[i] for i in self._covered_generators(basis))

    def observe(self, design: ShotDesign, parities: np.ndarray) -> None:
        local = design.basis[self._sites]
        masks = axis_masks(local)[0]
        self.pauli_counts += coverage(masks[None, :], self.pool.masks)[0]
        x, z = masks[0] | masks[1], masks[1] | masks[2]
        self.anti_counts += parity((x & self._gz) ^ (z & self._gx))
        self.local_counts[np.arange(len(self.subsystem)), local] += 1
        covered = self._covered_generators(design.basis)
        if len(parities) != len(covered):
            raise ValueError("One parity is required for every covered generator")
        self.generator_counts[covered] += 1
        self.generator_sums[covered] += parities
        self.shot_count += 1
