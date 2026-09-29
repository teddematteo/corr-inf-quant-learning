from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster_state import RotatedClusterState
from .pauli import Pauli, axis_at, pauli_from_basis_subset


@dataclass(slots=True)
class ShotDesign:
    basis: np.ndarray
    exploration: float = 1.0
    local_sites: tuple[int, ...] | None = None
    local_inclusion: np.ndarray | None = None
    candidate_bases: np.ndarray | None = None
    candidate_policy: np.ndarray | None = None
    probe_inclusion: np.ndarray | None = None

    def coverage_probability(self, pauli: Pauli) -> float:
        if pauli.weight == 0:
            return 1.0
        uniform_probability = 3.0 ** (-pauli.weight)
        if self.candidate_bases is None:
            return uniform_probability
        covering = np.ones(len(self.candidate_bases), dtype=bool)
        remaining = pauli.support
        while remaining:
            low = remaining & -remaining
            qubit = low.bit_length() - 1
            covering &= self.candidate_bases[:, qubit] == axis_at(pauli, qubit)
            remaining ^= low
        return float(
            explored_inclusion(
                float(self.candidate_policy @ covering), uniform_probability, self.exploration
            )
        )


@dataclass(slots=True)
class LocalShot:
    design: ShotDesign
    outcomes: np.ndarray


@dataclass(slots=True)
class PilotData:
    generators: tuple[Pauli, ...]
    settings: tuple[np.ndarray, ...]
    epsilon: np.ndarray
    parity_paulis: tuple[Pauli, ...]
    parity_outcomes: np.ndarray


def _complete_pauli_basis(pauli: Pauli, n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    basis = rng.integers(0, 3, size=n_qubits, dtype=np.int8)
    remaining = pauli.support
    while remaining:
        low = remaining & -remaining
        qubit = low.bit_length() - 1
        basis[qubit] = int(axis_at(pauli, qubit))
        remaining ^= low
    return basis


def run_common_pilot(
    oracle: RotatedClusterState, shots_per_generator: int, rng: np.random.Generator
) -> PilotData:
    if shots_per_generator < 1:
        raise ValueError("shots_per_generator must be positive")
    settings: list[np.ndarray] = []
    outcome_sums = np.zeros(oracle.n_qubits, dtype=np.int64)
    parity_paulis: list[Pauli] = []
    parity_outcomes: list[int] = []
    for generator_index, generator in enumerate(oracle.generators):
        settings.append(_complete_pauli_basis(generator, oracle.n_qubits, rng))
        for _ in range(shots_per_generator):
            parity = oracle.sample_pauli(generator, rng)
            outcome_sums[generator_index] += parity
            parity_paulis.append(generator)
            parity_outcomes.append(parity)
    means = outcome_sums.astype(float) / shots_per_generator
    return PilotData(
        generators=oracle.generators,
        settings=tuple(settings),
        epsilon=np.clip(1.0 - np.abs(means), 0.0, 1.0),
        parity_paulis=tuple(parity_paulis),
        parity_outcomes=np.asarray(parity_outcomes, dtype=np.int8),
    )


class UniformSelector:
    def __init__(self, n_qubits: int, rng: np.random.Generator) -> None:
        self.n_qubits = int(n_qubits)
        self.rng = rng

    def draw(self) -> ShotDesign:
        return ShotDesign(basis=self.rng.integers(0, 3, size=self.n_qubits, dtype=np.int8))

    def parity_probes(self, _: np.ndarray) -> tuple[Pauli, ...]:
        return ()

    def observe(self, *_: object) -> None:
        return


def generator_probes(generators: tuple[Pauli, ...]) -> tuple[Pauli, ...]:
    probes = []
    for index, generator in enumerate(generators):
        probes.append(generator)
        probes.append(Pauli(generator.x, generator.z | (1 << index)))
        probes.append(Pauli(0, 1 << index))
    return tuple(probes)


def probe_axes(probes: tuple[Pauli, ...], n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(probe.weight for probe in probes)
    sites = np.full((len(probes), width), -1, dtype=np.int64)
    axes = np.full((len(probes), width), -1, dtype=np.int8)
    for row, probe in enumerate(probes):
        support = [qubit for qubit in range(n_qubits) if (probe.support >> qubit) & 1]
        sites[row, : len(support)] = support
        axes[row, : len(support)] = [axis_at(probe, qubit) for qubit in support]
    return sites, axes


def covered_probes(basis: np.ndarray, sites: np.ndarray, axes: np.ndarray) -> np.ndarray:
    measured = np.asarray(basis)[np.where(sites >= 0, sites, 0)]
    return np.all((axes < 0) | (measured == axes), axis=1)


def softmax_policy_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = (scores - scores.max()) / temperature
    policy = np.exp(logits)
    return policy / policy.sum()


def explored_inclusion(
    covered_mass: np.ndarray, uniform_inclusion: np.ndarray, exploration: float
) -> np.ndarray:
    return (1.0 - exploration) * covered_mass + exploration * uniform_inclusion


class MagicProbeSelector:
    def __init__(
        self,
        generators: tuple[Pauli, ...],
        epsilon: np.ndarray | None,
        rng: np.random.Generator,
        *,
        candidates: int = 64,
        temperature: float = 0.05,
        exploration: float = 0.20,
        pilot_repetitions: int = 0,
    ) -> None:
        if epsilon is not None and len(generators) != len(epsilon):
            raise ValueError("epsilon must contain one value per generator")
        if int(candidates) < 1:
            raise ValueError("candidates must be a positive integer")
        if not temperature > 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < exploration <= 1.0:
            raise ValueError("exploration must lie in (0, 1]")
        self.n_qubits = len(generators)
        self.generators = tuple(generators)
        self.online = epsilon is None
        self.generator_counts = np.zeros(self.n_qubits, dtype=np.int64)
        self.generator_sums = np.zeros(self.n_qubits, dtype=np.int64)
        self.rng = rng
        self.candidates = int(candidates)
        self.temperature = float(temperature)
        self.exploration = float(exploration)
        self.probes = generator_probes(self.generators)
        self._sites, self._axes = probe_axes(self.probes, self.n_qubits)
        self._uniform_inclusion = np.power(
            3.0, -np.asarray([probe.weight for probe in self.probes], dtype=float)
        )
        self.counts = np.zeros(len(self.probes), dtype=np.int64)
        self.counts[0::3] = int(pilot_repetitions)
        self._set_weights(
            self.online_epsilon() if self.online else np.asarray(epsilon, dtype=float)
        )

    def _set_weights(self, epsilon: np.ndarray) -> None:
        mean_square = (1.0 - np.clip(epsilon, 0.0, 1.0)) ** 2
        self.weights = np.repeat((1.0 - mean_square) ** 2, 3)
        self._active = np.flatnonzero(self.weights > 0.0)

    def generator_epsilon(self) -> np.ndarray:
        epsilon = np.ones(self.n_qubits, dtype=float)
        seen = self.generator_counts > 0
        epsilon[seen] = 1.0 - np.abs(self.generator_sums[seen] / self.generator_counts[seen])
        return epsilon

    def online_epsilon(self) -> np.ndarray:
        margin = 1.0 / np.sqrt(1.0 + self.generator_counts)
        return np.minimum(1.0, self.generator_epsilon() + margin)

    def rewards(self) -> np.ndarray:
        return self.weights / (1.0 + self.counts)

    def covered(self, bases: np.ndarray) -> np.ndarray:
        measured = np.asarray(bases)[:, np.where(self._sites >= 0, self._sites, 0)]
        return np.all((self._axes < 0) | (measured == self._axes), axis=2)

    def candidate_bases(self, rewards: np.ndarray) -> np.ndarray:
        count = self.candidates
        bases = np.full((count, self.n_qubits), -1, dtype=np.int8)
        active = self._active
        if len(active):
            ties = self.rng.random((count, len(active)))
            keys = np.log(rewards[active])[None, :] + np.vstack(
                (np.zeros((1, len(active))), self.rng.gumbel(size=(count - 1, len(active))))
            )
            order = active[np.lexsort((ties, -keys), axis=1)]
            rows = np.arange(count)
            for position in range(len(active)):
                probe = order[:, position]
                sites = self._sites[probe]
                axes = self._axes[probe]
                valid = sites >= 0
                current = bases[rows[:, None], np.where(valid, sites, 0)]
                fits = np.all(~valid | (current < 0) | (current == axes), axis=1)
                write_rows, write_columns = np.nonzero(fits[:, None] & valid)
                bases[write_rows, sites[write_rows, write_columns]] = axes[
                    write_rows, write_columns
                ]
        free = bases < 0
        bases[free] = self.rng.integers(0, 3, size=int(free.sum()), dtype=np.int8)
        return bases

    def draw(self) -> ShotDesign:
        if self.online:
            self._set_weights(self.online_epsilon())
        rewards = self.rewards()
        bases = self.candidate_bases(rewards)
        covered = self.covered(bases)
        total = float(rewards.sum())
        scores = covered @ rewards / total if total > 0.0 else np.zeros(len(bases))
        policy = softmax_policy_weights(scores, self.temperature)
        probe_inclusion = explored_inclusion(
            policy @ covered, self._uniform_inclusion, self.exploration
        )
        if self.rng.random() < self.exploration:
            basis = self.rng.integers(0, 3, size=self.n_qubits, dtype=np.int8)
        else:
            basis = bases[int(self.rng.choice(len(bases), p=policy))].copy()
        return ShotDesign(
            basis=basis,
            exploration=self.exploration,
            candidate_bases=bases,
            candidate_policy=policy,
            probe_inclusion=probe_inclusion,
        )

    def _covered_generators(self, basis: np.ndarray) -> np.ndarray:
        return np.flatnonzero(covered_probes(basis, self._sites[0::3], self._axes[0::3]))

    def parity_probes(self, basis: np.ndarray) -> tuple[Pauli, ...]:
        if not self.online:
            return ()
        return tuple(self.generators[index] for index in self._covered_generators(basis))

    def observe(self, basis: np.ndarray, parities: np.ndarray | None = None) -> None:
        self.counts += covered_probes(basis, self._sites, self._axes)
        if not self.online:
            return
        covered = self._covered_generators(basis)
        parities = np.zeros(0, dtype=np.int64) if parities is None else np.asarray(parities)
        if len(parities) != len(covered):
            raise ValueError("One parity is required for every covered generator")
        self.generator_counts[covered] += 1
        self.generator_sums[covered] += parities.astype(np.int64)


def _pauli_to_basis(values: np.ndarray, size: int, combine=np.add) -> np.ndarray:
    tensor = np.asarray(values).reshape((4,) * size)
    for axis in range(size):
        tensor = np.moveaxis(tensor, axis, 0)
        tensor = np.moveaxis(combine(tensor[:1], tensor[1:]), 0, axis)
    return tensor.reshape(-1)


def _basis_to_pauli(values: np.ndarray, size: int) -> np.ndarray:
    tensor = np.asarray(values).reshape((3,) * size)
    for axis in range(size):
        tensor = np.moveaxis(tensor, axis, 0)
        tensor = np.concatenate((tensor.sum(axis=0, keepdims=True), tensor), axis=0)
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def _local_digits(count: int, powers: np.ndarray, base: int) -> np.ndarray:
    codes = np.arange(count, dtype=np.int64)
    digits = np.empty((count, len(powers)), dtype=np.int8)
    for index, power in enumerate(powers):
        digits[:, index] = (codes // power) % base
    return digits


class OnlineSubsystemSelector:
    def __init__(
        self,
        generators: tuple[Pauli, ...],
        subsystem: tuple[int, ...],
        rng: np.random.Generator,
        *,
        exploration: float = 0.20,
        purity_weight: float = 1.0,
        generator_weight: float = 1.0,
        temperature: float = 0.3,
        pauli_floor: float = 0.1,
        commuting_bias: float = 1.0,
        local_regularization: float = 0.05,
        update_every: int = 1,
    ) -> None:
        if not subsystem or len(set(subsystem)) != len(subsystem):
            raise ValueError("subsystem must contain distinct qubits")
        if int(update_every) < 1:
            raise ValueError("update_every must be a positive integer")
        if not 0.0 < exploration <= 1.0:
            raise ValueError("exploration must lie in (0, 1]")
        if purity_weight < 0.0 or generator_weight < 0.0:
            raise ValueError("score weights must be non-negative")
        if not temperature > 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= pauli_floor <= 1.0:
            raise ValueError("pauli_floor must lie in [0, 1]")
        if not np.isfinite(commuting_bias) or commuting_bias < 0.0:
            raise ValueError("commuting_bias must be finite and non-negative")
        self.temperature = float(temperature)
        self.pauli_floor = float(pauli_floor)
        self.commuting_bias = float(commuting_bias)
        self.local_regularization = float(local_regularization)
        self.n_qubits = len(generators)
        self.subsystem = tuple(int(qubit) for qubit in subsystem)
        self.rng = rng
        self.exploration = float(exploration)
        self.purity_weight = float(purity_weight)
        self.generator_weight = float(generator_weight)
        self.update_every = int(update_every)
        self._frozen_design: tuple[np.ndarray, np.ndarray] | None = None
        self._last_update = 0
        self._sites = np.asarray(self.subsystem, dtype=int)
        size = len(self.subsystem)
        self._size = size

        self._basis_powers = np.power(3, np.arange(size), dtype=np.int64)
        self.local_bases = _local_digits(3**size, self._basis_powers, 3)
        self._digit_powers = np.power(4, np.arange(size), dtype=np.int64)
        pauli_digits = _local_digits(4**size, self._digit_powers, 4)
        self._uniform_inclusion = np.power(3.0, -np.count_nonzero(pauli_digits, axis=1))
        subset_ids = np.arange(1 << size, dtype=np.int64)
        self._membership = ((subset_ids[:, None] >> np.arange(size)[None, :]) & 1).astype(bool)

        touching = tuple(
            index
            for index, generator in enumerate(generators)
            if any((generator.support >> qubit) & 1 for qubit in self.subsystem)
        )
        self.generators = tuple(generators[index] for index in touching)
        region = sorted(
            {
                qubit
                for generator in self.generators
                for qubit in range(self.n_qubits)
                if (generator.support >> qubit) & 1
            }
        )
        self._region = np.asarray(region, dtype=int)
        self._generator_axes = np.full((len(touching), len(region)), -1, dtype=np.int8)
        for row, generator in enumerate(self.generators):
            for column, qubit in enumerate(region):
                axis = axis_at(generator, qubit)
                if axis is not None:
                    self._generator_axes[row, column] = axis
        self._subsystem_columns = np.searchsorted(self._region, self._sites)
        local_axes = self._generator_axes[:, self._subsystem_columns]
        self._pauli_anti = np.zeros((len(pauli_digits), len(touching)), dtype=bool, order="F")
        self._basis_anti = np.zeros((len(self.local_bases), len(touching)), dtype=bool)
        for row, generator_axes in enumerate(local_axes):
            for site, axis in enumerate(generator_axes):
                if axis < 0:
                    continue
                self._pauli_anti[:, row] ^= (pauli_digits[:, site] > 0) & (
                    pauli_digits[:, site] != axis + 1
                )
                self._basis_anti[:, row] ^= self.local_bases[:, site] != axis
        del pauli_digits
        self._basis_anti_float = self._basis_anti.astype(float)
        self._build_generator_cliques()

        self.policy = np.full(len(self.local_bases), 1.0 / len(self.local_bases))
        self.shot_count = 0
        self.generator_counts = np.zeros(len(touching), dtype=np.int64)
        self.generator_sums = np.zeros(len(touching), dtype=np.int64)
        self.pauli_counts = np.zeros(4**size, dtype=np.int64)
        self.anti_counts = np.zeros(len(touching), dtype=np.int64)
        self.local_counts = np.zeros((size, 3), dtype=np.int64)

    def _build_generator_cliques(self) -> None:
        axes = self._generator_axes
        count = len(axes)
        compatible = [0] * count
        for left in range(count):
            for right in range(count):
                if left == right:
                    continue
                shared = (axes[left] >= 0) & (axes[right] >= 0)
                if np.all(axes[left][shared] == axes[right][shared]):
                    compatible[left] |= 1 << right
        cliques = [0]
        for index in range(count):
            cliques += [
                clique | (1 << index) for clique in cliques if not (clique & ~compatible[index])
            ]
        self._clique_membership = np.asarray(
            [[(clique >> index) & 1 for index in range(count)] for clique in cliques],
            dtype=float,
        ).reshape(len(cliques), count)
        clique_axes = np.full((len(cliques), axes.shape[1]), -1, dtype=np.int8)
        for row, clique in enumerate(cliques):
            for index in range(count):
                if (clique >> index) & 1:
                    clique_axes[row] = np.where(axes[index] >= 0, axes[index], clique_axes[row])
        self._clique_axes = clique_axes
        patterns, inverse = np.unique(
            clique_axes[:, self._subsystem_columns], axis=0, return_inverse=True
        )
        self._clique_pattern = inverse.reshape(-1)
        self._patterns = patterns
        self._clique_codes = ((patterns.astype(np.int64) + 1) @ self._digit_powers)[
            self._clique_pattern
        ]

    def _pattern_fits(self, basis_id: int) -> np.ndarray:
        return np.all((self._patterns < 0) | (self._patterns == self.local_bases[basis_id]), axis=1)

    def _anticommuting_minimum(self, bounds: np.ndarray) -> np.ndarray:
        result = np.ones(self._pauli_anti.shape[0])
        for column, bound in enumerate(bounds):
            if bound < 1.0:
                np.minimum(result, bound, out=result, where=self._pauli_anti[:, column])
        return result

    def generator_epsilon(self) -> np.ndarray:
        epsilon = np.ones(len(self.generators), dtype=float)
        seen = self.generator_counts > 0
        epsilon[seen] = 1.0 - np.abs(self.generator_sums[seen] / self.generator_counts[seen])
        return epsilon

    def _exploration_rewards(self) -> np.ndarray:
        return 1.0 / np.sqrt(1.0 + self.generator_counts)

    def pauli_bounds(self, margin: bool = True) -> np.ndarray:
        epsilon = self.generator_epsilon()
        if margin:
            epsilon = np.minimum(1.0, epsilon + self._exploration_rewards())
        bounds = epsilon * (2.0 - epsilon)
        return self._anticommuting_minimum(bounds)

    def inclusion(self, policy: np.ndarray) -> np.ndarray:
        return explored_inclusion(
            _basis_to_pauli(policy, self._size), self._uniform_inclusion, self.exploration
        )

    def _clique_scores(self) -> np.ndarray:
        return self._clique_membership @ self._exploration_rewards()

    def generator_scores(self, clique_scores: np.ndarray | None = None) -> np.ndarray:
        if clique_scores is None:
            clique_scores = self._clique_scores()
        pattern_best = np.full(len(self._uniform_inclusion), -np.inf)
        np.maximum.at(pattern_best, self._clique_codes, clique_scores)
        return _pauli_to_basis(pattern_best, self._size, np.maximum)

    def coverage_weights(self) -> np.ndarray:
        epsilon = self.generator_epsilon()
        bounds = epsilon * (2.0 - epsilon)
        weights = self._anticommuting_minimum(bounds)
        weights = np.maximum(weights, self.pauli_floor)
        weights[0] = 0.0
        return weights

    def coverage_scores(self) -> np.ndarray:
        epsilon = self.generator_epsilon()
        pauli_rewards = self.coverage_weights() / (1.0 + self.pauli_counts)
        anti_rewards = epsilon / (1.0 + self.anti_counts)
        pauli_total = float(pauli_rewards.sum())
        anti_total = float(anti_rewards.sum())
        biased_total = self.commuting_bias * pauli_total
        if biased_total + anti_total <= 1e-15:
            mixing = 0.0 if self.commuting_bias == 0.0 else 0.5
        else:
            mixing = biased_total / (biased_total + anti_total)
        coverage = _pauli_to_basis(pauli_rewards, self._size)
        anti_scores = self._basis_anti_float @ anti_rewards
        axis_rewards = 1.0 / (1.0 + self.local_counts.astype(float))
        axis_scores = np.sum(
            axis_rewards[np.arange(len(self.subsystem))[None, :], self.local_bases], axis=1
        ) / len(self.subsystem)
        normalized_coverage = coverage / pauli_total if pauli_total > 0.0 else 0.0
        normalized_anti = anti_scores / anti_total if anti_total > 0.0 else 0.0
        return (
            mixing * normalized_coverage
            + (1.0 - mixing) * normalized_anti
            + self.local_regularization * axis_scores
        )

    def softmax_policy(self, generator_scores: np.ndarray | None = None) -> np.ndarray:
        if generator_scores is None:
            generator_scores = self.generator_scores()
        generator_maximum = float(generator_scores.max())
        normalized_generator = (
            generator_scores / generator_maximum
            if generator_maximum > 0.0
            else np.zeros_like(generator_scores)
        )
        scores = (
            self.purity_weight * self.coverage_scores()
            + self.generator_weight * normalized_generator
        )
        return softmax_policy_weights(scores, self.temperature)

    def design_policy(self, generator_scores: np.ndarray | None = None) -> np.ndarray:
        self.policy = self.softmax_policy(generator_scores)
        return self.policy

    def draw(self) -> ShotDesign:
        clique_scores = self._clique_scores()
        if self._frozen_design is None or self.shot_count - self._last_update >= self.update_every:
            policy = self.design_policy(self.generator_scores(clique_scores))
            self._frozen_design = (policy, self.inclusion(policy))
            self._last_update = self.shot_count
        policy, inclusion = self._frozen_design

        if self.rng.random() < self.exploration:
            basis = self.rng.integers(0, 3, size=self.n_qubits, dtype=np.int8)
        else:
            basis_id = int(self.rng.choice(len(policy), p=policy))
            fits = self._pattern_fits(basis_id)[self._clique_pattern]
            best = np.max(clique_scores[fits])
            clique = int(self.rng.choice(np.flatnonzero(fits & np.isclose(clique_scores, best))))
            basis = self.rng.integers(0, 3, size=self.n_qubits, dtype=np.int8)
            clique_axes = self._clique_axes[clique]
            basis[self._region[clique_axes >= 0]] = clique_axes[clique_axes >= 0]
            basis[self._sites] = self.local_bases[basis_id]
        return ShotDesign(basis=basis, local_sites=self.subsystem, local_inclusion=inclusion)

    def _covered_generators(self, basis: np.ndarray) -> np.ndarray:
        on_region = np.asarray(basis)[self._region]
        return np.flatnonzero(
            np.all(
                (self._generator_axes < 0) | (self._generator_axes == on_region[None, :]),
                axis=1,
            )
        )

    def parity_probes(self, basis: np.ndarray) -> tuple[Pauli, ...]:
        return tuple(self.generators[index] for index in self._covered_generators(basis))

    def observe(
        self,
        design: ShotDesign,
        parities: np.ndarray | None = None,
    ) -> None:
        if design.local_inclusion is None:
            raise ValueError("OnlineSubsystemSelector observes only its own designs")
        local = np.asarray(design.basis, dtype=np.int64)[self._sites]
        codes = self._membership @ ((local + 1) * self._digit_powers)
        self.shot_count += 1
        basis_id = int(local @ self._basis_powers)
        self.pauli_counts[codes] += 1
        self.anti_counts += self._basis_anti[basis_id]
        self.local_counts[np.arange(len(self.subsystem)), local] += 1

        covered = self._covered_generators(design.basis)
        parities = np.zeros(0, dtype=np.int64) if parities is None else np.asarray(parities)
        if len(parities) != len(covered):
            raise ValueError("One parity is required for every covered generator")
        self.generator_counts[covered] += 1
        self.generator_sums[covered] += parities.astype(np.int64)


def shared_magic_probe(
    designs: tuple[ShotDesign, ShotDesign, ShotDesign, ShotDesign],
    rng: np.random.Generator,
) -> tuple[Pauli, int]:
    bases = np.stack([design.basis for design in designs])
    matching = np.all(bases == bases[0], axis=0)
    sites = np.flatnonzero(matching)
    include = rng.integers(0, 2, size=len(sites), dtype=np.int8).astype(bool)
    selected = tuple(int(site) for site in sites[include])
    return pauli_from_basis_subset(bases[0], selected), int(len(sites))
