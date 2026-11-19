"""Preparation and exact measurement simulators, separate from learning code."""

from __future__ import annotations

from functools import lru_cache
from typing import Union

import numpy as np

from ._compat import bit_count
from .cluster_state import RotatedClusterState, grid_adjacency, random_rotation_vertices
from .generators import ghz_generators
from .pauli import Pauli, axis_at
from .rotations import sample_rotation_angles

STATE_TYPES = ("CLUSTER", "GHZ")
_EIGENVECTORS = {
    0: np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    1: np.array([[1, 1j], [1, -1j]], dtype=complex) / np.sqrt(2),
    2: np.eye(2, dtype=complex),
}


class RotatedSparseState:
    """GHZ oracle with analytic moments and bond-two sequential Born sampling."""

    def __init__(
        self,
        n_qubits: int,
        k_rotations: int,
        theta: float = np.pi / 8,
        vertex_seed: int = 1729,
        *,
        state: str,
        rows: int | None = None,
        columns: int | None = None,
        rotation_vertices: tuple[int, ...] | None = None,
        theta_range: tuple[float, float] | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.state = state.upper()
        if self.state != "GHZ":
            raise ValueError("Sparse state must be GHZ")
        self.adjacency = grid_adjacency(n_qubits, rows=rows, columns=columns)
        if not np.isfinite(theta):
            raise ValueError("theta must be finite")
        if not 0 <= k_rotations <= n_qubits:
            raise ValueError("k_rotations must lie between 0 and n_qubits")
        self.theta = float(theta)
        self.rotation_vertices = (
            random_rotation_vertices(n_qubits, k_rotations, np.random.default_rng(vertex_seed))
            if rotation_vertices is None
            else tuple(rotation_vertices)
        )
        if (
            len(self.rotation_vertices) != k_rotations
            or len(set(self.rotation_vertices)) != k_rotations
            or any(
                not isinstance(v, int) or isinstance(v, bool) or (not 0 <= v < n_qubits)
                for v in self.rotation_vertices
            )
        ):
            raise ValueError("rotation_vertices must contain k_rotations distinct valid indices")
        self.k_rotations = k_rotations
        self._rotation_mask = sum(1 << site for site in self.rotation_vertices)
        self.rotation_angles = sample_rotation_angles(
            k_rotations, self.theta, theta_range, vertex_seed
        )
        self._angles = np.zeros(n_qubits, dtype=float)
        self._angles[list(self.rotation_vertices)] = self.rotation_angles
        self._cosines = np.cos(2 * self._angles)
        self._sines = np.sin(2 * self._angles)
        self.generators = ghz_generators(n_qubits)
        self._tensors = self._preparation_tensors()
        self._right = [np.ones((1, 1), dtype=complex)] * (n_qubits + 1)
        for site in reversed(range(n_qubits)):
            tensor = self._tensors[site]
            self._right[site] = sum(
                tensor[axis] @ self._right[site + 1] @ tensor[axis].conj().T for axis in range(2)
            )
        self._measurement_tensors = tuple(
            tuple(np.tensordot(kets.conj(), tensor, axes=(1, 0)) for kets in _EIGENVECTORS.values())
            for tensor in self._tensors
        )

    def _preparation_tensors(self) -> tuple[np.ndarray, ...]:
        tensors = []
        for site in range(self.n_qubits):
            tensor = np.zeros((2, 2, 2), dtype=complex)
            tensor[0, 0, 0] = tensor[1, 1, 1] = 1.0
            left = np.array([[1, 1]], dtype=complex) / np.sqrt(2)
            right = np.array([[1], [1]], dtype=complex)
            if site == 0:
                tensor = np.stack([left @ matrix for matrix in tensor])
            if site == self.n_qubits - 1:
                tensor = tensor @ right
            if self._rotation_mask >> site & 1:
                angle = self._angles[site]
                if self.state == "GHZ" and site > 0:
                    tensor = np.cos(angle) * tensor + 1j * np.sin(angle) * tensor[::-1]
                else:
                    tensor *= np.exp(1j * angle * np.array([1, -1]))[:, None, None]
            tensors.append(tensor)
        return tuple(tensors)

    @lru_cache(maxsize=1 << 18)
    def expectation(self, pauli: Pauli) -> float:
        leaves = (1 << self.n_qubits) - 1 ^ 1
        x = pauli.x & 1 | pauli.z & leaves
        z = pauli.z & 1 | pauli.x & leaves
        sign = (-1) ** bit_count(pauli.x & pauli.z & leaves)
        leaf_x_count = bit_count(x & leaves)
        neighbours = (leaves if x & 1 else 0) | leaf_x_count & 1
        transformed_z = z ^ neighbours
        phase = bit_count(x & z) - bit_count(x & transformed_z)
        sign *= (-1) ** ((x & 1) * leaf_x_count) * (1j**phase).real
        if transformed_z & ~x:
            return 0.0
        y = x & transformed_z
        if y & ~self._rotation_mask:
            return 0.0
        coefficient = float(sign)
        active = x & self._rotation_mask
        while active:
            bit = active & -active
            site = bit.bit_length() - 1
            coefficient *= -self._sines[site] if y & bit else self._cosines[site]
            active ^= bit
        return float(coefficient)

    def sample_outcomes(
        self, basis: np.ndarray, sites: tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        left = np.ones((1,), dtype=complex)
        outcomes = np.empty(self.n_qubits, dtype=np.int8)
        for site, axis in enumerate(basis):
            candidates = left @ self._measurement_tensors[site][int(axis)]
            probabilities = np.array(
                [
                    (candidate @ self._right[site + 1] @ candidate.conj()).real
                    for candidate in candidates
                ]
            )
            probabilities = np.maximum(probabilities, 0)
            total = probabilities.sum()
            chosen = int(rng.random() >= probabilities[0] / total)
            left = candidates[chosen] / np.sqrt(probabilities[chosen])
            outcomes[site] = 1 if chosen == 0 else -1
        return outcomes[np.asarray(sites, dtype=int)]

    def sample_local_and_parities(
        self,
        basis: np.ndarray,
        sites: tuple[int, ...],
        paulis: tuple[Pauli, ...],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        full = self.sample_outcomes(basis, tuple(range(self.n_qubits)), rng)
        parities = []
        for pauli in paulis:
            support = [site for site in range(self.n_qubits) if pauli.support >> site & 1]
            if any(axis_at(pauli, site) != basis[site] for site in support):
                raise ValueError("Parities must be compatible with the local measurement basis")
            parities.append(int(np.prod(full[support])))
        return (full[np.asarray(sites, dtype=int)], np.asarray(parities, dtype=np.int8))

    def sample_mode_parities(
        self, paulis: tuple[Pauli, ...], rng: np.random.Generator
    ) -> np.ndarray:
        basis = np.full(self.n_qubits, 2, dtype=np.int8)
        assigned = np.zeros(self.n_qubits, dtype=bool)
        for pauli in paulis:
            for site in range(self.n_qubits):
                axis = axis_at(pauli, site)
                if axis is not None:
                    if assigned[site] and basis[site] != axis:
                        raise ValueError("Joint parities must be locally compatible")
                    basis[site] = axis
                    assigned[site] = True
        return self.sample_local_and_parities(basis, (), paulis, rng)[1]

    def sample_pauli(self, pauli: Pauli, rng: np.random.Generator) -> int:
        return 1 if rng.random() < (1 + self.expectation(pauli)) / 2 else -1

    def local_purity(self, sites: tuple[int, ...]) -> float:
        return 1.0 if len(sites) in (0, self.n_qubits) else 0.5

    def stabilizer_renyi_magic(self) -> float:
        sites = list(self.rotation_vertices)
        local_moments = (1 + self._cosines[sites] ** 4 + self._sines[sites] ** 4) / 2
        return float(-np.sum(np.log2(local_moments)))


StateOracle = Union[RotatedClusterState, RotatedSparseState]


def make_state(state: str, n_qubits: int, k_rotations: int, **kwargs) -> StateOracle:
    state = state.upper()
    if state == "CLUSTER":
        return RotatedClusterState(n_qubits, k_rotations, **kwargs)
    if state == "GHZ":
        return RotatedSparseState(n_qubits, k_rotations, state=state, **kwargs)
    raise ValueError(f"state must be one of {STATE_TYPES}")
