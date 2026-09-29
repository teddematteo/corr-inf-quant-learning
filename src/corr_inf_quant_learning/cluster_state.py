from __future__ import annotations

from collections import deque
from functools import lru_cache
from math import ceil, floor, sqrt

import numpy as np

from .pauli import Pauli, pauli_from_basis_subset


def grid_shape(n_qubits: int) -> tuple[int, int]:
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    rows = max(1, floor(sqrt(n_qubits)))
    columns = ceil(n_qubits / rows)
    return rows, columns


def grid_adjacency(n_qubits: int) -> tuple[int, ...]:
    _, columns = grid_shape(n_qubits)
    rows_and_columns = [divmod(index, columns) for index in range(n_qubits)]
    adjacency = [0] * n_qubits
    for left in range(n_qubits):
        row_left, col_left = rows_and_columns[left]
        for right in range(left + 1, n_qubits):
            row_right, col_right = rows_and_columns[right]
            if abs(row_left - row_right) + abs(col_left - col_right) == 1:
                adjacency[left] |= 1 << right
                adjacency[right] |= 1 << left
    return tuple(adjacency)


def choose_connected_subsystem(adjacency: tuple[int, ...], size: int) -> tuple[int, ...]:
    if not 1 <= size <= len(adjacency):
        raise ValueError("subsystem size must lie between 1 and n_qubits")
    rows, columns = grid_shape(len(adjacency))
    centre = min(len(adjacency) - 1, (rows // 2) * columns + columns // 2)
    selected: list[int] = []
    queue: deque[int] = deque((centre,))
    seen = {centre}
    while queue and len(selected) < size:
        vertex = queue.popleft()
        selected.append(vertex)
        neighbours = adjacency[vertex]
        while neighbours:
            low = neighbours & -neighbours
            neighbour = low.bit_length() - 1
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
            neighbours ^= low
    if len(selected) < size:
        selected.extend(index for index in range(len(adjacency)) if index not in seen)
    return tuple(selected[:size])


def random_rotation_vertices(
    n_qubits: int, count: int, rng: np.random.Generator
) -> tuple[int, ...]:
    if not 0 <= count <= n_qubits:
        raise ValueError("count must lie between zero and n_qubits")
    order = np.arange(n_qubits)
    rng.shuffle(order)
    return tuple(int(vertex) for vertex in order[:count])


def _fwht(values: np.ndarray) -> np.ndarray:
    transformed = np.asarray(values, dtype=float).copy()
    block = 1
    while block < len(transformed):
        pairs = transformed.reshape(-1, 2, block)
        left = pairs[:, 0, :].copy()
        pairs[:, 0, :] += pairs[:, 1, :]
        pairs[:, 1, :] = left - pairs[:, 1, :]
        block *= 2
    return transformed


class RotatedClusterState:
    def __init__(
        self,
        n_qubits: int,
        k_rotations: int,
        theta: float = np.pi / 8,
        vertex_seed: int = 1729,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.theta = float(theta)
        self.adjacency = grid_adjacency(self.n_qubits)
        self.rotation_vertices = random_rotation_vertices(
            self.n_qubits, int(k_rotations), np.random.default_rng(vertex_seed)
        )
        self.k_rotations = len(self.rotation_vertices)
        self._rotation_mask = sum(1 << vertex for vertex in self.rotation_vertices)
        self._cosine = float(np.cos(2 * self.theta))
        self._sine = float(np.sin(2 * self.theta))
        self.generators = tuple(
            Pauli(1 << vertex, self.adjacency[vertex]) for vertex in range(self.n_qubits)
        )

    def _graph_multiply(self, mask: int) -> int:
        result = 0
        remaining = mask
        while remaining:
            low = remaining & -remaining
            result ^= self.adjacency[low.bit_length() - 1]
            remaining ^= low
        return result

    def _stabilizer_sign(self, x_mask: int, z_mask: int) -> int:
        twice_edges = 0
        remaining = x_mask
        while remaining:
            low = remaining & -remaining
            vertex = low.bit_length() - 1
            twice_edges += (self.adjacency[vertex] & x_mask).bit_count()
            remaining ^= low
        edges = twice_edges // 2
        y_count = (x_mask & z_mask).bit_count()
        return -1 if ((edges + y_count // 2) & 1) else 1

    @lru_cache(maxsize=1 << 18)
    def expectation(self, pauli: Pauli) -> float:
        toggles = pauli.z ^ self._graph_multiply(pauli.x)
        allowed_toggles = self._rotation_mask & pauli.x
        if toggles & ~allowed_toggles:
            return 0.0

        coefficient = 1.0
        for vertex in self.rotation_vertices:
            if not ((pauli.x >> vertex) & 1):
                continue
            if (toggles >> vertex) & 1:
                local_sign = -1.0 if ((pauli.z >> vertex) & 1) else 1.0
                coefficient *= local_sign * self._sine
            else:
                coefficient *= self._cosine

        stabilizer_z = pauli.z ^ toggles
        coefficient *= self._stabilizer_sign(pauli.x, stabilizer_z)
        if abs(coefficient) < 1e-15:
            return 0.0
        return float(np.clip(coefficient, -1.0, 1.0))

    @lru_cache(maxsize=1 << 15)
    def logical_image(self, pauli: Pauli) -> tuple[tuple[int, ...], int] | None:
        toggles = pauli.z ^ self._graph_multiply(pauli.x)
        if toggles & ~self._rotation_mask:
            return None

        axes = [-1] * self.k_rotations
        sign = 1
        for logical, vertex in enumerate(self.rotation_vertices):
            toggled = (toggles >> vertex) & 1
            if not ((pauli.x >> vertex) & 1):
                if toggled:
                    axes[logical] = 2
                continue
            if toggled:
                axes[logical] = 1
                if (pauli.z >> vertex) & 1:
                    sign *= -1
            else:
                axes[logical] = 0
        stabilizer_z = pauli.z ^ toggles
        sign *= self._stabilizer_sign(pauli.x, stabilizer_z)
        return tuple(axes), sign

    def _moments_for_axes(self, basis: np.ndarray, sites: tuple[int, ...]) -> np.ndarray:
        moments = np.empty(1 << len(sites), dtype=float)
        for subset in range(1 << len(sites)):
            selected = (sites[index] for index in range(len(sites)) if (subset >> index) & 1)
            moments[subset] = self.expectation(pauli_from_basis_subset(basis, selected))
        return moments

    @lru_cache(maxsize=4096)
    def _outcome_probabilities(
        self, sites: tuple[int, ...], local_axes: tuple[int, ...]
    ) -> np.ndarray:
        basis = np.zeros(self.n_qubits, dtype=np.int8)
        for site, axis in zip(sites, local_axes, strict=True):
            basis[site] = axis
        probabilities = _fwht(self._moments_for_axes(basis, sites)) / (1 << len(sites))
        probabilities[np.abs(probabilities) < 1e-14] = 0.0
        if np.min(probabilities) < -1e-10:
            raise RuntimeError("Correlator oracle produced a non-positive marginal")
        probabilities = np.clip(probabilities, 0.0, None)
        return probabilities / probabilities.sum()

    def sample_outcomes(
        self, basis: np.ndarray, sites: tuple[int, ...], rng: np.random.Generator
    ) -> np.ndarray:
        local_axes = tuple(int(basis[site]) for site in sites)
        probabilities = self._outcome_probabilities(sites, local_axes)
        outcome_index = int(rng.choice(len(probabilities), p=probabilities))
        return np.asarray(
            [1 if not ((outcome_index >> index) & 1) else -1 for index in range(len(sites))],
            dtype=np.int8,
        )

    @lru_cache(maxsize=1 << 10)
    def _joint_local_parities_probabilities(
        self,
        sites: tuple[int, ...],
        local_axes: tuple[int, ...],
        paulis: tuple[Pauli, ...],
    ) -> np.ndarray:
        basis = np.zeros(self.n_qubits, dtype=np.int8)
        for site, axis in zip(sites, local_axes, strict=True):
            basis[site] = axis
        local_dimension = 1 << len(sites)
        local_paulis = [
            pauli_from_basis_subset(
                basis,
                (sites[index] for index in range(len(sites)) if (subset >> index) & 1),
            )
            for subset in range(local_dimension)
        ]
        moments = np.empty(local_dimension << len(paulis), dtype=float)
        for parity_subset in range(1 << len(paulis)):
            product = Pauli()
            for index, pauli in enumerate(paulis):
                if (parity_subset >> index) & 1:
                    product = product ^ pauli
            offset = parity_subset * local_dimension
            for subset, local_pauli in enumerate(local_paulis):
                moments[offset + subset] = self.expectation(local_pauli ^ product)
        probabilities = _fwht(moments) / len(moments)
        probabilities[np.abs(probabilities) < 1e-14] = 0.0
        if np.min(probabilities) < -1e-10:
            raise RuntimeError("Joint marginal is not positive")
        probabilities = np.clip(probabilities, 0.0, None)
        return probabilities / probabilities.sum()

    def sample_local_and_parities(
        self,
        basis: np.ndarray,
        sites: tuple[int, ...],
        paulis: tuple[Pauli, ...],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_axes = tuple(int(basis[site]) for site in sites)
        probabilities = self._joint_local_parities_probabilities(sites, local_axes, tuple(paulis))
        outcome_index = int(rng.choice(len(probabilities), p=probabilities))
        outcomes = np.asarray(
            [1 if not ((outcome_index >> index) & 1) else -1 for index in range(len(sites))],
            dtype=np.int8,
        )
        parity_bits = outcome_index >> len(sites)
        parities = np.asarray(
            [1 if not ((parity_bits >> index) & 1) else -1 for index in range(len(paulis))],
            dtype=np.int8,
        )
        return outcomes, parities

    def sample_mode_parities(
        self, paulis: tuple[Pauli, ...], rng: np.random.Generator
    ) -> np.ndarray:
        modes: set[int] = set()
        for pauli in paulis:
            image = self.logical_image(pauli)
            if image is None:
                raise ValueError("A Pauli anticommuting with a stabilizer has no joint law here")
            active = [mode for mode, axis in enumerate(image[0]) if axis >= 0]
            if not active:
                continue
            if len(active) != 1 or active[0] in modes:
                raise ValueError("Each Pauli must decode to its own single logical mode")
            modes.add(active[0])
        means = np.asarray([self.expectation(pauli) for pauli in paulis], dtype=float)
        return np.where(rng.random(len(paulis)) < 0.5 * (1.0 + means), 1, -1).astype(np.int8)

    def sample_pauli(self, pauli: Pauli, rng: np.random.Generator) -> int:
        mean = self.expectation(pauli)
        return 1 if rng.random() < 0.5 * (1.0 + mean) else -1

    def local_purity(self, sites: tuple[int, ...]) -> float:
        total = 0.0
        for code in range(4 ** len(sites)):
            x = 0
            z = 0
            for index, site in enumerate(sites):
                digit = (code >> (2 * index)) & 3
                if digit in (1, 2):
                    x |= 1 << site
                if digit in (2, 3):
                    z |= 1 << site
            total += self.expectation(Pauli(x, z)) ** 2
        return float(total / (1 << len(sites)))

    def stabilizer_renyi_magic(self) -> float:
        if not self.k_rotations:
            return 0.0
        local_fourth_moment = (1.0 + self._cosine**4 + self._sine**4) / 2.0
        return float(-self.k_rotations * np.log2(local_fourth_moment))
