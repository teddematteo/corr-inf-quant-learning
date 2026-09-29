from __future__ import annotations

from functools import lru_cache

import numpy as np

_EIGENVECTORS = {
    0: (
        np.asarray([1.0, 1.0], dtype=complex) / np.sqrt(2.0),
        np.asarray([1.0, -1.0], dtype=complex) / np.sqrt(2.0),
    ),
    1: (
        np.asarray([1.0, 1.0j], dtype=complex) / np.sqrt(2.0),
        np.asarray([1.0, -1.0j], dtype=complex) / np.sqrt(2.0),
    ),
    2: (
        np.asarray([1.0, 0.0], dtype=complex),
        np.asarray([0.0, 1.0], dtype=complex),
    ),
}

_PAULI = (
    np.eye(2, dtype=complex),
    np.asarray([[0, 1], [1, 0]], dtype=complex),
    np.asarray([[0, -1j], [1j, 0]], dtype=complex),
    np.diag([1.0, -1.0]).astype(complex),
)

_BETA_ONE, _BETA_TWO = 0.9, 0.999


@lru_cache(maxsize=4096)
def _measurement_ket(basis: tuple[int, ...], outcomes: tuple[int, ...]) -> np.ndarray:
    vector = np.asarray([1.0 + 0.0j])
    for axis, outcome in zip(basis, outcomes, strict=True):
        vector = np.kron(vector, _EIGENVECTORS[int(axis)][0 if outcome > 0 else 1])
    return vector


@lru_cache(maxsize=4096)
def _pauli_matrix(code: int, n_qubits: int) -> np.ndarray:
    matrix = np.asarray([1.0 + 0.0j])
    work = int(code)
    for _ in range(n_qubits):
        matrix = np.kron(matrix, _PAULI[work & 3])
        work >>= 2
    return matrix


class AGHDO:
    def __init__(
        self,
        n_qubits: int,
        rank: int = 16,
        learning_rate: float = 0.025,
        seed: int = 0,
    ) -> None:
        if not 1 <= n_qubits <= 8:
            raise ValueError("The local AGHDO head supports 1 to 8 qubits")
        self.n_qubits = int(n_qubits)
        self.dimension = 1 << self.n_qubits
        self.rank = min(max(1, int(rank)), self.dimension)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        if self.rank == self.dimension:
            factor = np.eye(self.dimension, dtype=complex)
            factor += 1e-3 * (
                self.rng.normal(size=factor.shape) + 1j * self.rng.normal(size=factor.shape)
            )
        else:
            factor = self.rng.normal(size=(self.dimension, self.rank))
            factor = factor + 1j * self.rng.normal(size=factor.shape)
        self.factor = factor / np.linalg.norm(factor)
        self._adam_m = np.zeros_like(self.factor)
        self._adam_v = np.zeros_like(self.factor.real)
        self._adam_step = 0
        self._states: list[np.ndarray] = []

    def add_measurement(self, basis: np.ndarray, outcomes: np.ndarray) -> None:
        if len(basis) != self.n_qubits or len(outcomes) != self.n_qubits:
            raise ValueError("Local basis/outcome size does not match n_qubits")
        key_basis = tuple(int(axis) for axis in basis)
        key_outcomes = tuple(int(outcome) for outcome in outcomes)
        self._states.append(_measurement_ket(key_basis, key_outcomes))

    def fit(self, steps: int, batch_size: int) -> None:
        if not self._states or steps <= 0:
            return
        data = np.asarray(self._states, dtype=complex)
        for _ in range(int(steps)):
            indices = self.rng.integers(0, len(data), size=min(batch_size, len(data)))
            states = data[indices]
            normalization = float(np.vdot(self.factor, self.factor).real)
            amplitudes = states.conj() @ self.factor
            unnormalized = np.sum(np.abs(amplitudes) ** 2, axis=1).real
            weighted_amplitudes = amplitudes / np.clip(unnormalized[:, None], 1e-12, None)
            gradient = -(states.T @ weighted_amplitudes) / len(states)
            gradient += self.factor / normalization
            self._adam_step += 1
            self._adam_m = _BETA_ONE * self._adam_m + (1.0 - _BETA_ONE) * gradient
            self._adam_v = _BETA_TWO * self._adam_v + (1.0 - _BETA_TWO) * np.abs(gradient) ** 2
            m_hat = self._adam_m / (1.0 - _BETA_ONE**self._adam_step)
            v_hat = self._adam_v / (1.0 - _BETA_TWO**self._adam_step)
            self.factor -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.factor /= max(np.linalg.norm(self.factor), 1e-12)

    def density_matrix(self) -> np.ndarray:
        rho = self.factor @ self.factor.conj().T
        return rho / np.trace(rho).real

    def sample_purity(self, samples: int, rng: np.random.Generator) -> float:
        rho = self.density_matrix()
        total_paulis = 4**self.n_qubits
        if total_paulis == 1:
            return 1.0
        codes = rng.integers(1, total_paulis, size=max(1, int(samples)))
        moments = np.empty(len(codes), dtype=float)
        for index, code in enumerate(codes):
            matrix = _pauli_matrix(int(code), self.n_qubits)
            moments[index] = float(np.trace(rho @ matrix).real)
        return (1.0 + (total_paulis - 1) * float(np.mean(moments**2))) / self.dimension


class LogicalAGHDO:
    def __init__(self, n_logical: int, learning_rate: float = 0.03, seed: int = 0) -> None:
        self.n_logical = int(n_logical)
        self.learning_rate = float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.raw = np.ones((self.n_logical, 3), dtype=float)
        self.raw += 0.02 * self.rng.normal(size=self.raw.shape)
        self._adam_m = np.zeros_like(self.raw)
        self._adam_v = np.zeros_like(self.raw)
        self._adam_step = 0
        self._modes: list[int] = []
        self._axes: list[int] = []
        self._values: list[int] = []

    def bloch_vectors(self) -> np.ndarray:
        if not self.n_logical:
            return np.empty((0, 3), dtype=float)
        norms = np.linalg.norm(self.raw, axis=1, keepdims=True)
        return self.raw / np.clip(norms, 1e-12, None)

    def add_encoded(self, image: tuple[tuple[int, ...], int] | None, outcome: int) -> None:
        if image is None:
            return
        axes, sign = image
        if len(axes) != self.n_logical:
            raise ValueError("Logical image has the wrong size")
        axis_array = np.asarray(axes, dtype=np.int8)
        active = np.flatnonzero(axis_array >= 0)
        if not len(active):
            return
        if len(active) != 1:
            raise ValueError("LogicalAGHDO records single-mode observables only")
        self._modes.append(int(active[0]))
        self._axes.append(int(axis_array[active[0]]))
        self._values.append(int(sign) * int(outcome))

    @property
    def record_count(self) -> int:
        return len(self._modes)

    def fit(self, steps: int) -> None:
        if not self.record_count or not self.n_logical or steps <= 0:
            return
        modes = np.asarray(self._modes, dtype=np.int64)
        axes = np.asarray(self._axes, dtype=np.int64)
        values = np.asarray(self._values, dtype=float)
        for _ in range(int(steps)):
            bloch = self.bloch_vectors()
            gradient_bloch = np.zeros_like(bloch)
            signed = np.clip(values * bloch[modes, axes], -1 + 1e-9, 1 - 1e-9)
            np.add.at(gradient_bloch, (modes, axes), -values / (1.0 + signed))
            gradient_bloch /= len(modes)
            norms = np.linalg.norm(self.raw, axis=1, keepdims=True)
            radial = np.sum(gradient_bloch * bloch, axis=1, keepdims=True)
            gradient_raw = (gradient_bloch - radial * bloch) / np.clip(norms, 1e-12, None)
            self._adam_step += 1
            self._adam_m = _BETA_ONE * self._adam_m + (1.0 - _BETA_ONE) * gradient_raw
            self._adam_v = _BETA_TWO * self._adam_v + (1.0 - _BETA_TWO) * gradient_raw**2
            m_hat = self._adam_m / (1.0 - _BETA_ONE**self._adam_step)
            v_hat = self._adam_v / (1.0 - _BETA_TWO**self._adam_step)
            self.raw -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.raw /= np.clip(np.linalg.norm(self.raw, axis=1, keepdims=True), 1e-12, None)

    def sample_magic(self, samples_per_qubit: int, rng: np.random.Generator) -> float:
        if not self.n_logical:
            return 0.0
        bloch = self.bloch_vectors()
        samples = max(8, int(samples_per_qubit))
        axes = rng.integers(0, 4, size=(self.n_logical, samples))
        values = np.ones_like(axes, dtype=float)
        for pauli_axis in range(1, 4):
            mask = axes == pauli_axis
            rows = np.nonzero(mask)[0]
            values[mask] = bloch[rows, pauli_axis - 1]
        local_fourth = np.clip(2.0 * np.mean(values**4, axis=1), 0.5, 1.0)
        return float(-np.sum(np.log2(local_fourth)))
