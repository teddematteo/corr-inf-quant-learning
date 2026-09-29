from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Pauli:
    x: int = 0
    z: int = 0

    @property
    def support(self) -> int:
        return self.x | self.z

    @property
    def weight(self) -> int:
        return self.support.bit_count()

    def __xor__(self, other: Pauli) -> Pauli:
        return Pauli(self.x ^ other.x, self.z ^ other.z)


def axis_at(pauli: Pauli, qubit: int) -> int | None:
    x = (pauli.x >> qubit) & 1
    z = (pauli.z >> qubit) & 1
    if not (x or z):
        return None
    if x and z:
        return Y_AXIS
    return X_AXIS if x else Z_AXIS


def pauli_from_basis_subset(basis: np.ndarray, qubits: Iterable[int]) -> Pauli:
    x = 0
    z = 0
    for qubit in qubits:
        axis = int(basis[qubit])
        if axis in (X_AXIS, Y_AXIS):
            x |= 1 << qubit
        if axis in (Y_AXIS, Z_AXIS):
            z |= 1 << qubit
    return Pauli(x, z)
