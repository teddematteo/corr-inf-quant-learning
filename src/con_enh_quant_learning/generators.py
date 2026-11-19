"""Pauli reference generators; their means are learned from measurements."""

from __future__ import annotations

from .pauli import Pauli


def ghz_generators(n_qubits: int) -> tuple[Pauli, ...]:
    """Star basis: X on every qubit, then Z_0 Z_i for each leaf."""
    return (Pauli((1 << n_qubits) - 1),) + tuple(
        Pauli(0, 1 | (1 << site)) for site in range(1, n_qubits)
    )


def measurement_generators(
    generators: tuple[Pauli, ...], *, max_weight: int | None = 2
) -> tuple[Pauli, ...]:
    """Retain all Pauli constraints, including GHZ's global X word."""
    if max_weight is not None and (
        not isinstance(max_weight, int) or isinstance(max_weight, bool) or max_weight < 1
    ):
        raise ValueError("generator_max_weight must be a positive integer or None")
    if not generators or not all(isinstance(word, Pauli) for word in generators):
        raise ValueError("Expected a nonempty tuple of Pauli generators")
    return tuple(dict.fromkeys(generators))
