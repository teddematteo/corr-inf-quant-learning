"""Deterministic, inspectable execution budgets; no state information is used."""

from __future__ import annotations

from dataclasses import asdict
from math import ceil
from typing import TYPE_CHECKING

from ._compat import dataclass

if TYPE_CHECKING:
    from .simulation import SimulationConfig


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    backend: str
    dense_memory_mib: float
    pauli_samples: int
    candidates: int
    max_cliques: int
    update_every: int

    def as_dict(self) -> dict:
        return asdict(self)


def plan_resources(config: SimulationConfig, generator_count: int | None = None) -> ResourcePlan:
    """None means automatic; explicit positive budgets always take precedence."""
    size = config.n_qubits if config.property == "MAGIC" else config.subsystem_size
    if config.property == "MAGIC" and not any(
        m in config.methods for m in ("CS", "ConEnh", "SI-CS")
    ):
        return ResourcePlan("magic", 0.0, 0, config.conenh_candidates or 64,
                            config.conenh_max_cliques or 4096, config.conenh_update_every or 1)
    generators = config.n_qubits if generator_count is None else generator_count
    dense_bytes = (
        4**size * (generators + size + 80 + (16 if config.property == "MAGIC" else 0))
        + 3**size * (2 * generators + 2 * size + 48)
        + 2**size * (24 * size + 32)
        + 4096 * (config.n_qubits + 8 * generators)
    )
    dense_mib = dense_bytes / 2**20
    backend = config.subsystem_backend.lower()
    if backend == "auto":
        backend = "dense" if size <= 8 and dense_mib <= config.memory_budget_mib else "sampled"
    if backend == "dense" and (size > 12 or dense_mib > config.memory_budget_mib):
        raise ValueError(
            f"Dense |A|={size} needs approximately {dense_mib:.1f} MiB; "
            "use subsystem_backend='sampled' or increase memory_budget_mib "
            "(the dense backend supports at most 12 qubits)."
        )
    samples = config.pauli_samples or min(4096, max(512, 32 * size))
    candidates = config.conenh_candidates or (
        64 if backend == "dense" else min(256, max(64, 4 * size))
    )

    automatic_cliques = 4096 if backend == "dense" else min(2048, max(256, 16 * generators))
    bytes_per_clique = max(1, config.n_qubits + 8 * generators)
    memory_cliques = max(2, int(config.memory_budget_mib * 2**20 / (16 * bytes_per_clique)))
    cliques = config.conenh_max_cliques or min(automatic_cliques, memory_cliques)
    if config.conenh_update_every is not None:
        update = config.conenh_update_every
    elif backend == "dense":
        update = 1
    else:
        work = candidates * (samples + min(cliques, 4096))
        update = min(256, max(1, ceil(work / 8192)))

        update = min(update, max(1, config.post_shots // 8))
    return ResourcePlan(backend, dense_mib, samples, candidates, cliques, update)
