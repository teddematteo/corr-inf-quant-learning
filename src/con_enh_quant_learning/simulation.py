from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import field, replace
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from ._compat import dataclass, strict_zip
from .aghdo import AGHDO, FocusedAGHDO
from .cluster_state import choose_connected_subsystem, grid_adjacency
from .compute import ComputeBackend
from .estimators import (
    FactorizedUniformPurityUStatistic,
    IncrementalPauliMomentUStatistic,
    IncrementalPurityUStatistic,
    magic_from_fourth_moment,
)
from .generators import measurement_generators
from .logical import magic_focus, sample_focus_parities
from .measurement import (
    LocalShot,
    MagicProbeSelector,
    OnlineSubsystemSelector,
    PilotData,
    ShotDesign,
    UniformSelector,
    covered_probes,
    probe_axes,
    run_common_pilot,
)
from .pauli import Pauli
from .resources import ResourcePlan, plan_resources
from .rotations import validate_theta_range
from .sampled import (
    PauliPool,
    SampledPauliMomentUStatistic,
    SampledPurityUStatistic,
    SampledSubsystemSelector,
)
from .stabilizer_informed import StabilizerInformedSelector
from .states import STATE_TYPES, StateOracle, make_state

METHODS = ("CS", "ConEnh", "CS-AGHDO", "ConEnh-AGHDO", "SI-CS", "SI-CS-AGHDO")
STREAMS = (
    ("uniform", "CS", "CS-AGHDO"), ("bound", "ConEnh", "ConEnh-AGHDO"),
    ("stabilizer", "SI-CS", "SI-CS-AGHDO"),
)
MAX_SUBSYSTEM_SIZE = 100
_REPORTS_PER_STREAM = 50


@dataclass(slots=True)
class SimulationConfig:
    n_qubits: int = 100
    k_rotations: int = 50
    subsystem_size: int = 5
    methods: tuple[str, ...] = METHODS
    property: Literal["PURITY", "RENYI2", "MAGIC"] = "PURITY"
    theta: float = float(np.pi / 8)

    pilot_shots: int = 8
    post_shots: int = 256
    repetitions: int = 3
    evaluation_points: int = 9
    seed: int = 7
    rotation_seed: int = 1729
    count_pilot_cost: bool = True

    conenh_exploration: float = 0.20
    conenh_commuting_bias: float = 1.0
    conenh_candidates: int | None = None
    conenh_local_regularization: float = 0.05
    conenh_purity_weight: float = 1.0
    conenh_generator_weight: float = 1.0
    conenh_softmax_temperature: float = 0.3
    conenh_pauli_floor: float = 0.1
    conenh_update_every: int | None = None
    conenh_clip_bound: str = "POINT"

    aghdo_rank: int = 8
    aghdo_learning_rate: float = 0.005
    aghdo_fit_steps: int = 12
    aghdo_batch_size: int = 128
    aghdo_feature_densities: tuple[int, ...] = (8, 4)
    aghdo_init_std: float = 0.01
    qns_property_samples: int = 512

    progress_details: bool = True
    subsystem: tuple[int, ...] | None = None
    rows: int | None = None
    columns: int | None = None
    rotation_vertices: tuple[int, ...] | None = None
    conenh_clipping: bool = True
    magic_clipping: bool = True
    state: Literal["CLUSTER", "GHZ"] = "CLUSTER"
    generator_max_weight: int | None = 2
    subsystem_backend: Literal["auto", "dense", "sampled"] = "auto"
    memory_budget_mib: int = 256
    pauli_samples: int | None = None
    pauli_uniform_fraction: float = 0.20
    conenh_max_cliques: int | None = None
    theta_range: tuple[float, float] | None = None

    property_clipping: bool | None = None
    device: Literal["cpu", "cuda", "auto"] = "cpu"

    si_cs_exploration: float | None = None

    def resource_plan(self) -> ResourcePlan:
        """Preview automatic budgets; run events include the actual local-generator count."""
        return plan_resources(self)

    def validate(self) -> None:
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be 'cpu', 'cuda' or 'auto'")
        if not isinstance(self.state, str) or self.state.upper() not in STATE_TYPES:
            raise ValueError(f"state must be one of {STATE_TYPES}")
        self.state = self.state.upper()
        if self.generator_max_weight is not None and (
            not isinstance(self.generator_max_weight, int)
            or isinstance(self.generator_max_weight, bool)
            or self.generator_max_weight < 1
        ):
            raise ValueError("generator_max_weight must be a positive integer or None")
        for name in ("conenh_clipping", "magic_clipping"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.property_clipping is not None and not isinstance(self.property_clipping, bool):
            raise ValueError("property_clipping must be a boolean or None (legacy defaults)")
        integer_fields = (
            "n_qubits",
            "k_rotations",
            "subsystem_size",
            "pilot_shots",
            "post_shots",
            "repetitions",
            "evaluation_points",
            "seed",
            "rotation_seed",
            "aghdo_rank",
            "aghdo_fit_steps",
            "aghdo_batch_size",
            "qns_property_samples",
            "memory_budget_mib",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        optional_budgets = (
            "conenh_candidates", "conenh_update_every", "conenh_max_cliques", "pauli_samples"
        )
        for name in optional_budgets:
            value = getattr(self, name)
            minimum = 2 if name == "conenh_max_cliques" else 1
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < minimum
            ):
                raise ValueError(f"{name} must be None (automatic) or an integer >= {minimum}")
        if self.subsystem_backend not in ("auto", "dense", "sampled"):
            raise ValueError("subsystem_backend must be 'auto', 'dense' or 'sampled'")
        if self.memory_budget_mib < 16:
            raise ValueError("memory_budget_mib must be at least 16")
        if not 0 < self.pauli_uniform_fraction <= 1:
            raise ValueError("pauli_uniform_fraction must lie in (0, 1]")
        if not 0 < self.conenh_exploration <= 1:
            raise ValueError("conenh_exploration must lie in (0, 1]")
        if self.si_cs_exploration is not None and (
            not np.isfinite(self.si_cs_exploration) or not 0 < self.si_cs_exploration <= 1
        ):
            raise ValueError("si_cs_exploration must be None or lie in (0, 1]")
        if not 0 <= self.conenh_pauli_floor <= 1:
            raise ValueError("conenh_pauli_floor must lie in [0, 1]")
        if self.seed < 0 or self.rotation_seed < 0:
            raise ValueError("seeds must be non-negative")
        if not np.isfinite(self.theta):
            raise ValueError("theta must be finite")
        self.theta_range = validate_theta_range(self.theta_range)
        self.property = self.property.upper()
        if self.property not in ("PURITY", "RENYI2", "MAGIC"):
            raise ValueError("property must be 'PURITY', 'RENYI2' or 'MAGIC'")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if not self.methods:
            raise ValueError("At least one method must be active")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        grid_adjacency(self.n_qubits, rows=self.rows, columns=self.columns)
        if not 0 <= self.k_rotations <= self.n_qubits:
            raise ValueError("k_rotations must lie between 0 and n_qubits")
        if not 1 <= self.subsystem_size <= min(MAX_SUBSYSTEM_SIZE, self.n_qubits):
            raise ValueError(
                f"subsystem_size must lie between 1 and min({MAX_SUBSYSTEM_SIZE}, n_qubits)"
            )
        if self.pilot_shots < 0 or self.post_shots < 4:
            raise ValueError("pilot_shots must be >=0 and post_shots must be >=4")
        if self.property == "MAGIC" and self.post_shots % 4:
            raise ValueError("post_shots must be divisible by 4 for the order-4 U-statistic")
        if self.repetitions < 1 or self.evaluation_points < 2:
            raise ValueError("repetitions must be >=1 and evaluation_points must be >=2")
        if not np.isfinite(self.conenh_commuting_bias) or self.conenh_commuting_bias < 0.0:
            raise ValueError("conenh_commuting_bias must be finite and non-negative")
        self.conenh_clip_bound = self.conenh_clip_bound.upper()
        if self.conenh_clip_bound not in ("POINT", "UCB"):
            raise ValueError("conenh_clip_bound must be 'POINT' or 'UCB'")
        if not self.conenh_softmax_temperature > 0.0:
            raise ValueError("conenh_softmax_temperature must be positive")
        if self.conenh_purity_weight < 0.0 or self.conenh_generator_weight < 0.0:
            raise ValueError(
                "conenh_purity_weight and conenh_generator_weight must be non-negative"
            )
        if self.rotation_vertices is not None:
            vertices = self.rotation_vertices
            if (
                len(vertices) != self.k_rotations
                or len(set(vertices)) != len(vertices)
                or any(
                    not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < self.n_qubits
                    for v in vertices
                )
            ):
                raise ValueError(
                    "rotation_vertices must contain k_rotations distinct valid indices"
                )
        if self.subsystem is not None:
            if (
                len(self.subsystem) != self.subsystem_size
                or len(set(self.subsystem)) != len(self.subsystem)
                or any(
                    not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < self.n_qubits
                    for v in self.subsystem
                )
            ):
                raise ValueError("subsystem must contain subsystem_size distinct valid indices")
        if self.property != "MAGIC" and any("AGHDO" in method for method in self.methods):
            if self.subsystem_size > 8:
                raise ValueError("Local AGHDO supports at most 8 subsystem qubits")
        if self.aghdo_rank < 1 or self.aghdo_fit_steps < 0 or self.aghdo_batch_size < 1:
            raise ValueError("Invalid AGHDO training parameters")
        if not np.isfinite(self.aghdo_learning_rate) or self.aghdo_learning_rate <= 0:
            raise ValueError("aghdo_learning_rate must be finite and positive")
        if not self.aghdo_feature_densities or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.aghdo_feature_densities
        ):
            raise ValueError("aghdo_feature_densities must contain positive integers")
        if not np.isfinite(self.aghdo_init_std) or not 0.001 <= self.aghdo_init_std <= 0.1:
            raise ValueError("aghdo_init_std must lie in [0.001, 0.1]")
        if self.qns_property_samples < 1:
            raise ValueError("qns_property_samples must be positive")
        if self.property != "MAGIC" or any(m in self.methods for m in ("CS", "ConEnh", "SI-CS")):
            self.resource_plan()


@dataclass(slots=True)
class BenchmarkResult:
    config: SimulationConfig
    shots: np.ndarray
    truth: float
    estimates: dict[str, np.ndarray]
    subsystem: tuple[int, ...]
    pilot_cost: int
    execution: dict[str, object] = field(default_factory=dict)
    truths: np.ndarray | None = None

    @property
    def repetition_truths(self) -> np.ndarray:
        """Exact value for each realization; ``truth`` is their mean."""
        if self.truths is None:
            return np.full(self.config.repetitions, self.truth)
        return np.asarray(self.truths, dtype=float)

    def as_renyi2(self) -> BenchmarkResult:
        """Convert purity truth and estimates to S_2 = -log2(purity), in bits."""
        if self.config.property != "PURITY":
            raise ValueError("Renyi-2 entropy can only be derived from a PURITY result")
        with np.errstate(divide="ignore", invalid="ignore"):
            truths = -np.log2(self.repetition_truths)
            execution = {**self.execution}
            if "repetitions" in execution:
                execution["repetitions"] = [
                    {**item, "truth": float(truth)}
                    for item, truth in strict_zip(execution["repetitions"], truths)
                ]
            return replace(
                self,
                config=replace(self.config, property="RENYI2"),
                truth=float(truths.mean()),
                truths=truths,
                execution=execution,
                estimates={
                    method: _purity_to_renyi2(values, self.config)
                    for method, values in self.estimates.items()
                },
            )

    def errors(self, mode: str = "absolute") -> dict[str, np.ndarray]:
        mode = mode.lower()
        if mode not in ("absolute", "relative"):
            raise ValueError("mode must be 'absolute' or 'relative'")
        truths = self.repetition_truths[:, None]
        denominator = np.abs(truths)
        if mode == "relative" and np.any(denominator <= 1e-14):
            raise ValueError("Relative error is undefined because the exact property is zero")
        return {
            method: np.abs(values - truths) / (denominator if mode == "relative" else 1.0)
            for method, values in self.estimates.items()
        }

    def final_summary(self, error_mode: str = "absolute") -> list[dict[str, float | str]]:
        errors = self.errors(error_mode)
        summary = []
        for method in self.config.methods:
            values = self.estimates[method][:, -1]
            summary.append(
                {
                    "method": method,
                    "estimate_mean": float(np.nanmean(values)),
                    "estimate_sem": float(np.nanstd(values, ddof=1) / np.sqrt(len(values)))
                    if len(values) > 1
                    else 0.0,
                    f"{error_mode}_error_mean": float(np.nanmean(errors[method][:, -1])),
                }
            )
        return summary


def _purity_to_renyi2(values, config: SimulationConfig):
    if config.property_clipping is True:
        values = np.clip(values, 2.0**(-config.subsystem_size), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log2(values)


def _clip_property_estimate(value: float, config: SimulationConfig) -> float:
    """Project the reported property only; acquisition and training are unaffected."""
    if config.property_clipping is not True:
        return value
    if config.property == "PURITY":
        lower, upper = 2.0**(-config.subsystem_size), 1.0
    elif config.property == "RENYI2":
        lower, upper = 0.0, float(config.subsystem_size)
    else:
        lower, upper = 0.0, float(config.n_qubits)
    return float(np.clip(value, lower, upper))


def _clip_magic_moment(config: SimulationConfig) -> bool:
    """An explicit global switch overrides the legacy direct-MAGIC option."""
    return config.magic_clipping if config.property_clipping is None else config.property_clipping


def _evaluation_checkpoints(config: SimulationConfig) -> np.ndarray:
    minimum = 4 if config.property == "MAGIC" else 2
    raw = np.geomspace(minimum, config.post_shots, config.evaluation_points)
    checkpoints = np.unique(np.rint(raw).astype(int))
    if config.property == "MAGIC":
        checkpoints = np.maximum(4, 4 * np.rint(checkpoints / 4).astype(int))
    checkpoints = np.unique(np.clip(checkpoints, minimum, config.post_shots))
    if checkpoints[-1] != config.post_shots:
        checkpoints = np.append(checkpoints, config.post_shots)
    return checkpoints


def _make_moment_selector(
    kind: str,
    config: SimulationConfig,
    generators: tuple[Pauli, ...],
    subsystem: tuple[int, ...],
    rng: np.random.Generator,
    plan: ResourcePlan,
    pool: PauliPool | None = None,
    pilot: PilotData | None = None,
    compute: ComputeBackend | None = None,
) -> (
    UniformSelector | OnlineSubsystemSelector | SampledSubsystemSelector
    | StabilizerInformedSelector
):
    if kind == "uniform":
        return UniformSelector(config.n_qubits, rng)
    if kind == "stabilizer":
        return StabilizerInformedSelector(
            generators, config.n_qubits, rng, subsystem=subsystem, pool=pool, pilot=pilot,
            exploration=(config.conenh_exploration if config.si_cs_exploration is None
                         else config.si_cs_exploration),
        )
    common = dict(
        exploration=config.conenh_exploration,
        purity_weight=config.conenh_purity_weight,
        generator_weight=config.conenh_generator_weight,
        temperature=config.conenh_softmax_temperature,
        pauli_floor=config.conenh_pauli_floor,
        commuting_bias=config.conenh_commuting_bias,
        local_regularization=config.conenh_local_regularization,
        update_every=plan.update_every,
        n_qubits=config.n_qubits,
        max_cliques=plan.max_cliques,
        moment_order=4 if config.property == "MAGIC" else 2,
        compute=compute,
    )
    if pool is not None:
        selector = SampledSubsystemSelector(
            generators, pool, rng, candidates=plan.candidates, **common
        )
    else:
        selector = OnlineSubsystemSelector(generators, subsystem, rng, **common)
    if pilot is not None:

        indices = {word: i for i, word in enumerate(selector.generators)}
        for word, outcome in strict_zip(pilot.parity_paulis, pilot.parity_outcomes):
            if word in indices:
                i = indices[word]
                selector.generator_counts[i] += 1
                selector.generator_sums[i] += int(outcome)
    return selector


def _make_magic_selector(
    kind: str,
    config: SimulationConfig,
    pilot: PilotData | None,
    generators: tuple[Pauli, ...],
    rng: np.random.Generator,
    *, probes: tuple[Pauli, ...] | None = None, compute: ComputeBackend | None = None,
) -> UniformSelector | MagicProbeSelector | StabilizerInformedSelector:
    if kind == "uniform":
        return UniformSelector(config.n_qubits, rng)
    if kind == "stabilizer":
        return StabilizerInformedSelector(
            generators, config.n_qubits, rng, pilot=pilot,
            exploration=(config.conenh_exploration if config.si_cs_exploration is None
                         else config.si_cs_exploration),
        )
    return MagicProbeSelector(
        generators,
        None if pilot is None else pilot.epsilon,
        rng,
        candidates=config.conenh_candidates or 64,
        temperature=config.conenh_softmax_temperature,
        exploration=config.conenh_exploration,
        pilot_repetitions=0 if pilot is None else config.pilot_shots,
        n_qubits=config.n_qubits,
        probes=probes, compute=compute,
    )


def _report_due(shot_number: int, config: SimulationConfig, at_checkpoint: bool) -> bool:
    return at_checkpoint or shot_number % max(1, config.post_shots // _REPORTS_PER_STREAM) == 0


def _purity_diagnostics(
    truth: float,
    estimate: float,
    statistic: (
        IncrementalPauliMomentUStatistic | SampledPauliMomentUStatistic
        | FactorizedUniformPurityUStatistic | None
    ),
    selector: (
        UniformSelector | OnlineSubsystemSelector | SampledSubsystemSelector
        | StabilizerInformedSelector
    ),
    design: ShotDesign,
    qns_value: float,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    if statistic is not None:
        stats.update(est=estimate, err=abs(estimate - truth), wmax=statistic.max_weight)
    if not np.isnan(qns_value):
        stats.update(qns=qns_value, qns_err=abs(qns_value - truth))
    if isinstance(selector, (
        OnlineSubsystemSelector, SampledSubsystemSelector, StabilizerInformedSelector,
    )):
        seen = selector.generator_counts > 0
        stats["Qmin"] = float(design.local_inclusion.min())
        stats["H_A"] = f"{int(seen.sum())}/{len(seen)}"
        stats["eps"] = (
            float(selector.generator_epsilon()[seen].mean()) if seen.any() else float("nan")
        )
    return stats


def _run_moment_stream(
    kind: str,
    config: SimulationConfig,
    oracle: StateOracle,
    generators: tuple[Pauli, ...],
    subsystem: tuple[int, ...],
    checkpoints: np.ndarray,
    direct_active: bool,
    qns_active: bool,
    seed: int,
    advance_progress: Callable[[], None],
    report: Callable[[dict[str, object]], None] | None,
    truth: float,
    plan: ResourcePlan,
    pool: PauliPool | None,
    pilot: PilotData | None = None,
    compute: ComputeBackend | None = None,
) -> Generator[dict[str, object], None, tuple[np.ndarray | None, np.ndarray | None]]:
    moment_order = 4 if config.property == "MAGIC" else 2
    selector_seed, outcome_seed, _, model_seed = np.random.SeedSequence(seed).spawn(4)
    selector = _make_moment_selector(
        kind, config, generators, subsystem, np.random.default_rng(selector_seed),
        plan, pool, pilot, compute,
    )
    outcome_rng = np.random.default_rng(outcome_seed)
    qns_rng = np.random.default_rng(model_seed)
    model = (
        AGHDO(
            config.subsystem_size,
            rank=config.aghdo_rank,
            learning_rate=config.aghdo_learning_rate,
            seed=int(qns_rng.integers(2**31)),
            device=compute.device if compute is not None else "cpu",
            feature_densities=config.aghdo_feature_densities,
            init_std=config.aghdo_init_std,
        )
        if qns_active
        else None
    )
    statistic = None
    if direct_active:
        if pool is not None and kind == "uniform" and moment_order == 2:
            statistic = FactorizedUniformPurityUStatistic(subsystem, config.post_shots)
        elif pool is not None:
            statistic = (
                SampledPurityUStatistic(pool) if moment_order == 2
                else SampledPauliMomentUStatistic(pool, moment_order=4)
            )
        else:
            statistic = (
                IncrementalPurityUStatistic(subsystem) if moment_order == 2
                else IncrementalPauliMomentUStatistic(subsystem, moment_order=4)
            )
    clipped = isinstance(
        selector, (OnlineSubsystemSelector, SampledSubsystemSelector, StabilizerInformedSelector)
    ) and config.conenh_clipping
    direct_curve = np.full(len(checkpoints), np.nan) if direct_active else None
    qns_curve = np.full(len(checkpoints), np.nan) if qns_active else None
    checkpoint_lookup = {int(value): index for index, value in enumerate(checkpoints)}
    qns_value = float("nan")

    for shot_number in range(1, config.post_shots + 1):
        design = selector.draw()
        probes = selector.parity_probes(design.basis)
        if probes:
            outcomes, parities = oracle.sample_local_and_parities(
                design.basis, subsystem, probes, outcome_rng
            )
        else:
            outcomes = oracle.sample_outcomes(design.basis, subsystem, outcome_rng)
            parities = np.zeros(0, dtype=np.int8)
        selector.observe(design, parities)
        if statistic is not None:
            statistic.add(LocalShot(design=design, outcomes=outcomes))
        if model is not None:
            model.add_measurement(design.basis[np.asarray(subsystem, dtype=int)], outcomes)

        point = checkpoint_lookup.get(shot_number)
        due = report is not None and _report_due(shot_number, config, point is not None)
        estimate = float("nan")
        if statistic is not None and (point is not None or due):
            bounds = (
                selector.pauli_bounds(margin=config.conenh_clip_bound == "UCB") if clipped else None
            )
            if bounds is not None and moment_order == 4:
                bounds = bounds**2
            estimate = statistic.value(bounds)
            if moment_order == 4:
                estimate = magic_from_fourth_moment(
                    estimate, config.n_qubits, clip=_clip_magic_moment(config),
                )
            estimate = _clip_property_estimate(estimate, config)
        if point is not None:
            if direct_curve is not None:
                direct_curve[point] = estimate
            if qns_curve is not None and model is not None:
                model.fit(config.aghdo_fit_steps, config.aghdo_batch_size)
                qns_curve[point] = _clip_property_estimate(
                    model.sample_purity(config.qns_property_samples, qns_rng), config
                )
                qns_value = qns_curve[point]
        if due:
            report(_purity_diagnostics(truth, estimate, statistic, selector, design, qns_value))
        advance_progress()
        if _report_due(shot_number, config, point is not None):
            if isinstance(statistic, SampledPauliMomentUStatistic):
                diagnostics = {
                    "sampled_paulis_with_pairs": int(np.count_nonzero(statistic.hits >= 2)),
                    "sampled_paulis_with_quartets": int(np.count_nonzero(statistic.hits >= 4)),
                    "sampled_paulis": len(pool.paulis),
                    "max_inverse_inclusion": statistic.max_weight,
                    "policy_updates": getattr(selector, "policy_updates", 0),
                    "retained_cliques": getattr(selector, "clique_pool_size", 0),
                    "estimator": "importance_sampled_pauli_sum", "moment_order": moment_order,
                }
            elif isinstance(statistic, FactorizedUniformPurityUStatistic):
                diagnostics = {
                    "estimator": "exact_factorized_uniform", "pairs": statistic.pair_count,
                }
            elif statistic is not None:
                diagnostics = {
                    "estimator": "exact_enumerated_pauli_sum", "moment_order": moment_order,
                }
            else:
                diagnostics = {}
            if isinstance(selector, StabilizerInformedSelector):
                diagnostics.update(selector.diagnostics())
            yield {
                "shot": shot_number,
                "basis": design.basis.tolist(),
                "estimates": {
                    name: float(curve[point])
                    for name, curve in (("direct", direct_curve), ("qns", qns_curve))
                    if point is not None and curve is not None
                },
                "diagnostics": diagnostics,
            }
    return direct_curve, qns_curve


def _run_magic_stream(
    kind: str,
    config: SimulationConfig,
    pilot: PilotData | None,
    oracle: StateOracle,
    generators: tuple[Pauli, ...],
    checkpoints: np.ndarray,
    direct_active: bool,
    qns_active: bool,
    seed: int,
    advance_progress: Callable[[], None],
    report: Callable[[dict[str, object]], None] | None,
    truth: float,
    compute: ComputeBackend | None = None,
) -> Generator[dict[str, object], None, tuple[None, np.ndarray]]:

    focus = magic_focus(config.state, oracle.adjacency)
    words = tuple(item.pauli for item in focus)
    focus_sites, focus_axes = probe_axes(words, config.n_qubits)
    by_word = {item.pauli: item for item in focus}
    selector_seed, outcome_seed, model_seed = np.random.SeedSequence(seed).spawn(3)
    selector = _make_magic_selector(
        kind, config, pilot, generators, np.random.default_rng(selector_seed),
        probes=words, compute=compute,
    )
    outcome_rng = np.random.default_rng(outcome_seed)
    model_rng = np.random.default_rng(model_seed)
    model = FocusedAGHDO(
        config.n_qubits, rank=config.aghdo_rank,
        learning_rate=config.aghdo_learning_rate, seed=int(model_rng.integers(2**31)),
        device=compute.device if compute is not None else "cpu",
        feature_densities=config.aghdo_feature_densities, init_std=config.aghdo_init_std,
    )
    if pilot is not None:
        for word, outcome in strict_zip(pilot.parity_paulis, pilot.parity_outcomes):
            item = by_word[word]
            model.add_encoded(item.mode, item.axis, item.sign * int(outcome))
    curve = np.full(len(checkpoints), np.nan)
    checkpoint_lookup = {int(value): index for index, value in enumerate(checkpoints)}
    value = float("nan")
    for shot_number in range(1, config.post_shots + 1):
        design = selector.draw()
        selected = tuple(focus[i] for i in np.flatnonzero(
            covered_probes(design.basis, focus_sites, focus_axes)
        ))


        parities = sample_focus_parities(oracle, selected, outcome_rng)
        measured = {item.pauli: int(parity) for item, parity in strict_zip(selected, parities)}
        selector_probes = selector.parity_probes(design.basis)
        selector.observe(design.basis, np.asarray(
            [measured[word] for word in selector_probes], dtype=np.int8
        ))
        for item, parity in strict_zip(selected, parities):
            model.add_encoded(item.mode, item.axis, item.sign * int(parity))
        point = checkpoint_lookup.get(shot_number)
        if point is not None:
            model.fit(config.aghdo_fit_steps, config.aghdo_batch_size)
            value = _clip_property_estimate(model.sample_magic(
                config.qns_property_samples, model_rng, clip=config.property_clipping is None,
            ), config)
            curve[point] = value
        advance_progress()
        if _report_due(shot_number, config, point is not None):
            diagnostics = {
                **(selector.diagnostics()
                   if isinstance(selector, StabilizerInformedSelector) else {}),
                "estimator": "neural_AGHDO_product_Clifford_frame",
                "focus_observables": len(focus), "logical_modes": config.n_qubits,
                "observed_focus_observables": int(np.count_nonzero(model.counts.sum(-1))),
                "logical_modes_with_records": int(np.count_nonzero(model.counts.sum((1, 2)))),
                "records": model.record_count, "loss": model.last_loss,
                "aghdo_device": str(model.network.device),
                "neural_parameters": sum(p.numel() for p in model.network.parameters()),
            }
            if report is not None:
                report({"qns": value, "qns_err": abs(value - truth),
                        "recs": model.record_count})
            yield {
                "shot": shot_number, "basis": design.basis.tolist(),
                "estimates": {"qns": float(curve[point])} if point is not None else {},
                "diagnostics": diagnostics,
            }
    return None, curve


def _iter_realization(
    config: SimulationConfig, *, repetition_index: int, show_progress: bool = True
) -> Generator[dict[str, object], None, BenchmarkResult]:
    """Yield acquisition updates and return the result via StopIteration.value."""
    config.validate()
    if config.property == "RENYI2":
        stream = _iter_realization(
            replace(config, property="PURITY"), repetition_index=repetition_index,
            show_progress=show_progress,
        )
        try:
            while True:
                try:
                    event = next(stream)
                except StopIteration as done:
                    result = done.value.as_renyi2()
                    break
                with np.errstate(divide="ignore", invalid="ignore"):
                    event["truth"] = float(-np.log2(event["truth"]))
                    event["estimates"] = {
                        method: float(_purity_to_renyi2(value, config))
                        for method, value in event["estimates"].items()
                    }
                yield event
        finally:
            stream.close()
        result.config = config
        return result
    compute = ComputeBackend(config.device)
    oracle = make_state(
        config.state,
        config.n_qubits,
        config.k_rotations,
        theta=config.theta,
        theta_range=config.theta_range,
        vertex_seed=(config.rotation_seed if repetition_index == 0 else int(
            np.random.SeedSequence(config.rotation_seed, spawn_key=(repetition_index,))
            .generate_state(1, dtype=np.uint64)[0]
        )),
        rows=config.rows,
        columns=config.columns,
        rotation_vertices=config.rotation_vertices,
    )
    generators = measurement_generators(
        oracle.generators, max_weight=config.generator_max_weight
    )
    subsystem = (
        tuple(config.subsystem)
        if config.subsystem is not None
        else choose_connected_subsystem(
            oracle.adjacency, config.subsystem_size, columns=config.columns
        )
    )
    if len(subsystem) != config.subsystem_size or len(set(subsystem)) != len(subsystem):
        raise ValueError("subsystem must contain subsystem_size distinct qubits")

    active_register = tuple(range(config.n_qubits)) if config.property == "MAGIC" else subsystem
    subsystem_mask = sum(1 << q for q in active_register)
    local_generator_count = sum(bool(g.support & subsystem_mask) for g in generators)
    plan = plan_resources(config, local_generator_count)
    execution = plan.as_dict()
    execution["compute"] = compute.as_dict()
    execution["property_clipping"] = config.property_clipping
    execution["moment_order"] = 4 if config.property == "MAGIC" else 2
    execution["rotation_vertices"] = list(oracle.rotation_vertices)
    execution["rotation_angles"] = list(oracle.rotation_angles)
    if any("AGHDO" in method for method in config.methods):
        execution["aghdo"] = {
            "architecture": "complex_masked_dense_SELU_AGHDO",
            "paper": "https://arxiv.org/html/2206.13488v1",
            "local_rank": config.aghdo_rank,
            "feature_densities": list(config.aghdo_feature_densities),
            "init_std": config.aghdo_init_std, "device": compute.device,
            "training": "measurement_negative_log_likelihood",
            "model": "product_in_reference_Clifford_frame"
            if config.property == "MAGIC" else "unrestricted_subsystem_AGHDO",
            "focus_observables": 3 * config.n_qubits if config.property == "MAGIC" else None,
            "rotation_sites_supplied_to_learner": False,
        }
    if config.property == "MAGIC":
        execution["direct_register_size"] = config.n_qubits
        execution["aghdo_acquisition"] = "reference_Clifford_XYZ_focus"
    base_execution = execution.copy()

    checkpoints = _evaluation_checkpoints(config)
    use_pilot = config.property != "PURITY" and config.pilot_shots > 0
    pilot_cost = len(generators) * config.pilot_shots if use_pilot else 0
    shots = checkpoints + (pilot_cost if config.count_pilot_cost else 0)
    truth = (
        oracle.local_purity(subsystem)
        if config.property == "PURITY"
        else oracle.stabilizer_renyi_magic()
    )
    estimates = {
        method: np.full((config.repetitions, len(checkpoints)), np.nan, dtype=float)
        for method in config.methods
    }
    repetition_seeds = [np.random.SeedSequence(config.seed, spawn_key=(repetition_index,))]
    execution_history: list[dict[str, object]] = []
    stream_jobs = []
    for index, (kind, direct_name, qns_name) in enumerate(STREAMS):
        direct_active = direct_name in config.methods
        qns_active = qns_name in config.methods
        if config.property == "PURITY":
            if direct_active or qns_active:
                stream_jobs.append((index, kind, direct_name, qns_name,
                                    direct_active, qns_active, "moment"))
        else:
            if direct_active:
                stream_jobs.append((index, kind, direct_name, qns_name, True, False, "moment"))
            if qns_active:


                stream_jobs.append((index, kind, direct_name, qns_name,
                                    direct_active, True, "focused_aghdo"))
    active_streams = len(stream_jobs)
    target = f" |A|={len(subsystem)}" if config.property == "PURITY" else ""

    with tqdm(
        total=config.repetitions * active_streams * config.post_shots,
        desc=f"{config.property}{target} exact={truth:.4g} [{compute.device}]",
        unit="shot",
        disable=not show_progress,
    ) as progress:
        completed = 0
        total = config.repetitions * active_streams * config.post_shots
        yield {
            "phase": "initializing",
            "truth": float(truth),
            "estimates": {},
            "completed": 0,
            "total": total,
            "shot": 0,
            "shots": pilot_cost if config.count_pilot_cost else 0,
            "repetition": 0,
            "stream": "",
            "subsystem": list(subsystem),
            "rotation_vertices": list(oracle.rotation_vertices),
            "rotation_angles": list(oracle.rotation_angles),
            "basis": [],
            "execution": execution,
        }
        for repetition, repetition_seed in enumerate(repetition_seeds):
            execution = base_execution.copy()
            pilot_seed, *stream_seeds = repetition_seed.spawn(1 + len(STREAMS))
            pilot = (
                run_common_pilot(
                    oracle, config.pilot_shots, np.random.default_rng(pilot_seed),
                    generators=generators,
                )
                if use_pilot
                else None
            )
            pool = None
            if plan.backend == "sampled":

                pool_rng = np.random.default_rng([config.seed, repetition_index, 0x5041554C])
                pool = PauliPool.sample(
                    generators, active_register, config.n_qubits, plan.pauli_samples,
                    config.pauli_uniform_fraction, pool_rng,
                )
                execution = {
                    **base_execution, "unique_paulis": len(pool.paulis),
                    "proposal_span_rank": pool.span_rank,
                    "pauli_uniform_fraction": config.pauli_uniform_fraction,
                    "effective_uniform_fraction": config.pauli_uniform_fraction
                    if pool.span_rank else 1.0,
                    "approximation": "sampled Pauli sum, bounded candidates and clique pool",
                }
            label = {"rep": f"{repetition + 1}/{config.repetitions}"}
            if not config.progress_details:
                progress.set_postfix(label, refresh=False)
            for index, kind, direct_name, qns_name, direct_active, qns_active, mode in stream_jobs:
                compute.clear_cache()
                stream_seed = stream_seeds[index]
                integer_seed = int(np.random.default_rng(stream_seed).integers(2**31))
                stream_label = {
                    **label, "run": qns_name if mode == "focused_aghdo" else (
                        direct_name if direct_active else qns_name
                    ),
                }

                def report(stats: dict[str, object], stream_label=stream_label) -> None:
                    progress.set_postfix({**stream_label, **stats}, refresh=False)

                stream_report = report if config.progress_details else None
                if mode == "moment":
                    stream = _run_moment_stream(
                        kind,
                        config,
                        oracle,
                        generators,
                        active_register,
                        checkpoints,
                        direct_active,
                        qns_active,
                        integer_seed,
                        progress.update,
                        stream_report,
                        truth,
                        plan,
                        pool,
                        pilot,
                        compute,
                    )
                else:
                    stream = _run_magic_stream(
                        kind,
                        config,
                        pilot,
                        oracle,
                        generators,
                        checkpoints,
                        direct_active,
                        qns_active,
                        integer_seed,
                        progress.update,
                        stream_report,
                        truth,
                        compute,
                    )
                while True:
                    try:
                        event = next(stream)
                    except StopIteration as done:
                        direct_curve, qns_curve = done.value
                        if mode == "focused_aghdo":
                            direct_curve = None
                        break
                    if mode == "focused_aghdo":
                        event["estimates"].pop("direct", None)
                    stream_key = f"{kind}_aghdo" if mode == "focused_aghdo" else kind
                    execution["compute"] = compute.as_dict()
                    event.update(
                        phase="acquiring",
                        truth=float(truth),
                        repetition=repetition,
                        stream=stream_key,
                        completed=completed + event["shot"],
                        total=total,
                        execution=execution,
                        shots=event["shot"] + (pilot_cost if config.count_pilot_cost else 0),
                    )
                    if event.get("diagnostics"):
                        execution = {
                            **execution,
                            "streams": {
                                **execution.get("streams", {}), stream_key: event["diagnostics"],
                            },
                        }
                        event["execution"] = execution
                    event["estimates"] = {
                        direct_name if name == "direct" else qns_name: value
                        for name, value in event["estimates"].items()
                    }
                    yield event
                completed += config.post_shots
                if direct_curve is not None:
                    estimates[direct_name][repetition] = direct_curve
                if qns_curve is not None:
                    estimates[qns_name][repetition] = qns_curve
                compute.clear_cache()
            execution_history.append({"repetition": repetition_index, "truth": float(truth),
                                      **execution})

    return BenchmarkResult(
        config=config,
        shots=shots,
        truth=float(truth),
        estimates=estimates,
        subsystem=subsystem,
        pilot_cost=pilot_cost,
        execution={**execution, "repetitions": execution_history},
    )


def iter_benchmark(
    config: SimulationConfig, *, show_progress: bool = True, repetition_start: int = 0,
) -> Generator[dict[str, object], None, BenchmarkResult]:
    """Sample independent rotation realizations, shared by all active methods."""
    config.validate()
    if (not isinstance(repetition_start, int) or isinstance(repetition_start, bool)
            or repetition_start < 0):
        raise ValueError("repetition_start must be a nonnegative integer")
    results = []
    completed = 0
    total = None
    with tqdm(desc=f"{config.property} · {config.state}", unit="shot",
              disable=not show_progress) as progress:
        for index in range(repetition_start, repetition_start + config.repetitions):
            stream = _iter_realization(
                replace(config, repetitions=1), repetition_index=index, show_progress=False,
            )
            local_completed = 0
            try:
                while True:
                    try:
                        event = next(stream)
                    except StopIteration as done:
                        results.append(done.value)
                        break
                    if total is None:
                        total = int(event["total"]) * config.repetitions
                        progress.total = total
                    current = int(event["completed"])
                    progress.update(current - local_completed)
                    local_completed = current
                    event.update(
                        repetition=index - repetition_start, repetition_index=index,
                        completed=completed + current, total=total,
                    )
                    yield event
            finally:
                stream.close()
            completed += local_completed
    truths = np.array([result.truth for result in results])
    return BenchmarkResult(
        config=config, shots=results[0].shots, truth=float(truths.mean()), truths=truths,
        estimates={method: np.concatenate([result.estimates[method] for result in results])
                   for method in config.methods},
        subsystem=results[0].subsystem, pilot_cost=results[0].pilot_cost,
        execution={**results[0].execution,
                   "rotation_sampling": "independent_uniform_per_repetition",
                   "repetitions": [item for result in results
                                   for item in result.execution["repetitions"]]},
    )


def run_benchmark(
    config: SimulationConfig,
    on_update: Callable[[dict[str, object]], None] | None = None,
    *,
    show_progress: bool = True,
    repetition_start: int = 0,
) -> BenchmarkResult:
    """Run synchronously, optionally observing the same events as iter_benchmark."""
    stream = iter_benchmark(
        config, show_progress=show_progress, repetition_start=repetition_start,
    )
    try:
        while True:
            try:
                event = next(stream)
            except StopIteration as done:
                return done.value
            if on_update is not None:
                on_update(event)
    finally:
        stream.close()
