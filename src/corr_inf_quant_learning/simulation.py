from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from tqdm.auto import tqdm

from .aghdo import AGHDO, LogicalAGHDO
from .cluster_state import RotatedClusterState, choose_connected_subsystem
from .estimators import (
    IncrementalPurityUStatistic,
    fourth_moment_from_kernels,
    magic_block_kernel,
    magic_from_fourth_moment,
)
from .measurement import (
    LocalShot,
    MagicProbeSelector,
    OnlineSubsystemSelector,
    PilotData,
    ShotDesign,
    UniformSelector,
    covered_probes,
    generator_probes,
    probe_axes,
    run_common_pilot,
)
from .pauli import Pauli

METHODS = ("CS", "CorInf", "CS-AGHDO", "CorInf-AGHDO")
STREAMS = (("uniform", "CS", "CS-AGHDO"), ("bound", "CorInf", "CorInf-AGHDO"))
MAX_SUBSYSTEM_SIZE = 12
_REPORTS_PER_STREAM = 50


@dataclass(slots=True)
class SimulationConfig:
    n_qubits: int = 100
    k_rotations: int = 50
    subsystem_size: int = 5
    methods: tuple[str, ...] = METHODS
    property: Literal["PURITY", "MAGIC"] = "PURITY"
    theta: float = float(np.pi / 8)

    pilot_shots: int = 8
    post_shots: int = 256
    repetitions: int = 3
    evaluation_points: int = 9
    seed: int = 7
    rotation_seed: int = 1729
    count_pilot_cost: bool = True

    corinf_exploration: float = 0.20
    corinf_commuting_bias: float = 1.0
    corinf_candidates: int = 64
    corinf_local_regularization: float = 0.05
    corinf_purity_weight: float = 1.0
    corinf_generator_weight: float = 1.0
    corinf_softmax_temperature: float = 0.3
    corinf_pauli_floor: float = 0.1
    corinf_update_every: int = 1
    corinf_clip_bound: str = "POINT"

    aghdo_rank: int = 16
    aghdo_learning_rate: float = 0.025
    aghdo_fit_steps: int = 12
    aghdo_batch_size: int = 128
    qns_property_samples: int = 512

    progress_details: bool = True
    subsystem: tuple[int, ...] | None = None

    def validate(self) -> None:
        self.property = self.property.upper()
        if self.property not in ("PURITY", "MAGIC"):
            raise ValueError("property must be 'PURITY' or 'MAGIC'")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")
        if not self.methods:
            raise ValueError("At least one method must be active")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
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
        if not np.isfinite(self.corinf_commuting_bias) or self.corinf_commuting_bias < 0.0:
            raise ValueError("corinf_commuting_bias must be finite and non-negative")
        if self.corinf_update_every < 1:
            raise ValueError("corinf_update_every must be a positive integer")
        self.corinf_clip_bound = self.corinf_clip_bound.upper()
        if self.corinf_clip_bound not in ("POINT", "UCB"):
            raise ValueError("corinf_clip_bound must be 'POINT' or 'UCB'")
        if self.corinf_candidates < 1:
            raise ValueError("corinf_candidates must be a positive integer")
        if not self.corinf_softmax_temperature > 0.0:
            raise ValueError("corinf_softmax_temperature must be positive")
        if self.corinf_purity_weight < 0.0 or self.corinf_generator_weight < 0.0:
            raise ValueError(
                "corinf_purity_weight and corinf_generator_weight must be non-negative"
            )


@dataclass(slots=True)
class BenchmarkResult:
    config: SimulationConfig
    shots: np.ndarray
    truth: float
    estimates: dict[str, np.ndarray]
    subsystem: tuple[int, ...]
    pilot_cost: int

    def errors(self, mode: str = "absolute") -> dict[str, np.ndarray]:
        mode = mode.lower()
        if mode not in ("absolute", "relative"):
            raise ValueError("mode must be 'absolute' or 'relative'")
        denominator = abs(self.truth)
        if mode == "relative" and denominator <= 1e-14:
            raise ValueError("Relative error is undefined because the exact property is zero")
        return {
            method: np.abs(values - self.truth) / (denominator if mode == "relative" else 1.0)
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


def _make_purity_selector(
    kind: str,
    config: SimulationConfig,
    oracle: RotatedClusterState,
    subsystem: tuple[int, ...],
    rng: np.random.Generator,
) -> UniformSelector | OnlineSubsystemSelector:
    if kind == "uniform":
        return UniformSelector(config.n_qubits, rng)
    return OnlineSubsystemSelector(
        oracle.generators,
        subsystem,
        rng,
        exploration=config.corinf_exploration,
        purity_weight=config.corinf_purity_weight,
        generator_weight=config.corinf_generator_weight,
        temperature=config.corinf_softmax_temperature,
        pauli_floor=config.corinf_pauli_floor,
        commuting_bias=config.corinf_commuting_bias,
        local_regularization=config.corinf_local_regularization,
        update_every=config.corinf_update_every,
    )


def _make_magic_selector(
    kind: str,
    config: SimulationConfig,
    pilot: PilotData | None,
    generators: tuple[Pauli, ...],
    rng: np.random.Generator,
) -> UniformSelector | MagicProbeSelector:
    if kind == "uniform":
        return UniformSelector(config.n_qubits, rng)
    return MagicProbeSelector(
        generators,
        None if pilot is None else pilot.epsilon,
        rng,
        candidates=config.corinf_candidates,
        temperature=config.corinf_softmax_temperature,
        exploration=config.corinf_exploration,
        pilot_repetitions=0 if pilot is None else config.pilot_shots,
    )


def _logical_model(
    config: SimulationConfig,
    pilot: PilotData | None,
    oracle: RotatedClusterState,
    seed: int,
) -> LogicalAGHDO:
    model = LogicalAGHDO(config.k_rotations, learning_rate=config.aghdo_learning_rate, seed=seed)
    if pilot is not None:
        for pauli, outcome in zip(pilot.parity_paulis, pilot.parity_outcomes, strict=True):
            model.add_encoded(oracle.logical_image(pauli), int(outcome))
    return model


def _report_due(shot_number: int, config: SimulationConfig, at_checkpoint: bool) -> bool:
    return at_checkpoint or shot_number % max(1, config.post_shots // _REPORTS_PER_STREAM) == 0


def _purity_diagnostics(
    truth: float,
    estimate: float,
    statistic: IncrementalPurityUStatistic | None,
    selector: UniformSelector | OnlineSubsystemSelector,
    design: ShotDesign,
    qns_value: float,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    if statistic is not None:
        stats.update(est=estimate, err=abs(estimate - truth), wmax=statistic.max_weight)
    if not np.isnan(qns_value):
        stats.update(qns=qns_value, qns_err=abs(qns_value - truth))
    if isinstance(selector, OnlineSubsystemSelector):
        seen = selector.generator_counts > 0
        stats["Qmin"] = float(design.local_inclusion[1:].min())
        stats["H_A"] = f"{int(seen.sum())}/{len(seen)}"
        stats["eps"] = (
            float(selector.generator_epsilon()[seen].mean()) if seen.any() else float("nan")
        )
    return stats


def _magic_diagnostics(
    config: SimulationConfig,
    truth: float,
    kernels: list[float],
    selector: UniformSelector | MagicProbeSelector,
    qns_value: float,
    model: LogicalAGHDO | None,
    design: ShotDesign,
) -> dict[str, object]:
    stats: dict[str, object] = {}
    if kernels:
        estimate = magic_from_fourth_moment(fourth_moment_from_kernels(kernels), config.n_qubits)
        stats.update(
            est=estimate,
            err=abs(estimate - truth),
            kmax=float(np.max(np.abs(kernels))),
            blocks=len(kernels),
        )
    if not np.isnan(qns_value):
        stats.update(qns=qns_value, qns_err=abs(qns_value - truth))
    if model is not None:
        stats["recs"] = model.record_count
    if isinstance(selector, MagicProbeSelector):
        if selector.online:
            seen = int(np.count_nonzero(selector.generator_counts))
            stats["H_seen"] = f"{seen}/{selector.n_qubits}"
        if len(selector._active):
            stats["Qmin"] = float(design.probe_inclusion[selector._active].min())
        policy = design.candidate_policy[design.candidate_policy > 0.0]
        stats["cands"] = float(np.exp(-np.sum(policy * np.log(policy))))
    return stats


def _run_purity_stream(
    kind: str,
    config: SimulationConfig,
    oracle: RotatedClusterState,
    subsystem: tuple[int, ...],
    checkpoints: np.ndarray,
    direct_active: bool,
    qns_active: bool,
    seed: int,
    advance_progress: Callable[[], None],
    report: Callable[[dict[str, object]], None] | None,
    truth: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    selector_seed, outcome_seed, _, model_seed = np.random.SeedSequence(seed).spawn(4)
    selector = _make_purity_selector(
        kind, config, oracle, subsystem, np.random.default_rng(selector_seed)
    )
    outcome_rng = np.random.default_rng(outcome_seed)
    qns_rng = np.random.default_rng(model_seed)
    model = (
        AGHDO(
            config.subsystem_size,
            rank=config.aghdo_rank,
            learning_rate=config.aghdo_learning_rate,
            seed=int(qns_rng.integers(2**31)),
        )
        if qns_active
        else None
    )
    statistic = IncrementalPurityUStatistic(subsystem) if direct_active else None
    clipped = isinstance(selector, OnlineSubsystemSelector)
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
                selector.pauli_bounds(margin=config.corinf_clip_bound == "UCB")
                if clipped
                else None
            )
            estimate = statistic.value(bounds)
        if point is not None:
            if direct_curve is not None:
                direct_curve[point] = estimate
            if qns_curve is not None and model is not None:
                model.fit(config.aghdo_fit_steps, config.aghdo_batch_size)
                qns_curve[point] = model.sample_purity(config.qns_property_samples, qns_rng)
                qns_value = qns_curve[point]
        if due:
            report(_purity_diagnostics(truth, estimate, statistic, selector, design, qns_value))
        advance_progress()
    return direct_curve, qns_curve


def _run_magic_stream(
    kind: str,
    config: SimulationConfig,
    pilot: PilotData | None,
    oracle: RotatedClusterState,
    checkpoints: np.ndarray,
    direct_active: bool,
    qns_active: bool,
    seed: int,
    advance_progress: Callable[[], None],
    report: Callable[[dict[str, object]], None] | None,
    truth: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    selector_seed, outcome_seed, model_seed = np.random.SeedSequence(seed).spawn(3)
    selector = _make_magic_selector(
        kind, config, pilot, oracle.generators, np.random.default_rng(selector_seed)
    )
    outcome_rng = np.random.default_rng(outcome_seed)
    qns_rng = np.random.default_rng(model_seed)
    model = (
        _logical_model(config, pilot, oracle, int(qns_rng.integers(2**31))) if qns_active else None
    )
    if model is not None:
        probes = tuple(
            probe
            for probe in generator_probes(oracle.generators)
            if (image := oracle.logical_image(probe)) is not None
            and sum(axis >= 0 for axis in image[0]) == 1
        )
        probe_sites, probe_axis_values = probe_axes(probes, config.n_qubits)

    direct_curve = np.full(len(checkpoints), np.nan) if direct_active else None
    qns_curve = np.full(len(checkpoints), np.nan) if qns_active else None
    kernels: list[float] = []
    pending: list[ShotDesign] = []
    checkpoint_lookup = {int(value): index for index, value in enumerate(checkpoints)}
    qns_value = float("nan")

    for shot_number in range(1, config.post_shots + 1):
        design = selector.draw()
        selector_probes = selector.parity_probes(design.basis)
        qns_probes = (
            tuple(
                probes[index]
                for index in np.flatnonzero(
                    covered_probes(design.basis, probe_sites, probe_axis_values)
                )
            )
            if model is not None
            else ()
        )
        read = tuple(dict.fromkeys(selector_probes + qns_probes))
        outcomes = (
            dict(zip(read, oracle.sample_mode_parities(read, outcome_rng).tolist())) if read else {}
        )
        selector.observe(
            design.basis, np.asarray([outcomes[probe] for probe in selector_probes], dtype=np.int8)
        )
        if model is not None:
            for probe in qns_probes:
                model.add_encoded(oracle.logical_image(probe), outcomes[probe])
        pending.append(design)
        if len(pending) == 4:
            if direct_active:
                kernels.append(magic_block_kernel(tuple(pending), oracle, outcome_rng))
            pending.clear()

        point = checkpoint_lookup.get(shot_number)
        if point is not None:
            if direct_curve is not None:
                moment = fourth_moment_from_kernels(kernels)
                direct_curve[point] = magic_from_fourth_moment(moment, config.n_qubits)
            if qns_curve is not None and model is not None:
                model.fit(config.aghdo_fit_steps)
                qns_curve[point] = model.sample_magic(config.qns_property_samples, qns_rng)
                qns_value = qns_curve[point]
        if report is not None and _report_due(shot_number, config, point is not None):
            report(_magic_diagnostics(config, truth, kernels, selector, qns_value, model, design))
        advance_progress()
    return direct_curve, qns_curve


def run_benchmark(config: SimulationConfig) -> BenchmarkResult:
    config.validate()
    oracle = RotatedClusterState(
        config.n_qubits,
        config.k_rotations,
        theta=config.theta,
        vertex_seed=config.rotation_seed,
    )
    subsystem = (
        tuple(config.subsystem)
        if config.subsystem is not None
        else choose_connected_subsystem(oracle.adjacency, config.subsystem_size)
    )
    if len(subsystem) != config.subsystem_size or len(set(subsystem)) != len(subsystem):
        raise ValueError("subsystem must contain subsystem_size distinct qubits")

    checkpoints = _evaluation_checkpoints(config)
    use_pilot = config.property != "PURITY" and config.pilot_shots > 0
    pilot_cost = config.n_qubits * config.pilot_shots if use_pilot else 0
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
    repetition_seeds = np.random.SeedSequence(config.seed).spawn(config.repetitions)
    active_streams = sum(any(method in config.methods for method in names) for _, *names in STREAMS)
    target = f" |A|={len(subsystem)}" if config.property == "PURITY" else ""

    with tqdm(
        total=config.repetitions * active_streams * config.post_shots,
        desc=f"{config.property}{target} exact={truth:.4g}",
        unit="shot",
    ) as progress:
        for repetition, repetition_seed in enumerate(repetition_seeds):
            pilot_seed, *stream_seeds = repetition_seed.spawn(3)
            pilot = (
                run_common_pilot(oracle, config.pilot_shots, np.random.default_rng(pilot_seed))
                if use_pilot
                else None
            )
            label = {"rep": f"{repetition + 1}/{config.repetitions}"}
            if not config.progress_details:
                progress.set_postfix(label, refresh=False)
            for (kind, direct_name, qns_name), stream_seed in zip(STREAMS, stream_seeds):
                direct_active = direct_name in config.methods
                qns_active = qns_name in config.methods
                if not (direct_active or qns_active):
                    continue
                integer_seed = int(np.random.default_rng(stream_seed).integers(2**31))
                stream_label = {**label, "run": direct_name if direct_active else qns_name}

                def report(stats: dict[str, object], stream_label=stream_label) -> None:
                    progress.set_postfix({**stream_label, **stats}, refresh=False)

                stream_report = report if config.progress_details else None
                if config.property == "PURITY":
                    direct_curve, qns_curve = _run_purity_stream(
                        kind,
                        config,
                        oracle,
                        subsystem,
                        checkpoints,
                        direct_active,
                        qns_active,
                        integer_seed,
                        progress.update,
                        stream_report,
                        truth,
                    )
                else:
                    direct_curve, qns_curve = _run_magic_stream(
                        kind,
                        config,
                        pilot,
                        oracle,
                        checkpoints,
                        direct_active,
                        qns_active,
                        integer_seed,
                        progress.update,
                        stream_report,
                        truth,
                    )
                if direct_curve is not None:
                    estimates[direct_name][repetition] = direct_curve
                if qns_curve is not None:
                    estimates[qns_name][repetition] = qns_curve

    return BenchmarkResult(
        config=config,
        shots=shots,
        truth=float(truth),
        estimates=estimates,
        subsystem=subsystem,
        pilot_cost=pilot_cost,
    )
