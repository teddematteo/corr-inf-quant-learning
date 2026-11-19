"""Content-addressed scientific results, one method and realization per file."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .cluster_state import choose_connected_subsystem, grid_adjacency
from .resources import plan_resources
from .simulation import BenchmarkResult, SimulationConfig, _evaluation_checkpoints

CACHE_VERSION = "uniform-realizations-v1"
SCHEMA_VERSION = 2


def scientific_configuration(config: SimulationConfig, method: str) -> dict:
    """Resolve defaults and retain only settings used by this method's path."""
    single = replace(config, methods=(method,), repetitions=1)
    neural = "AGHDO" in method
    focused = neural and config.property == "MAGIC"
    bound = method.startswith("ConEnh")
    informed = method.startswith("SI-CS")
    names = {
        "state", "n_qubits", "k_rotations", "property", "seed", "device",
        "post_shots", "property_clipping",
    }
    if neural:
        names |= {
            "aghdo_rank", "aghdo_feature_densities", "aghdo_init_std",
            "aghdo_learning_rate", "aghdo_fit_steps",
        }
        if not focused:
            names |= {"aghdo_batch_size", "qns_property_samples"}
    if bound:
        names |= {"conenh_exploration", "conenh_softmax_temperature"}
        if not focused:
            names |= {
                "conenh_commuting_bias", "conenh_local_regularization",
                "conenh_purity_weight", "conenh_generator_weight", "conenh_pauli_floor",
            }
    values = asdict(single)
    selected = {name: values[name] for name in sorted(names)}

    selected["checkpoints"] = _evaluation_checkpoints(single).tolist()
    if config.k_rotations:
        bounds = config.theta_range or (config.theta, config.theta)
        selected["theta_range"] = tuple(float(value) for value in bounds)
        selected["rotation_vertices"] = config.rotation_vertices
        if config.rotation_vertices is None or bounds[0] != bounds[1]:
            selected["rotation_seed"] = config.rotation_seed
    adjacency = grid_adjacency(config.n_qubits, rows=config.rows, columns=config.columns)
    if config.state == "CLUSTER":
        selected["adjacency"] = adjacency
    if config.property != "MAGIC":
        subsystem = config.subsystem if config.subsystem is not None else (
            choose_connected_subsystem(adjacency, config.subsystem_size, columns=config.columns)
        )
        selected["subsystem"] = subsystem
        register_mask = sum(1 << site for site in subsystem)
    else:
        register_mask = (1 << config.n_qubits) - 1
        selected["pilot_shots"] = config.pilot_shots
    if informed:
        selected["si_cs_exploration"] = (
            config.conenh_exploration if config.si_cs_exploration is None
            else config.si_cs_exploration
        )
    if focused:
        if bound:
            selected["candidates"] = config.conenh_candidates or 64
    else:
        supports = (
            [(1 << site) | neighbours for site, neighbours in enumerate(adjacency)]
            if config.state == "CLUSTER" else
            [(1 << config.n_qubits) - 1]
            + [1 | (1 << site) for site in range(1, config.n_qubits)]
        )
        plan = plan_resources(single, sum(bool(word & register_mask) for word in supports))
        selected["backend"] = plan.backend
        if plan.backend == "sampled" and (informed or bound or config.property == "MAGIC"):
            selected["pauli_samples"] = plan.pauli_samples
            selected["pauli_uniform_fraction"] = config.pauli_uniform_fraction
        if bound:
            selected.update(max_cliques=plan.max_cliques, update_every=plan.update_every)
            if plan.backend == "sampled":
                selected["candidates"] = plan.candidates
        if (bound or informed) and not neural:
            selected["conenh_clipping"] = config.conenh_clipping
            if config.conenh_clipping:
                selected["conenh_clip_bound"] = config.conenh_clip_bound
        if config.property == "MAGIC" and not neural and config.property_clipping is None:
            selected["magic_clipping"] = config.magic_clipping
    return safe_json(selected)


def fingerprint(config: SimulationConfig, method: str, repetition: int) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "method": method,
        "repetition": repetition,
        "scientific_configuration": scientific_configuration(config, method),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_path(directory: Path, key: str) -> Path:
    return directory / key[:2] / f"{key}.json"


def engine_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp",
            delete=False, encoding="utf-8",
        ) as stream:
            temporary = Path(stream.name)
            json.dump(safe_json(document), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_realization(path, result, method, repetition, *, engine, seconds):
    config = result.config
    write_json(path, {
        "schema_version": SCHEMA_VERSION, "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint(config, method, repetition),
        "scientific_configuration": scientific_configuration(config, method),
        "method": method, "repetition": repetition, "config": asdict(config),
        "engine_sha256": engine, "package_version": __version__,
        "python": platform.python_version(), "seconds": seconds,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": result.shots - (result.pilot_cost if config.count_pilot_cost else 0),
        "truth": result.truth, "estimates": result.estimates[method][0],
        "subsystem": result.subsystem, "pilot_cost": result.pilot_cost,
        "execution": result.execution,
    })


def load_realization(path, config, method, repetition) -> BenchmarkResult:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid cache document: {path}")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("cache_version") != CACHE_VERSION
        or document.get("fingerprint") != fingerprint(config, method, repetition)
        or document.get("scientific_configuration") != scientific_configuration(config, method)
        or document.get("method") != method or document.get("repetition") != repetition
    ):
        raise ValueError(f"Incompatible scientific configuration in cache: {path}")
    checkpoints = np.asarray(document["checkpoints"], dtype=int)
    estimates = np.asarray(document["estimates"], dtype=float)
    truth = float(document["truth"])
    execution = document["execution"]
    if not isinstance(execution, dict):
        raise ValueError(f"Invalid cached execution metadata: {path}")
    history = execution.get("repetitions")
    if (
        not isinstance(history, list) or len(history) != 1 or not isinstance(history[0], dict)
        or history[0].get("repetition") != repetition or history[0].get("truth") != truth
        or len(history[0].get("rotation_vertices", [])) != config.k_rotations
        or len(history[0].get("rotation_angles", [])) != config.k_rotations
    ):
        raise ValueError(f"Invalid cached realization metadata: {path}")
    if (
        not np.array_equal(checkpoints, _evaluation_checkpoints(config))
        or estimates.shape != checkpoints.shape or not np.isfinite(truth)
    ):
        raise ValueError(f"Invalid cached estimates: {path}")
    pilot_cost = int(document["pilot_cost"])
    return BenchmarkResult(
        replace(config, methods=(method,), repetitions=1),
        checkpoints + (pilot_cost if config.count_pilot_cost else 0), truth,
        {method: estimates[None, :]}, tuple(document["subsystem"]), pilot_cost,
        execution, truths=np.array([truth]),
    )
