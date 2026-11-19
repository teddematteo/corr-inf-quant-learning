from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .plotting import plot_benchmark
from .simulation import METHODS, BenchmarkResult, SimulationConfig, run_benchmark
from .states import STATE_TYPES

K_ROTATIONS = (10, 30, 50)
SUBSYSTEM_SIZES = (3, 5, 7)
REPETITIONS = 10
ALL_METHODS = METHODS
COMPARISONS = (
    ("conenh-vs-cs", ("CS", "ConEnh")),
    ("conenh-aghdo-vs-cs-aghdo", ("CS-AGHDO", "ConEnh-AGHDO")),
    ("conenh-vs-si-cs", ("SI-CS", "ConEnh")),
    ("conenh-aghdo-vs-si-cs-aghdo", ("SI-CS-AGHDO", "ConEnh-AGHDO")),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CS/SI-CS/ConEnh and their AGHDO purity, Renyi-2 and magic "
            "comparison matrix (10 repetitions per configuration)."
        )
    )
    parser.add_argument("--n-qubits", type=int, default=100)
    parser.add_argument("--state", choices=STATE_TYPES, default="CLUSTER")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="Device for ConEnh policy calculations (default: cpu).")
    parser.add_argument(
        "--post-shots",
        type=int,
        default=10_000,
        help="Measurement shots per method (must be divisible by 4 for magic).",
    )
    parser.add_argument("--evaluation-points", type=int, default=9)
    parser.add_argument("--pilot-shots", type=int, default=0)
    parser.add_argument(
        "--si-cs-exploration", type=float, default=None,
        help="SI-CS uniform exploration eta; defaults to ConEnh's eta (0.001 in this suite).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--aghdo-local-rank", type=int, default=8)
    parser.add_argument("--aghdo-features", type=int, nargs="+", default=[8, 4])
    parser.add_argument("--aghdo-init-std", type=float, default=0.01)
    parser.add_argument("--aghdo-learning-rate", type=float, default=0.005)
    parser.add_argument("--aghdo-fit-steps", type=int, default=12)
    parser.add_argument(
        "--property-clipping", action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable final physical property clipping for every method and property. "
             "Omit to retain legacy defaults.",
    )
    parser.add_argument("--theta", type=float, default=float(np.pi / 8),
                        help="Fixed local rotation angle in radians (default: pi/8).")
    parser.add_argument(
        "--theta-range", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="Sample one uniform angle per rotation in this range (radians); overrides --theta.",
    )
    parser.add_argument("--rotation-seed", type=int, default=1729,
                        help="Seed for rotation sites and angles (default: 1729).")
    parser.add_argument(
        "--error-mode",
        choices=("absolute", "relative"),
        default="relative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulations") / "output" / "comparisons",
        help="Directory that will receive every PDF and PNG figure.",
    )
    return parser.parse_args()


def _config(
    args: argparse.Namespace,
    *,
    property_name: str,
    k_rotations: int,
    subsystem_size: int,
) -> SimulationConfig:
    """Construct a benchmark configuration for one comparison."""
    return SimulationConfig(
        state=args.state,
        n_qubits=args.n_qubits,
        device=args.device,
        k_rotations=k_rotations,
        subsystem_size=subsystem_size,
        property=property_name,
        methods=ALL_METHODS,
        pilot_shots=args.pilot_shots,
        post_shots=args.post_shots,
        repetitions=REPETITIONS,
        evaluation_points=args.evaluation_points,
        seed=args.seed,
        property_clipping=args.property_clipping,
        theta=args.theta,
        theta_range=tuple(args.theta_range) if args.theta_range is not None else None,
        rotation_seed=args.rotation_seed,
        count_pilot_cost=True,
        conenh_exploration=0.001,
        si_cs_exploration=args.si_cs_exploration,
        conenh_commuting_bias=1.0,
        conenh_candidates=100,
        conenh_purity_weight=1.0,
        conenh_generator_weight=1.0,
        conenh_softmax_temperature=0.01,
        conenh_pauli_floor=0.0,
        conenh_update_every=1,
        conenh_clip_bound="POINT",
        aghdo_rank=args.aghdo_local_rank,
        aghdo_feature_densities=tuple(args.aghdo_features),
        aghdo_init_std=args.aghdo_init_std,
        aghdo_learning_rate=args.aghdo_learning_rate,
        aghdo_fit_steps=args.aghdo_fit_steps,
        qns_property_samples=4096,
        progress_details=False,
    )


def _comparison_result(result: BenchmarkResult, methods: tuple[str, str]) -> BenchmarkResult:
    """Make a plotting view without rerunning an already computed benchmark."""
    return replace(
        result,
        config=replace(result.config, methods=methods),
        estimates={method: result.estimates[method] for method in methods},
    )


def _save_comparisons(
    result: BenchmarkResult,
    output_dir: Path,
    error_mode: str,
) -> tuple[Path, ...]:
    saved: list[Path] = []
    property_name = result.config.property.lower()
    k_rotations = result.config.k_rotations
    target = (
        f"_a{result.config.subsystem_size:02d}"
        if result.config.property in ("PURITY", "RENYI2")
        else ""
    )

    for comparison_name, methods in COMPARISONS:
        plotting_result = _comparison_result(result, methods)
        stem = output_dir / (
            f"{property_name}_{comparison_name}{target}_k{k_rotations:02d}_{error_mode}"
        )
        figure, _, paths = plot_benchmark(
            plotting_result,
            error_mode=error_mode,
            log_x=True,
            log_y=True,
            smooth_window=1,
            show_band=True,
            output_stem=stem,
        )
        saved.extend(paths)
        plt.close(figure)
    return tuple(saved)


def main() -> None:
    args = _arguments()
    if args.post_shots % 4:
        raise SystemExit("--post-shots must be divisible by 4 because the suite includes magic")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("PURITY", k_rotations, subsystem_size)
        for subsystem_size in SUBSYSTEM_SIZES
        for k_rotations in K_ROTATIONS
    ] + [("MAGIC", k_rotations, 5) for k_rotations in K_ROTATIONS]

    saved: list[Path] = []
    with tqdm(jobs, desc="Complete comparison suite", unit="test") as suite_progress:
        for property_name, k_rotations, subsystem_size in suite_progress:
            target = f", |A|={subsystem_size}" if property_name == "PURITY" else ""
            suite_progress.set_postfix_str(f"{property_name}, k={k_rotations}{target}")
            result = run_benchmark(
                _config(
                    args,
                    property_name=property_name,
                    k_rotations=k_rotations,
                    subsystem_size=subsystem_size,
                )
            )
            saved.extend(_save_comparisons(result, output_dir, args.error_mode))
            if property_name == "PURITY":
                saved.extend(_save_comparisons(result.as_renyi2(), output_dir, args.error_mode))

    tqdm.write(f"Saved {len(saved)} files in {output_dir}")
    for path in saved:
        tqdm.write(f"  {path.name}")


if __name__ == "__main__":
    main()
