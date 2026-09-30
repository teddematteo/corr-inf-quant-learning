from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .plotting import plot_benchmark
from .simulation import BenchmarkResult, SimulationConfig, run_benchmark

K_ROTATIONS = (10, 30, 50)
SUBSYSTEM_SIZES = (3, 5, 7)
REPETITIONS = 10
ALL_METHODS = ("CS", "CorInf", "CS-AGHDO", "CorInf-AGHDO")
COMPARISONS = (
    ("corinf-vs-cs", ("CS", "CorInf")),
    ("corinf-aghdo-vs-cs-aghdo", ("CS-AGHDO", "CorInf-AGHDO")),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the requested CS/CorInf and CS-AGHDO/CorInf-AGHDO purity and magic "
            "comparison matrix (10 repetitions per configuration)."
        )
    )
    parser.add_argument("--n-qubits", type=int, default=100)
    parser.add_argument(
        "--post-shots",
        type=int,
        default=10_000,
        help="Measurement shots per method (must be divisible by 4 for magic).",
    )
    parser.add_argument("--evaluation-points", type=int, default=9)
    parser.add_argument("--pilot-shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--error-mode",
        choices=("absolute", "relative"),
        default="relative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures") / "comparison_suite_10_reps",
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
    """Use the tuned parameters from the project notebook for every run."""
    return SimulationConfig(
        n_qubits=args.n_qubits,
        k_rotations=k_rotations,
        subsystem_size=subsystem_size,
        property=property_name,
        methods=ALL_METHODS,
        pilot_shots=args.pilot_shots,
        post_shots=args.post_shots,
        repetitions=REPETITIONS,
        evaluation_points=args.evaluation_points,
        seed=args.seed,
        count_pilot_cost=True,
        corinf_exploration=0.001,
        corinf_commuting_bias=1.0,
        corinf_candidates=100,
        corinf_purity_weight=1.0,
        corinf_generator_weight=1.0,
        corinf_softmax_temperature=0.01,
        corinf_pauli_floor=0.0,
        corinf_update_every=1,
        corinf_clip_bound="POINT",
        aghdo_fit_steps=12,
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
    target = f"_a{result.config.subsystem_size:02d}" if result.config.property == "PURITY" else ""

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

    tqdm.write(f"Saved {len(saved)} files in {output_dir}")
    for path in saved:
        tqdm.write(f"  {path.name}")


if __name__ == "__main__":
    main()
