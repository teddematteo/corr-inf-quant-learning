"""Editable configuration, execution, caching and rendering of the paper suite."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import asdict, field, fields, replace
from os.path import relpath
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from . import __version__
from ._compat import dataclass
from .paper_figures import PaperFigure, PaperPlotSettings, plot_paper_figure
from .simulation import BenchmarkResult, SimulationConfig, run_benchmark
from .simulation_cache import (
    SCHEMA_VERSION,
    cache_path,
    engine_digest,
    fingerprint,
    load_realization,
    save_realization,
    write_json,
)

_SCENARIO_FIELDS = {
    "state",
    "property",
    "methods",
    "n_qubits",
    "k_rotations",
    "subsystem_size",
    "subsystem",
    "rotation_vertices",
    "rows",
    "columns",
}
_TUPLE_FIELDS = {
    "methods",
    "theta_range",
    "subsystem",
    "rotation_vertices",
    "aghdo_feature_densities",
}
_FIGURE_NAMES = {1: "figure_1_cluster_renyi2", 2: "figure_2_ghz_renyi2", 3: "figure_3_magic"}


@dataclass(slots=True)
class PanelJob:
    figure: int
    row: int
    column: int
    config: SimulationConfig

    @property
    def key(self):
        target = (
            f"a{self.config.subsystem_size:02d}" if self.config.property == "RENYI2" else "magic"
        )
        return (
            f"f{self.figure}_r{self.row + 1}_c{self.column + 1}_"
            f"{self.config.state.lower()}_{target}_k{self.config.k_rotations:02d}"
        )

    @property
    def work(self):
        return self.config.post_shots * self.config.repetitions * len(self.config.methods)


@dataclass(slots=True)
class PaperSuite:
    config_path: Path
    output_dir: Path
    jobs: list[PanelJob]
    plot: PaperPlotSettings
    resume: bool = True
    smoke: bool = False
    configuration: dict = field(default_factory=dict)
    cache_dir: Path | None = None


def _unknown(values, allowed, section):
    extra = set(values) - set(allowed)
    if extra:
        raise ValueError(f"Unknown {section} option(s): {', '.join(sorted(extra))}")


def _optional(value):
    return None if isinstance(value, str) and value.lower() == "none" else value


def _simulation_values(values):
    known = {item.name for item in fields(SimulationConfig)}
    _unknown(values, known, "simulation")
    forbidden = set(values) & _SCENARIO_FIELDS
    if forbidden:
        raise ValueError(f"Scenario fields belong in [suite]/[selections]: {sorted(forbidden)}")
    result = {}
    for name, value in values.items():
        value = _optional(value)
        if name in _TUPLE_FIELDS and value is not None:
            if not isinstance(value, list):
                raise ValueError(f"simulation.{name} must be an array or 'none'")
            value = tuple(value)
        result[name] = value
    return result


def _integers(values, name, minimum):
    if (
        not isinstance(values, list) or not values
        or any(not isinstance(v, int) or isinstance(v, bool) or v < minimum for v in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain one or more distinct integers >= {minimum}")
    return values


def _indices(mapping, size):
    value = _optional(mapping.get(str(size), "none"))
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Selection '{size}' must be an index array or 'none'")
    return tuple(value)


def _comparison_methods(layout, name, allowed, default):
    methods = layout.get(name, list(default))
    if (
        not isinstance(methods, list) or not 2 <= len(methods) <= len(allowed)
        or any(not isinstance(method, str) or method not in allowed for method in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError(
            f"suite.{name} must contain two or three distinct methods from {allowed}"
        )
    return tuple(methods)


def load_paper_suite(path: str | Path, *, smoke=False, figures=None) -> PaperSuite:
    """Read a TOML-format .txt safely; no Python expressions or code are executed."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    _unknown(document, {"suite", "simulation", "selections", "plot"}, "section")
    layout = document.get("suite", {})
    _unknown(
        layout,
        {"n_qubits", "k_values", "subsystem_sizes", "figures", "output_dir", "resume",
         "direct_methods", "magic_methods", "magic_k_values", "cache_dir"},
        "suite",
    )
    direct_methods = _comparison_methods(
        layout, "direct_methods", ("CS", "SI-CS", "ConEnh"), ("CS", "ConEnh"),
    )
    magic_methods = _comparison_methods(
        layout, "magic_methods", ("CS-AGHDO", "SI-CS-AGHDO", "ConEnh-AGHDO"),
        ("CS-AGHDO", "ConEnh-AGHDO"),
    )
    n_qubits = layout.get("n_qubits", 100)
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError("suite.n_qubits must be a positive integer")
    k_values = _integers(layout.get("k_values", [1, 10, 50]), "suite.k_values", 0)
    magic_k_values = _integers(layout.get("magic_k_values", k_values), "suite.magic_k_values", 0)
    sizes = _integers(layout.get("subsystem_sizes", [5, 50]), "suite.subsystem_sizes", 1)
    selected = figures if figures is not None else layout.get("figures", [1, 2, 3])
    if (
        not isinstance(selected, (list, tuple))
        or not selected
        or any(
            not isinstance(v, int) or isinstance(v, bool) or v not in (1, 2, 3) for v in selected
        )
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("suite.figures must select distinct figure numbers 1, 2 and/or 3")
    resume = layout.get("resume", True)
    if not isinstance(resume, bool):
        raise ValueError("suite.resume must be true or false")
    output = layout.get("output_dir", "output")
    if not isinstance(output, str) or not output:
        raise ValueError("suite.output_dir must be a nonempty path string")
    output_dir = Path(output).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    cache = layout.get("cache_dir", "cache")
    if not isinstance(cache, str) or not cache:
        raise ValueError("suite.cache_dir must be a nonempty path string")
    cache_dir = Path(cache).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = path.parent / cache_dir
    if smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    options = _simulation_values(document.get("simulation", {}))
    if smoke:
        options.update(
            post_shots=32,
            repetitions=1,
            evaluation_points=5,
            pilot_shots=0,
            pauli_samples=128,
            conenh_max_cliques=128,
            conenh_candidates=16,
            conenh_update_every=4,
            aghdo_fit_steps=2,
            qns_property_samples=32,
        )
    selections = document.get("selections", {})
    _unknown(selections, {"rows", "columns", "subsystems", "rotation_vertices"}, "selections")
    for key in ("subsystems", "rotation_vertices"):
        if not isinstance(selections.get(key, {}), dict):
            raise ValueError(f"selections.{key} must be a table")
    shared = dict(
        n_qubits=n_qubits,
        rows=_optional(selections.get("rows", "none")),
        columns=_optional(selections.get("columns", "none")),
        **options,
    )
    jobs = []
    for number in sorted(selected):
        figure_k_values = magic_k_values if number == 3 else k_values
        for row in range(2 if number == 3 else len(sizes)):
            state = "CLUSTER" if number == 1 or (number == 3 and row == 0) else "GHZ"
            for column, k in enumerate(figure_k_values):
                size = sizes[row] if number != 3 else min(50, n_qubits)

                config = SimulationConfig(
                    **shared,
                    state=state,
                    k_rotations=k,
                    subsystem_size=size,
                    subsystem=_indices(selections.get("subsystems", {}), size)
                    if number != 3
                    else None,
                    rotation_vertices=_indices(selections.get("rotation_vertices", {}), k),
                    property="MAGIC" if number == 3 else "RENYI2",
                    methods=magic_methods if number == 3 else direct_methods,
                )
                config.validate()
                jobs.append(PanelJob(number, row, column, config))
    plot_values = document.get("plot", {})
    _unknown(plot_values, {f.name for f in fields(PaperPlotSettings)}, "plot")
    plot_values = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in plot_values.items()
    }
    for key in ("x_min", "figsize", "y_limits"):
        if key in plot_values:
            plot_values[key] = _optional(plot_values[key])
    plot = PaperPlotSettings(**plot_values)
    plot.validate()
    for job in jobs:
        palette_name = "magic_colors" if job.figure == 3 else "colors"
        if len(getattr(plot, palette_name)) < len(job.config.methods):
            raise ValueError(
                f"plot.{palette_name} must provide at least one color per selected method"
            )
    return PaperSuite(
        path, output_dir.resolve(), jobs, plot, resume, smoke, document, cache_dir.resolve(),
    )


def _figure_specification(number, jobs, results, smoke):
    selected = [job for job in jobs if job.figure == number]
    panels = [results[job.key] for job in selected]
    n = selected[0].config.n_qubits
    if number in (1, 2):
        state = "Cluster" if number == 1 else "GHZ"
        title = rf"Bipartite 2-Renyi entanglement entropy · {state} state · $n={n}$"
        rows = tuple(
            rf"$|A|={next(job.config.subsystem_size for job in selected if job.row == row)}$"
            for row in sorted({job.row for job in selected})
        )
    else:
        title = rf"Stabilizer 2-Renyi entropy · $n={n}$"
        rows = ("Cluster state", "GHZ state")
    return PaperFigure(number, title, rows, panels, smoke)


def _assemble_panel(job, pieces):
    """Combine cached method/realization results, preserving each exact value."""
    config = job.config
    first_method = config.methods[0]
    reference = [pieces[(first_method, rep)] for rep in range(config.repetitions)]
    truths = np.array([item.truth for item in reference])
    history = []
    for rep, item in enumerate(reference):
        streams = {}
        reference_details = item.execution["repetitions"][0]
        for method in config.methods:
            candidate = pieces[(method, rep)]
            if (not np.array_equal(candidate.shots, item.shots)
                    or candidate.truth != item.truth):
                raise ValueError("Cached methods refer to different realizations or checkpoints")
            details = candidate.execution["repetitions"][0]
            if (details["rotation_vertices"] != reference_details["rotation_vertices"]
                    or details["rotation_angles"] != reference_details["rotation_angles"]):
                raise ValueError("Cached methods have inconsistent rotation realizations")
            streams[method] = details
        history.append({**item.execution["repetitions"][0], "repetition": rep,
                        "truth": item.truth, "methods": streams})
    return BenchmarkResult(
        config, reference[0].shots, float(truths.mean()),
        {method: np.concatenate([pieces[(method, rep)].estimates[method]
                                 for rep in range(config.repetitions)])
         for method in config.methods},
        reference[0].subsystem, reference[0].pilot_cost,
        {**reference[0].execution, "repetitions": history}, truths=truths,
    )


def run_paper_suite(suite: PaperSuite, *, plot_only=False, force=False, show_progress=True) -> dict:
    """Reuse compatible realizations, acquire missing ones, and render the layout."""
    if plot_only and force:
        raise ValueError("--plot-only and --force cannot be combined")
    engine = engine_digest()
    reuse = plot_only or (suite.resume and not force)
    directory = suite.cache_dir or suite.config_path.parent / "cache"

    entries, loaded, missing = {}, {}, []
    for job in suite.jobs:
        for method in job.config.methods:
            for rep in range(job.config.repetitions):
                key = fingerprint(job.config, method, rep)
                path = cache_path(directory, key)
                entries[(job.key, method, rep)] = (key, path)
                if key in loaded:
                    continue
                result = None
                if reuse and path.is_file():
                    try:
                        result = load_realization(path, job.config, method, rep)
                    except (ValueError, KeyError, TypeError, OverflowError):
                        if show_progress:
                            tqdm.write(f"Invalid cache entry, recomputing: {path}")
                loaded[key] = result
                if result is None:
                    missing.append(str(path))
    if plot_only and missing:
        raise FileNotFoundError("Missing saved simulations for --plot-only:\n" + "\n".join(missing))
    suite.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = suite.output_dir / "configs"
    snapshots.mkdir(exist_ok=True)
    snapshot_hash = hashlib.sha256(suite.config_path.read_bytes()).hexdigest()[:12]
    snapshot = snapshots / f"{snapshot_hash}.txt"
    if not snapshot.exists():
        snapshot.write_bytes(suite.config_path.read_bytes())
    results, data_files, panel_entries = {}, set(), []
    reused_panels = computed_panels = reused_entries = computed_entries = computed_shots = 0
    total = sum(job.work for job in suite.jobs)
    with tqdm(total=total, desc="Simulations", unit="shot", dynamic_ncols=True,
              disable=not show_progress, mininterval=0.25) as progress:
        for index, job in enumerate(suite.jobs, start=1):
            label = (f"{index}/{len(suite.jobs)} · Fig.{job.figure} · {job.config.state} · "
                     f"k={job.config.k_rotations}")
            if job.figure != 3:
                label += f" · |A|={job.config.subsystem_size}"
            progress.set_description(label, refresh=False)
            pieces, computed = {}, False
            files = []
            for rep in range(job.config.repetitions):
                for method in job.config.methods:
                    key, path = entries[(job.key, method, rep)]
                    result = loaded[key]
                    if result is not None:

                        result = replace(result, shots=(
                            result.shots
                            - (result.pilot_cost if result.config.count_pilot_cost else 0)
                            + (result.pilot_cost if job.config.count_pilot_cost else 0)
                        ), config=replace(job.config, methods=(method,), repetitions=1))
                        reused_entries += 1
                        progress.update(job.config.post_shots)
                    else:
                        progress.set_postfix_str(
                            f"{method}, rep {rep + 1}/{job.config.repetitions}", refresh=False,
                        )
                        completed = 0

                        def update(event):
                            nonlocal completed
                            count = int(event["completed"])
                            if count < completed or count > job.config.post_shots:
                                raise RuntimeError("Inconsistent benchmark progress")
                            progress.update(count - completed)
                            completed = count

                        single = replace(job.config, methods=(method,), repetitions=1)
                        start = perf_counter()
                        result = run_benchmark(
                            single, on_update=update, show_progress=False, repetition_start=rep,
                        )
                        if completed != single.post_shots:
                            raise RuntimeError("Benchmark finished with an incomplete shot count")
                        save_realization(path, result, method, rep, engine=engine,
                                         seconds=perf_counter() - start)
                        loaded[key] = result
                        computed_entries += 1
                        computed_shots += single.post_shots
                        computed = True
                    pieces[(method, rep)] = result
                    relative = relpath(path, suite.output_dir)
                    data_files.add(relative)
                    files.append({"method": method, "repetition": rep, "file": relative})
            computed_panels += int(computed)
            reused_panels += int(not computed)
            results[job.key] = _assemble_panel(job, pieces)
            panel_entries.append({"key": job.key, "figure": job.figure, "row": job.row,
                                  "column": job.column, "entries": files})
    manifest = {
        "schema_version": SCHEMA_VERSION, "smoke": suite.smoke,
        "package_version": __version__, "engine_sha256": engine,
        "source_configuration": str(suite.config_path),
        "configuration_snapshot": str(snapshot.relative_to(suite.output_dir)),
        "cache_dir": str(directory), "plot_settings": asdict(suite.plot),
        "panel_count": len(suite.jobs), "reused_panel_count": reused_panels,
        "computed_panel_count": computed_panels,
        "reused_entry_count": reused_entries, "computed_entry_count": computed_entries,
        "post_measurement_shots": total,
        "computed_post_measurement_shots": computed_shots,
        "data_files": sorted(data_files), "panels": panel_entries, "figures": [],
    }
    import matplotlib.pyplot as plt

    for number in sorted({job.figure for job in suite.jobs}):
        specification = _figure_specification(number, suite.jobs, results, suite.smoke)
        figure, _, paths, metadata = plot_paper_figure(
            specification, suite.plot, suite.output_dir / "figures" / _FIGURE_NAMES[number],
        )
        plt.close(figure)
        manifest["figures"].append({
            "number": number,
            "files": [str(path.relative_to(suite.output_dir)) for path in paths], **metadata,
        })
        if metadata["zero_truth_fallback"] and show_progress:
            tqdm.write(f"Figure {number}: using absolute error because an exact value is zero.")
    write_json(suite.output_dir / "manifest.json", manifest)
    return manifest


def main(argv=None, *, default_config=None):
    parser = argparse.ArgumentParser(
        description="Generate research figures with arbitrary rows and columns.",
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=default_config,
        help="Editable TOML-format .txt configuration",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the panel plan")
    parser.add_argument("--smoke", action="store_true", help="32 shots; separate *_smoke output")
    parser.add_argument("--plot-only", action="store_true", help="Render matching saved results")
    parser.add_argument("--force", action="store_true", help="Rerun even matching saved panels")
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        help="Select figure numbers without editing the configuration",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("provide a configuration .txt path")
    if args.force and args.plot_only:
        parser.error("--force and --plot-only cannot be combined")
    try:
        suite = load_paper_suite(args.config, smoke=args.smoke, figures=args.figures)
        print(f"Configuration: {suite.config_path}\nOutput: {suite.output_dir}")
        print(
            f"{len(suite.jobs)} panels, "
            f"{sum(job.work for job in suite.jobs):,} total post-pilot shots"
        )
        if args.dry_run:
            for job in suite.jobs:
                print(
                    f"  {job.key}: {job.config.methods}, "
                    f"N={job.config.post_shots}, R={job.config.repetitions}, "
                    f"device={job.config.device}, backend={job.config.resource_plan().backend}"
                )
            return
        manifest = run_paper_suite(suite, plot_only=args.plot_only, force=args.force)
        print(
            f"Panels reused: {manifest['reused_panel_count']}; "
            f"computed: {manifest['computed_panel_count']}\n"
            f"Method/repetition results reused: {manifest['reused_entry_count']}; "
            f"computed: {manifest['computed_entry_count']}"
        )
        for item in manifest["figures"]:
            for filename in item["files"]:
                print(suite.output_dir / filename)
    except KeyboardInterrupt:
        print(
            "\nInterrupted: completed method/repetition results are saved and will be reused.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
