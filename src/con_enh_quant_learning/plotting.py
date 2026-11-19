from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from ._compat import strict_zip
from .simulation import BenchmarkResult, SimulationConfig

_STYLES = {
    "CS": {"color": "#0072B2", "marker": "o"},
    "ConEnh": {"color": "#D55E00", "marker": "s"},
    "CS-AGHDO": {"color": "#0072B2", "marker": "o"},
    "ConEnh-AGHDO": {"color": "#D55E00", "marker": "s"},
    "SI-CS": {"color": "#009E73", "marker": "^"},
    "SI-CS-AGHDO": {"color": "#009E73", "marker": "^"},
}

_THETA_COLORS = ("#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf")

_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12.0,
    "axes.titlesize": 12.5,
    "axes.labelsize": 13.0,
    "legend.fontsize": 11.5,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7,
    "ytick.minor.width": 0.7,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 2.8,
    "ytick.minor.size": 2.8,
    "lines.linewidth": 2.2,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = min(int(window), len(values))
    smoothed = np.full_like(values, np.nan, dtype=float)
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        start = max(0, index - window + 1)
        local = values[start : index + 1]
        smoothed[index] = local[np.isfinite(local)].mean()
    return smoothed


def _error_statistics(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Summarize finite repetitions, leaving unavailable checkpoints as gaps."""
    finite = np.isfinite(errors)
    counts = finite.sum(axis=0)
    values = np.where(finite, errors, 0.0)
    mean = np.full(errors.shape[1], np.nan)
    np.divide(values.sum(axis=0), counts, out=mean, where=counts > 0)
    centered = np.where(finite, values - np.where(counts > 0, mean, 0.0), 0.0)
    sem_squared = np.zeros_like(mean)
    np.divide(
        (centered**2).sum(axis=0), counts * (counts - 1),
        out=sem_squared, where=counts > 1,
    )
    sem_squared[counts == 0] = np.nan
    return mean, np.sqrt(sem_squared)


def _interpolate_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_x: bool,
    log_y: bool,
    points_per_interval: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    if log_x:
        valid &= x > 0
    if log_y:
        valid &= y > 0
    if not np.all(valid):
        indices = np.flatnonzero(valid)
        segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        curve_x, curve_y = [], []
        for segment in segments:
            if not len(segment):
                continue
            if curve_x:
                curve_x.append(np.array([np.nan]))
                curve_y.append(np.array([np.nan]))
            local_x, local_y = _interpolate_curve(
                x[segment], y[segment], log_x=log_x, log_y=log_y,
                points_per_interval=points_per_interval,
            )
            curve_x.append(local_x)
            curve_y.append(local_y)
        if not curve_x:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(curve_x), np.concatenate(curve_y)
    if len(x) < 2:
        return x, y
    transformed_x = np.log10(x) if log_x else x.astype(float)
    transformed_y = np.log10(y) if log_y else y.astype(float)

    dense_x = np.concatenate([
        np.linspace(left, right, points_per_interval, endpoint=False)
        for left, right in strict_zip(transformed_x[:-1], transformed_x[1:])
    ] + [transformed_x[-1:]])
    steps = np.diff(transformed_x)
    secants = np.diff(transformed_y) / steps
    slopes = np.empty_like(transformed_y)
    slopes[0], slopes[-1] = secants[0], secants[-1]
    for index in range(1, len(x) - 1):
        left, right = secants[index - 1], secants[index]
        if left * right <= 0.0:
            slopes[index] = 0.0
        else:
            weight_left = 2.0 * steps[index] + steps[index - 1]
            weight_right = steps[index] + 2.0 * steps[index - 1]
            slopes[index] = (weight_left + weight_right) / (
                weight_left / left + weight_right / right
            )
    interval = np.clip(np.searchsorted(transformed_x, dense_x, side="right") - 1, 0, len(x) - 2)
    width = transformed_x[interval + 1] - transformed_x[interval]
    position = (dense_x - transformed_x[interval]) / width
    h00 = 2 * position**3 - 3 * position**2 + 1
    h10 = position**3 - 2 * position**2 + position
    h01 = -2 * position**3 + 3 * position**2
    h11 = position**3 - position**2
    dense_y = (
        h00 * transformed_y[interval]
        + h10 * width * slopes[interval]
        + h01 * transformed_y[interval + 1]
        + h11 * width * slopes[interval + 1]
    )
    return (
        np.power(10.0, dense_x) if log_x else dense_x,
        np.power(10.0, dense_y) if log_y else dense_y,
    )


def _configuration_title(result: BenchmarkResult) -> str:
    config = result.config
    common = rf"$n={config.n_qubits}$, $k={config.k_rotations}$"
    if config.state != "CLUSTER":
        common = f"{config.state}, {common}"
    if config.theta_range is not None:
        common += rf", ${_rotation_label(config)}$"
    repetitions = f"{config.repetitions} repetitions"
    if config.property_clipping is True:
        repetitions += ", property clipping"
    if config.property == "PURITY":
        return rf"Local purity: {common}, $|A|={len(result.subsystem)}$, {repetitions}"
    if config.property == "RENYI2":
        return (
            rf"Bipartite 2-Renyi entanglement entropy $\mathcal{{E}}_2$ (bits): {common}, "
            rf"$|A|={len(result.subsystem)}$, {repetitions}"
        )
    pilot = "no pilot" if result.pilot_cost == 0 else f"pilot {config.pilot_shots} shots/generator"
    return rf"Stabilizer 2-Renyi entropy $\mathcal{{M}}_2$: {common}, {pilot}, {repetitions}"


def _truth_label(result: BenchmarkResult) -> str:
    prefix = "mean exact" if not np.allclose(result.repetition_truths, result.truth) else "exact"
    if result.config.property == "PURITY":
        return rf"{prefix} $\mathrm{{Tr}}\,\rho_A^2 = {result.truth:.4g}$"
    if result.config.property == "RENYI2":
        return rf"{prefix} $\mathcal{{E}}_2 = {result.truth:.4g}$ bits"
    return rf"{prefix} $\mathcal{{M}}_2 = {result.truth:.4g}$"


def _theta_label(theta: float) -> str:
    multiple = Fraction(float(theta) / np.pi).limit_denominator(64)
    if not np.isclose(float(multiple) * np.pi, theta, rtol=0, atol=1e-10):
        return f"{theta:.6g}"
    numerator, denominator = multiple.numerator, multiple.denominator
    if numerator == 0:
        return "0"
    sign = "-" if numerator < 0 else ""
    coefficient = "" if abs(numerator) == 1 else str(abs(numerator))
    label = rf"{sign}{coefficient}\pi"
    return label if denominator == 1 else rf"{label}/{denominator}"


def _rotation_label(config: SimulationConfig) -> str:
    if config.theta_range is None:
        return rf"\theta={_theta_label(config.theta)}"
    lower, upper = config.theta_range
    return rf"\theta_i\sim U[{_theta_label(lower)}, {_theta_label(upper)}]"


def plot_benchmark(
    result: BenchmarkResult,
    *,
    error_mode: str = "absolute",
    log_x: bool = True,
    log_y: bool = True,
    smooth_window: int = 1,
    output_stem: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
    show_band: bool = True,
) -> tuple[plt.Figure, plt.Axes, tuple[Path, ...]]:
    return _plot_benchmarks(
        (result,), error_mode=error_mode, log_x=log_x, log_y=log_y,
        smooth_window=smooth_window, output_stem=output_stem, title=title,
        figsize=figsize, show_band=show_band,
    )


def plot_theta_sweep(
    results: Sequence[BenchmarkResult],
    *,
    error_mode: str = "absolute",
    log_x: bool = True,
    log_y: bool = True,
    smooth_window: int = 1,
    output_stem: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 5.5),
    show_band: bool = True,
    colors: Sequence[str] | None = None,
) -> tuple[plt.Figure, plt.Axes, tuple[Path, ...]]:
    """Compare rotations with light/medium/dark ConEnh/SI-CS/CS curves per hue."""
    results = tuple(results)
    if not results:
        raise ValueError("At least one theta result is required")
    reference = results[0]
    for result in results:
        if (
            replace(result.config, theta=reference.config.theta,
                    theta_range=reference.config.theta_range) != reference.config
            or result.subsystem != reference.subsystem
            or result.pilot_cost != reference.pilot_cost
            or not np.array_equal(result.shots, reference.shots)
        ):
            raise ValueError("Theta sweep results must share all settings except theta/theta_range")
    settings = [
        ("fixed", result.config.theta) if result.config.theta_range is None
        else ("range", *result.config.theta_range) for result in results
    ]
    if len(set(settings)) != len(results):
        raise ValueError("Theta sweep angles or ranges must be distinct")
    if colors is None:
        colors = (
            _THETA_COLORS[:len(results)] if len(results) <= len(_THETA_COLORS)
            else tuple(plt.get_cmap("hsv")(i / len(results)) for i in range(len(results)))
        )
    if len(colors) != len(results):
        raise ValueError("Provide one base color per theta")
    base_colors = tuple(to_rgb(color) for color in colors)
    return _plot_benchmarks(
        results, error_mode=error_mode, log_x=log_x, log_y=log_y,
        smooth_window=smooth_window, output_stem=output_stem,
        title=title if title is not None else (
            f"{_configuration_title(reference)}, {len(results)} angles"
        ),
        figsize=figsize, show_band=show_band, theta_colors=base_colors,
    )


def _plot_benchmarks(
    results: Sequence[BenchmarkResult],
    *,
    error_mode: str,
    log_x: bool,
    log_y: bool,
    smooth_window: int,
    output_stem: str | Path | None,
    title: str | None,
    figsize: tuple[float, float],
    show_band: bool,
    theta_colors: Sequence[tuple[float, float, float]] | None = None,
) -> tuple[plt.Figure, plt.Axes, tuple[Path, ...]]:

    all_errors = [result.errors(error_mode) for result in results]
    relative = error_mode.lower() == "relative"
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        has_finite_estimates = False
        curves = (
            (index, result, method, errors[method])
            for index, (result, errors) in enumerate(strict_zip(results, all_errors))
            for method in result.config.methods
        )
        for index, result, method, method_errors in curves:
            mean, sem = _error_statistics(method_errors)
            mean = _moving_average(mean, smooth_window)
            sem = _moving_average(sem, smooth_window)
            has_finite_estimates |= bool(np.any(np.isfinite(mean)))
            floor = 1e-14
            if log_y:
                positive = mean[np.isfinite(mean) & (mean > 0)]
                floor = max(1e-14, float(positive.min()) * 1e-3) if len(positive) else 1e-14
                mean = np.clip(mean, floor, None)
            style = _STYLES[method]
            label = method
            linestyle = "-"
            if theta_colors is not None:
                base = np.asarray(theta_colors[index])
                if method.startswith("SI-CS"):
                    color = base
                else:
                    color = 0.64 * base + 0.36 if method.startswith("ConEnh") else 0.68 * base
                style = {"color": color, "marker": style["marker"]}
                if "AGHDO" in method:
                    linestyle = "--"
                    style["marker"] = "D" if method.startswith("ConEnh") else "^"
                label = rf"{method} (${_rotation_label(result.config)}$)"
            if show_band and len(method_errors) > 1:
                lower_x, lower_y = _interpolate_curve(
                    result.shots, np.clip(mean - sem, floor, None), log_x=log_x, log_y=log_y
                )
                _, upper_y = _interpolate_curve(result.shots, mean + sem, log_x=log_x, log_y=log_y)
                axis.fill_between(
                    lower_x, lower_y, upper_y, color=style["color"], alpha=0.16, linewidth=0
                )
            curve_x, curve_y = _interpolate_curve(result.shots, mean, log_x=log_x, log_y=log_y)
            axis.plot(
                curve_x, curve_y, color=style["color"], linestyle=linestyle, label=label, zorder=3
            )
            axis.plot(
                result.shots,
                mean,
                linestyle="none",
                marker=style["marker"],
                markersize=7.0,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=1.0,
                color=style["color"],
                zorder=4,
            )

        result = results[0]
        property_label = (
            r"local purity $\mathrm{Tr}\,\rho_A^2$"
            if result.config.property == "PURITY"
            else r"stabilizer 2-Renyi entropy $\mathcal{M}_2$"
        )
        if result.config.property == "RENYI2":
            property_label = r"bipartite 2-Renyi entanglement entropy $\mathcal{E}_2$ (bits)"
        axis.set_ylabel(f"{'Relative' if relative else 'Absolute'} error in {property_label}")
        if result.pilot_cost == 0:
            shot_suffix = ""
        elif result.config.count_pilot_cost:
            shot_suffix = " (pilot included)"
        else:
            shot_suffix = " after pilot"
        axis.set_xlabel(r"Measurement shots $N$" + shot_suffix)
        if not has_finite_estimates:
            axis.set_xlim(min(item.shots[0] for item in results),
                          max(item.shots[-1] for item in results))
            if log_y:
                axis.set_ylim(0.1, 1.0)
            axis.text(
                0.5, 0.5, "No finite estimates available", transform=axis.transAxes,
                ha="center", va="center",
            )
        if log_x:
            axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.minorticks_on()
        axis.grid(which="major", color="0.88", linewidth=0.7)
        axis.grid(which="minor", color="0.95", linewidth=0.5)
        axis.set_axisbelow(True)
        legend = axis.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
            fontsize=10 if theta_colors is not None else _RC["legend.fontsize"],
            frameon=True, fancybox=False, edgecolor="0.3",
        )
        legend.get_frame().set_linewidth(0.8)
        truth_label = _truth_label(result)
        if theta_colors is not None and not all(
            np.isclose(item.truth, result.truth, rtol=1e-10, atol=1e-14) for item in results
        ):
            truth_label = "\n".join(
                rf"${_rotation_label(item.config)}$: {_truth_label(item)}"
                for item in results
            )


        heading = title if title is not None else _configuration_title(result)
        axis.set_title(
            f"{heading}\n{truth_label}", pad=12, linespacing=1.5,
        )

        saved: list[Path] = []
        if output_stem is not None:
            stem = Path(output_stem)
            stem.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".pdf", ".png"):
                path = stem.with_suffix(suffix)
                figure.savefig(path, bbox_inches="tight")
                saved.append(path)
    return figure, axis, tuple(saved)
