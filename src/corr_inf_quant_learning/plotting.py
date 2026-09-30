from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .simulation import BenchmarkResult

_STYLES = {
    "CS": {"color": "#0072B2", "marker": "o"},
    "CorInf": {"color": "#D55E00", "marker": "s"},
    "CS-AGHDO": {"color": "#0072B2", "marker": "o"},
    "CorInf-AGHDO": {"color": "#D55E00", "marker": "s"},
}

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
    smoothed = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        smoothed[index] = np.nanmean(values[start : index + 1])
    return smoothed


def _interpolate_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    log_x: bool,
    log_y: bool,
    points_per_interval: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 2:
        return x, y
    transformed_x = np.log10(x) if log_x else x.astype(float)
    transformed_y = np.log10(y) if log_y else y.astype(float)
    dense_x = np.linspace(
        transformed_x[0], transformed_x[-1], (len(x) - 1) * points_per_interval + 1
    )
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
    repetitions = f"{config.repetitions} repetitions"
    if config.property == "PURITY":
        return rf"Local purity: {common}, $|A|={len(result.subsystem)}$, {repetitions}"
    pilot = "no pilot" if result.pilot_cost == 0 else f"pilot {config.pilot_shots} shots/generator"
    return rf"Stabilizer Rényi entropy $M_2$: {common}, {pilot}, {repetitions}"


def _truth_label(result: BenchmarkResult) -> str:
    if result.config.property == "PURITY":
        return rf"exact $\mathrm{{Tr}}\,\rho_A^2 = {result.truth:.4g}$"
    return rf"exact $M_2 = {result.truth:.4g}$"


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
    errors = result.errors(error_mode)
    relative = error_mode.lower() == "relative"
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        for method in result.config.methods:
            method_errors = errors[method]
            mean = _moving_average(np.nanmean(method_errors, axis=0), smooth_window)
            counts = np.sum(np.isfinite(method_errors), axis=0)
            spread = np.nanstd(method_errors, axis=0, ddof=1) if len(method_errors) > 1 else 0.0
            sem = _moving_average(
                np.where(counts > 1, spread / np.sqrt(counts), 0.0), smooth_window
            )
            floor = 1e-14
            if log_y:
                positive = mean[mean > 0]
                floor = max(1e-14, float(positive.min()) * 1e-3) if len(positive) else 1e-14
                mean = np.clip(mean, floor, None)
            style = _STYLES[method]
            if show_band and len(method_errors) > 1:
                lower_x, lower_y = _interpolate_curve(
                    result.shots, np.clip(mean - sem, floor, None), log_x=log_x, log_y=log_y
                )
                _, upper_y = _interpolate_curve(result.shots, mean + sem, log_x=log_x, log_y=log_y)
                axis.fill_between(
                    lower_x, lower_y, upper_y, color=style["color"], alpha=0.16, linewidth=0
                )
            curve_x, curve_y = _interpolate_curve(result.shots, mean, log_x=log_x, log_y=log_y)
            axis.plot(curve_x, curve_y, color=style["color"], label=method, zorder=3)
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

        property_label = (
            r"local purity $\mathrm{Tr}\,\rho_A^2$"
            if result.config.property == "PURITY"
            else r"stabilizer Rényi entropy $M_2$"
        )
        axis.set_ylabel(f"{'Relative' if relative else 'Absolute'} error in {property_label}")
        if result.pilot_cost == 0:
            shot_suffix = ""
        elif result.config.count_pilot_cost:
            shot_suffix = " (pilot included)"
        else:
            shot_suffix = " after pilot"
        axis.set_xlabel(r"Measurement shots $N$" + shot_suffix)
        if log_x:
            axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_title(title if title is not None else _configuration_title(result), pad=10)
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.minorticks_on()
        axis.grid(which="major", color="0.88", linewidth=0.7)
        axis.grid(which="minor", color="0.95", linewidth=0.5)
        axis.set_axisbelow(True)
        legend = axis.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.3")
        legend.get_frame().set_linewidth(0.8)
        axis.text(
            0.03,
            0.04,
            _truth_label(result),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.3",
                "linewidth": 0.8,
            },
            zorder=5,
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
