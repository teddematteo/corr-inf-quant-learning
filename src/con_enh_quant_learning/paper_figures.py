"""Publication-style grids from saved, reproducible benchmark results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    LogFormatterSciNotation,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
    StrMethodFormatter,
)

from ._compat import dataclass
from .plotting import _error_statistics, _interpolate_curve, _moving_average
from .simulation import BenchmarkResult


@dataclass(slots=True)
class PaperPlotSettings:
    error_mode: str = "relative"
    log_x: bool = True
    log_y: bool = True
    smooth_window: int = 1
    interpolate: bool = True
    show_error_band: bool = True
    show_last_values: bool = True
    share_y: str = "none"
    figsize: tuple[float, float] | None = None
    font_size: float = 16.0
    line_width: float = 2.2
    marker_size: float = 5.2
    colors: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73")
    magic_colors: tuple[str, ...] = ("#E69F00", "#C43C39", "#8C564B")
    formats: tuple[str, ...] = ("pdf", "svg", "png")
    png_dpi: int = 600
    x_min: float | None = 100.0
    rescale_y: bool = True
    y_limits: tuple[float, float] | None = None

    def validate(self) -> None:
        if self.error_mode not in ("relative", "absolute"):
            raise ValueError("plot.error_mode must be 'relative' or 'absolute'")
        for name in (
            "log_x", "log_y", "rescale_y", "interpolate", "show_error_band", "show_last_values",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"plot.{name} must be a boolean")
        if self.x_min is not None and (
            isinstance(self.x_min, bool)
            or not isinstance(self.x_min, (int, float))
            or not np.isfinite(self.x_min)
            or self.x_min < 0
        ):
            raise ValueError("plot.x_min must be a finite nonnegative number or 'none'")
        if self.log_x and self.x_min == 0:
            raise ValueError("plot.x_min must be positive or 'none' when plot.log_x is true")
        if self.y_limits is not None and (
            len(self.y_limits) != 2
            or any(isinstance(v, bool) or not np.isfinite(v) for v in self.y_limits)
            or not 0 <= self.y_limits[0] < self.y_limits[1]
            or (self.log_y and self.y_limits[0] == 0)
        ):
            raise ValueError(
                "plot.y_limits must be increasing nonnegative limits, positive for log y"
            )
        if self.share_y not in ("row", "all", "none"):
            raise ValueError("plot.share_y must be 'row', 'all' or 'none'")
        if not isinstance(self.smooth_window, int) or self.smooth_window < 1:
            raise ValueError("plot.smooth_window must be a positive integer")
        if self.figsize is not None and (
            len(self.figsize) != 2 or any(not np.isfinite(v) or v <= 0 for v in self.figsize)
        ):
            raise ValueError("plot.figsize must contain two positive finite dimensions")
        for name in ("colors", "magic_colors"):
            palette = getattr(self, name)
            if len(palette) < 2 or not all(matplotlib.colors.is_color_like(c) for c in palette):
                raise ValueError(f"plot.{name} must contain at least two valid Matplotlib colors")
        if not self.formats or not set(self.formats) <= {"pdf", "svg", "png"}:
            raise ValueError("plot.formats must use pdf, svg and/or png")
        if len(set(self.formats)) != len(self.formats):
            raise ValueError("plot.formats must not contain duplicates")
        if not isinstance(self.png_dpi, int) or self.png_dpi < 72:
            raise ValueError("plot.png_dpi must be an integer >= 72")
        for name in ("font_size", "line_width", "marker_size"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"plot.{name} must be finite and positive")


@dataclass(slots=True)
class PaperFigure:
    number: int
    title: str
    row_labels: tuple[str, ...]
    results: Sequence[BenchmarkResult]
    smoke: bool = False
    notes: list[str] = field(default_factory=list)


def _curve(x, y, settings):
    if settings.interpolate:
        return _interpolate_curve(x, y, log_x=settings.log_x, log_y=settings.log_y)
    return x, y


def _visible_ordinates(vertices, xlim, settings):
    """Clip rendered segments in axis coordinates, retaining boundary crossings."""
    x, y = np.asarray(vertices, dtype=float).T
    valid = np.isfinite(x) & np.isfinite(y)
    if settings.log_x:
        valid &= x > 0
    if settings.log_y:
        valid &= y > 0
    left, right = xlim
    values = [y[valid & (x >= left) & (x <= right)]]
    for boundary in xlim:
        crossing = valid[:-1] & valid[1:] & (
            ((x[:-1] < boundary) & (x[1:] > boundary))
            | ((x[:-1] > boundary) & (x[1:] < boundary))
        )
        indices = np.flatnonzero(crossing)
        if not len(indices):
            continue
        x0, x1 = x[indices], x[indices + 1]
        y0, y1 = y[indices], y[indices + 1]
        position = boundary
        if settings.log_x:
            x0, x1, position = np.log10(x0), np.log10(x1), np.log10(boundary)
        if settings.log_y:
            y0, y1 = np.log10(y0), np.log10(y1)
        interpolated = y0 + (position - x0) / (x1 - x0) * (y1 - y0)
        values.append(10.0 ** interpolated if settings.log_y else interpolated)
    return np.concatenate(values)


def _rescale_visible_y(axes, settings):
    if settings.share_y == "all":
        groups = [list(axes.flat)]
    elif settings.share_y == "row":
        groups = axes
    else:
        groups = [[axis] for axis in axes.flat]
    for group in groups:
        values = []
        for axis in group:
            vertices = [line.get_xydata() for line in axis.lines]
            vertices.extend(
                path.vertices
                for collection in axis.collections
                if isinstance(collection, PolyCollection)
                for path in collection.get_paths()
            )
            values.extend(
                _visible_ordinates(points, axis.get_xlim(), settings) for points in vertices
            )
        visible = np.concatenate(values) if values else np.empty(0)
        if not len(visible):
            continue
        limits = np.array([visible.min(), visible.max()])
        if settings.log_y:
            limits = np.log10(limits)
        minimum_span = 0.02 if settings.log_y else max(float(np.abs(limits).max()), 1.0) * 0.02
        padding = 0.08 * max(float(limits[1] - limits[0]), minimum_span)
        limits += [-padding, padding]
        limits = 10.0 ** limits if settings.log_y else np.maximum(limits, 0)
        group[0].set_ylim(*limits)


def _format_log_y(axis, rescaled):
    lower, upper = axis.get_ylim()
    ratio = upper / lower
    if rescaled and ratio < 3:
        ticks = MaxNLocator(nbins=5, min_n_ticks=3).tick_values(lower, upper)
        axis.yaxis.set_major_locator(FixedLocator(ticks[(ticks >= lower) & (ticks <= upper)]))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.4g}"))
        axis.yaxis.set_minor_locator(NullLocator())
    else:
        subdivisions = (1, 2, 5) if ratio < (100 if rescaled else 12) else (1,)
        axis.yaxis.set_major_locator(
            LogLocator(base=10, subs=subdivisions, numticks=12 if rescaled else 6)
        )
        options = {"minor_thresholds": (np.inf, np.inf)} if rescaled else {}
        axis.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False, **options))
    axis.yaxis.set_minor_formatter(NullFormatter())


def _label_last_values(figure, endpoints, settings):
    """Place endpoint values in display coordinates, rejecting text collisions."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = renderer.points_to_pixels(2)
    occupied = [
        text.get_window_extent(renderer).padded(gap)
        for text in figure.findobj(match=Text)
        if text.get_visible() and text.get_text().strip()
    ]
    for index, (axis, details, color) in enumerate(endpoints):
        value = details["last_value"]
        if value is None or value < 0:
            continue
        xy = (details["last_checkpoint"], value)
        transform = axis.transData
        if settings.log_y and value == 0:
            xy = (xy[0], 0.018)
            transform = axis.get_xaxis_transform()
        point = transform.transform(xy)
        if not axis.bbox.padded(0.5).contains(*point):
            continue
        for dy, alignment in ((7, "bottom"), (-7, "top")):
            label = axis.annotate(
                f"{value:.3g}", xy=xy, xycoords=transform,
                xytext=(-7, dy), textcoords="offset points",
                ha="right", va=alignment, color=color, annotation_clip=False,
                fontsize=0.85 * settings.font_size, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1},
            )
            box = label.get_window_extent(renderer).padded(gap)
            fits = axis.bbox.contains(box.x0, box.y0) and axis.bbox.contains(box.x1, box.y1)
            if fits and not any(box.overlaps(other) for other in occupied):
                label.set_gid(f"last-value-{index}")
                occupied.append(box)
                details["last_value_label"] = label.get_text()
                break
            label.remove()


def plot_paper_figure(
    specification: PaperFigure,
    settings: PaperPlotSettings,
    output_stem: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray, tuple[Path, ...], dict]:
    """Render benchmark errors, exact property values and a shared method legend."""
    settings.validate()
    results = tuple(specification.results)
    rows = len(specification.row_labels)
    if not rows or not results or len(results) % rows:
        raise ValueError("Results must form a nonempty rectangular grid in row-major order")
    columns = len(results) // rows
    methods = results[0].config.methods
    if (
        len(methods) < 2 or len(set(methods)) != len(methods)
        or any(item.config.methods != methods for item in results)
    ):
        raise ValueError("All panels must compare the same distinct methods (at least two)")
    palette_name = "magic_colors" if specification.number == 3 else "colors"
    colors = getattr(settings, palette_name)
    if len(colors) < len(methods):
        raise ValueError(f"plot.{palette_name} must provide at least one color per selected method")
    if any(item.config.property != results[0].config.property for item in results):
        raise ValueError("All panels must estimate the same property")
    symbol = r"\mathcal{E}_2" if results[0].config.property == "RENYI2" else r"\mathcal{M}_4"
    truth_symbol = rf"{symbol}^\star"
    relative = settings.error_mode == "relative"
    fallback = relative and any(np.any(np.abs(item.repetition_truths) <= 1e-14) for item in results)
    mode = "absolute" if fallback else settings.error_mode
    metadata = {
        "requested_error_mode": settings.error_mode,
        "effective_error_mode": mode,
        "zero_truth_fallback": fallback,
        "panels": [],
    }
    line_styles = ("-", (0, (5, 2.4)), "-.", ":")
    markers = ("o", "s", "^", "D")
    rc = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": settings.font_size,
        "axes.labelsize": settings.font_size,
        "xtick.labelsize": settings.font_size - 1,
        "ytick.labelsize": settings.font_size - 1,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.unicode_minus": True,
    }
    with plt.rc_context(rc):
        figsize = settings.figsize or (max(5.4, 1.1 + 3.6 * columns), 1.0 + 3.05 * rows)
        width, height = figsize
        font_scale = settings.font_size / 11.0
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize,
            squeeze=False,
            sharex=True,
            sharey={"row": "row", "all": True, "none": False}[settings.share_y],
        )
        figure.subplots_adjust(
            left=1.1 * font_scale / width, right=1 - 0.25 / width,
            bottom=0.35 * font_scale / height, top=1 - 0.75 * font_scale / height,
            wspace=0.28 if settings.share_y == "none" else 0.14, hspace=0.27 * font_scale,
        )
        center = (figure.subplotpars.left + figure.subplotpars.right) / 2
        legend_labels = {"SI-CS": "CS", "SI-CS-AGHDO": "CS-AGHDO"}
        handles = [
            Line2D(
                [],
                [],
                color=colors[index],
                lw=settings.line_width,
                ls=line_styles[index % len(line_styles)],
                marker=markers[index % len(markers)],
                markersize=settings.marker_size,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=(
                    legend_labels.get(method, method)
                    if legend_labels.get(method) not in methods else method
                ),
            )
            for index, method in enumerate(methods)
        ]
        has_zeros = False
        endpoints = []
        for index, (axis, result) in enumerate(zip(axes.flat, results)):
            row, column = divmod(index, columns)
            errors = result.errors(mode)
            panel = {
                "state": result.config.state,
                "k": result.config.k_rotations,
                "subsystem_size": result.config.subsystem_size,
                "truth": result.truth,
                "truths": result.repetition_truths.tolist(),
                "methods": {},
            }
            has_positive = False
            for number, method in enumerate(methods):
                values = errors[method]
                finite = np.isfinite(values)
                mean, sem = _error_statistics(values)
                mean = _moving_average(mean, settings.smooth_window)
                sem = _moving_average(sem, settings.smooth_window)
                positive = np.isfinite(mean) & (mean > 0)
                zeros = np.isfinite(mean) & (mean == 0)
                has_positive |= bool(np.any(positive))
                has_zeros |= settings.log_y and bool(np.any(zeros))
                display = np.where(positive, mean, np.nan) if settings.log_y else mean
                color = colors[number]
                curve_x, curve_y = _curve(result.shots, display, settings)
                axis.plot(
                    curve_x,
                    curve_y,
                    color=color,
                    lw=settings.line_width,
                    ls=line_styles[number % len(line_styles)],
                    zorder=3,
                )
                axis.plot(
                    result.shots,
                    display,
                    ls="none",
                    color=color,
                    marker=markers[number % len(markers)],
                    markersize=settings.marker_size,
                    markeredgecolor="white",
                    markeredgewidth=0.75,
                    zorder=4,
                )
                if settings.log_y and np.any(zeros):
                    axis.scatter(
                        result.shots[zeros],
                        np.full(np.count_nonzero(zeros), 0.018),
                        transform=axis.get_xaxis_transform(),
                        marker="v",
                        s=24,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1,
                        zorder=5,
                    )
                enough = finite.sum(0) > 1
                if settings.show_error_band and np.any(enough):
                    lower, upper = mean - sem, mean + sem
                    if settings.log_y:
                        floor = float(mean[positive].min()) * 0.05 if np.any(positive) else 1e-12
                        lower = np.maximum(lower, floor)
                        lower = np.where(positive & enough, lower, np.nan)
                        upper = np.where(positive & enough, upper, np.nan)
                    else:
                        lower = np.where(enough, np.maximum(lower, 0), np.nan)
                        upper = np.where(enough, upper, np.nan)
                    x_lower, y_lower = _curve(result.shots, lower, settings)
                    x_upper, y_upper = _curve(result.shots, upper, settings)


                    if not np.array_equal(x_lower, x_upper, equal_nan=True):
                        x_lower, y_lower, y_upper = result.shots, lower, upper
                    axis.fill_between(
                        x_lower, y_lower, y_upper, color=color, alpha=0.14, linewidth=0, zorder=1
                    )
                panel["methods"][method] = {
                    "finite_repetitions": finite.sum(0).tolist(),
                    "zero_error_checkpoints": int(np.count_nonzero(zeros)),
                    "nonfinite_values": int(np.count_nonzero(~finite)),
                    "last_checkpoint": int(result.shots[-1]),
                    "last_value": float(mean[-1]) if np.isfinite(mean[-1]) else None,
                    "last_value_label": None,
                }
                endpoints.append((axis, panel["methods"][method], color))
            if settings.log_x:
                axis.set_xscale("log")
                axis.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
                axis.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
                axis.xaxis.set_minor_formatter(NullFormatter())
            else:
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                axis.xaxis.set_minor_locator(AutoMinorLocator(2))
                axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 4), useMathText=True)
            if settings.log_y:
                axis.set_yscale("log")
                axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
                axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=12))
                if not has_positive and settings.share_y == "none":
                    axis.set_ylim(1e-3, 1)
            else:
                axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
                axis.yaxis.set_minor_locator(AutoMinorLocator(2))
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(which="both", direction="out", length=3.5, width=0.7)
            axis.tick_params(axis="x", which="both", labelbottom=True)
            axis.tick_params(which="minor", length=2)
            axis.grid(axis="y", which="major", color="#dfe3e8", linewidth=0.65)
            axis.grid(axis="y", which="minor", color="#edf0f3", linewidth=0.4)
            axis.set_axisbelow(True)
            axis.set_xlim(float(result.shots[0]), float(result.shots[-1]))
            axis.text(
                0.035,
                1.035,
                rf"$k={result.config.k_rotations}$",
                transform=axis.transAxes,
                va="bottom",
                fontsize=settings.font_size,
            )
            axis.text(
                0.965,
                1.035,
                rf"${truth_symbol}={result.truth:.4g}$",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=settings.font_size - 1.3,
            )
            if not any(np.isfinite(errors[m]).any() for m in methods):
                axis.text(
                    0.5,
                    0.5,
                    "No finite estimates",
                    transform=axis.transAxes,
                    ha="center",
                    color="0.4",
                )
            if column == 0:
                axis.set_ylabel(
                    specification.row_labels[row],
                    labelpad=8,
                    fontsize=settings.font_size + 1,
                    fontweight="semibold",
                )
            metadata["panels"].append(panel)
        left, right = axes[0, 0].get_xlim()
        metadata["x_min_fallback"] = settings.x_min is not None and settings.x_min >= right
        if settings.x_min is not None and not metadata["x_min_fallback"]:
            left = settings.x_min
            axes[0, 0].set_xlim(left, right)
            ticks = axes[0, 0].get_xticks()
            ticks = np.unique(np.r_[left, ticks[(ticks > left) & (ticks <= right)]])
            axes[0, 0].xaxis.set_major_locator(FixedLocator(ticks))
        if settings.y_limits is not None:
            for axis in axes.flat:
                axis.set_ylim(*settings.y_limits)
        elif settings.rescale_y:
            _rescale_visible_y(axes, settings)
        for axis in axes.flat:
            if settings.log_y:
                _format_log_y(axis, settings.rescale_y)
        metadata["axis_limits"] = [
            {"xlim": list(axis.get_xlim()), "ylim": list(axis.get_ylim())}
            for axis in axes.flat
        ]
        if has_zeros:
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="v",
                    markerfacecolor="none",
                    color="0.4",
                    markersize=5,
                    label="Zero error (bottom edge)",
                )
            )
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(center, 1 - 0.08 * font_scale / height),
            ncol=len(handles),
            frameon=False,
            fontsize=settings.font_size,
            handlelength=3.1,
            columnspacing=2.4,
            handletextpad=0.75,
        )
        if settings.show_last_values:
            _label_last_values(figure, endpoints, settings)
        saved = []
        if output_stem is not None:
            stem = Path(output_stem)
            stem.parent.mkdir(parents=True, exist_ok=True)
            for extension in settings.formats:
                path = stem.with_suffix("." + extension)
                figure.savefig(
                    path,
                    dpi=settings.png_dpi if extension == "png" else None,
                    bbox_inches="tight",
                    pad_inches=0.06,
                )
                saved.append(path)
    return figure, axes, tuple(saved), metadata
