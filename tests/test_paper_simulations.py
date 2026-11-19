import json
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from con_enh_quant_learning.paper_figures import PaperFigure, PaperPlotSettings, plot_paper_figure
from con_enh_quant_learning.paper_simulations import (
    _SCENARIO_FIELDS,
    load_paper_suite,
    run_paper_suite,
)
from con_enh_quant_learning.simulation import BenchmarkResult, SimulationConfig, run_benchmark
from con_enh_quant_learning.simulation_cache import (
    fingerprint,
    load_realization,
    save_realization,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "simulations" / "config.txt"


def small_config(path, *, figures="[1, 2, 3]", si=False, k_values=(1, 2, 3), sizes=(1, 2)):
    methods = (
        'direct_methods = ["CS", "SI-CS", "ConEnh"]\n'
        'magic_methods = ["CS-AGHDO", "SI-CS-AGHDO", "ConEnh-AGHDO"]\n'
        if si else ""
    )
    path.write_text(f"""[suite]
n_qubits = 4
k_values = {list(k_values)}
subsystem_sizes = {list(sizes)}
figures = {figures}
output_dir = "outputs"
resume = true
{methods}
[simulation]
device = "cpu"
post_shots = 8
pilot_shots = 0
repetitions = 1
evaluation_points = 3
property_clipping = true
pauli_samples = 16
conenh_max_cliques = 16
conenh_candidates = 8
aghdo_fit_steps = 1
aghdo_rank = 2
aghdo_feature_densities = [2, 2]
[plot]
formats = ["png"]
png_dpi = 72
""")


def figure_results(*, truth=1.0, repetitions=2, k_values=(1, 2, 3)):
    results = []
    for size in (1, 2):
        for k in k_values:
            config = SimulationConfig(
                n_qubits=4,
                k_rotations=k,
                subsystem_size=size,
                property="RENYI2",
                methods=("CS", "ConEnh"),
                repetitions=repetitions,
                post_shots=16,
            )


            estimates = {
                "CS": np.array([[truth - 1] * 4, [truth + 1] * 4], dtype=float)[:repetitions],
                "ConEnh": np.array([[truth - 0.2] * 4, [truth + 0.2] * 4])[:repetitions],
            }
            results.append(
                BenchmarkResult(
                    config, np.array([2, 4, 8, 16]), truth, estimates, tuple(range(size)), 0
                )
            )
    return results


class SuiteConfigurationTests(unittest.TestCase):
    def test_arbitrary_nonempty_row_and_column_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            for k_values, sizes in (((1,), (1,)), ((1, 2), (1, 2, 3)),
                                    ((0, 1, 2, 3, 4), (4, 2, 1))):
                with self.subTest(k_values=k_values, sizes=sizes):
                    small_config(path, k_values=k_values, sizes=sizes)
                    suite = load_paper_suite(path)
                    for number in (1, 2, 3):
                        jobs = [job for job in suite.jobs if job.figure == number]
                        rows = 2 if number == 3 else len(sizes)
                        self.assertEqual(len(jobs), rows * len(k_values))
                        self.assertEqual([job.config.k_rotations for job in jobs],
                                         list(k_values) * rows)
                        self.assertEqual([job.row for job in jobs],
                                         [row for row in range(rows) for _ in k_values])

    def test_requested_panels_and_all_hyperparameters(self):
        suite = load_paper_suite(CONFIG)
        layout = suite.configuration["suite"]
        simulation = suite.configuration["simulation"]
        column_counts = {
            number: len(layout.get("magic_k_values", layout["k_values"]))
            if number == 3 else len(layout["k_values"])
            for number in layout["figures"]
        }
        row_counts = {
            number: 2 if number == 3 else len(layout["subsystem_sizes"])
            for number in layout["figures"]
        }
        self.assertEqual(len(suite.jobs), sum(
            column_counts[number] * row_counts[number] for number in layout["figures"]
        ))
        self.assertEqual(suite.output_dir, CONFIG.parent / "output")
        self.assertEqual(suite.plot.error_mode, "relative")
        self.assertEqual(
            sum(job.work for job in suite.jobs),
            sum(
                column_counts[number] * row_counts[number]
                * len(layout["magic_methods" if number == 3 else "direct_methods"])
                for number in layout["figures"]
            )
            * simulation["post_shots"] * simulation["repetitions"],
        )
        self.assertEqual(
            set(suite.configuration["simulation"]),
            {f.name for f in fields(SimulationConfig)} - _SCENARIO_FIELDS,
        )
        for number in layout["figures"]:
            columns = column_counts[number]
            k_values = (layout.get("magic_k_values", layout["k_values"])
                        if number == 3 else layout["k_values"])
            jobs = [job for job in suite.jobs if job.figure == number]
            self.assertEqual([job.config.k_rotations for job in jobs],
                             k_values * row_counts[number])
            self.assertTrue(all(job.config.n_qubits == layout["n_qubits"] for job in jobs))
            angles = simulation["theta_range"]
            expected_angles = None if angles == "none" else tuple(angles)
            self.assertTrue(all(job.config.theta_range == expected_angles for job in jobs))
            if number == 3:
                self.assertEqual(
                    [job.config.state for job in jobs], ["CLUSTER"] * columns + ["GHZ"] * columns
                )
                self.assertTrue(
                    all(job.config.methods == tuple(layout["magic_methods"])
                        for job in jobs)
                )
            else:
                self.assertEqual(
                    [job.config.subsystem_size for job in jobs],
                    [size for size in layout["subsystem_sizes"] for _ in range(columns)],
                )
                self.assertTrue(all(job.config.property == "RENYI2" for job in jobs))
                self.assertTrue(all(job.config.methods == tuple(layout["direct_methods"])
                                    for job in jobs))

    def test_four_columns_preserve_configured_order_and_two_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, k_values=(4, 1, 2, 3))
            suite = load_paper_suite(path)
            self.assertEqual(len(suite.jobs), 24)
            for number in (1, 2, 3):
                jobs = [job for job in suite.jobs if job.figure == number]
                self.assertEqual([job.column for job in jobs], list(range(4)) * 2)
                self.assertEqual([job.row for job in jobs], [0] * 4 + [1] * 4)
                self.assertEqual([job.config.k_rotations for job in jobs], [4, 1, 2, 3] * 2)

    def test_magic_columns_are_independent_and_invalid_lists_fail_before_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, k_values=(1, 2))
            original = path.read_text()
            path.write_text(original.replace("[simulation]",
                                             "magic_k_values = [3, 0]\n[simulation]"))
            suite = load_paper_suite(path)
            for number in (1, 2, 3):
                jobs = [job for job in suite.jobs if job.figure == number]
                expected = [3, 0] if number == 3 else [1, 2]
                self.assertEqual([job.config.k_rotations for job in jobs], expected * 2)
            path.write_text(path.read_text().replace("k_values = [1, 2]", "k_values = [4]"))
            suite = load_paper_suite(path)
            self.assertEqual([job.config.k_rotations for job in suite.jobs if job.figure == 3],
                             [3, 0, 3, 0])
            for invalid in ("[]", "[1, 1]", "[true]", "[-1]", "[1.5]", "[5]", '"none"'):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    path.write_text(original.replace("[simulation]",
                                                     f"magic_k_values = {invalid}\n[simulation]"))
                    load_paper_suite(path)

    def test_plot_limits_are_configurable_without_changing_simulation_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path)
            original = load_paper_suite(path)
            path.write_text(path.read_text() + '\nx_min = "none"\nrescale_y = false\n')
            updated = load_paper_suite(path)
            self.assertIsNone(updated.plot.x_min)
            self.assertFalse(updated.plot.rescale_y)
            self.assertEqual(
                [fingerprint(job.config, job.config.methods[0], 0) for job in original.jobs],
                [fingerprint(job.config, job.config.methods[0], 0) for job in updated.jobs],
            )
        for invalid in (0, -1, np.inf, np.nan, True, "1000"):
            with self.subTest(x_min=invalid), self.assertRaisesRegex(ValueError, "x_min"):
                PaperPlotSettings(x_min=invalid).validate()
        with self.assertRaisesRegex(ValueError, "rescale_y"):
            PaperPlotSettings(rescale_y="true").validate()

    def test_configurable_si_pairs_preserve_order_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path)
            legacy = load_paper_suite(path)
            self.assertEqual(legacy.jobs[0].config.methods, ("CS", "ConEnh"))
            self.assertEqual(legacy.jobs[-1].config.methods, ("CS-AGHDO", "ConEnh-AGHDO"))
            path.write_text(path.read_text().replace(
                "[simulation]",
                'direct_methods = ["SI-CS", "ConEnh"]\n'
                'magic_methods = ["ConEnh-AGHDO", "SI-CS-AGHDO"]\n[simulation]',
            ))
            suite = load_paper_suite(path)
            self.assertEqual(suite.jobs[0].config.methods, ("SI-CS", "ConEnh"))
            self.assertEqual(suite.jobs[-1].config.methods, ("ConEnh-AGHDO", "SI-CS-AGHDO"))
            self.assertEqual(sum(job.work for job in suite.jobs), 18 * 2 * 8)
            self.assertEqual(fingerprint(legacy.jobs[0].config, "ConEnh", 0),
                             fingerprint(suite.jobs[0].config, "ConEnh", 0))

    def test_invalid_methods_and_insufficient_colors_fail_before_acquisition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path)
            original = path.read_text()
            for option in (
                'direct_methods = ["SI-CS"]',
                'direct_methods = ["SI-CS", "SI-CS"]',
                'direct_methods = ["SI-CS", "unknown"]',
                'direct_methods = ["SI-CS-AGHDO", "ConEnh"]',
                'direct_methods = "SI-CS"',
                'direct_methods = ["SI-CS", ["CS"]]',
                'magic_methods = ["SI-CS", "ConEnh-AGHDO"]',
                'magic_methods = ["SI-CS-AGHDO", "SI-CS-AGHDO"]',
            ):
                path.write_text(original.replace("[simulation]", option + "\n[simulation]"))
                with self.subTest(option=option), self.assertRaisesRegex(ValueError, "methods"):
                    load_paper_suite(path)
            small_config(path, si=True)
            path.write_text(path.read_text() + '\ncolors = ["blue", "orange"]\n')
            with self.assertRaisesRegex(ValueError, "one color per selected method"):
                load_paper_suite(path)

    def test_smoke_keeps_register_and_layout_but_separates_budget_and_output(self):
        production = load_paper_suite(CONFIG)
        smoke = load_paper_suite(CONFIG, smoke=True)
        self.assertEqual(len(smoke.jobs), len(production.jobs))
        self.assertNotEqual(production.output_dir, smoke.output_dir)
        for original, short in zip(production.jobs, smoke.jobs):
            self.assertEqual(original.key, short.key)
            self.assertEqual(short.config.n_qubits, 100)
            self.assertEqual(short.config.post_shots, 32)
            self.assertEqual(
                original.config.post_shots, production.configuration["simulation"]["post_shots"]
            )

    def test_unknown_options_and_wrong_panel_grids_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path)
            original = path.read_text()
            for modified in (
                original.replace('device = "cpu"', 'device_typo = "cpu"'),
                original.replace("k_values = [1, 2, 3]", "k_values = []"),
                original.replace("subsystem_sizes = [1, 2]", "subsystem_sizes = []"),
                original.replace("subsystem_sizes = [1, 2]", "subsystem_sizes = [2, 2]"),
                original.replace("k_values = [1, 2, 3]", "k_values = [1, 1, 3]"),
                original.replace("k_values = [1, 2, 3]", "k_values = [0, 1, 2, 2]"),
                original.replace("k_values = [1, 2, 3]", "k_values = [true, 1, 2, 3]"),
                original.replace("k_values = [1, 2, 3]", "k_values = [-1, 1, 2, 3]"),
                original.replace("n_qubits = 4", "n_qubits = 2"),
                original.replace("error_mode", "error_typo") + "\ninvalid_plot_key=4",
            ):
                path.write_text(modified)
                with self.subTest(modified=modified), self.assertRaises(ValueError):
                    load_paper_suite(path)

    def test_explicit_selections_are_routed_by_subsystem_size_and_rotation_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, figures="[1]")
            path.write_text(
                path.read_text()
                + """
[selections]
subsystems = { "1" = [0], "2" = [0, 1] }
rotation_vertices = { "1" = [3], "2" = [2, 3], "3" = [1, 2, 3] }
"""
            )
            suite = load_paper_suite(path)
            self.assertEqual(suite.jobs[0].config.subsystem, (0,))
            self.assertEqual(suite.jobs[3].config.subsystem, (0, 1))
            self.assertEqual(suite.jobs[1].config.rotation_vertices, (2, 3))


class PaperFigureTests(unittest.TestCase):
    def test_arbitrary_grids_and_automatic_figure_sizes(self):
        for rows, columns in ((1, 1), (1, 5), (3, 1), (3, 5)):
            with self.subTest(rows=rows, columns=columns):
                sample = figure_results(repetitions=1)[0]
                results = [replace(sample, config=replace(
                    sample.config, k_rotations=column, subsystem_size=row + 1,
                )) for row in range(rows) for column in range(columns)]
                spec = PaperFigure(1, "Arbitrary grid", tuple(map(str, range(rows))), results)
                figure, axes, _, metadata = plot_paper_figure(spec, PaperPlotSettings())
                figure.canvas.draw()
                self.assertEqual(axes.shape, (rows, columns))
                self.assertEqual(len(metadata["panels"]), rows * columns)
                self.assertGreaterEqual(figure.get_figwidth(), 3.6 * columns)
                self.assertGreaterEqual(figure.get_figheight(), 3.0 * rows)
                plt.close(figure)

    def test_relative_fallback_checks_every_realization_and_errors_use_own_truth(self):
        results = figure_results(truth=2)
        for result in results:
            result.truths = np.array([1.0, 3.0])
            result.estimates["CS"] = np.array([[2.0] * 4, [6.0] * 4])
        spec = PaperFigure(3, "Realization truths", ("A", "B"), results)
        figure, axes, _, metadata = plot_paper_figure(
            spec, PaperPlotSettings(log_y=False, interpolate=False),
        )
        np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(), 1.0)
        self.assertEqual(metadata["panels"][0]["truths"], [1.0, 3.0])
        self.assertTrue(any("star" in text.get_text() for text in axes[0, 0].texts))
        self.assertFalse(any("langle" in text.get_text() for text in axes[0, 0].texts))
        plt.close(figure)
        results[0].truths = np.array([0.0, 4.0])
        _, _, _, metadata = plot_paper_figure(spec, PaperPlotSettings())
        self.assertTrue(metadata["zero_truth_fallback"])

    def tearDown(self):
        plt.close("all")

    def test_configured_log_x_starts_at_100_and_labels_both_rows_without_changing_data(self):
        results = figure_results()
        for result in results:
            result.shots = np.array([2, 10, 100, 1000])
            result.config = replace(result.config, post_shots=1000)
        before = [
            (
                result.shots.copy(),
                {name: values.copy() for name, values in result.estimates.items()},
            )
            for result in results
        ]
        spec = PaperFigure(1, "Logarithmic shots", ("A", "B"), results)
        figure, axes, _, metadata = plot_paper_figure(spec, load_paper_suite(CONFIG).plot)
        figure.canvas.draw()
        for axis, result, (shots, estimates) in zip(axes.flat, results, before):
            self.assertEqual(axis.get_xlim(), (100, 1000))
            self.assertEqual(axis.get_xscale(), "log")
            np.testing.assert_array_equal(axis.get_xticks(), [100, 1000])
            labels = axis.get_xticklabels()
            self.assertEqual(len(labels), 2)
            self.assertTrue(all(label.get_visible() and label.get_text() for label in labels))
            positions = axis.xaxis.get_transform().transform(np.array([1, 10, 100, 1000]))
            np.testing.assert_allclose(np.diff(positions), 1)
            for index, (name, values) in enumerate(estimates.items()):
                np.testing.assert_array_equal(axis.lines[2 * index + 1].get_xdata(), shots)
                np.testing.assert_allclose(
                    axis.lines[2 * index + 1].get_ydata(), result.errors("relative")[name].mean(0)
                )
                self.assertGreaterEqual(np.nanmin(axis.lines[2 * index].get_xdata()), shots[0])
                np.testing.assert_array_equal(result.estimates[name], values)
            np.testing.assert_array_equal(result.shots, shots)
        self.assertTrue(all(panel["xlim"] == [100, 1000] for panel in metadata["axis_limits"]))

    def test_minimal_labels_legend_six_truths_and_mean_per_repetition_errors(self):
        settings = PaperPlotSettings(log_y=False, interpolate=False)
        spec = PaperFigure(1, "One common title", ("$|A|=1$", "$|A|=2$"), figure_results())
        figure, axes, _, metadata = plot_paper_figure(spec, settings)
        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(len(figure.legends), 1)
        self.assertIsNone(figure._suptitle)
        self.assertIsNone(figure._supxlabel)
        self.assertIsNone(figure._supylabel)
        self.assertEqual(len(figure.texts), 0)
        for axis in axes.flat:
            self.assertIsNone(axis.get_legend())
            self.assertEqual(axis.get_title(), "")
            self.assertTrue(any(text.get_text().startswith("$k=") for text in axis.texts))
            self.assertTrue(
                any(
                    r"\mathcal{E}_2" in text.get_text() and "star" in text.get_text()
                    for text in axis.texts
                )
            )
            np.testing.assert_allclose(axis.lines[0].get_ydata(), 1)
            np.testing.assert_allclose(axis.lines[2].get_ydata(), 0.2)
        self.assertEqual(metadata["effective_error_mode"], "relative")

    def test_three_method_figure_includes_si_curves_and_distinct_styles(self):
        results = figure_results(repetitions=1)
        for result in results:
            result.config = replace(result.config, methods=("CS", "SI-CS", "ConEnh"))
            result.estimates["SI-CS"] = np.full_like(result.estimates["CS"], result.truth + 0.4)
        settings = PaperPlotSettings(
            log_y=False, interpolate=False, colors=("blue", "green", "orange"),
        )
        spec = PaperFigure(1, "SI comparison", ("A", "B"), results)
        figure, axes, _, metadata = plot_paper_figure(spec, settings)
        self.assertEqual([t.get_text() for t in figure.legends[0].get_texts()],
                         ["CS", "SI-CS", "ConEnh"])
        for axis in axes.flat:
            np.testing.assert_allclose(axis.lines[2].get_ydata(), 0.4)
            self.assertEqual([axis.lines[i].get_color() for i in (0, 2, 4)],
                             ["blue", "green", "orange"])
            self.assertEqual(len({axis.lines[i].get_linestyle() for i in (0, 2, 4)}), 3)
            self.assertEqual(len({axis.lines[i].get_marker() for i in (1, 3, 5)}), 3)
        self.assertIn("SI-CS", metadata["panels"][0]["methods"])
        with self.assertRaisesRegex(ValueError, "one color per selected method"):
            plot_paper_figure(spec, replace(settings, colors=("blue", "orange")))

    def test_coincident_final_values_omit_collisions_at_different_fonts_and_export_dpi(self):
        for font_size, dpi in ((11, 72), (20, 144)):
            with self.subTest(font_size=font_size, dpi=dpi):
                results = figure_results(repetitions=1, k_values=(1,))
                for result in results:
                    result.config = replace(result.config, methods=("CS", "SI-CS", "ConEnh"))
                    result.estimates = {
                        method: np.full((1, 4), result.truth + 1)
                        for method in result.config.methods
                    }
                spec = PaperFigure(1, "Coincident endpoints", ("A", "B"), results)
                figure, axes, _, metadata = plot_paper_figure(
                    spec, PaperPlotSettings(
                        log_y=False, interpolate=False, y_limits=(0, 2),
                        font_size=font_size, figsize=(5.4, 5.4),
                    ),
                )
                figure.set_dpi(dpi)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                for axis, panel in zip(axes.flat, metadata["panels"]):
                    labels = [text for text in axis.texts
                              if (text.get_gid() or "").startswith("last-value-")]
                    self.assertEqual(len(labels), 2)
                    self.assertEqual(sum(item["last_value_label"] is not None
                                         for item in panel["methods"].values()), 2)
                    other_texts = [*axis.texts, *axis.get_xticklabels(), *axis.get_yticklabels()]
                    for label in labels:
                        self.assertEqual(label.get_text(), "1")
                        box = label.get_window_extent(renderer)
                        self.assertTrue(axis.bbox.contains(box.x0, box.y0))
                        self.assertTrue(axis.bbox.contains(box.x1, box.y1))
                        for text in other_texts:
                            if text is not label and text.get_visible() and text.get_text():
                                self.assertFalse(box.overlaps(text.get_window_extent(renderer)))
                    for line in axis.lines[1::2]:
                        np.testing.assert_array_equal(line.get_ydata(), np.ones(4))
                plt.close(figure)

    def test_final_value_labels_skip_missing_checkpoint_and_anchor_log_zero_to_marker(self):
        results = figure_results(repetitions=1, k_values=(1,))
        for result in results:
            result.estimates["CS"][:, -1] = np.nan
            result.estimates["ConEnh"][:, -1] = result.truth
        figure, axes, _, metadata = plot_paper_figure(
            PaperFigure(1, "Final checkpoints", ("A", "B"), results),
            PaperPlotSettings(interpolate=False, y_limits=(0.01, 2)),
        )
        for axis, panel in zip(axes.flat, metadata["panels"]):
            labels = [text for text in axis.texts
                      if (text.get_gid() or "").startswith("last-value-")]
            self.assertEqual([label.get_text() for label in labels], ["0"])
            self.assertIsNone(panel["methods"]["CS"]["last_value"])
            self.assertIsNone(panel["methods"]["CS"]["last_value_label"])
            self.assertEqual(panel["methods"]["ConEnh"]["last_value"], 0)
            self.assertEqual(panel["methods"]["ConEnh"]["last_value_label"], "0")
            self.assertTrue(np.isnan(axis.lines[2].get_ydata()[-1]))
        plt.close(figure)

    def test_four_columns_draw_all_panels_with_correct_row_labels(self):
        results = figure_results(k_values=(4, 1, 2, 3))
        spec = PaperFigure(1, "Four columns", ("Upper row", "Lower row"), results)
        figure, axes, _, metadata = plot_paper_figure(
            spec, PaperPlotSettings(log_y=False, interpolate=False)
        )
        figure.canvas.draw()
        self.assertEqual(axes.shape, (2, 4))
        self.assertEqual([panel["k"] for panel in metadata["panels"]], [4, 1, 2, 3] * 2)
        self.assertEqual(len(figure.legends), 1)
        for row, label in enumerate(spec.row_labels):
            for column in range(4):
                self.assertEqual(axes[row, column].get_ylabel(), label if column == 0 else "")

    def test_zero_errors_on_log_scale_are_marked_without_fabricating_positive_values(self):
        results = figure_results(repetitions=1)
        for result in results:
            result.estimates["CS"][:] = result.truth
        spec = PaperFigure(1, "Zero handling", ("A", "B"), results)
        figure, axes, _, metadata = plot_paper_figure(spec, PaperPlotSettings(interpolate=False))
        for axis in axes.flat:
            self.assertTrue(np.all(np.isnan(axis.lines[0].get_ydata())))
        self.assertIn("Zero error", figure.legends[0].get_texts()[-1].get_text())
        self.assertEqual(metadata["panels"][0]["methods"]["CS"]["zero_error_checkpoints"], 4)
        self.assertEqual(results[0].estimates["CS"][0, 0], results[0].truth)

    def test_zero_truth_falls_back_for_entire_figure_and_nonfinite_data_remain_gaps(self):
        results = figure_results(truth=0)
        results[0].estimates["CS"][:, 1] = np.nan
        spec = PaperFigure(1, "Fallback", ("A", "B"), results)
        _, axes, _, metadata = plot_paper_figure(
            spec, PaperPlotSettings(log_y=False, interpolate=False)
        )
        self.assertTrue(metadata["zero_truth_fallback"])
        self.assertEqual(metadata["effective_error_mode"], "absolute")
        self.assertTrue(np.isnan(axes[0, 0].lines[0].get_ydata()[1]))
        self.assertEqual(metadata["panels"][0]["methods"]["CS"]["nonfinite_values"], 2)

    def test_rescaling_excludes_hidden_outliers_and_includes_sem_with_shared_limits(self):
        for log_y in (False, True):
            for share_y in ("row", "all", "none"):
                with self.subTest(log_y=log_y, share_y=share_y):
                    results = figure_results()
                    for index, result in enumerate(results, start=1):
                        result.shots = np.array([2, 500, 2000, 10000])
                        result.config = replace(result.config, post_shots=10000)
                        for method, error in (("CS", 1.0), ("ConEnh", 0.1)):
                            errors = np.array([[1000, 1, 1, 1], [1000, 3, 3, 3]], float)
                            errors[:, 1:] *= index * error
                            result.estimates[method] = result.truth + errors
                    saved = results[0].estimates["CS"].copy()
                    spec = PaperFigure(1, "Visible range", ("A", "B"), results)
                    figure, axes, _, metadata = plot_paper_figure(
                        spec, PaperPlotSettings(
                            x_min=1000, log_y=log_y, share_y=share_y, interpolate=False
                        )
                    )
                    for index, axis in enumerate(axes.flat, start=1):
                        self.assertEqual(axis.get_xlim(), (1000, 10000))
                        low, high = axis.get_ylim()
                        self.assertLessEqual(low, 0.1 * index)
                        self.assertGreaterEqual(high, 3 * index)
                        self.assertLess(high, 100)
                        self.assertEqual(axis.lines[0].get_ydata()[0], 1000)
                    limits = [axis.get_ylim() for axis in axes.flat]
                    if share_y == "all":
                        self.assertEqual(len(set(limits)), 1)
                    elif share_y == "row":
                        self.assertEqual(len(set(limits[:3])), 1)
                        self.assertEqual(len(set(limits[3:])), 1)
                        self.assertNotEqual(limits[0], limits[3])
                    else:
                        self.assertEqual(len(set(limits)), 6)
                    self.assertFalse(metadata["x_min_fallback"])
                    np.testing.assert_array_equal(results[0].estimates["CS"], saved)
                    plt.close(figure)

    def test_rescaling_includes_curve_crossing_at_x_min_in_linear_and_log_axes(self):
        for log_x in (False, True):
            for log_y in (False, True):
                with self.subTest(log_x=log_x, log_y=log_y):
                    results = figure_results(repetitions=1)
                    for result in results:
                        result.shots = np.array([500, 1500, 5000, 10000])
                        result.config = replace(result.config, post_shots=10000)
                        result.estimates["CS"] = result.truth + np.array([[100, 1, 1, 1]])
                        result.estimates["ConEnh"] = np.full((1, 4), result.truth + 0.5)
                    spec = PaperFigure(1, "Boundary", ("A", "B"), results)
                    figure, axes, _, _ = plot_paper_figure(
                        spec, PaperPlotSettings(
                            x_min=1000, log_x=log_x, log_y=log_y, interpolate=False
                        )
                    )
                    fraction = np.log(2) / np.log(3) if log_x else 0.5
                    boundary = 100 ** (1 - fraction) if log_y else 100 - 99 * fraction
                    for axis in axes.flat:
                        self.assertLessEqual(axis.get_ylim()[0], 0.5)
                        self.assertGreaterEqual(axis.get_ylim()[1], boundary)
                        self.assertLess(axis.get_ylim()[1], 100)
                    plt.close(figure)

    def test_short_runs_keep_all_checkpoints_and_full_range_can_be_requested(self):
        spec = PaperFigure(1, "Short run", ("A", "B"), figure_results())
        for x_min, fallback in ((1000, True), (None, False)):
            figure, axes, _, metadata = plot_paper_figure(
                spec, PaperPlotSettings(x_min=x_min, rescale_y=False)
            )
            self.assertEqual(metadata["x_min_fallback"], fallback)
            self.assertTrue(all(axis.get_xlim() == (2, 16) for axis in axes.flat))
            plt.close(figure)


class SuiteExecutionTests(unittest.TestCase):
    def test_separate_magic_columns_reuse_existing_data_and_acquire_only_missing_magic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, figures="[1, 3]", k_values=(1,), sizes=(1,))
            initial = load_paper_suite(path)
            first = run_paper_suite(initial, show_progress=False)
            before = {name: (initial.output_dir / name).read_bytes()
                      for name in first["data_files"]}
            path.write_text(path.read_text().replace("[simulation]",
                                                     "magic_k_values = [2, 1]\n[simulation]"))
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                expanded = run_paper_suite(load_paper_suite(path), show_progress=False)
            self.assertEqual(expanded["reused_entry_count"], 6)
            self.assertEqual(expanded["computed_entry_count"], 4)
            self.assertEqual(benchmark.call_count, 4)
            self.assertTrue(all(call.args[0].property == "MAGIC"
                                and call.args[0].k_rotations == 2
                                for call in benchmark.call_args_list))
            path.write_text(path.read_text().replace("magic_k_values = [2, 1]",
                                                     "magic_k_values = [1, 2]"))
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       side_effect=AssertionError("reordering must reuse all results")):
                reordered = run_paper_suite(load_paper_suite(path), plot_only=True,
                                            show_progress=False)
            self.assertEqual(reordered["reused_entry_count"], 10)
            self.assertEqual(reordered["computed_entry_count"], 0)
            self.assertEqual([panel["k"] for panel in reordered["figures"][1]["panels"]],
                             [1, 2, 1, 2])
            for name, contents in before.items():
                self.assertEqual((initial.output_dir / name).read_bytes(), contents)

    def test_real_suite_resumes_and_plot_only_never_simulates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, si=True, k_values=(1,))
            suite = load_paper_suite(path)
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                manifest = run_paper_suite(suite, show_progress=False)
            self.assertEqual(benchmark.call_count, 18)
            self.assertEqual(manifest["panel_count"], 6)
            self.assertEqual(manifest["computed_entry_count"], 18)
            self.assertEqual(manifest["computed_post_measurement_shots"], 18 * 8)
            self.assertEqual(len(manifest["data_files"]), 18)
            self.assertEqual(len(manifest["figures"]), 3)
            before = {name: (suite.output_dir / name).read_bytes()
                      for name in manifest["data_files"]}
            for filename in manifest["data_files"]:
                saved = json.loads(before[filename])
                self.assertEqual(len(saved["config"]["methods"]), 1)
                self.assertIn(saved["method"], saved["config"]["methods"])
            for item in manifest["figures"]:
                self.assertTrue((suite.output_dir / item["files"][0]).is_file())


            path.write_text(path.read_text().replace('output_dir = "outputs"',
                                                       'output_dir = "other_output"')
                            + "\nx_min = 4\nrescale_y = true\n")
            updated = load_paper_suite(path)
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       side_effect=AssertionError("saved results must be reused")):
                for plot_only in (True, False):
                    resumed = run_paper_suite(updated, plot_only=plot_only, show_progress=False)
                    self.assertEqual(resumed["computed_entry_count"], 0)
                    self.assertEqual(resumed["reused_entry_count"], 18)
                    self.assertTrue(all(axis["xlim"][0] == 4
                                        for figure in resumed["figures"]
                                        for axis in figure["axis_limits"]))
            for name, contents in before.items():
                self.assertEqual((suite.output_dir / name).read_bytes(), contents)

    def test_add_reorder_rows_columns_methods_and_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, figures="[1]", k_values=(1,), sizes=(1,))
            initial = load_paper_suite(path)
            initial.jobs[0].config.theta_range = (0.1, 0.4)
            first = run_paper_suite(initial, show_progress=False)
            before = {
                name: (initial.output_dir / name).read_bytes() for name in first["data_files"]
            }
            small_config(path, figures="[1]", si=True, k_values=(2, 1), sizes=(2, 1, 3))
            path.write_text(path.read_text().replace('repetitions = 1', 'repetitions = 3')
                            .replace('[simulation]', '[simulation]\ntheta_range = [0.1, 0.4]'))
            expanded = load_paper_suite(path)
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                manifest = run_paper_suite(expanded, show_progress=False)
            self.assertEqual(manifest["computed_entry_count"], 6 * 3 * 3 - 2)
            self.assertEqual(manifest["reused_entry_count"], 2)
            self.assertEqual(benchmark.call_count, 52)
            self.assertTrue(set(first["data_files"]) <= set(manifest["data_files"]))
            for call in benchmark.call_args_list:
                cfg = call.args[0]
                if cfg.k_rotations == cfg.subsystem_size == 1:
                    self.assertTrue(call.kwargs["repetition_start"] > 0
                                    or cfg.methods == ("SI-CS",))
            path.write_text(path.read_text().replace('k_values = [2, 1]', 'k_values = [1, 2]')
                            .replace('subsystem_sizes = [2, 1, 3]', 'subsystem_sizes = [3, 1, 2]')
                            .replace('["CS", "SI-CS", "ConEnh"]',
                                     '["ConEnh", "CS", "SI-CS"]'))
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       side_effect=AssertionError("layout must not invalidate data")):
                rerender = run_paper_suite(load_paper_suite(path), show_progress=False)
            self.assertEqual(rerender["computed_entry_count"], 0)
            for name, contents in before.items():
                self.assertEqual((initial.output_dir / name).read_bytes(), contents)

    def test_changes_invalidate_only_affected_methods_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, si=True, k_values=(1,), sizes=(1,), figures="[1, 3]")
            run_paper_suite(load_paper_suite(path), show_progress=False)
            path.write_text(path.read_text().replace('aghdo_fit_steps = 1', 'aghdo_fit_steps = 2'))
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                updated = run_paper_suite(load_paper_suite(path), show_progress=False)
            self.assertEqual(updated["computed_entry_count"], 6)
            self.assertTrue(all(call.args[0].property == "MAGIC"
                                for call in benchmark.call_args_list))
            path.write_text(path.read_text().replace('[simulation]',
                                                     '[simulation]\nconenh_purity_weight = 3.0'))
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                updated = run_paper_suite(load_paper_suite(path), show_progress=False)
            self.assertEqual(benchmark.call_count, 1)
            self.assertEqual(benchmark.call_args.args[0].methods, ("ConEnh",))
            self.assertEqual(updated["reused_entry_count"], 8)

    def test_interruption_preserves_completed_realizations_and_force_recomputes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, figures="[1]", k_values=(1,), sizes=(1,))
            suite = load_paper_suite(path)
            calls = 0

            def interrupt(config, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt
                return run_benchmark(config, **kwargs)

            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                run_paper_suite(suite, show_progress=False)
            self.assertEqual(len(list(suite.cache_dir.rglob("*.json"))), 1)
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                resumed = run_paper_suite(suite, show_progress=False)
            self.assertEqual(benchmark.call_count, 1)
            self.assertEqual(resumed["reused_entry_count"], 1)
            forced = run_paper_suite(suite, force=True, show_progress=False)
            self.assertEqual(forced["computed_entry_count"], 2)

    def test_missing_or_corrupted_cache_plot_only_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, figures="[1]", k_values=(1,), sizes=(1,))
            suite = load_paper_suite(path)
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       side_effect=AssertionError("plot-only must not simulate")):
                with self.assertRaises(FileNotFoundError):
                    run_paper_suite(suite, plot_only=True, show_progress=False)
            self.assertFalse(suite.output_dir.exists())
            manifest = run_paper_suite(suite, show_progress=False)
            corrupt = suite.output_dir / manifest["data_files"][0]
            corrupt.write_text("{broken json")
            with self.assertRaises(FileNotFoundError):
                run_paper_suite(suite, plot_only=True, show_progress=False)
            with patch("con_enh_quant_learning.paper_simulations.run_benchmark",
                       wraps=run_benchmark) as benchmark:
                repaired = run_paper_suite(suite, show_progress=False)
            self.assertEqual(benchmark.call_count, 1)
            self.assertEqual(repaired["computed_entry_count"], 1)

    def test_cache_roundtrip_preserves_nonfinite_estimates_and_checks_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.txt"
            small_config(path, figures="[1]", k_values=(1,))
            config = replace(load_paper_suite(path).jobs[0].config, methods=("CS",))
            result = run_benchmark(config, show_progress=False)
            result.estimates["CS"][0, 0] = np.nan
            result.estimates["CS"][0, -1] = np.inf
            saved = Path(directory) / "result.json"
            save_realization(saved, result, "CS", 0, engine="engine", seconds=1.0)
            raw = json.loads(saved.read_text(),
                             parse_constant=lambda _: self.fail("Nonstandard JSON"))
            self.assertEqual(raw["estimates"][0], "NaN")
            loaded = load_realization(saved, config, "CS", 0)
            np.testing.assert_array_equal(loaded.estimates["CS"], result.estimates["CS"])
            np.testing.assert_array_equal(loaded.repetition_truths, result.repetition_truths)
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_realization(saved, replace(config, seed=config.seed + 1), "CS", 0)


if __name__ == "__main__":
    unittest.main()
