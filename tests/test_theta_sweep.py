import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import matplotlib

from con_enh_quant_learning._compat import strict_zip

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from con_enh_quant_learning import BenchmarkResult, SimulationConfig
from con_enh_quant_learning.plotting import plot_benchmark, plot_theta_sweep


class ThetaSweepPlotTests(unittest.TestCase):
    def result(self, theta, truth):
        config = SimulationConfig(
            n_qubits=4, k_rotations=2, subsystem_size=2, property="MAGIC",
            methods=("CS", "ConEnh"), theta=theta, post_shots=12, repetitions=2,
            evaluation_points=3,
        )
        return BenchmarkResult(
            config=config,
            shots=np.array([4, 8, 12]),
            truth=truth,
            estimates={
                "CS": np.array([[0.5, 1.5, 2.5], [1.5, 2.5, 3.5]]),
                "ConEnh": np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
            },
            subsystem=(0, 1), pilot_cost=0,
        )

    def tearDown(self):
        plt.close("all")

    def test_each_curve_uses_its_own_truth_and_angle_color(self):
        results = [self.result(np.pi / 16, 1.0), self.result(np.pi / 8, 2.0)]
        _, axis, saved = plot_theta_sweep(
            results, error_mode="relative", log_x=False, log_y=False,
        )
        curves = [line for line in axis.lines if not line.get_label().startswith("_")]
        self.assertEqual(len(curves), 4)
        self.assertEqual(saved, ())
        self.assertEqual(
            [line.get_label() for line in curves],
            [r"CS ($\theta=\pi/16$)", r"ConEnh ($\theta=\pi/16$)",
             r"CS ($\theta=\pi/8$)", r"ConEnh ($\theta=\pi/8$)"],
        )
        for result, pair in strict_zip(results, (curves[:2], curves[2:])):
            for method, curve in strict_zip(result.config.methods, pair):
                expected = np.mean(np.abs(result.estimates[method] - result.truth) / result.truth,
                                   axis=0)
                np.testing.assert_allclose(
                    np.interp(result.shots, curve.get_xdata(), curve.get_ydata()), expected,
                )
            dark, light = (np.asarray(line.get_color()) for line in pair)
            np.testing.assert_allclose(dark / 0.68, (light - 0.36) / 0.64)
            self.assertGreater(float(light.mean()), float(dark.mean()))
        self.assertFalse(np.allclose(curves[0].get_color(), curves[2].get_color()))
        self.assertEqual(len(axis.collections), 4)
        self.assertIn(r"\theta=\pi/16", axis.get_title())

    def test_original_plot_labels_styles_and_statistics_are_preserved(self):
        result = self.result(np.pi / 8, 1.0)
        _, axis, _ = plot_benchmark(result, error_mode="absolute", log_x=False, log_y=False)
        curves = [line for line in axis.lines if not line.get_label().startswith("_")]
        self.assertEqual([line.get_label() for line in curves], ["CS", "ConEnh"])
        self.assertEqual([line.get_color() for line in curves], ["#0072B2", "#D55E00"])
        for method, curve in strict_zip(result.config.methods, curves):
            np.testing.assert_allclose(
                np.interp(result.shots, curve.get_xdata(), curve.get_ydata()),
                np.mean(np.abs(result.estimates[method] - result.truth), axis=0),
            )

    def test_aghdo_and_single_angle_exports(self):
        result = self.result(0.123456, 1.0)
        methods = ("CS", "ConEnh", "CS-AGHDO", "ConEnh-AGHDO")
        result.config = replace(result.config, methods=methods)
        result.estimates.update({"CS-AGHDO": result.estimates["CS"],
                                 "ConEnh-AGHDO": result.estimates["ConEnh"]})
        with tempfile.TemporaryDirectory() as directory:
            _, axis, saved = plot_theta_sweep(
                [result], colors=["red"], output_stem=Path(directory) / "sweep",
                log_x=True, log_y=True, show_band=False,
            )
            curves = [line for line in axis.lines if not line.get_label().startswith("_")]
            self.assertEqual([line.get_linestyle() for line in curves], ["-", "-", "--", "--"])
            self.assertIn(r"\theta=0.123456", curves[0].get_label())
            self.assertEqual(len(axis.collections), 0)
            self.assertEqual([path.suffix for path in saved], [".pdf", ".png"])
            self.assertTrue(all(path.stat().st_size > 1000 for path in saved))

    def test_rejects_empty_duplicate_and_incompatible_comparisons(self):
        result = self.result(0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "At least one"):
            plot_theta_sweep([])
        with self.assertRaisesRegex(ValueError, "distinct"):
            plot_theta_sweep([result, result])
        other = self.result(np.pi / 8, 2.0)
        other.config = replace(other.config, seed=42)
        with self.assertRaisesRegex(ValueError, "share all settings"):
            plot_theta_sweep([result, other])
        with self.assertRaisesRegex(ValueError, "one base color"):
            plot_theta_sweep([result], colors=[])
        with self.assertRaisesRegex(ValueError, "Relative error is undefined"):
            plot_theta_sweep([result], error_mode="relative")

    def test_range_labels_and_distinct_distributions(self):
        results = [self.result(np.pi / 8, 1.0), self.result(np.pi / 8, 2.0)]
        results[0].config = replace(results[0].config, theta_range=(0, np.pi / 16))
        results[1].config = replace(results[1].config, theta_range=(0, np.pi / 8))
        _, axis, _ = plot_theta_sweep(results, log_x=False, log_y=False)
        curves = [line for line in axis.lines if not line.get_label().startswith("_")]
        self.assertIn(r"\theta_i\sim U[0, \pi/16]", curves[0].get_label())
        self.assertIn(r"\theta_i\sim U[0, \pi/8]", curves[2].get_label())
        self.assertIn(r"\theta_i\sim U[0, \pi/16]", axis.get_title())
        _, single_axis, _ = plot_benchmark(results[1])
        self.assertIn(r"\theta_i\sim U[0, \pi/8]", single_axis.get_title())
        duplicate = replace(results[0], config=replace(results[0].config, theta=99))
        with self.assertRaisesRegex(ValueError, "distinct"):
            plot_theta_sweep([results[0], duplicate])


if __name__ == "__main__":
    unittest.main()
