import itertools
import unittest
import warnings

import matplotlib

from con_enh_quant_learning._compat import strict_zip

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from con_enh_quant_learning import BenchmarkResult, SimulationConfig
from con_enh_quant_learning.plotting import (
    _error_statistics,
    _interpolate_curve,
    _moving_average,
    plot_benchmark,
)


class MissingPlotDataTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_latest_cs_renyi_curve_retains_all_finite_segments(self):

        shots = np.array([2, 6, 17, 49, 141, 410, 1189, 3448, 10000])
        estimates = np.array([[
            -2.995732273553991, np.nan, -2.749871695955549, np.nan,
            -0.11984945705590815, 1.6640176398649689, np.nan,
            2.4893604203751103, 2.2783047956413354,
        ]]) / np.log(2.0)
        result = BenchmarkResult(
            config=SimulationConfig(
                property="RENYI2", methods=("CS",), repetitions=1,
                n_qubits=100, k_rotations=0, subsystem_size=5,
            ),
            shots=shots, truth=4.0, estimates={"CS": estimates},
            subsystem=(55, 45, 54, 56, 65), pilot_cost=0,
        )
        original = estimates.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            figure, axis, _ = plot_benchmark(
                result, error_mode="relative", log_x=False, log_y=True,
            )
            figure.canvas.draw()
        curve = next(line for line in axis.lines if line.get_label() == "CS")
        x, y = curve.get_data()
        indices = np.flatnonzero(np.isfinite(y))
        segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)

        spans = [(x[group[0]], x[group[-1]]) for group in segments if len(group)]
        self.assertEqual(spans, [(2, 2), (17, 17), (141, 410), (3448, 10000)])
        errors = result.errors("relative")["CS"][0]
        for shot, error in strict_zip(shots, errors):
            if np.isfinite(error):
                indices = np.flatnonzero(np.isclose(x, shot, rtol=0, atol=1e-8))
                self.assertGreater(len(indices), 0)
                np.testing.assert_allclose(y[indices], error)
        np.testing.assert_array_equal(estimates, original)

    def test_statistics_use_available_repetitions_without_empty_slice_warnings(self):
        errors = np.array([
            [np.nan, np.inf, 0.4, 0.2, 0.0, np.nan],
            [np.nan, 0.8, 0.8, np.nan, 0.0, 0.7],
            [np.inf, 0.4, np.inf, 0.6, 0.0, np.nan],
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            mean, sem = _error_statistics(errors)
        np.testing.assert_allclose(mean, [np.nan, 0.6, 0.6, 0.4, 0.0, 0.7])
        np.testing.assert_allclose(sem, [np.nan, 0.2, 0.2, 0.2, 0.0, 0.0])
        result = BenchmarkResult(
            config=SimulationConfig(methods=("CS",), repetitions=3, property="RENYI2"),
            shots=np.array([2, 6, 17, 49, 141, 410]), truth=2.0,
            estimates={"CS": 2.0 + errors}, subsystem=(0, 1), pilot_cost=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            figure, axis, _ = plot_benchmark(result)
            figure.canvas.draw()
        self.assertGreater(len(axis.collections[0].get_paths()), 0)
        for path in axis.collections[0].get_paths():
            self.assertTrue(np.all(np.isfinite(path.vertices)))

    def test_smoothing_preserves_missing_checkpoints(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            smoothed = _moving_average(np.array([np.nan, np.inf, 0.4, np.nan, 0.2]), 3)
        np.testing.assert_allclose(smoothed, [np.nan, np.nan, 0.4, np.nan, 0.3])

    def test_all_undefined_curves_can_be_rendered_with_bands(self):
        result = BenchmarkResult(
            config=SimulationConfig(methods=("CS",), repetitions=2, property="RENYI2"),
            shots=np.array([2, 6, 17]), truth=2.0,
            estimates={"CS": np.array([[np.nan, np.inf, np.nan]] * 2)},
            subsystem=(0, 1), pilot_cost=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            figure, axis, _ = plot_benchmark(result, smooth_window=3)
            figure.canvas.draw()
        curve = next(line for line in axis.lines if line.get_label() == "CS")
        self.assertEqual(len(curve.get_xdata()), 0)


class InterpolationTests(unittest.TestCase):
    def test_uneven_checkpoints_remain_exact_in_all_axis_scales(self):
        x = np.array([2, 6, 17, 49, 141, 410, 1189, 3448, 10000])
        y = np.array([2.0, 1.4, 0.7, 0.1, 0.2, 0.01, 0.3, 0.2, 0.1])
        for log_x, log_y in itertools.product((False, True), repeat=2):
            curve_x, curve_y = _interpolate_curve(x, y, log_x=log_x, log_y=log_y)
            for measured_x, measured_y in strict_zip(x, y):
                indices = np.flatnonzero(np.isclose(curve_x, measured_x, rtol=1e-12))
                self.assertGreater(len(indices), 0)
                np.testing.assert_allclose(curve_y[indices], measured_y, atol=1e-12)

    def test_logarithms_skip_invalid_values_without_bridging_gaps(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            x, y = _interpolate_curve(
                np.arange(1, 8), np.array([1.0, 0.5, 0, -1, np.inf, 0.1, 0.05]),
                log_x=True, log_y=True,
            )
        self.assertEqual(np.count_nonzero(np.isnan(x)), 1)
        split = np.flatnonzero(np.isnan(x))[0]
        self.assertAlmostEqual(x[split - 1], 2)
        self.assertAlmostEqual(x[split + 1], 6)
        self.assertTrue(np.all(y[np.isfinite(y)] > 0))


if __name__ == "__main__":
    unittest.main()
