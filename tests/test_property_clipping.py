import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from con_enh_quant_learning import METHODS, BenchmarkResult, SimulationConfig, run_benchmark
from con_enh_quant_learning.aghdo import AGHDO, FocusedAGHDO
from con_enh_quant_learning.cli import _arguments, _config


class PropertyClippingTests(unittest.TestCase):
    def config(self, prop="PURITY", backend="dense", state="CLUSTER"):
        return SimulationConfig(
            n_qubits=4, k_rotations=2, subsystem_size=2, state=state,
            property=prop, methods=METHODS, pilot_shots=0, post_shots=16,
            repetitions=1, evaluation_points=3, aghdo_fit_steps=1,
            qns_property_samples=8, progress_details=False,
            subsystem_backend=backend, pauli_samples=32, conenh_candidates=8,
            conenh_max_cliques=16, conenh_update_every=2,
        )

    def assert_events_match(self, result, events):
        for event in events:
            for method, value in event["estimates"].items():
                index = list(result.shots).index(event["shots"])
                np.testing.assert_equal(value, result.estimates[method][event["repetition"], index])

    def test_validation_and_cli_boolean_override(self):
        for value in (True, False, None):
            replace(self.config(), property_clipping=value).validate()
        for value in (0, 1, "True", np.bool_(True)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "property_clipping"):
                replace(self.config(), property_clipping=value).validate()
        for flag, expected in (([], None), (["--property-clipping"], True),
                               (["--no-property-clipping"], False)):
            with patch("sys.argv", ["con-enh-benchmark", *flag]):
                args = _arguments()
            config = _config(args, property_name="RENYI2", k_rotations=2, subsystem_size=2)
            self.assertIs(config.property_clipping, expected)

    def test_saved_purity_conversion_projects_before_log_for_every_method(self):
        values = np.array([[-1.0, 0.0, 0.01, 0.5, 2.0]])
        result = BenchmarkResult(
            config=replace(self.config(), property_clipping=True),
            shots=np.arange(2, 7), truth=0.5,
            estimates={method: values.copy() for method in METHODS},
            subsystem=(0, 1), pilot_cost=0,
        )
        renyi = result.as_renyi2()
        self.assertEqual(renyi.truth, 1.0)
        for method in METHODS:
            np.testing.assert_allclose(
                renyi.estimates[method], [[2.0] * 3 + [1.0, 0]], atol=1e-15,
            )
            np.testing.assert_array_equal(result.estimates[method], values)
        raw = replace(result, config=replace(result.config, property_clipping=False)).as_renyi2()
        self.assertTrue(np.isnan(raw.estimates["CS"][0, 0]))
        self.assertTrue(np.isposinf(raw.estimates["CS"][0, 1]))
        self.assertLess(raw.estimates["CS"][0, -1], 0)

    def test_extreme_purity_predictions_for_all_methods_and_backends(self):
        for backend in ("dense", "sampled"):
            for value, expected in ((-0.5, 0.25), (2.0, 1.0)):
                with (
                    self.subTest(backend=backend, value=value),
                    patch("con_enh_quant_learning.simulation.IncrementalPurityUStatistic.value",
                          return_value=value),
                    patch("con_enh_quant_learning.simulation.FactorizedUniformPurityUStatistic.value",
                          return_value=value),
                    patch("con_enh_quant_learning.simulation.SampledPurityUStatistic.value",
                          return_value=value),
                    patch.object(AGHDO, "sample_purity", return_value=value),
                ):
                    for prop in ("PURITY", "RENYI2"):
                        events = []
                        result = run_benchmark(
                            replace(self.config(prop, backend), property_clipping=True),
                            events.append, show_progress=False,
                        )
                        target = expected if prop == "PURITY" else -np.log2(expected)
                        for method in METHODS:
                            np.testing.assert_allclose(result.estimates[method], target)
                        self.assert_events_match(result, events)
                    raw = run_benchmark(
                        replace(self.config(backend=backend), property_clipping=False),
                        show_progress=False,
                    )
                    for method in METHODS:
                        np.testing.assert_array_equal(raw.estimates[method], value)

    def test_all_properties_preserve_acquisition_and_clip_live_and_final_estimates(self):
        for backend in ("dense", "sampled"):
            for prop in ("PURITY", "RENYI2", "MAGIC"):
                with self.subTest(backend=backend, property=prop):
                    base = self.config(prop, backend)
                    raw_events, clipped_events = [], []
                    raw = run_benchmark(replace(base, property_clipping=False), raw_events.append,
                                        show_progress=False)
                    clipped = run_benchmark(replace(base, property_clipping=True),
                                            clipped_events.append, show_progress=False)
                    self.assertEqual([e["basis"] for e in raw_events],
                                     [e["basis"] for e in clipped_events])
                    lower, upper = {"PURITY": (0.25, 1), "RENYI2": (0, 2),
                                    "MAGIC": (0, 4)}[prop]
                    for method in METHODS:
                        values = clipped.estimates[method]
                        self.assertTrue(np.all(np.isfinite(values)), method)
                        self.assertTrue(np.all((values >= lower) & (values <= upper)), method)
                        if prop != "RENYI2":
                            expected = np.clip(raw.estimates[method], lower, upper)
                            if prop == "MAGIC":

                                expected = np.where(np.isnan(expected), upper, expected)
                            np.testing.assert_allclose(values, expected)
                    self.assert_events_match(clipped, clipped_events)
                    self.assert_events_match(raw, raw_events)
                    self.assertIs(clipped.execution["property_clipping"], True)

    def test_explicit_magic_switch_overrides_legacy_setting(self):
        for state in ("CLUSTER", "GHZ"):
            for setting in (True, False):
                with self.subTest(state=state, setting=setting):
                    config = replace(self.config("MAGIC", state=state), property_clipping=setting)
                    a = run_benchmark(replace(config, magic_clipping=True), show_progress=False)
                    b = run_benchmark(replace(config, magic_clipping=False), show_progress=False)
                    for method in METHODS:
                        np.testing.assert_array_equal(a.estimates[method], b.estimates[method])

    def test_aghdo_magic_heads_can_disable_their_property_projection(self):
        rng = unittest.mock.Mock()
        generic = AGHDO(1, rank=1)
        generic.density_matrix = lambda: np.array([[1, 0], [0, 0]], dtype=complex)
        rng.integers.return_value = np.full(8, 3)
        self.assertEqual(generic.sample_magic(8, rng), 0)
        self.assertEqual(generic.sample_magic(8, rng, clip=False), -1)
        logical = FocusedAGHDO(1)
        logical.bloch_vectors = lambda: np.array([[0, 0, 3**0.25]])
        rng.integers.return_value = np.zeros((1, 8), dtype=int)
        self.assertEqual(logical.sample_magic(8, rng), 0)
        self.assertAlmostEqual(logical.sample_magic(8, rng, clip=False), -1)


if __name__ == "__main__":
    unittest.main()
