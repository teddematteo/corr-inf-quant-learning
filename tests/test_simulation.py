import unittest
from dataclasses import replace

import numpy as np

from con_enh_quant_learning import SimulationConfig, iter_benchmark, run_benchmark
from con_enh_quant_learning._compat import bit_count
from con_enh_quant_learning.cluster_state import RotatedClusterState, grid_adjacency
from con_enh_quant_learning.estimators import IncrementalPurityUStatistic, magic_from_fourth_moment
from con_enh_quant_learning.measurement import LocalShot, ShotDesign
from con_enh_quant_learning.pauli import Pauli


class ClusterTests(unittest.TestCase):
    def test_rectangle_edges(self):
        adjacency = grid_adjacency(6, rows=2, columns=3)
        self.assertEqual(adjacency, (10, 21, 34, 17, 42, 20))
        for vertex, neighbours in enumerate(adjacency):
            self.assertFalse(neighbours & (1 << vertex))
            for other in range(6):
                self.assertEqual(
                    bool(neighbours & (1 << other)), bool(adjacency[other] & (1 << vertex))
                )

    def test_rectangular_oracle_against_dense_state(self):
        oracle = RotatedClusterState(6, 2, rows=2, columns=3, rotation_vertices=(0, 4))
        basis = np.arange(64)
        phase = np.ones(64, dtype=complex)
        for left, neighbours in enumerate(oracle.adjacency):
            for right in range(left + 1, 6):
                if neighbours & (1 << right):
                    phase *= (-1.0) ** (((basis >> left) & 1) * ((basis >> right) & 1))
        for vertex in oracle.rotation_vertices:
            phase *= np.exp(1j * oracle.theta * (1 - 2 * ((basis >> vertex) & 1)))
        state = phase / 8
        for x in range(64):
            for z in range(64):
                signs = np.array([(-1) ** (bit_count(int(b) & z)) for b in basis])
                exact = np.sum(
                    state[basis ^ x].conj() * state * signs * 1j ** (bit_count(x & z))
                )
                self.assertAlmostEqual(oracle.expectation(Pauli(x, z)), exact.real, places=12)

    def test_invalid_geometry_and_selections(self):
        for kwargs in ({"rows": 2}, {"rows": 2, "columns": 4}, {"rows": 0, "columns": 3}):
            with self.assertRaises(ValueError):
                grid_adjacency(6, **kwargs)
        for vertices in ((0, 0), (-1, 1), (0, 6), (0,)):
            with self.assertRaises(ValueError):
                RotatedClusterState(6, 2, rotation_vertices=vertices)
        with self.assertRaises(ValueError):
            SimulationConfig(
                n_qubits=4, k_rotations=0, subsystem_size=2, subsystem=(0, 4)
            ).validate()


class BenchmarkTests(unittest.TestCase):
    def config(self, prop="PURITY"):
        return SimulationConfig(
            n_qubits=4,
            k_rotations=2,
            subsystem_size=2,
            property=prop,
            post_shots=16,
            repetitions=1,
            evaluation_points=3,
            pilot_shots=1,
            aghdo_fit_steps=2,
            qns_property_samples=16,
            progress_details=False,
        )

    def test_preexisting_seeded_results(self):


        expected = {
            "PURITY": {
                "CS": [0.25, -2.3, -0.7625000000000002],
                "ConEnh": [0.25, 0.30434348937382305, 0.25],
            },
            "MAGIC": {
                "CS": [4, 4, 1.8536956948962238],
                "ConEnh": [4, 4, 3.1094712406286997],
            },
        }
        for prop, curves in expected.items():
            events = []
            result = run_benchmark(self.config(prop), events.append, show_progress=False)
            for method, values in curves.items():
                np.testing.assert_allclose(result.estimates[method][0], values, rtol=1e-12)
            checkpoints = [event for event in events if event["estimates"]]
            self.assertEqual(len(checkpoints), 18 if prop == "MAGIC" else 9)
            self.assertEqual(events[-1]["completed"], events[-1]["total"])
            for event in checkpoints:
                index = list(result.shots).index(event["shots"])
                for method, value in event["estimates"].items():
                    self.assertEqual(value, result.estimates[method][event["repetition"], index])

    def test_explicit_grid_matches_legacy_and_iterator(self):
        config = self.config()
        reference = run_benchmark(config, show_progress=False)
        vertices = RotatedClusterState(4, 2).rotation_vertices
        stream = iter_benchmark(
            replace(config, rows=2, columns=2, rotation_vertices=vertices), show_progress=False
        )
        while True:
            try:
                next(stream)
            except StopIteration as done:
                result = done.value
                break
        for method in reference.estimates:
            np.testing.assert_array_equal(reference.estimates[method], result.estimates[method])

    def test_renyi_events_preserve_undefined_values(self):
        purity = run_benchmark(self.config(), show_progress=False)
        events = []
        renyi = run_benchmark(self.config("RENYI2"), events.append, show_progress=False)
        self.assertEqual(renyi.truth, -np.log2(purity.truth))
        for event in events:
            self.assertEqual(event["truth"], renyi.truth)
        for method, values in purity.as_renyi2().estimates.items():
            np.testing.assert_array_equal(renyi.estimates[method], values)
        self.assertTrue(np.isnan(next(e for e in events if e["shot"] == 6)["estimates"]["CS"]))

    def test_conenh_clipping_changes_only_the_direct_estimate(self):
        config = self.config()
        clipped_events, raw_events = [], []
        clipped = run_benchmark(config, clipped_events.append, show_progress=False)
        raw = run_benchmark(
            replace(config, conenh_clipping=False), raw_events.append, show_progress=False
        )
        self.assertEqual([e["basis"] for e in clipped_events], [e["basis"] for e in raw_events])
        for method in ("CS", "CS-AGHDO", "ConEnh-AGHDO"):
            np.testing.assert_array_equal(clipped.estimates[method], raw.estimates[method])
        self.assertTrue(np.any(raw.estimates["ConEnh"] < 0))
        self.assertFalse(np.array_equal(clipped.estimates["ConEnh"], raw.estimates["ConEnh"]))
        renyi = run_benchmark(
            replace(config, property="RENYI2", conenh_clipping=False), show_progress=False
        )
        for method, values in raw.as_renyi2().estimates.items():
            np.testing.assert_array_equal(renyi.estimates[method], values)

    def test_unclipped_correlator_and_magic_conversions(self):
        statistic = IncrementalPurityUStatistic((0,))
        basis = ShotDesign(basis=np.array([0], dtype=np.int8))
        statistic.add(LocalShot(basis, np.array([1], dtype=np.int8)))
        statistic.add(LocalShot(basis, np.array([-1], dtype=np.int8)))

        self.assertEqual(statistic.value(), -4)
        self.assertEqual(statistic.value(np.ones(4)), 0.5)
        for moment, raw_value in ((2.0, -1.0), (0.125, 3.0), (0.0, float("inf"))):
            self.assertEqual(magic_from_fourth_moment(moment, 2, clip=False), raw_value)
            self.assertTrue(0 <= magic_from_fourth_moment(moment, 2) <= 2)
        self.assertTrue(np.isnan(magic_from_fourth_moment(-0.5, 2, clip=False)))

    def test_magic_clipping_preserves_acquisition_aghdo_and_live_values(self):
        config = self.config("MAGIC")
        clipped_events, raw_events = [], []
        clipped = run_benchmark(config, clipped_events.append, show_progress=False)
        raw = run_benchmark(
            replace(config, magic_clipping=False), raw_events.append, show_progress=False
        )
        self.assertEqual([e["basis"] for e in clipped_events], [e["basis"] for e in raw_events])
        for method in ("CS-AGHDO", "ConEnh-AGHDO"):
            np.testing.assert_array_equal(clipped.estimates[method], raw.estimates[method])
        self.assertTrue(np.isnan(raw.estimates["CS"][0, 1]))
        self.assertEqual(clipped.estimates["CS"][0, 1], 4)
        for event in raw_events:
            for method, value in event["estimates"].items():
                point = list(raw.shots).index(event["shots"])
                np.testing.assert_equal(value, raw.estimates[method][0, point])

    def test_100_qubit_grid_and_eight_qubit_subsystem_run(self):
        config = SimulationConfig(
            n_qubits=100,
            rows=10,
            columns=10,
            k_rotations=3,
            rotation_vertices=(0, 27, 99),
            subsystem_size=8,
            subsystem=tuple(range(8)),
            property="PURITY",
            methods=("CS", "ConEnh"),
            post_shots=4,
            repetitions=1,
            evaluation_points=32,
            theta=float(np.pi / 8),
            seed=7,
            progress_details=False,
        )
        result = run_benchmark(config, show_progress=False)
        self.assertAlmostEqual(result.truth, 1 / 256)
        for estimates in result.estimates.values():
            self.assertTrue(np.all(np.isfinite(estimates)))


if __name__ == "__main__":
    unittest.main()
