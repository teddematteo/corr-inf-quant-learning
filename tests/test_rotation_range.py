import itertools
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from con_enh_quant_learning import SimulationConfig, run_benchmark
from con_enh_quant_learning._compat import bit_count, strict_zip
from con_enh_quant_learning.cli import _arguments, _config
from con_enh_quant_learning.pauli import Pauli
from con_enh_quant_learning.states import STATE_TYPES, make_state


def dense_vector(oracle, state):
    n = oracle.n_qubits
    indices = np.arange(1 << n)
    vector = np.zeros(1 << n, dtype=complex)
    if state == "CLUSTER":
        vector[:] = 1 / np.sqrt(1 << n)
        for left, neighbours in enumerate(oracle.adjacency):
            for right in range(left + 1, n):
                if neighbours >> right & 1:
                    vector *= (-1.0)**(((indices >> left) & 1) * ((indices >> right) & 1))
    elif state == "GHZ":
        vector[[0, -1]] = 1 / np.sqrt(2)
    for site, angle in strict_zip(oracle.rotation_vertices, oracle.rotation_angles):
        if state == "GHZ" and site > 0:
            vector = np.cos(angle) * vector + 1j * np.sin(angle) * vector[indices ^ (1 << site)]
        else:
            vector *= np.exp(1j * angle * (1 - 2 * ((indices >> site) & 1)))
    return vector


class RotationRangeTests(unittest.TestCase):
    def test_dense_reduced_state_ignores_boundary_z_rotations_but_detects_interior_ones(self):
        sites = (0, 1, 3, 4)

        def reduced_state(rotations):
            state = make_state(
                "CLUSTER", 6, len(rotations), rows=2, columns=3,
                rotation_vertices=rotations, theta_range=(0.17, 0.31),
            )
            vector = dense_vector(state, "CLUSTER")
            outside = tuple(site for site in range(6) if site not in sites)
            indices = np.arange(64).reshape((2,) * 6).transpose(
                tuple(5 - site for site in sites + outside)
            ).reshape(16, 4)
            amplitudes = vector[indices]
            return amplitudes @ amplitudes.conj().T

        reference = reduced_state(())
        boundary_and_outside = reduced_state((1, 2, 4, 5))
        interior = reduced_state((0,))
        np.testing.assert_allclose(boundary_and_outside, reference, atol=1e-14)
        self.assertGreater(np.linalg.norm(interior - reference), 0.05)

        for density in (boundary_and_outside, interior):
            self.assertAlmostEqual(np.trace(density @ density).real,
                                   np.trace(reference @ reference).real, places=13)

    def test_correlators_purity_and_magic_against_dense_vectors(self):
        for state, (n, sites) in itertools.product(
            STATE_TYPES, ((1, (0,)), (3, ()), (3, (0, 2)), (4, (0, 1, 2, 3))),
        ):
            with self.subTest(state=state, n=n, sites=sites):
                oracle = make_state(
                    state, n, len(sites), rotation_vertices=sites, theta_range=(-0.21, 0.67),
                )
                vector = dense_vector(oracle, state)
                indices = np.arange(1 << n)
                fourth = local_squares = 0.0
                for x, z in itertools.product(range(1 << n), repeat=2):
                    signs = np.array([(-1)**bit_count(int(i) & z) for i in indices])
                    expected = np.sum(
                        vector[indices ^ x].conj() * vector * signs * 1j**bit_count(x & z)
                    ).real
                    self.assertAlmostEqual(oracle.expectation(Pauli(x, z)), expected, places=12)
                    fourth += expected**4
                    if (x | z) < 1 << min(2, n):
                        local_squares += expected**2
                self.assertAlmostEqual(
                    oracle.stabilizer_renyi_magic(), -np.log2(fourth / (1 << n)), places=12,
                )
                self.assertAlmostEqual(
                    oracle.local_purity(tuple(range(min(2, n)))),
                    local_squares / (1 << min(2, n)), places=12,
                )

    def test_seeds_override_and_site_selection(self):
        for state in STATE_TYPES:
            fixed = make_state(state, 12, 7)
            ranged = make_state(state, 12, 7, theta_range=(-0.2, 0.5))
            replay = make_state(state, 12, 7, theta=-99, theta_range=(-0.2, 0.5))
            explicit = make_state(
                state, 12, 7, theta_range=(-0.2, 0.5), rotation_vertices=fixed.rotation_vertices,
            )
            other = make_state(state, 12, 7, theta_range=(-0.2, 0.5), vertex_seed=1730)
            self.assertEqual(ranged.rotation_vertices, fixed.rotation_vertices)
            self.assertEqual(ranged.rotation_angles, replay.rotation_angles)
            self.assertEqual(ranged.rotation_angles, explicit.rotation_angles)
            self.assertNotEqual(ranged.rotation_angles, other.rotation_angles)
            self.assertEqual(len(set(ranged.rotation_angles)), 7)
            self.assertTrue(all(-0.2 <= angle <= 0.5 for angle in ranged.rotation_angles))

    def test_invalid_ranges_are_rejected_by_config_and_oracles(self):
        for bounds in ((0.5, 0.1), (0, np.inf), (np.nan, 0), (), (0,), (0, 1, 2), 0.5,
                       ((0, 1),), (-1e308, 1e308)):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "theta_range"):
                    SimulationConfig(theta_range=bounds).validate()
                for state in STATE_TYPES:
                    with self.assertRaisesRegex(ValueError, "theta_range"):
                        make_state(state, 3, 2, theta_range=bounds)

    def test_equal_bounds_match_fixed_angle_for_all_methods(self):
        for state, prop in itertools.product(STATE_TYPES, ("PURITY", "MAGIC")):
            config = SimulationConfig(
                state=state, n_qubits=4, k_rotations=3, subsystem_size=2, property=prop,
                theta=0.173, post_shots=8, repetitions=1, evaluation_points=2,
                pilot_shots=1, aghdo_fit_steps=1, qns_property_samples=16,
                progress_details=False,
            )
            fixed = run_benchmark(config, show_progress=False)
            ranged = run_benchmark(replace(config, theta_range=(0.173, 0.173)), show_progress=False)
            self.assertEqual(fixed.truth, ranged.truth)
            for method in config.methods:
                np.testing.assert_array_equal(fixed.estimates[method], ranged.estimates[method])

    def test_benchmark_samples_realizations_and_replays_each_repetition(self):
        for state in STATE_TYPES:
            config = SimulationConfig(
                state=state, n_qubits=8, k_rotations=3, subsystem_size=2,
                theta_range=(0.05, 0.41), post_shots=8, repetitions=3,
                evaluation_points=2, methods=("CS", "ConEnh"), progress_details=False,
            )
            events = []
            result = run_benchmark(config, events.append, show_progress=False)
            replay = run_benchmark(config, show_progress=False)
            history = result.execution["repetitions"]
            self.assertEqual(len({tuple(item["rotation_angles"]) for item in history}), 3)
            self.assertEqual(len({tuple(item["rotation_vertices"]) for item in history}), 3)
            for event in events:
                realization = history[event["repetition"]]
                self.assertEqual(event["execution"]["rotation_angles"],
                                 realization["rotation_angles"])
                self.assertEqual(event["execution"]["rotation_vertices"],
                                 realization["rotation_vertices"])
                self.assertEqual(event["truth"], realization["truth"])
                self.assertTrue(all(0.05 <= angle <= 0.41
                                    for angle in realization["rotation_angles"]))
            for method in config.methods:
                np.testing.assert_array_equal(result.estimates[method], replay.estimates[method])
            other = run_benchmark(replace(config, seed=42), show_progress=False)
            for original, changed in zip(history, other.execution["repetitions"]):
                self.assertEqual(original["rotation_angles"], changed["rotation_angles"])
                self.assertEqual(original["rotation_vertices"], changed["rotation_vertices"])
            for rep in range(3):
                resumed = run_benchmark(replace(config, repetitions=1),
                                        repetition_start=rep, show_progress=False)
                for method in config.methods:
                    np.testing.assert_array_equal(resumed.estimates[method][0],
                                                  result.estimates[method][rep])

    def test_magic_errors_use_each_realizations_truth_and_renyi_conversion(self):
        config = SimulationConfig(
            n_qubits=4, k_rotations=2, subsystem_size=2, property="MAGIC",
            theta_range=(0.05, 0.41), post_shots=8, repetitions=3,
            evaluation_points=2, methods=("CS", "ConEnh"), progress_details=False,
        )
        result = run_benchmark(config, show_progress=False)
        self.assertGreater(np.ptp(result.repetition_truths), 0.01)
        self.assertEqual(result.truth, result.repetition_truths.mean())
        for method in config.methods:
            np.testing.assert_allclose(result.errors("relative")[method],
                                       np.abs(result.estimates[method]
                                              - result.repetition_truths[:, None])
                                       / np.abs(result.repetition_truths[:, None]))
        purity = run_benchmark(replace(config, property="PURITY"), show_progress=False)
        renyi = purity.as_renyi2()
        np.testing.assert_allclose(renyi.repetition_truths, -np.log2(purity.repetition_truths))
        np.testing.assert_allclose([item["truth"] for item in renyi.execution["repetitions"]],
                                   renyi.repetition_truths)

    def test_cli_passes_range_and_seed(self):
        with patch("sys.argv", ["benchmark", "--theta-range", "-0.1", "0.4",
                                "--theta", "0.2", "--rotation-seed", "19"]):
            config = _config(_arguments(), property_name="MAGIC", k_rotations=2, subsystem_size=3)
        config.validate()
        self.assertEqual(config.theta_range, (-0.1, 0.4))
        self.assertEqual(config.theta, 0.2)
        self.assertEqual(config.rotation_seed, 19)


class BornRangeTests(unittest.TestCase):
    def test_every_sequential_born_probability_against_dense_state(self):
        class ZeroProbability(Exception):
            pass

        class ForcedRng:
            def __init__(self, bits):
                self.bits = iter(bits)
                self.probability = 1.0

            def random(self):
                owner = self
                bit = next(self.bits)

                class Decision:
                    def __lt__(self, threshold):
                        probability = threshold if bit == 0 else 1 - threshold
                        if probability < 1e-14:
                            raise ZeroProbability
                        owner.probability *= probability
                        return bit == 0

                    def __ge__(self, threshold):
                        return not self.__lt__(threshold)

                return Decision()

        eigenvectors = (
            np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            np.array([[1, 1j], [1, -1j]]) / np.sqrt(2),
            np.eye(2),
        )
        for state in STATE_TYPES:
            oracle = make_state(state, 3, 3, theta_range=(-0.3, 0.7))
            vector = dense_vector(oracle, state)
            for basis, bits in itertools.product(
                itertools.product(range(3), repeat=3), itertools.product(range(2), repeat=3),
            ):
                ket = np.ones(1, dtype=complex)
                for site in reversed(range(3)):
                    ket = np.kron(ket, eigenvectors[basis[site]][bits[site]])
                expected = abs(np.vdot(ket, vector))**2
                rng = ForcedRng(bits)
                try:

                    if state == "CLUSTER":
                        full = oracle._sample_full_record(np.array(basis), rng)
                    else:
                        full = oracle.sample_outcomes(np.array(basis), (0, 1, 2), rng)
                    np.testing.assert_array_equal(full, 1 - 2 * np.array(bits))
                    actual = rng.probability
                except ZeroProbability:
                    actual = 0.0
                self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
