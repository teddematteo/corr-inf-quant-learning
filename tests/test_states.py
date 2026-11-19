import itertools
import unittest
from dataclasses import replace

import numpy as np

from con_enh_quant_learning import METHODS, SimulationConfig, run_benchmark
from con_enh_quant_learning._compat import bit_count
from con_enh_quant_learning.generators import measurement_generators
from con_enh_quant_learning.measurement import MagicProbeSelector, OnlineSubsystemSelector
from con_enh_quant_learning.pauli import Pauli
from con_enh_quant_learning.states import make_state


def dense_state(state, n, rotated, theta):
    vector = np.zeros(1 << n, dtype=complex)
    if state == "GHZ":
        vector[[0, -1]] = 1 / np.sqrt(2)
    indices = np.arange(1 << n)
    for site in rotated:
        if state == "GHZ" and site > 0:
            vector = np.cos(theta) * vector + 1j * np.sin(theta) * vector[indices ^ (1 << site)]
        else:
            vector *= np.exp(1j * theta * (1 - 2 * ((indices >> site) & 1)))
    return vector


def dense_pauli(pauli, n):
    basis = np.arange(1 << n)
    matrix = np.zeros((1 << n, 1 << n), dtype=complex)
    signs = np.array([(-1)**bit_count(int(index) & pauli.z) for index in basis])
    matrix[basis ^ pauli.x, basis] = signs * 1j**bit_count(pauli.x & pauli.z)
    return matrix


class StateOracleTests(unittest.TestCase):
    def test_all_correlators_purity_and_magic_against_dense_states(self):
        for state, (n, k), theta in itertools.product(
            ("GHZ",), [(n, k) for n in range(1, 5) for k in range(n + 1)],
            (0.0, 0.173, np.pi / 8, np.pi / 4),
        ):
            with self.subTest(state=state, n=n, k=k, theta=theta):
                oracle = make_state(state, n, k, theta=theta)
                vector = dense_state(state, n, oracle.rotation_vertices, theta)
                fourth = 0
                local_squares = np.zeros(n + 1)
                for x in range(1 << n):
                    for z in range(1 << n):
                        pauli = Pauli(x, z)
                        expected = np.vdot(vector, dense_pauli(pauli, n) @ vector).real
                        self.assertAlmostEqual(oracle.expectation(pauli), expected, places=12)
                        fourth += expected**4
                        for size in range(n + 1):
                            if pauli.support < (1 << size):
                                local_squares[size] += expected**2
                self.assertAlmostEqual(
                    oracle.stabilizer_renyi_magic(), -np.log2(fourth / (1 << n)), places=12
                )
                for size in range(n + 1):
                    self.assertAlmostEqual(
                        oracle.local_purity(tuple(range(size))),
                        local_squares[size] / (1 << size), places=12,
                    )

    def test_reference_generators_are_commuting_involutions_and_specify_unique_state(self):
        for state, n in itertools.product(("GHZ",), range(1, 6)):
            oracle = make_state(state, n, 0)
            vector = dense_state(state, n, (), 0)
            matrices = []
            projector = np.eye(1 << n, dtype=complex)
            for generator in oracle.generators:
                matrix = dense_pauli(generator, n)
                np.testing.assert_allclose(matrix @ vector, vector, atol=1e-13)
                np.testing.assert_allclose(matrix @ matrix, np.eye(1 << n), atol=1e-13)
                for previous in matrices:
                    np.testing.assert_allclose(matrix @ previous, previous @ matrix, atol=1e-13)
                matrices.append(matrix)
                projector = projector @ (np.eye(1 << n) + matrix) / 2
            np.testing.assert_allclose(projector, np.outer(vector, vector.conj()), atol=1e-13)

    def test_joint_sampling_matches_born_law_and_uses_one_copy(self):
        rng = np.random.default_rng(923)
        for state in ("GHZ",):
            oracle = make_state(state, 3, 2, theta=0.247, rotation_vertices=(0, 2))
            for basis in (np.array([0, 1, 0]), np.array([2, 2, 2])):
                probes = tuple(Pauli(
                    sum(1 << q for q in subset if basis[q] in (0, 1)),
                    sum(1 << q for q in subset if basis[q] in (1, 2)),
                ) for subset in ((0, 1), (1, 2), (0, 1, 2)))
                products = []
                for _ in range(2200):
                    full, parities = oracle.sample_local_and_parities(
                        basis, (0, 1, 2), probes, rng
                    )
                    np.testing.assert_array_equal(parities, [
                        full[0] * full[1], full[1] * full[2], np.prod(full)
                    ])
                    products.append(parities)
                np.testing.assert_allclose(
                    np.mean(products, axis=0), [oracle.expectation(p) for p in probes], atol=0.065
                )

    def test_incompatible_joint_parities_are_rejected(self):
        oracle = make_state("GHZ", 3, 1)
        with self.assertRaisesRegex(ValueError, "compatible"):
            oracle.sample_mode_parities((Pauli(1), Pauli(0, 1)), np.random.default_rng(1))

    def test_ghz_each_rotation_removes_one_independent_stabilizer(self):
        n = 4
        paulis = tuple(Pauli(x, z) for x, z in itertools.product(range(1 << n), repeat=2))
        for selected in range(1 << n):
            sites = tuple(q for q in range(n) if selected >> q & 1)
            with self.subTest(sites=sites):
                oracle = make_state("GHZ", n, len(sites), rotation_vertices=sites)
                expected_means = [np.cos(np.pi / 4) if q in sites else 1.0 for q in range(n)]
                np.testing.assert_allclose(
                    [oracle.expectation(g) for g in oracle.generators], expected_means,
                    atol=1e-13,
                )


                stabilizers = sum(np.isclose(abs(oracle.expectation(p)), 1) for p in paulis)
                self.assertEqual(stabilizers, 2**(n - len(sites)))
                self.assertAlmostEqual(
                    oracle.stabilizer_renyi_magic(), len(sites) * np.log2(4 / 3), places=12,
                )

    def test_ghz_sampler_matches_every_dense_born_probability(self):
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
                    def __ge__(self, threshold):
                        probability = threshold if bit == 0 else 1 - threshold
                        if probability < 1e-14:
                            raise ZeroProbability
                        owner.probability *= probability
                        return bit == 1

                return Decision()

        eigenvectors = (
            np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            np.array([[1, 1j], [1, -1j]]) / np.sqrt(2),
            np.eye(2),
        )
        n = 3
        for sites in ((), (0,), (1, 2), (0, 1, 2)):
            oracle = make_state("GHZ", n, len(sites), theta=0.247, rotation_vertices=sites)
            vector = dense_state("GHZ", n, sites, oracle.theta)
            for basis in itertools.product(range(3), repeat=n):
                for bits in itertools.product(range(2), repeat=n):
                    ket = np.ones(1, dtype=complex)
                    for q in reversed(range(n)):
                        ket = np.kron(ket, eigenvectors[basis[q]][bits[q]])
                    expected = abs(np.vdot(ket, vector))**2
                    rng = ForcedRng(bits)
                    try:
                        full = oracle.sample_outcomes(np.array(basis), tuple(range(n)), rng)
                        np.testing.assert_array_equal(full, 1 - 2 * np.array(bits))
                        actual = rng.probability
                    except ZeroProbability:
                        actual = 0.0
                    self.assertAlmostEqual(actual, expected, places=12)


class StateBenchmarkTests(unittest.TestCase):
    def config(self, state, prop="PURITY", **kwargs):
        return SimulationConfig(
            state=state, n_qubits=4, k_rotations=2, subsystem_size=2,
            property=prop, post_shots=8, repetitions=1, evaluation_points=2,
            pilot_shots=1, aghdo_fit_steps=1, qns_property_samples=16,
            progress_details=False, **kwargs,
        )

    def test_all_methods_all_properties_reproducible_and_entropy_uses_same_stream(self):
        for state in ("GHZ",):
            purity = run_benchmark(self.config(state), show_progress=False)
            renyi = run_benchmark(self.config(state, "RENYI2"), show_progress=False)
            magic = run_benchmark(self.config(state, "MAGIC"), show_progress=False)
            replay = run_benchmark(self.config(state, "MAGIC"), show_progress=False)
            for method in METHODS:
                self.assertTrue(np.all(np.isfinite(purity.estimates[method])))
                self.assertTrue(np.all(np.isfinite(magic.estimates[method])))
                np.testing.assert_array_equal(magic.estimates[method], replay.estimates[method])
                np.testing.assert_array_equal(
                    purity.as_renyi2().estimates[method], renyi.estimates[method]
                )
            oracle = make_state(state, 4, 2)
            count = len(measurement_generators(oracle.generators))
            self.assertEqual(magic.pilot_cost, count)

    def test_hundred_qubits_with_direct_estimators(self):
        for state, prop in itertools.product(("GHZ",), ("PURITY", "MAGIC")):
            config = replace(
                self.config(state, prop), n_qubits=100, k_rotations=50, subsystem_size=3,
                methods=("CS", "ConEnh"), post_shots=4, pilot_shots=0,
            )
            result = run_benchmark(config, show_progress=False)
            if state == "GHZ" and prop == "MAGIC":
                self.assertAlmostEqual(result.truth, 50 * np.log2(4 / 3))
            for values in result.estimates.values():
                self.assertTrue(np.all(np.isfinite(values)))

    def test_unrotated_magic_all_methods_including_single_qubit(self):
        for state, n in itertools.product(("CLUSTER", "GHZ"), (1, 4)):
            config = replace(
                self.config(state, "MAGIC"), n_qubits=n, k_rotations=0,
                subsystem_size=1, pilot_shots=0,
            )
            result = run_benchmark(config, show_progress=False)
            for values in result.estimates.values():
                self.assertTrue(np.all(np.isfinite(values)))

    def test_generators_do_not_change_with_rotation_and_selectors_receive_only_paulis(self):
        for state in ("GHZ",):
            reference = make_state(state, 5, 0)
            rotated = make_state(state, 5, 4)
            self.assertEqual(reference.generators, rotated.generators)
            words = measurement_generators(reference.generators)
            self.assertTrue(all(isinstance(word, Pauli) for word in words))

            local = OnlineSubsystemSelector(words, (0, 1), np.random.default_rng(4), n_qubits=5)
            global_ = MagicProbeSelector(words, None, np.random.default_rng(4), n_qubits=5)
            self.assertEqual(len(local.draw().basis), 5)
            self.assertEqual(len(global_.draw().basis), 5)

    def test_only_supported_states_and_pauli_generators(self):
        for state in ("CLUSTER", "GHZ"):
            oracle = make_state(state, 5, 0)
            self.assertEqual(measurement_generators(oracle.generators, max_weight=1),
                             tuple(oracle.generators))
            replace(self.config(state, "MAGIC"), n_qubits=100).validate()
        for kwargs in ({"state": "unknown"}, {"generator_max_weight": 0}):
            with self.assertRaises(ValueError):
                replace(self.config("GHZ"), **kwargs).validate()


if __name__ == "__main__":
    unittest.main()
