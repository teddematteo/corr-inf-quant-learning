import itertools
import unittest
from dataclasses import replace

import numpy as np

from con_enh_quant_learning import SimulationConfig, run_benchmark
from con_enh_quant_learning._compat import bit_count
from con_enh_quant_learning.cluster_state import RotatedClusterState
from con_enh_quant_learning.estimators import (
    FactorizedUniformPurityUStatistic,
    IncrementalPurityUStatistic,
)
from con_enh_quant_learning.generators import ghz_generators
from con_enh_quant_learning.graph_sampling import GraphStateSampler
from con_enh_quant_learning.measurement import LocalShot, ShotDesign
from con_enh_quant_learning.pauli import Pauli
from con_enh_quant_learning.sampled import (
    PauliPool,
    SampledPurityUStatistic,
    SampledSubsystemSelector,
    _in_span,
    local_generator_span,
)
from con_enh_quant_learning.states import make_state


def exhaustive_pool(size):
    words = []
    masks = []
    x, z = [], []
    for digits in itertools.product(range(4), repeat=size):
        px = sum(1 << j for j, d in enumerate(digits) if d in (1, 2))
        pz = sum(1 << j for j, d in enumerate(digits) if d in (2, 3))
        if px or pz:
            words.append(Pauli(px, pz))
            x.append(px)
            z.append(pz)
            masks.append((px & ~pz, px & pz, pz & ~px))
    x, z = np.array(x, dtype=np.uint64), np.array(z, dtype=np.uint64)
    support = x | z
    return PauliPool(
        tuple(range(size)), tuple(words), np.array(masks, dtype=np.uint64), x, z, support,
        np.ones(len(words)), np.array([3.0**(-bit_count(int(s))) for s in support]),
        len(words), 0,
    )


class EstimatorScalingTests(unittest.TestCase):
    def test_factorized_50_site_kernel_matches_direct_product(self):
        for size in (50, 63, 64, 100):
            with self.subTest(size=size):
                count = 31
                rng = np.random.default_rng(62)
                statistic = FactorizedUniformPurityUStatistic(tuple(range(size)), count)
                bases = rng.integers(0, 3, (count, size), dtype=np.int8)
                outcomes = 1 - 2 * rng.integers(0, 2, (count, size), dtype=np.int8)
                total = 0.0
                for i in range(count):
                    for j in range(i):
                        total += np.prod(
                            (1 + 9 * (bases[i] == bases[j]) * outcomes[i] * outcomes[j]) / 2
                        )
                    statistic.add(LocalShot(ShotDesign(bases[i]), outcomes[i]))
                self.assertAlmostEqual(
                    statistic.value() / (total / statistic.pair_count), 1, places=12,
                )

    def test_factorized_wide_masks_keep_high_axis_and_outcome_bits(self):
        size = 100
        for site in (0, 63, 64, 99):
            for change_axis in (False, True):
                with self.subTest(site=site, change_axis=change_axis):
                    statistic = FactorizedUniformPurityUStatistic(tuple(range(size)), 2)
                    basis = np.zeros(size, dtype=np.int8)
                    outcomes = np.ones(size, dtype=np.int8)
                    statistic.add(LocalShot(ShotDesign(basis.copy()), outcomes.copy()))
                    if change_axis:
                        basis[site] = 1
                        expected = 0.5 * 5.0**99
                    else:
                        outcomes[site] = -1
                        expected = -4.0 * 5.0**99
                    statistic.add(LocalShot(ShotDesign(basis), outcomes))
                    self.assertAlmostEqual(statistic.value() / expected, 1, places=12)

    def test_factorized_and_sampled_full_pool_equal_original_u_statistic(self):
        rng = np.random.default_rng(438)
        size, shots = 4, 19
        pool = exhaustive_pool(size)
        dense = IncrementalPurityUStatistic(pool.subsystem)
        factorized = FactorizedUniformPurityUStatistic(pool.subsystem, shots)
        sampled = SampledPurityUStatistic(pool)
        for _ in range(shots):
            shot = LocalShot(
                ShotDesign(basis=rng.integers(0, 3, size, dtype=np.int8)),
                rng.choice(np.array([-1, 1], dtype=np.int8), size=size),
            )
            for statistic in (dense, sampled, factorized):
                statistic.add(shot)
            if dense.pair_count:
                self.assertAlmostEqual(dense.value(), factorized.value(), places=11)
                self.assertAlmostEqual(dense.value(), sampled.value(), places=11)
        self.assertAlmostEqual(
            dense.value(np.ones(4**size)), sampled.value(np.ones(len(pool.paulis)))
        )

    def test_sampled_inclusions_are_exact_and_snapshots_survive_refresh(self):
        pool = exhaustive_pool(3)
        selector = SampledSubsystemSelector(
            ghz_generators(3), pool, np.random.default_rng(72), n_qubits=3,
            candidates=20, max_cliques=64, update_every=1, exploration=0.12,
            temperature=0.3, pauli_floor=0.1, commuting_bias=1, purity_weight=1,
            generator_weight=1, local_regularization=0.05,
        )
        shot = selector.draw()
        original = shot.local_inclusion.copy()
        for i, word in enumerate(pool.paulis):
            covered = []
            for basis in selector._bases:
                covered.append(all(
                    not ((word.support >> q) & 1)
                    or basis[q] == (1 if (word.x & word.z) >> q & 1 else (
                        0 if word.x >> q & 1 else 2
                    )) for q in range(3)
                ))
            expected = 0.88 * (selector.policy @ covered) + 0.12 * 3.0**(-word.weight)
            self.assertAlmostEqual(shot.local_inclusion[i], expected)
            self.assertAlmostEqual(shot.coverage_probability(word), expected)
        probes = selector.parity_probes(shot.basis)
        selector.observe(shot, np.ones(len(probes), dtype=np.int8))
        selector.draw()
        np.testing.assert_array_equal(original, shot.local_inclusion)
        self.assertEqual(selector.policy_updates, 2)

    def test_generator_span_includes_products_cancelling_outside_A(self):

        span = local_generator_span((Pauli(0, 3), Pauli(0, 6)), (0, 2), 3)
        self.assertEqual(len(span), 1)
        self.assertTrue(_in_span(3 << 2, span))
        self.assertFalse(_in_span(1 << 2, span))

    def test_importance_pool_corrects_nonuniform_proposal(self):


        values = []
        for seed in range(80):
            pool = PauliPool.sample(
                ghz_generators(3), (0, 1), 3, 128, 0.2, np.random.default_rng(seed)
            )
            second = np.array([float(word == Pauli(0, 3)) for word in pool.paulis])
            values.append((1 + pool.population_weights @ second) / 4)
        self.assertAlmostEqual(np.mean(values), 0.5, delta=0.008)


class GraphSamplingTests(unittest.TestCase):
    def test_large_ghz_sampler_matches_hub_and_leaf_rotations(self):
        oracle = make_state("GHZ", 12, 5, theta=0.193, rotation_vertices=(0, 2, 5, 9, 11))
        rng = np.random.default_rng(93)
        sites = tuple(range(12))
        z_basis = np.full(12, 2, dtype=np.int8)
        samples = np.array([oracle.sample_outcomes(z_basis, sites, rng) for _ in range(2000)])
        unrotated = (0, 1, 3, 4, 6, 7, 8, 10)
        self.assertTrue(np.all(samples[:, unrotated] == samples[:, :1]))
        for site in (1, 2, 5, 11):
            self.assertAlmostEqual(
                (samples[:, 0] * samples[:, site]).mean(),
                oracle.expectation(Pauli(0, 1 | (1 << site))), delta=0.07,
            )
        basis = np.zeros(12, dtype=np.int8)
        basis[:3] = 1
        samples = np.array([oracle.sample_outcomes(basis, sites, rng) for _ in range(2000)])
        expected = oracle.expectation(Pauli((1 << 12) - 1, 7))
        self.assertAlmostEqual(np.prod(samples, axis=1).mean(), expected, delta=0.07)
        self.assertLess(np.max(np.abs(samples.mean(axis=0))), 0.08)

        unrotated_oracle = make_state("GHZ", 12, 0)
        for _ in range(20):
            full = unrotated_oracle.sample_outcomes(z_basis, sites, rng)
            self.assertTrue(np.all(full == full[0]))

    def test_frontier_probabilities_match_every_dense_born_probability(self):
        oracle = RotatedClusterState(4, 2, rows=2, columns=2, rotation_vertices=(0, 3))
        sampler = GraphStateSampler(oracle.adjacency, oracle.rotation_vertices, oracle.theta)
        indices = np.arange(16)
        state = np.ones(16, dtype=complex) / 4
        for q, neighbours in enumerate(oracle.adjacency):
            for r in range(q + 1, 4):
                if neighbours >> r & 1:
                    state *= (-1.0)**(((indices >> q) & 1) * ((indices >> r) & 1))
        for q in oracle.rotation_vertices:
            state *= np.exp(1j * oracle.theta * (1 - 2 * ((indices >> q) & 1)))
        eigenvectors = (
            np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            np.array([[1, 1j], [1, -1j]]) / np.sqrt(2),
            np.eye(2),
        )

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

                return Decision()

        for basis in itertools.product(range(3), repeat=4):
            for bits in itertools.product(range(2), repeat=4):
                ket = np.ones(1, dtype=complex)
                for q in reversed(range(4)):
                    ket = np.kron(ket, eigenvectors[basis[q]][bits[q]])
                expected = abs(np.vdot(ket, state))**2
                rng = ForcedRng(bits)
                try:
                    sampler.sample(np.array(basis), rng)
                    actual = rng.probability
                except ZeroProbability:
                    actual = 0.0
                self.assertAlmostEqual(actual, expected, places=12)

    def test_cut_rank_purity_matches_original_pauli_sum(self):
        oracle = RotatedClusterState(9, 4, rows=3, columns=3)


        self.assertEqual(oracle.local_purity(tuple(range(9))), 1.0)
        for sites in ((0, 1), (0, 1, 3, 4), (0, 2, 4, 6, 8)):
            outside = ((1 << 9) - 1) ^ sum(1 << q for q in sites)
            rows = {0}
            for q in sites:
                rows |= {row ^ (oracle.adjacency[q] & outside) for row in tuple(rows)}
            self.assertAlmostEqual(oracle.local_purity(sites), 1 / len(rows))


class ResourceScalingTests(unittest.TestCase):
    def test_subsystem_limit_is_100_and_cannot_exceed_the_register(self):
        config = SimulationConfig(
            n_qubits=100, subsystem_size=100, methods=("CS", "SI-CS", "ConEnh"),
        )
        config.validate()
        self.assertEqual(config.resource_plan().backend, "sampled")
        for invalid in (
            replace(config, subsystem_size=101),
            replace(config, n_qubits=50, subsystem_size=51),
        ):
            with self.assertRaisesRegex(ValueError, "min\\(100, n_qubits\\)"):
                invalid.validate()

    def test_100_site_purity_and_renyi_run_all_direct_methods(self):
        for state in ("CLUSTER", "GHZ"):
            for prop in ("PURITY", "RENYI2"):
                with self.subTest(state=state, property=prop):
                    config = SimulationConfig(
                        state=state, n_qubits=100, k_rotations=5, subsystem_size=100,
                        property=prop, methods=("CS", "SI-CS", "ConEnh"),
                        post_shots=4, repetitions=1, evaluation_points=2, pilot_shots=0,
                        pauli_samples=64, conenh_candidates=8, conenh_max_cliques=32,
                        progress_details=False, property_clipping=True,
                    )
                    result = run_benchmark(config, show_progress=False)
                    self.assertEqual(len(result.subsystem), 100)
                    self.assertAlmostEqual(result.truth, 1 if prop == "PURITY" else 0)
                    for estimates in result.estimates.values():
                        self.assertTrue(np.all(np.isfinite(estimates)))

    def test_automatic_budgets_and_manual_overrides(self):
        small = SimulationConfig(n_qubits=100, subsystem_size=5)
        large = replace(small, subsystem_size=50, post_shots=10000, methods=("CS", "ConEnh"))
        self.assertEqual(small.resource_plan().backend, "dense")
        self.assertEqual(small.resource_plan().update_every, 1)
        plan = large.resource_plan()
        self.assertEqual(plan.backend, "sampled")
        self.assertGreater(plan.update_every, 1)
        explicit = replace(large, conenh_max_cliques=300, conenh_candidates=17,
                           conenh_update_every=7, pauli_samples=37).resource_plan()
        self.assertEqual((explicit.max_cliques, explicit.candidates, explicit.update_every,
                          explicit.pauli_samples), (300, 17, 7, 37))
        with self.assertRaisesRegex(ValueError, "Dense"):
            replace(large, subsystem_backend="dense").validate()
        with self.assertRaisesRegex(ValueError, "8 subsystem"):
            replace(large, methods=("ConEnh-AGHDO",)).validate()

    def test_100_qubits_50_subsystem_end_to_end_and_reproducibility(self):
        for state in ("CLUSTER", "GHZ"):
            config = SimulationConfig(
                state=state, n_qubits=100, k_rotations=50, subsystem_size=50,
                methods=("CS", "ConEnh"), post_shots=8, repetitions=1,
                evaluation_points=2, pauli_samples=128, conenh_candidates=32,
                conenh_max_cliques=128, pilot_shots=0, progress_details=False,
            )
            events = []
            first = run_benchmark(config, events.append, show_progress=False)
            second = run_benchmark(config, show_progress=False)
            self.assertEqual(first.execution["backend"], "sampled")
            self.assertLessEqual(first.execution["unique_paulis"], 128)
            self.assertEqual(first.shots[-1], 8)
            self.assertEqual(events[-1]["completed"], events[-1]["total"])
            for method in config.methods:
                np.testing.assert_array_equal(first.estimates[method], second.estimates[method])
                self.assertTrue(np.all(np.isfinite(first.estimates[method])))
            self.assertEqual(
                first.execution["streams"]["uniform"]["estimator"], "exact_factorized_uniform"
            )


if __name__ == "__main__":
    unittest.main()
