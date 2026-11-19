import itertools
import unittest
from dataclasses import replace
from math import comb
from unittest.mock import patch

import numpy as np
from test_scaling import exhaustive_pool

from con_enh_quant_learning import METHODS, SimulationConfig, run_benchmark
from con_enh_quant_learning.estimators import IncrementalPauliMomentUStatistic
from con_enh_quant_learning.generators import ghz_generators
from con_enh_quant_learning.measurement import LocalShot, OnlineSubsystemSelector, ShotDesign
from con_enh_quant_learning.pauli import Pauli, axis_at
from con_enh_quant_learning.sampled import (
    PauliPool,
    SampledPauliMomentUStatistic,
    SampledSubsystemSelector,
    axis_masks,
    coverage,
)
from con_enh_quant_learning.states import RotatedSparseState, make_state


class FourthMomentTests(unittest.TestCase):
    def test_dense_and_sampled_equal_explicit_distinct_shot_quadruples(self):
        rng = np.random.default_rng(432)
        sites = (0, 1)
        pool = exhaustive_pool(2)
        dense = IncrementalPauliMomentUStatistic(sites, moment_order=4)
        sampled = SampledPauliMomentUStatistic(pool, moment_order=4)
        features = []
        for _ in range(8):
            basis = rng.integers(0, 3, size=2, dtype=np.int8)
            outcomes = rng.choice(np.array([-1, 1], dtype=np.int8), size=2)
            shot = LocalShot(ShotDesign(basis=basis), outcomes)
            dense.add(shot)
            sampled.add(shot)
            row = []
            for word in pool.paulis:
                covered = all(
                    axis_at(word, q) is None or axis_at(word, q) == basis[q] for q in sites
                )
                sign = np.prod([outcomes[q] for q in sites if word.support >> q & 1])
                row.append(sign * 3**word.weight if covered else 0)
            features.append(row)
            if len(features) >= 4:
                terms = sum(
                    np.prod(np.asarray(features)[list(indices)], axis=0)
                    for indices in itertools.combinations(range(len(features)), 4)
                ) / comb(len(features), 4)
                expected = (1 + terms.sum()) / 4
                self.assertAlmostEqual(dense.value(), expected, places=11)
                self.assertAlmostEqual(sampled.value(), expected, places=11)

    def test_order_four_uses_distinct_copies_and_optional_projection(self):
        statistic = IncrementalPauliMomentUStatistic((0,), moment_order=4)
        for sign in (1, 1, 1):
            statistic.add(LocalShot(ShotDesign(np.array([0])), np.array([sign])))
        self.assertTrue(np.isnan(statistic.value()))
        statistic.add(LocalShot(ShotDesign(np.array([0])), np.array([-1])))

        self.assertEqual(statistic.value(), -40)
        self.assertEqual(statistic.value(np.ones(4)), 0.5)

    def test_raw_fourth_moment_unbiased_under_history_dependent_acquisition(self):
        oracle = make_state("GHZ", 1, 1, theta=0.193)
        expected = 2**(-oracle.stabilizer_renyi_magic())
        average = 0.0
        total_probability = 0.0

        for records in itertools.product(itertools.product(range(3), (-1, 1)), repeat=4):
            statistic = IncrementalPauliMomentUStatistic((0,), moment_order=4)
            probability = 1.0
            previous = 1
            for axis, outcome in records:
                policy = np.array([0.2, 0.3, 0.5]) if previous == 1 else np.array([0.6, 0.3, 0.1])
                word = (Pauli(1), Pauli(1, 1), Pauli(0, 1))[axis]
                probability *= policy[axis] * (1 + outcome * oracle.expectation(word)) / 2
                design = ShotDesign(
                    basis=np.array([axis]), local_sites=(0,),
                    local_inclusion=np.r_[1.0, policy],
                )
                statistic.add(LocalShot(design, np.array([outcome])))
                previous = outcome
            average += probability * statistic.value()
            total_probability += probability
        self.assertAlmostEqual(total_probability, 1.0, places=12)
        self.assertAlmostEqual(average, expected, places=12)

    def test_shared_acquisition_uses_squared_bounds_for_magic(self):
        selectors = [OnlineSubsystemSelector(
            ghz_generators(3), (0, 1, 2), np.random.default_rng(9), n_qubits=3,
            pauli_floor=0, moment_order=order,
        ) for order in (2, 4)]
        for selector in selectors:
            selector.generator_counts[:] = (8, 4, 4)
            selector.generator_sums[:] = (4, 4, 0)
        np.testing.assert_allclose(
            selectors[1].coverage_weights(), selectors[0].coverage_weights()**2,
        )


class WidePauliTests(unittest.TestCase):
    def test_masks_and_exact_inclusions_cross_uint64_boundary(self):
        n = 100
        generators = ghz_generators(n)
        pool = PauliPool.sample(generators, tuple(range(n)), n, 32, 0.2,
                                np.random.default_rng(833))
        self.assertEqual(pool.x.dtype, object)
        selector = SampledSubsystemSelector(
            generators, pool, np.random.default_rng(97), n_qubits=n, candidates=8,
            max_cliques=128, update_every=1, exploration=0.2, temperature=0.3,
            pauli_floor=0.1, commuting_bias=1, purity_weight=1, generator_weight=1,
            local_regularization=0.05, moment_order=4,
        )
        design = selector.draw()
        for index, word in enumerate(pool.paulis):
            covering = np.array([all(
                axis_at(word, q) is None or basis[q] == axis_at(word, q) for q in range(n)
            ) for basis in selector._bases])
            expected = 0.8 * (selector.policy @ covering) + 0.2 * 3.0**(-word.weight)
            self.assertAlmostEqual(design.local_inclusion[index], expected, places=13)
        original = design.local_inclusion.copy()
        selector.observe(design, np.ones(len(selector.parity_probes(design.basis)), dtype=np.int8))
        selector.draw()
        np.testing.assert_array_equal(original, design.local_inclusion)


        bases = np.zeros((2, n), dtype=np.int8)
        bases[1, 64] = 2
        word_axes = np.zeros((1, n), dtype=np.int8)
        self.assertEqual(coverage(axis_masks(bases), axis_masks(word_axes)).tolist(),
                         [[True], [False]])

    def test_wide_fourth_moment_matches_explicit_four_shot_parities(self):
        n = 100
        sites = tuple(range(n))
        support = sum(1 << q for q in (0, 63, 64, 99))
        word = Pauli(support)
        x, z = np.array([support], dtype=object), np.array([0], dtype=object)
        pool = PauliPool(sites, (word,), np.array([[support, 0, 0]], dtype=object),
                         x, z, x | z, np.ones(1), np.array([3.0**-4]), 1, 0)
        statistic = SampledPauliMomentUStatistic(pool, moment_order=4)
        for sign in (1, 1, -1, 1):
            outcomes = np.ones(n, dtype=np.int8)
            outcomes[99] = sign
            statistic.add(LocalShot(ShotDesign(np.zeros(n, dtype=np.int8)), outcomes))
        self.assertAlmostEqual(statistic.value() * 2**n, 1 - 3**16)


class UnifiedMagicTests(unittest.TestCase):
    def config(self, **kwargs):
        return SimulationConfig(
            state="GHZ", n_qubits=4, k_rotations=2, subsystem_size=2,
            property="MAGIC", methods=("CS", "ConEnh"), post_shots=8,
            repetitions=1, evaluation_points=2, pilot_shots=0,
            progress_details=False, **kwargs,
        )

    def test_direct_magic_uses_one_local_record_per_shot_without_extra_pauli_copies(self):
        original = RotatedSparseState.sample_outcomes
        with patch.object(RotatedSparseState, "sample_pauli", side_effect=AssertionError),\
             patch.object(RotatedSparseState, "sample_outcomes", autospec=True,
                          side_effect=original) as outcomes:
            result = run_benchmark(self.config(), show_progress=False)
        self.assertEqual(outcomes.call_count, 16)
        self.assertEqual(result.execution["moment_order"], 4)

    def test_all_families_support_global_magic_at_100_qubits(self):
        for state in ("CLUSTER", "GHZ"):
            config = replace(self.config(), state=state, n_qubits=100, k_rotations=50,
                             pauli_samples=32, conenh_candidates=8, conenh_max_cliques=128)
            events = []
            result = run_benchmark(config, events.append, show_progress=False)
            self.assertEqual(result.execution["backend"], "sampled")
            self.assertEqual(result.execution["direct_register_size"], 100)
            self.assertEqual(events[-1]["completed"], events[-1]["total"])
            for values in result.estimates.values():
                self.assertTrue(np.all(np.isfinite(values)))
            diagnostics = result.execution["streams"]["bound"]
            self.assertIn("sampled_paulis_with_quartets", diagnostics)
            self.assertEqual(diagnostics["moment_order"], 4)

    def test_new_magic_score_settings_leave_legacy_aghdo_predictions_unchanged(self):
        config = replace(self.config(), methods=METHODS, aghdo_fit_steps=1, qns_property_samples=16)
        before = run_benchmark(config, show_progress=False)
        changed = replace(config, conenh_generator_weight=4, conenh_purity_weight=0.1,
                          conenh_commuting_bias=0, conenh_pauli_floor=1,
                          conenh_update_every=4, conenh_max_cliques=16)
        after = run_benchmark(changed, show_progress=False)
        for method in ("CS-AGHDO", "ConEnh-AGHDO"):
            np.testing.assert_array_equal(before.estimates[method], after.estimates[method])

    def test_global_magic_resource_plan_uses_n_not_local_subsystem_size(self):
        config = replace(self.config(), n_qubits=100, subsystem_size=2)
        self.assertEqual(config.resource_plan().backend, "sampled")
        self.assertEqual(config.resource_plan().pauli_samples, 3200)
        with self.assertRaisesRegex(ValueError, "Dense"):
            replace(config, subsystem_backend="dense").validate()

        legacy = replace(config, state="CLUSTER", methods=("ConEnh-AGHDO",),
                         subsystem_backend="dense")
        legacy.validate()
        self.assertEqual(legacy.resource_plan().backend, "magic")


if __name__ == "__main__":
    unittest.main()
