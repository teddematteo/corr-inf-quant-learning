import itertools
import unittest
from dataclasses import replace

import numpy as np
from test_scaling import exhaustive_pool

from con_enh_quant_learning import SimulationConfig, run_benchmark
from con_enh_quant_learning.estimators import IncrementalPauliMomentUStatistic
from con_enh_quant_learning.generators import ghz_generators
from con_enh_quant_learning.measurement import LocalShot, OnlineSubsystemSelector
from con_enh_quant_learning.pauli import Pauli, axis_at
from con_enh_quant_learning.sampled import (
    PauliPool,
    SampledPauliMomentUStatistic,
    SampledSubsystemSelector,
)
from con_enh_quant_learning.stabilizer_informed import (
    StabilizerInformedSelector,
    stabilizer_bases,
)
from con_enh_quant_learning.states import make_state


def covers(basis, word):
    return all(
        axis_at(word, q) is None or basis[q] == axis_at(word, q)
        for q in range(len(basis))
    )


class StabilizerInformedDistributionTests(unittest.TestCase):
    def test_compatible_groups_cover_every_generator_and_balance_ghz(self):
        for family in ("GHZ", "CLUSTER"):
            oracle = make_state(family, 100, 50)
            bases = stabilizer_bases(oracle.generators, 100)
            self.assertEqual(bases.shape, (2, 100))
            self.assertTrue(all(any(covers(b, g) for b in bases) for g in oracle.generators))
        ghz = stabilizer_bases(ghz_generators(100), 100)
        np.testing.assert_array_equal(ghz, np.array([[0] * 100, [2] * 100]))

        self.assertEqual(len(stabilizer_bases((Pauli(3), Pauli(0, 3)), 2)), 2)

        np.testing.assert_array_equal(
            stabilizer_bases((Pauli(0, 1), Pauli(0, 2), Pauli(0, 1)), 2), [[2, 2]],
        )

    def test_exhaustive_basis_law_and_all_pauli_inclusions(self):
        eta = 0.23
        selector = StabilizerInformedSelector(
            ghz_generators(3), 3, np.random.default_rng(90), exploration=eta,
            subsystem=(0, 1, 2),
        )
        template = selector.draw()
        bases = np.array(list(itertools.product(range(3), repeat=3)))
        law = np.array([
            eta / 27 + (1 - eta) * sum(
                w for b, w in zip(selector.bases, selector.weights) if np.array_equal(b, basis)
            ) for basis in bases
        ])
        self.assertAlmostEqual(law.sum(), 1)
        pool = exhaustive_pool(3)
        for word in (Pauli(), *pool.paulis):
            expected = law @ np.array([covers(b, word) for b in bases])
            self.assertAlmostEqual(template.coverage_probability(word), expected, places=14)

        counts = np.zeros(27)
        for _ in range(12000):
            b = selector.draw().basis
            counts[int(b @ np.array([9, 3, 1]))] += 1
        np.testing.assert_allclose(counts / counts.sum(), law, atol=0.015, rtol=0)

    def test_policy_is_static_and_eta_one_is_uniform(self):
        args = (ghz_generators(3), 3)
        selectors = [StabilizerInformedSelector(
            *args, np.random.default_rng(123), subsystem=(2, 0), exploration=0.17,
        ) for _ in range(2)]
        original = selectors[0].draw().local_inclusion.copy()
        selectors[1].draw()
        for shot in range(40):
            designs = [s.draw() for s in selectors]
            np.testing.assert_array_equal(designs[0].basis, designs[1].basis)
            for index, (selector, design) in enumerate(zip(selectors, designs)):
                parities = np.full(len(selector.parity_probes(design.basis)),
                                   1 if index == 0 else (-1)**shot, dtype=np.int8)
                selector.observe(design, parities)
                np.testing.assert_array_equal(design.local_inclusion, original)
        uniform = StabilizerInformedSelector(
            *args, np.random.default_rng(3), subsystem=(0, 1, 2), exploration=1,
        ).draw()
        for word in exhaustive_pool(3).paulis:
            self.assertAlmostEqual(uniform.coverage_probability(word), 3.0**-word.weight)

    def test_sampled_inclusions_support_noncontiguous_and_wide_registers(self):
        for n, sites in ((5, (4, 0, 2)), (100, tuple(range(100)))):
            generators = ghz_generators(n)
            pool = PauliPool.sample(generators, sites, n, 64, 0.2, np.random.default_rng(7))
            selector = StabilizerInformedSelector(
                generators, n, np.random.default_rng(9), subsystem=sites,
                pool=pool, exploration=0.31,
            )
            design = selector.draw()
            for i, word in enumerate(pool.paulis):
                expected = 0.31 * 3.0**-word.weight + 0.69 * sum(
                    w for b, w in zip(selector.bases, selector.weights) if covers(b, word)
                )
                self.assertAlmostEqual(design.local_inclusion[i], expected, places=14)
                self.assertAlmostEqual(design.coverage_probability(word), expected, places=14)
            self.assertTrue(np.all(design.local_inclusion > 0))
            if n == 100:
                self.assertEqual(pool.x.dtype, object)


class StabilizerInformedEstimatorTests(unittest.TestCase):
    def test_raw_second_and_fourth_moments_are_unbiased_over_complete_experiment(self):
        oracle = make_state("GHZ", 1, 1, theta=0.193)
        selector = StabilizerInformedSelector(
            oracle.generators, 1, np.random.default_rng(1), subsystem=(0,), exploration=0.4,
        )
        template = selector.draw()
        words = (Pauli(1), Pauli(1, 1), Pauli(0, 1))
        mass = template.local_inclusion[1:]
        average = {2: 0.0, 4: 0.0}
        total = 0.0
        for records in itertools.product(itertools.product(range(3), (-1, 1)), repeat=4):
            statistics = {order: IncrementalPauliMomentUStatistic((0,), order) for order in (2, 4)}
            probability = 1.0
            for axis, outcome in records:
                probability *= mass[axis] * (1 + outcome * oracle.expectation(words[axis])) / 2
                shot = LocalShot(replace(template, basis=np.array([axis])), np.array([outcome]))
                for statistic in statistics.values():
                    statistic.add(shot)
            total += probability
            for order, statistic in statistics.items():
                average[order] += probability * statistic.value()
        self.assertAlmostEqual(total, 1.0, places=12)
        self.assertAlmostEqual(average[2], oracle.local_purity((0,)), places=12)
        self.assertAlmostEqual(average[4], 2**-oracle.stabilizer_renyi_magic(), places=12)

    def test_dense_sampled_statistics_and_bounds_match_conenh(self):
        sites = (2, 0)
        generators = ghz_generators(3)

        local_pool = exhaustive_pool(2)
        words = tuple(Pauli(
            sum(((g.x >> j) & 1) << q for j, q in enumerate(sites)),
            sum(((g.z >> j) & 1) << q for j, q in enumerate(sites)),
        ) for g in local_pool.paulis)
        pool = replace(local_pool, subsystem=sites, paulis=words)
        dense = StabilizerInformedSelector(
            generators, 3, np.random.default_rng(34), subsystem=sites,
        )
        sampled = StabilizerInformedSelector(
            generators, 3, np.random.default_rng(34), subsystem=sites, pool=pool,
        )
        bound = OnlineSubsystemSelector(generators, sites, np.random.default_rng(1), n_qubits=3)
        sampled_bound = SampledSubsystemSelector(
            generators, pool, np.random.default_rng(1), n_qubits=3, candidates=8,
            max_cliques=16, update_every=1, exploration=0.2, temperature=0.3,
            pauli_floor=0.1, commuting_bias=1, purity_weight=1, generator_weight=1,
            local_regularization=0.05,
        )
        rng = np.random.default_rng(70)
        stats = {order: (IncrementalPauliMomentUStatistic(sites, order),
                         SampledPauliMomentUStatistic(pool, order)) for order in (2, 4)}
        codes = [sum(
            (0 if axis_at(g, q) is None else axis_at(g, q) + 1) * 4**j
            for j, q in enumerate(sites)
        ) for g in words]
        for _ in range(12):
            d, s = dense.draw(), sampled.draw()
            np.testing.assert_array_equal(d.basis, s.basis)
            np.testing.assert_allclose(d.local_inclusion[codes], s.local_inclusion)
            outcomes = rng.choice(np.array([-1, 1], dtype=np.int8), size=2)
            parities = rng.choice(np.array([-1, 1], dtype=np.int8),
                                  size=len(dense.parity_probes(d.basis)))
            for selector, design in ((dense, d), (sampled, s), (bound, d), (sampled_bound, s)):
                selector.observe(design, parities)
            for margin in (False, True):
                np.testing.assert_array_equal(dense.pauli_bounds(margin),
                                              bound.pauli_bounds(margin))
                np.testing.assert_array_equal(sampled.pauli_bounds(margin),
                                              sampled_bound.pauli_bounds(margin))
                np.testing.assert_array_equal(dense.pauli_bounds(margin)[codes],
                                              sampled.pauli_bounds(margin))
            for order, (ds, ss) in stats.items():
                ds.add(LocalShot(d, outcomes))
                ss.add(LocalShot(s, outcomes))
                if ds.shot_count >= order:
                    self.assertAlmostEqual(ds.value(), ss.value(), places=10)
                    self.assertAlmostEqual(
                        ds.value(dense.pauli_bounds()**(order // 2)),
                        ss.value(sampled.pauli_bounds()**(order // 2)), places=10,
                    )


class StabilizerInformedBenchmarkTests(unittest.TestCase):
    def config(self, **kwargs):
        return SimulationConfig(
            n_qubits=4, k_rotations=2, subsystem_size=2, post_shots=8, repetitions=1,
            evaluation_points=2, pilot_shots=1, progress_details=False,
            aghdo_fit_steps=1, aghdo_rank=2, aghdo_feature_densities=(2,), **kwargs,
        )

    def test_stream_events_replay_and_legacy_predictions_do_not_depend_on_si(self):
        for family, prop in itertools.product(("GHZ", "CLUSTER"), ("PURITY", "MAGIC")):
            config = self.config(state=family, property=prop)
            events = []
            result = run_benchmark(config, events.append, show_progress=False)
            legacy = run_benchmark(replace(
                config, methods=("CS", "ConEnh", "CS-AGHDO", "ConEnh-AGHDO"),
            ), show_progress=False)
            replay = run_benchmark(config, show_progress=False)
            for name, values in result.estimates.items():
                np.testing.assert_array_equal(values, replay.estimates[name])
                self.assertTrue(np.all(np.isfinite(values)))
                if name in legacy.estimates:
                    np.testing.assert_array_equal(values, legacy.estimates[name])
            self.assertEqual(events[-1]["completed"], events[-1]["total"])
            for event in events:
                for name, value in event["estimates"].items():
                    point = list(result.shots).index(event["shots"])
                    self.assertEqual(value, result.estimates[name][event["repetition"], point])
            diag = result.execution["streams"]["stabilizer"]
            self.assertEqual(diag["weights"], [0.5, 0.5])
            self.assertEqual(diag["policy_updates"], 0)
            self.assertFalse(diag["exact_generator_means_supplied"])

    def test_hundred_qubit_sampled_and_focused_routes_and_resource_plans(self):
        for family in ("GHZ", "CLUSTER"):
            config = replace(
                self.config(state=family, property="MAGIC", methods=("SI-CS", "SI-CS-AGHDO")),
                n_qubits=100, k_rotations=50, pauli_samples=32, si_cs_exploration=0.3,
            )
            self.assertEqual(config.resource_plan().backend, "sampled")
            result = run_benchmark(config, show_progress=False)
            for values in result.estimates.values():
                self.assertTrue(np.all(np.isfinite(values)))
            for key in ("stabilizer", "stabilizer_aghdo"):
                self.assertEqual(result.execution["streams"][key]["exploration"], 0.3)
            focused = replace(config, methods=("SI-CS-AGHDO",), subsystem_backend="dense")
            focused.validate()
            self.assertEqual(focused.resource_plan().backend, "magic")
            solo = run_benchmark(focused, show_progress=False)
            np.testing.assert_array_equal(solo.estimates["SI-CS-AGHDO"],
                                          result.estimates["SI-CS-AGHDO"])

    def test_invalid_exploration(self):
        for eta in (0, -0.2, 1.1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "si_cs_exploration"):
                replace(self.config(), si_cs_exploration=eta).validate()
            with self.assertRaisesRegex(ValueError, "exploration"):
                StabilizerInformedSelector(ghz_generators(2), 2, np.random.default_rng(1),
                                          exploration=eta)


if __name__ == "__main__":
    unittest.main()
