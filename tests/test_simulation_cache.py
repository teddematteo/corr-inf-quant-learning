import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from con_enh_quant_learning.simulation import SimulationConfig, run_benchmark
from con_enh_quant_learning.simulation_cache import (
    fingerprint,
    load_realization,
    save_realization,
)


class ScientificCacheTests(unittest.TestCase):
    def config(self, method="ConEnh", **kwargs):
        return SimulationConfig(
            n_qubits=4, k_rotations=2, subsystem_size=2, methods=(method,),
            property="MAGIC" if "AGHDO" in method else "RENYI2",
            post_shots=8, repetitions=1, pilot_shots=0, evaluation_points=2,
            theta_range=(0.1, 0.4), aghdo_rank=2, aghdo_feature_densities=(2,),
            aghdo_fit_steps=1, progress_details=False, **kwargs,
        )

    def assert_equivalent(self, config, **changes):
        method, = config.methods
        changed = replace(config, **changes)
        self.assertEqual(fingerprint(config, method, 0), fingerprint(changed, method, 0))
        original = run_benchmark(config, show_progress=False)
        updated = run_benchmark(changed, show_progress=False)
        np.testing.assert_array_equal(original.estimates[method], updated.estimates[method])
        np.testing.assert_array_equal(original.repetition_truths, updated.repetition_truths)

    def test_direct_ignores_neural_legacy_logging_and_overridden_angle_parameters(self):
        config = self.config()
        self.assert_equivalent(
            config, theta=99, aghdo_rank=3, aghdo_fit_steps=10,
            aghdo_learning_rate=0.1, aghdo_batch_size=99, qns_property_samples=11,
            aghdo_feature_densities=(3, 2), aghdo_init_std=0.03,
            generator_max_weight=None, progress_details=True,
            pilot_shots=99, count_pilot_cost=False, magic_clipping=False,
        )

    def test_uniform_and_si_ignore_conenh_only_controls(self):
        for method in ("CS", "SI-CS"):
            with self.subTest(method=method):
                config = self.config(method)
                self.assert_equivalent(
                    config, conenh_purity_weight=3, conenh_generator_weight=2,
                    conenh_commuting_bias=0, conenh_softmax_temperature=2,
                    conenh_local_regularization=0.2, conenh_max_cliques=11,
                    conenh_candidates=11, conenh_update_every=9, conenh_pauli_floor=0.5,
                )
        config = self.config("SI-CS")
        self.assert_equivalent(config, si_cs_exploration=config.conenh_exploration)
        config = replace(config, si_cs_exploration=0.3)
        self.assert_equivalent(config, conenh_exploration=0.9)

    def test_magic_ignores_direct_selector_and_unused_training_parameters(self):
        for method in ("CS-AGHDO", "SI-CS-AGHDO", "ConEnh-AGHDO"):
            with self.subTest(method=method):
                self.assert_equivalent(
                    self.config(method), conenh_purity_weight=3, conenh_generator_weight=2,
                    conenh_commuting_bias=0, conenh_local_regularization=0.2,
                    conenh_max_cliques=11, conenh_update_every=9, conenh_pauli_floor=0.5,
                    conenh_clipping=False, conenh_clip_bound="UCB",
                    aghdo_batch_size=19, qns_property_samples=11,
                    pauli_samples=11, subsystem_backend="sampled", memory_budget_mib=32,
                    subsystem_size=3, subsystem=(0, 1, 2),
                )

    def test_effective_defaults_match_explicit_backend_budgets_geometry_and_angles(self):
        config = self.config()
        plan = config.resource_plan()
        self.assert_equivalent(
            config, subsystem_backend=plan.backend,
            conenh_max_cliques=plan.max_cliques,
            conenh_update_every=plan.update_every,
            pauli_samples=19, pauli_uniform_fraction=0.4,
            conenh_candidates=19, rows=2, columns=2,
        )
        config = replace(config, theta_range=None)
        self.assert_equivalent(config, theta_range=(config.theta, config.theta))
        config = replace(config, rotation_vertices=(0, 3))
        self.assert_equivalent(config, rotation_seed=999)
        config = replace(config, k_rotations=0, rotation_vertices=None)
        self.assert_equivalent(config, rotation_seed=999, theta=0.7, theta_range=(-1, 1))

    def test_relevant_changes_get_separate_keys(self):
        config = self.config(subsystem_backend="sampled")
        changes = (
            {"seed": 13}, {"rotation_seed": 99}, {"theta_range": (0.11, 0.4)},
            {"rotation_vertices": (0, 3)}, {"k_rotations": 3},
            {"subsystem_size": 1}, {"post_shots": 16}, {"evaluation_points": 3},
            {"property_clipping": True}, {"conenh_clipping": False},
            {"conenh_exploration": 0.4}, {"conenh_purity_weight": 2},
            {"conenh_update_every": 2}, {"conenh_max_cliques": 16},
            {"pauli_samples": 16}, {"pauli_uniform_fraction": 0.8},
            {"device": "cuda"},
        )
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(fingerprint(config, "ConEnh", 0),
                                    fingerprint(replace(config, **change), "ConEnh", 0))
        self.assertNotEqual(fingerprint(config, "ConEnh", 0), fingerprint(config, "ConEnh", 1))
        self.assertEqual(fingerprint(config, "ConEnh", 0), fingerprint(
            replace(config, repetitions=10, methods=("SI-CS", "ConEnh", "CS")), "ConEnh", 0,
        ))

    def test_pilot_axis_is_reconstructed_without_new_acquisitions(self):
        config = replace(self.config("ConEnh-AGHDO"), pilot_shots=1)
        result = run_benchmark(config, show_progress=False)
        self.assertGreater(result.pilot_cost, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            save_realization(path, result, "ConEnh-AGHDO", 0, engine="source", seconds=0)
            changed = replace(config, count_pilot_cost=False)
            loaded = load_realization(path, changed, "ConEnh-AGHDO", 0)
        np.testing.assert_array_equal(loaded.shots, result.shots - result.pilot_cost)
        np.testing.assert_array_equal(loaded.estimates["ConEnh-AGHDO"],
                                      result.estimates["ConEnh-AGHDO"])


if __name__ == "__main__":
    unittest.main()
