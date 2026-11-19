import itertools
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import torch

from con_enh_quant_learning import SimulationConfig, run_benchmark
from con_enh_quant_learning.aghdo import AGHDO, AGHDONetwork, FocusedAGHDO
from con_enh_quant_learning.logical import magic_focus, sample_focus_parities
from con_enh_quant_learning.measurement import covered_probes, probe_axes
from con_enh_quant_learning.pauli import Pauli, axis_at
from con_enh_quant_learning.states import STATE_TYPES, make_state


def pauli_matrix(word, n):
    matrices = (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))
    result = np.ones((1, 1), dtype=complex)

    for site in reversed(range(n)):
        axis = axis_at(word, site)
        result = np.kron(result, np.eye(2) if axis is None else matrices[axis])
    return result


class NeuralArchitectureTests(unittest.TestCase):
    def test_trace_positivity_hermiticity_and_exact_chain_rule(self):
        for rank in (1, 2, 8):
            with self.subTest(rank=rank):
                model = AGHDONetwork(3, rank, (3, 2), seed=21)
                sigma = model.basis_configurations()
                rho = model.dense_density_matrix().detach().numpy()
                np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
                self.assertAlmostEqual(np.trace(rho).real, 1, places=12)
                self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(), -1e-13)
                np.testing.assert_allclose(
                    np.diag(rho).real, model.diagonal_probability(sigma).detach(), atol=1e-14
                )
                amplitude = model.conditional_amplitudes(sigma)
                np.testing.assert_allclose(
                    amplitude.abs().square().sum((-1, -2)).detach(), 1, atol=1e-14
                )
                left = sigma.repeat_interleave(len(sigma), dim=0)
                right = sigma.repeat((len(sigma), 1))
                elements = model.density_elements(left, right).reshape(8, 8).detach().numpy()
                np.testing.assert_allclose(elements, rho, atol=1e-14)
                if rank == 1:
                    self.assertAlmostEqual(np.sum(np.abs(rho) ** 2), 1, places=12)

    def test_mask_excludes_current_and_future_inputs(self):
        model = AGHDONetwork(5, 3, (4, 2), seed=5)
        configurations = torch.zeros((6, 5), dtype=torch.long)
        for site in range(5):
            configurations[site + 1, site] = 1
        amplitudes = model.conditional_amplitudes(configurations).detach().numpy()
        for output in range(5):
            for changed in range(output, 5):
                np.testing.assert_array_equal(
                    amplitudes[0, output], amplitudes[changed + 1, output]
                )
        loss = model.conditional_amplitudes(configurations).real.sum()
        loss.backward()
        for layer in model.layers:
            self.assertTrue(torch.all(layer.weight_real.grad[layer.mask == 0] == 0))

    def test_direct_sampling_matches_diagonal_and_pair_proposal(self):
        model = AGHDONetwork(2, 2, (3, 2), seed=7)
        rng = np.random.default_rng(8)
        samples = model.sample_diagonal(16000, rng).cpu().numpy()
        codes = samples @ np.array([2, 1])
        probability = model.diagonal_probability(model.basis_configurations()).detach().numpy()
        np.testing.assert_allclose(
            np.bincount(codes, minlength=4) / len(codes), probability, atol=0.015
        )
        sigma, eta, log_mass = model.sample_pairs(10000, rng, alpha=0.4)
        left = model.diagonal_probability(sigma)
        conditionals = model.conditional_amplitudes(eta).abs().square().sum(-1)
        chosen = conditionals.gather(2, eta[:, :, None]).squeeze(-1)
        expected = left * (0.4 * chosen + 0.6 * (sigma == eta).to(torch.float64)).prod(-1)
        np.testing.assert_allclose(log_mass.exp(), expected.detach(), rtol=1e-12)
        importance = model.density_elements(sigma, eta).abs().square() / log_mass.exp()
        rho = model.dense_density_matrix()
        self.assertAlmostEqual(
            importance.mean().item(), rho.abs().square().sum().item(), delta=0.02
        )

    def test_neural_measurement_likelihood_learns_data(self):
        model = AGHDO(1, rank=2, learning_rate=0.002, seed=23, feature_densities=(4, 2))
        for _ in range(16):
            model.add_measurement(np.array([2]), np.array([-1]))
        initial = model.density_matrix()[1, 1].real
        model.fit(40, 16)
        final = model.density_matrix()[1, 1].real
        self.assertGreater(final, initial)
        self.assertGreater(final, 0.97)
        self.assertTrue(np.isfinite(model.last_loss))
        self.assertEqual(model.record_count, 16)

    def test_large_network_queries_do_not_allocate_a_density_matrix(self):
        model = AGHDONetwork(100, 2, (2, 1), seed=8)
        with patch.object(model, "dense_density_matrix", side_effect=AssertionError):
            x = torch.zeros((2, 100), dtype=torch.long)
            y = x.clone()
            y[1, 99] = 1
            result = model.density_elements(x, y)
            self.assertTrue(torch.isfinite(result).all())
            self.assertEqual(result.shape, (2,))
        with self.assertRaisesRegex(ValueError, "Dense"):
            model.dense_density_matrix()

    def test_seed_does_not_modify_global_torch_rng(self):
        before = torch.random.get_rng_state().clone()
        a, b = AGHDONetwork(2, seed=17), AGHDONetwork(2, seed=17)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for left, right in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(left, right))


class LogicalFocusTests(unittest.TestCase):
    def test_clifford_images_obey_pauli_algebra_for_both_families(self):
        for state in STATE_TYPES:
            n = 4
            oracle = make_state(state, n, 0)
            focus = magic_focus(state, oracle.adjacency)
            self.assertEqual(len(focus), 3 * n)
            matrices = [item.sign * pauli_matrix(item.pauli, n) for item in focus]
            for mode in range(n):
                x, y, z = matrices[3 * mode : 3 * mode + 3]
                np.testing.assert_allclose(x @ y, 1j * z, atol=1e-14)
                np.testing.assert_allclose(x @ x, np.eye(1 << n), atol=1e-14)
                for other in range(mode):
                    for left, right in itertools.product(
                        matrices[3 * mode : 3 * mode + 3], matrices[3 * other : 3 * other + 3]
                    ):
                        np.testing.assert_allclose(left @ right, right @ left, atol=1e-14)
            self.assertEqual(tuple(focus[3 * q].pauli for q in range(n)), oracle.generators)

    def test_focus_is_independent_of_unknown_rotations_and_has_correct_means(self):
        for state in STATE_TYPES:
            oracle = make_state(state, 5, 2, theta=0.173, rotation_vertices=(0, 3))
            focus = magic_focus(state, oracle.adjacency)
            reference = make_state(state, 5, 0)
            self.assertEqual(focus, magic_focus(state, reference.adjacency))
            expected = np.array([[1, 0, 0]] * 5, dtype=float)
            expected[[0, 3], 0] = np.cos(2 * 0.173)
            expected[[0, 3], 1] = -np.sin(2 * 0.173)
            actual = np.array([item.sign * oracle.expectation(item.pauli) for item in focus])
            np.testing.assert_allclose(actual.reshape(5, 3), expected, atol=1e-14)

    def test_joint_focus_law_matches_physical_correlators(self):
        for state in STATE_TYPES:
            oracle = make_state(state, 4, 2, theta=0.247)
            focus = magic_focus(state, oracle.adjacency)
            sites, axes = probe_axes(tuple(item.pauli for item in focus), 4)
            for basis in (np.zeros(4, dtype=int), np.full(4, 2), np.array([0, 1, 2, 0])):
                chosen = tuple(focus[i] for i in np.flatnonzero(covered_probes(basis, sites, axes)))
                if len(chosen) < 2:
                    continue
                rng = np.random.default_rng(91)
                records = np.array(
                    [sample_focus_parities(oracle, chosen, rng) for _ in range(5000)]
                )
                for first, second in itertools.combinations(range(len(chosen)), 2):
                    left, right = chosen[first].pauli, chosen[second].pauli


                    exact = oracle.expectation(left ^ right)
                    actual = np.mean(records[:, first] * records[:, second])
                    self.assertAlmostEqual(actual, exact, delta=0.06)

    def test_product_neural_model_matches_full_fourth_pauli_sum(self):
        model = FocusedAGHDO(3, rank=2, seed=45, feature_densities=(3, 2))
        rho = model.network.dense_density_matrix().detach().numpy()
        fourth = 0.0
        for x, z in itertools.product(range(8), repeat=2):

            fourth += np.trace(rho @ pauli_matrix(Pauli(x, z), 3)).real ** 4
        self.assertAlmostEqual(
            model.sample_magic(1, np.random.default_rng(1), clip=False),
            -np.log2(fourth / 8),
            places=12,
        )

    def test_focused_training_uses_only_record_counts_and_learns_parity_signs(self):
        model = FocusedAGHDO(2, rank=2, seed=6, learning_rate=0.002, feature_densities=(4, 2))
        for _ in range(24):
            model.add_encoded(0, 0, 1)
            model.add_encoded(1, 1, -1)
        initial = model.bloch_vectors()
        model.fit(60)
        final = model.bloch_vectors()
        self.assertGreater(final[0, 0], initial[0, 0])
        self.assertLess(final[1, 1], initial[1, 1])
        self.assertGreater(final[0, 0], 0.98)
        self.assertLess(final[1, 1], -0.98)
        self.assertFalse(hasattr(model, "raw"))
        self.assertFalse(hasattr(model, "factor"))

    def test_hundred_qubit_magic_uses_common_neural_model_and_300_observables(self):
        for state in STATE_TYPES:
            config = SimulationConfig(
                state=state,
                n_qubits=100,
                k_rotations=5,
                subsystem_size=2,
                property="MAGIC",
                methods=("CS-AGHDO", "ConEnh-AGHDO"),
                post_shots=8,
                pilot_shots=1,
                repetitions=1,
                evaluation_points=2,
                aghdo_fit_steps=1,
                aghdo_rank=2,
                aghdo_feature_densities=(3, 2),
            )
            result = run_benchmark(config, show_progress=False)
            self.assertEqual(result.execution["aghdo"]["focus_observables"], 300)
            self.assertFalse(result.execution["aghdo"]["rotation_sites_supplied_to_learner"])
            for stream in result.execution["streams"].values():
                self.assertEqual(stream["focus_observables"], 300)
                self.assertGreater(stream["neural_parameters"], 300)
                self.assertGreater(stream["records"], 0)
            for values in result.estimates.values():
                self.assertTrue(np.all(np.isfinite(values)))

    @unittest.skipUnless(os.environ.get("CONENH_TEST_CUDA") == "1", "requires GPU allocation")
    def test_hundred_qubit_training_actually_runs_on_cuda(self):
        for state in STATE_TYPES:
            config = SimulationConfig(
                state=state,
                n_qubits=100,
                k_rotations=5,
                subsystem_size=2,
                device="cuda",
                property="MAGIC",
                methods=("CS-AGHDO", "ConEnh-AGHDO"),
                post_shots=16,
                pilot_shots=0,
                repetitions=1,
                evaluation_points=3,
                aghdo_fit_steps=2,
                aghdo_rank=2,
                aghdo_feature_densities=(4, 2),
            )
            cpu = run_benchmark(replace(config, device="cpu"), show_progress=False)
            gpu = run_benchmark(config, show_progress=False)
            for stream in gpu.execution["streams"].values():
                self.assertEqual(stream["aghdo_device"], "cuda:0")
            for method in config.methods:
                np.testing.assert_allclose(
                    gpu.estimates[method], cpu.estimates[method], rtol=1e-8, atol=1e-8
                )


if __name__ == "__main__":
    unittest.main()
