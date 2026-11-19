import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from con_enh_quant_learning import SimulationConfig, run_benchmark
from con_enh_quant_learning._compat import bit_count, dataclass, strict_zip
from con_enh_quant_learning.compute import ComputeBackend


class KernelCompatibilityTests(unittest.TestCase):
    def test_backports_preserve_counts_lengths_and_dataclass_behavior(self):
        self.assertEqual(bit_count((1 << 100) | 7), 4)
        self.assertEqual(bit_count(-7), 3)
        self.assertEqual(list(strict_zip([1, 2], [3, 4])), [(1, 3), (2, 4)])
        with self.assertRaises(ValueError):
            list(strict_zip([1], [2, 3]))

        @dataclass(slots=True)
        class Value:
            count: int

        self.assertEqual(replace(Value(1), count=2), Value(2))

    def test_auto_reports_fallback_and_explicit_cuda_fails(self):
        with patch.dict("sys.modules", {"torch": None}):
            backend = ComputeBackend("auto")
            self.assertEqual(backend.device, "cpu")
            self.assertIn("torch", backend.reason)
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                ComputeBackend("cuda")

    def test_invalid_device_rejected_before_simulation(self):
        with self.assertRaisesRegex(ValueError, "device"):
            SimulationConfig(device="invalid").validate()


@unittest.skipUnless(os.environ.get("CONENH_TEST_CUDA") == "1", "requires a GPU allocation")
class CudaRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ComputeBackend("cuda")

    def test_cuda_reductions_and_clique_compatibility_match_numpy(self):
        rng = np.random.default_rng(35)
        matrix = rng.normal(size=(37, 21))
        vector = rng.normal(size=21)
        np.testing.assert_allclose(self.backend.matvec(matrix, vector), matrix @ vector,
                                   rtol=1e-12, atol=1e-12)
        weights = rng.normal(size=37)
        np.testing.assert_allclose(self.backend.matvec(matrix, weights, transpose=True),
                                   matrix.T @ weights, rtol=1e-12, atol=1e-12)
        bases = rng.integers(0, 3, (27, 9), dtype=np.int8)
        axes = rng.integers(-1, 3, (43, 9), dtype=np.int8)
        np.testing.assert_array_equal(self.backend.clique_fits(bases, axes),
                                      ComputeBackend().clique_fits(bases, axes))

    def test_dense_and_sampled_cuda_benchmarks_match_cpu(self):
        for backend in ("dense", "sampled"):
            for prop in ("PURITY", "RENYI2", "MAGIC"):
                with self.subTest(backend=backend, property=prop):
                    config = SimulationConfig(
                        n_qubits=4, k_rotations=2, subsystem_size=2,
                        methods=("CS", "ConEnh"), property=prop,
                        post_shots=24, repetitions=1, evaluation_points=3,
                        pilot_shots=0, subsystem_backend=backend, pauli_samples=64,
                        conenh_candidates=16, conenh_max_cliques=32,
                    )
                    cpu = run_benchmark(config, show_progress=False)
                    gpu = run_benchmark(replace(config, device="cuda"), show_progress=False)
                    self.assertEqual(gpu.execution["compute"]["device"], "cuda")
                    self.assertGreater(gpu.execution["compute"]["gpu_operations"], 0)
                    for method in config.methods:
                        np.testing.assert_allclose(gpu.estimates[method], cpu.estimates[method],
                                                   rtol=1e-10, atol=1e-10, equal_nan=True)
