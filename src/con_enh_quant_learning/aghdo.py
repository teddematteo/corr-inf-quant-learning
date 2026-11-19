"""Neural AGHDO of arXiv:2206.13488v1, Eqs. (13)--(17), Appendix E."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from ._compat import bit_count, strict_zip

_EIGENVECTORS = (
    np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    np.array([[1, 1j], [1, -1j]], dtype=complex) / np.sqrt(2),
    np.eye(2, dtype=complex),
)


@lru_cache(maxsize=4096)
def _measurement_ket(basis: tuple[int, ...], outcomes: tuple[int, ...]) -> np.ndarray:
    vector = np.ones(1, dtype=complex)
    for axis, outcome in strict_zip(basis, outcomes):
        vector = np.kron(vector, _EIGENVECTORS[axis][0 if outcome > 0 else 1])
    return vector


def _selu(value: torch.Tensor) -> torch.Tensor:
    return torch.complex(functional.selu(value.real), functional.selu(value.imag))


class ComplexMaskedDense(nn.Module):
    """A complex dense layer with an immutable autoregressive connectivity mask."""

    def __init__(self, mask: np.ndarray, generator: torch.Generator, std: float) -> None:
        super().__init__()
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float64))
        self.weight_real = nn.Parameter(torch.empty(mask.shape, dtype=torch.float64))
        self.weight_imag = nn.Parameter(torch.empty(mask.shape, dtype=torch.float64))
        self.bias_real = nn.Parameter(torch.empty(mask.shape[0], dtype=torch.float64))
        self.bias_imag = nn.Parameter(torch.empty(mask.shape[0], dtype=torch.float64))
        for value in self.parameters():
            nn.init.trunc_normal_(value, std=std, a=-2 * std, b=2 * std, generator=generator)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weight = torch.complex(self.weight_real * self.mask, self.weight_imag * self.mask)
        bias = torch.complex(self.bias_real, self.bias_imag)
        return values @ weight.T + bias


class _ComplexSiteDense(nn.Module):
    """Block diagonal masked dense layer, stored without its zero off-site blocks."""

    def __init__(self, sites, inputs, outputs, generator, std):
        super().__init__()
        self.weight_real = nn.Parameter(torch.empty(sites, inputs, outputs, dtype=torch.float64))
        self.weight_imag = nn.Parameter(torch.empty(sites, inputs, outputs, dtype=torch.float64))
        self.bias_real = nn.Parameter(torch.empty(sites, outputs, dtype=torch.float64))
        self.bias_imag = nn.Parameter(torch.empty(sites, outputs, dtype=torch.float64))
        for value in self.parameters():
            nn.init.trunc_normal_(value, std=std, a=-2 * std, b=2 * std, generator=generator)

    def forward(self, values):
        weights = torch.complex(self.weight_real, self.weight_imag)
        bias = torch.complex(self.bias_real, self.bias_imag)
        return torch.einsum("bni,nio->bno", values, weights) + bias


class AGHDONetwork(nn.Module):
    """Trace-one PSD density elements without allocating a Hilbert-space matrix."""

    def __init__(
        self,
        n_qubits: int,
        rank: int = 8,
        feature_densities=(8, 4),
        seed: int = 0,
        init_std: float = 0.01,
        *,
        product: bool = False,
    ) -> None:
        super().__init__()
        if n_qubits < 1 or rank < 1:
            raise ValueError("n_qubits and local rank must be positive")
        if not feature_densities or any(int(v) != v or v < 1 for v in feature_densities):
            raise ValueError("feature_densities must contain positive integers")
        if not 0.001 <= init_std <= 0.1:
            raise ValueError("init_std must be in the paper's [0.001, 0.1] interval")
        self.n_qubits, self.rank = int(n_qubits), int(rank)
        self.product = product
        self.feature_densities = tuple(feature_densities)
        generator = torch.Generator().manual_seed(seed)
        layers = []
        if product:
            inputs = 1
            for outputs in (*feature_densities, 2 * rank):
                layers.append(_ComplexSiteDense(n_qubits, inputs, outputs, generator, init_std))
                inputs = outputs
        else:
            degrees = np.arange(n_qubits)
            for index, width in enumerate((*feature_densities, 2 * rank)):
                following = np.repeat(np.arange(n_qubits), width)


                mask = (
                    degrees[None, :] < following[:, None]
                    if index == 0
                    else (degrees[None, :] <= following[:, None])
                )
                layers.append(ComplexMaskedDense(mask, generator, init_std))
                degrees = following
        self.layers = nn.ModuleList(layers)

    @property
    def device(self):
        return next(self.parameters()).device

    def conditional_amplitudes(self, configurations: torch.Tensor) -> torch.Tensor:
        """psi[h, spin, a], normalized over (spin,a), as in Eqs. (14),(17)."""
        configurations = torch.as_tensor(configurations, device=self.device)
        if configurations.ndim != 2 or configurations.shape[1] != self.n_qubits:
            raise ValueError("configurations must have shape (batch, n_qubits)")
        if self.product:
            values = torch.ones((1, self.n_qubits, 1), dtype=torch.complex128, device=self.device)
        else:
            values = (1 - 2 * configurations.to(torch.float64)).to(torch.complex128)
        for index, layer in enumerate(self.layers):
            values = layer(values)
            if index + 1 < len(self.layers):
                values = _selu(values)
        values = values.reshape(-1, self.n_qubits, 2, self.rank)
        normalization = values.abs().square().sum(dim=(-1, -2), keepdim=True).sqrt()
        amplitudes = values / normalization.clamp_min(1e-150)
        return amplitudes.expand(len(configurations), -1, -1, -1)

    def selected_amplitudes(self, configurations):
        configurations = torch.as_tensor(configurations, device=self.device, dtype=torch.long)
        all_values = self.conditional_amplitudes(configurations)
        return all_values[
            torch.arange(len(configurations), device=self.device)[:, None],
            torch.arange(self.n_qubits, device=self.device)[None, :],
            configurations,
        ]

    def density_elements(self, sigma, eta):
        left, right = self.selected_amplitudes(sigma), self.selected_amplitudes(eta)
        return (left * right.conj()).sum(dim=-1).prod(dim=-1)

    def diagonal_probability(self, sigma):
        return self.selected_amplitudes(sigma).abs().square().sum(dim=-1).prod(dim=-1)

    def basis_configurations(self):
        if self.n_qubits > 8:
            raise ValueError("Dense diagnostics/likelihood support at most 8 qubits")
        integers = torch.arange(1 << self.n_qubits, device=self.device)
        powers = torch.arange(self.n_qubits - 1, -1, -1, device=self.device)
        return (integers[:, None] >> powers[None, :]) & 1

    def dense_density_matrix(self):
        configurations = self.basis_configurations()
        selected = self.selected_amplitudes(configurations)
        rho = torch.ones((len(selected), len(selected)), dtype=torch.complex128, device=self.device)
        for site in range(self.n_qubits):
            factor = selected[:, site]
            rho = rho * (factor @ factor.conj().T)
        return rho

    @torch.no_grad()
    def sample_diagonal(self, samples: int, rng: np.random.Generator):
        configurations = torch.zeros((samples, self.n_qubits), dtype=torch.long, device=self.device)
        if self.product:
            probabilities = self.conditional_amplitudes(configurations[:1]).abs().square().sum(-1)
            draws = torch.as_tensor(rng.random((samples, self.n_qubits)), device=self.device)
            return (draws >= probabilities[0, :, 0]).to(torch.long)
        for site in range(self.n_qubits):
            probabilities = (
                self.conditional_amplitudes(configurations)[:, site].abs().square().sum(-1)
            )
            draws = torch.as_tensor(rng.random(samples), device=self.device)
            configurations[:, site] = (draws >= probabilities[:, 0]).to(torch.long)
        return configurations

    @torch.no_grad()
    def sample_pairs(self, samples: int, rng: np.random.Generator, alpha: float = 0.5):
        """Eq. (26): independent direct samples, with their exact proposal mass."""
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1] for full proposal support")
        sigma = self.sample_diagonal(samples, rng)
        eta = torch.zeros_like(sigma)
        log_probability = self.diagonal_probability(sigma).log()
        for site in range(self.n_qubits):
            probability = self.conditional_amplitudes(eta)[:, site].abs().square().sum(-1)
            mixture = alpha * probability
            mixture = mixture.scatter_add(
                1,
                sigma[:, site, None],
                torch.full((samples, 1), 1 - alpha, device=self.device, dtype=torch.float64),
            )
            draws = torch.as_tensor(rng.random(samples), device=self.device)
            eta[:, site] = (draws >= mixture[:, 0]).to(torch.long)
            log_probability += mixture.gather(1, eta[:, site, None]).squeeze(1).log()
        return sigma, eta, log_probability


class AGHDO:
    """Unrestricted neural AGHDO, fitted to local physical measurement records."""

    def __init__(
        self,
        n_qubits,
        rank=8,
        learning_rate=0.005,
        seed=0,
        *,
        device="cpu",
        feature_densities=(8, 4),
        init_std=0.01,
    ):
        if not 1 <= n_qubits <= 8:
            raise ValueError("Local measurement likelihood supports 1 to 8 subsystem qubits")
        self.n_qubits, self.rank = n_qubits, rank
        self.rng = np.random.default_rng(seed)
        self.network = AGHDONetwork(n_qubits, rank, feature_densities, seed, init_std).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self._states = []
        self.last_loss = float("nan")

    def add_measurement(self, basis, outcomes):
        if len(basis) != self.n_qubits or len(outcomes) != self.n_qubits:
            raise ValueError("Local basis/outcome size does not match n_qubits")
        self._states.append(_measurement_ket(tuple(map(int, basis)), tuple(map(int, outcomes))))

    @property
    def record_count(self):
        return len(self._states)

    def fit(self, steps, batch_size):
        if not self._states or steps <= 0:
            return
        for _ in range(int(steps)):
            indices = self.rng.integers(
                0, len(self._states), size=min(batch_size, len(self._states))
            )
            states = torch.as_tensor(
                np.asarray([self._states[i] for i in indices]),
                dtype=torch.complex128,
                device=self.network.device,
            )
            rho = self.network.dense_density_matrix()
            probability = torch.einsum("bi,ij,bj->b", states.conj(), rho, states).real
            loss = -probability.clamp_min(1e-12).log().mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = loss.item()

    def density_matrix(self):
        with torch.no_grad():
            return self.network.dense_density_matrix().cpu().numpy()

    def sample_purity(self, samples, rng):
        rho = self.density_matrix()
        return float(np.sum(np.abs(rho) ** 2))

    def sample_magic(self, samples, rng, *, clip=True):
        rho = self.density_matrix()
        dimension = 1 << self.n_qubits
        total = 4**self.n_qubits
        codes = rng.integers(1, total, size=max(1, int(samples)))
        basis = np.arange(dimension)
        fourth = []
        for code in codes:
            x = z = y_count = 0
            for site in range(self.n_qubits):
                digit = (int(code) >> (2 * site)) & 3
                bit = 1 << (self.n_qubits - 1 - site)
                if digit in (1, 2):
                    x |= bit
                if digit in (2, 3):
                    z |= bit
                y_count += digit == 2
            signs = np.array([(-1) ** bit_count(int(value) & z) for value in basis])
            mean = np.sum(rho[basis, basis ^ x] * signs * 1j**y_count).real
            fourth.append(mean**4)
        moment = (1 + (total - 1) * np.mean(fourth)) / dimension
        if clip:
            moment = np.clip(moment, 2.0 ** (-self.n_qubits), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(-np.log2(moment))


class FocusedAGHDO:
    """Paper AGHDO restricted to product modes in a known Clifford frame."""

    def __init__(
        self,
        n_qubits,
        rank=8,
        learning_rate=0.005,
        seed=0,
        *,
        device="cpu",
        feature_densities=(8, 4),
        init_std=0.01,
    ):
        self.n_qubits = n_qubits
        self.rng = np.random.default_rng(seed)
        self.network = AGHDONetwork(
            n_qubits, rank, feature_densities, seed, init_std, product=True
        ).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.counts = np.zeros((n_qubits, 3, 2), dtype=np.int64)
        self.last_loss = float("nan")

    def _local_density(self):
        inputs = torch.zeros((1, self.n_qubits), dtype=torch.long, device=self.network.device)
        amplitude = self.network.conditional_amplitudes(inputs)[0]
        return amplitude @ amplitude.conj().transpose(-1, -2)

    def _bloch(self):
        rho = self._local_density()
        return torch.stack(
            (2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, rho[:, 0, 0].real - rho[:, 1, 1].real),
            dim=1,
        )

    def add_encoded(self, mode, axis, outcome):
        if not 0 <= mode < self.n_qubits or axis not in (0, 1, 2) or outcome not in (-1, 1):
            raise ValueError("Invalid logical mode, Pauli axis or parity")
        self.counts[mode, axis, int(outcome < 0)] += 1

    @property
    def record_count(self):
        return int(self.counts.sum())

    def fit(self, steps, batch_size=128):


        if not self.record_count or steps <= 0:
            return
        counts = torch.as_tensor(self.counts, dtype=torch.float64, device=self.network.device)
        for _ in range(int(steps)):
            bloch = self._bloch()
            probabilities = torch.stack(((1 + bloch) / 2, (1 - bloch) / 2), dim=-1)
            loss = -(counts * probabilities.clamp_min(1e-12).log()).sum() / counts.sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = loss.item()

    def bloch_vectors(self):
        with torch.no_grad():
            return self._bloch().cpu().numpy()

    def sample_magic(self, samples, rng, *, clip=True):


        bloch = self.bloch_vectors()
        local_fourth = (1 + np.sum(bloch**4, axis=1)) / 2
        if clip:
            local_fourth = np.clip(local_fourth, 0.5, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(-np.sum(np.log2(local_fourth)))
