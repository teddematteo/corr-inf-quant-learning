"""Optional CUDA acceleration of ConEnh policy calculations."""

from __future__ import annotations

import numpy as np


class ComputeBackend:
    def __init__(self, device: str = "cpu") -> None:
        if device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be 'cpu', 'cuda' or 'auto'")
        self.requested = device
        self.device = "cpu"
        self.name = "CPU (NumPy)"
        self.reason = "CPU requested"
        self.operations = 0
        self._torch = None
        self._cache = {}
        if device == "cpu":
            return
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this environment")

            probe = torch.ones((2, 2), dtype=torch.float64, device="cuda")
            (probe @ probe).sum().item()
        except (ImportError, OSError, RuntimeError) as error:
            self.reason = str(error)
            if device == "cuda":
                raise RuntimeError(
                    "CUDA was requested but cannot execute. Use a GPU allocation "
                    "and a compatible PyTorch installation. "
                    f"Details: {error}"
                ) from error
            return
        self._torch = torch
        self.device = "cuda"
        self.name = torch.cuda.get_device_name(torch.cuda.current_device())
        self.reason = "CUDA calculation verified"

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested, "device": self.device, "name": self.name,
            "reason": self.reason, "gpu_operations": self.operations,
            "scope": "ConEnh policy matrix reductions and candidate/clique compatibility",
        }

    def _tensor(self, values: np.ndarray, *, cache: bool = False, integer: bool = False):
        torch = self._torch
        dtype = torch.int8 if integer else torch.float64
        key = (id(values), integer)
        if cache and key in self._cache:
            original, tensor = self._cache[key]
            if original is values:
                return tensor
        tensor = torch.as_tensor(np.ascontiguousarray(values), dtype=dtype, device="cuda")
        if cache:

            self._cache[key] = (values, tensor)
        return tensor

    def clear_cache(self) -> None:
        self._cache.clear()

    def matvec(
        self, matrix: np.ndarray, vector: np.ndarray, *, cache: bool = False,
        transpose: bool = False,
    ) -> np.ndarray:
        if self.device == "cpu":
            return np.einsum("ij,i->j" if transpose else "ij,j->i", matrix, vector)
        tensor = self._tensor(matrix, cache=cache)
        if transpose:
            tensor = tensor.T
        result = tensor @ self._tensor(vector)
        self.operations += 1
        return result.cpu().numpy()

    def clique_fits(self, bases: np.ndarray, axes: np.ndarray) -> np.ndarray:
        if self.device == "cuda":
            torch = self._torch
            device_bases = self._tensor(bases, integer=True)
            device_axes = self._tensor(axes, integer=True, cache=True)
            fits = torch.ones((len(bases), len(axes)), dtype=torch.bool, device="cuda")
            for site in range(bases.shape[1]):
                column = device_axes[:, site]
                fits &= (column[None, :] < 0) | (
                    column[None, :] == device_bases[:, site, None]
                )
            self.operations += 1
            return fits.cpu().numpy()
        fits = np.ones((len(bases), len(axes)), dtype=bool)
        for site in range(bases.shape[1]):
            column = axes[:, site]
            fits &= (column[None, :] < 0) | (column[None, :] == bases[:, site, None])
        return fits
