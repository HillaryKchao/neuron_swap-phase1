"""CaDynamics_E2: intracellular calcium pool driven by the calcium current.

Transcribed from ``CaDynamics_E2.mod`` (modified from Destexhe et al. 1994)::

    cai' = -(10000) * (ica * gamma / (2 * FARADAY * depth)) - (cai - minCai) / decay

``ica`` is the total calcium current density (mA/cm^2), ``gamma`` the fraction
of free (unbuffered) calcium, ``depth`` the shell depth (um), ``decay`` the
removal time constant (ms) and ``minCai`` the resting concentration (mM).

This is the "calcium pool integrator" singled out in the README as a hardware
challenge, and it is required to drive :class:`~neuron_swap.channels.SK_E2`.
"""

from __future__ import annotations

import numpy as np

from .base import FARADAY


class CaDynamics_E2:
    name = "CaDynamics_E2"
    reference = "Destexhe et al. 1994 (modified)"

    def __init__(
        self,
        gamma: float = 0.05,
        decay: float = 80.0,
        depth: float = 0.1,
        min_cai: float = 1e-4,
        tau_scale: float = 1.0,
    ) -> None:
        self.gamma = float(gamma)
        self.decay = float(decay)
        self.depth = float(depth)
        self.min_cai = float(min_cai)
        if tau_scale <= 0:
            raise ValueError("tau_scale must be positive")
        self.tau_scale = float(tau_scale)

    @property
    def tau(self) -> float:
        """Effective removal time constant (ms) after time scaling."""
        return self.decay * self.tau_scale

    def influx(self, ica) -> np.ndarray:
        """Free-calcium influx term (mM/ms) for a calcium current density."""
        ica = np.asarray(ica, dtype=float)
        # The influx is a rate, so it scales inversely with time.
        return -(10000) * (ica * self.gamma / (2 * FARADAY * self.depth)) / self.tau_scale

    def steady_state(self, ica=0.0) -> np.ndarray:
        """Concentration the pool relaxes to for a constant calcium current."""
        return self.min_cai + self.influx(ica) * self.tau

    def derivative(self, cai, ica) -> np.ndarray:
        cai = np.asarray(cai, dtype=float)
        return self.influx(ica) - (cai - self.min_cai) / self.tau

    def step(self, cai, ica, dt: float) -> np.ndarray:
        """Exact (``cnexp``) update of the linear ODE over ``dt`` ms."""
        cai = np.asarray(cai, dtype=float)
        cai_inf = self.steady_state(ica)
        return cai_inf + (cai - cai_inf) * np.exp(-dt / self.tau)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "gamma": self.gamma,
            "decay": self.decay,
            "depth": self.depth,
            "minCai": self.min_cai,
            "tau_scale": self.tau_scale,
            "reference": self.reference,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CaDynamics_E2(gamma={self.gamma:g}, decay={self.decay:g}, "
            f"depth={self.depth:g}, min_cai={self.min_cai:g}, tau_scale={self.tau_scale:g})"
        )
