"""Base classes for the Python golden model of Hodgkin-Huxley style channels.

Every L5PC channel mechanism has the same structure in its ``.mod`` file::

    BREAKPOINT  { SOLVE states METHOD cnexp ; g = gbar * prod(x_i ^ p_i) ; i = g * (v - e) }
    DERIVATIVE states { rates() ; x_i' = (xInf_i - x_i) / xTau_i }
    INITIAL     { rates() ; x_i = xInf_i }

so a channel is fully described by (a) its ``rates`` function mapping the
membrane potential (or, for SK_E2, the calcium concentration) to the
steady-state value and time constant of each gate, (b) the exponent of each
gate in the conductance, and (c) its maximal conductance density and reversal
potential. :class:`GatingChannel` implements everything else once:

* ``steady_state``  -- NEURON's ``INITIAL`` block
* ``derivatives``   -- the ODE right-hand side (for ``scipy.integrate``)
* ``step``          -- NEURON's exact ``cnexp`` update over one time step
* ``conductance`` / ``current`` -- the ``BREAKPOINT`` block

All functions are vectorised with numpy: ``V`` may be a scalar or an array of
voltages (e.g. one entry per voltage-clamp sweep), and the state array carries
the gates on its leading axis, ``state.shape == (n_gates,) + V.shape``.

Time scaling
------------
Phase 2/3 hardware runs the dynamics faster than biology.  ``tau_scale``
multiplies every time constant of the model (``tau_scale < 1`` accelerates)
so the golden model can be scaled together with the hardware, as required by
the project README ("as the hardware dynamics are scaled, the Python model
will be equivalently scaled").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (NEURON >= 8 "modern" units, nrnunits.lib)
# ---------------------------------------------------------------------------
FARADAY = 96485.33212  # C / mol
R_GAS = 8.314462618  # J / (mol K)

#: The BBP L5PC mechanisms hard-code a Q10 correction from the 21 degC
#: recording temperature to the 34 degC simulation temperature.
Q10 = 2.3
T_RECORDING = 21.0
T_SIMULATION = 34.0
QT = Q10 ** ((T_SIMULATION - T_RECORDING) / 10.0)

#: Default operating point of the L5PC model (biophysics.hoc / constants.hoc)
CELSIUS = 34.0
E_NA = 50.0  # mV
E_K = -85.0  # mV
E_HCN = -45.0  # mV  (ehcn in Ih.mod)
CA_I_DEFAULT = 5e-5  # mM, NEURON default intracellular Ca
CA_O_DEFAULT = 2.0  # mM, NEURON default extracellular Ca
DT_DEFAULT = 0.025  # ms


def nernst(charge: int, c_in: float, c_out: float, celsius: float = CELSIUS) -> float:
    """Nernst reversal potential in mV (this is how NEURON derives ``eca``)."""
    return 1e3 * R_GAS * (celsius + 273.15) / (charge * FARADAY) * np.log(c_out / c_in)


E_CA = float(nernst(2, CA_I_DEFAULT, CA_O_DEFAULT))  # ~140 mV at 34 degC


def _guard(v: np.ndarray, singular_value: float, eps: float = 1e-4) -> np.ndarray:
    """Replicates the ``if (v == x) { v = v + 0.0001 }`` guards of the .mod
    files that avoid 0/0 in expressions such as ``(v - x) / (1 - exp((v - x)/k))``."""
    return np.where(v == singular_value, v + eps, v)


def exprel_ratio(x: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Utility kept for symmetry with the C++ port; simply ``num / den``."""
    return num / den


@dataclass(frozen=True)
class GateSpec:
    """A gating variable and its exponent in the conductance."""

    name: str
    power: int


class GatingChannel:
    """A voltage- (or calcium-) gated conductance with first-order gates.

    Subclasses must define the class attributes ``name``, ``ion``, ``gates``
    (a tuple of :class:`GateSpec`), ``default_gbar`` and ``default_erev`` and
    implement :meth:`_rates`.
    """

    #: mechanism name (SUFFIX in the .mod file)
    name: str = ""
    #: ion carried: "na", "k", "ca" or "hcn"
    ion: str = ""
    #: gating variables in state order
    gates: tuple[GateSpec, ...] = ()
    #: maximal conductance density S/cm^2 (PARAMETER default of the .mod file)
    default_gbar: float = 1e-5
    #: reversal potential mV at the L5PC operating point
    default_erev: float = 0.0
    #: literature reference from the .mod header
    reference: str = ""
    #: True if the gates depend on intracellular calcium rather than voltage
    calcium_dependent: bool = False

    def __init__(
        self,
        gbar: float | None = None,
        erev: float | None = None,
        tau_scale: float = 1.0,
    ) -> None:
        self.gbar = float(self.default_gbar if gbar is None else gbar)
        self.erev = float(self.default_erev if erev is None else erev)
        if tau_scale <= 0:
            raise ValueError("tau_scale must be positive")
        self.tau_scale = float(tau_scale)

    # ------------------------------------------------------------------ meta
    @property
    def gate_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.gates)

    @property
    def powers(self) -> tuple[int, ...]:
        return tuple(g.power for g in self.gates)

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(gbar={self.gbar:g}, erev={self.erev:g}, "
            f"tau_scale={self.tau_scale:g})"
        )

    # --------------------------------------------------------------- kinetics
    def _rates(self, v: np.ndarray, cai: np.ndarray | None) -> Sequence[tuple[np.ndarray, np.ndarray]]:
        """Return ``[(xInf, xTau), ...]`` in gate order.  ``xTau`` in ms,
        *before* time scaling.  Must be implemented by subclasses and should be
        a line-by-line transcription of the ``PROCEDURE rates()`` block."""
        raise NotImplementedError

    def rates(self, V, cai=None) -> tuple[np.ndarray, np.ndarray]:
        """Steady-state values and (scaled) time constants of every gate.

        Returns ``(inf, tau)`` each of shape ``(n_gates,) + np.shape(V)``.
        """
        v = np.asarray(V, dtype=float)
        if self.calcium_dependent:
            if cai is None:
                raise ValueError(f"{self.name} is calcium dependent: pass cai (mM)")
            v, ca = np.broadcast_arrays(v, np.asarray(cai, dtype=float))
        else:
            ca = None
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            pairs = self._rates(v, ca)
        inf = np.stack([np.broadcast_to(p[0], v.shape) for p in pairs])
        tau = np.stack([np.broadcast_to(p[1], v.shape) for p in pairs]) * self.tau_scale
        return inf, tau

    def steady_state(self, V, cai=None) -> np.ndarray:
        """State the channel relaxes to at fixed ``V`` (NEURON ``INITIAL``)."""
        return self.rates(V, cai)[0]

    def derivatives(self, state, V, cai=None) -> np.ndarray:
        """Time derivative of the gates, ``(xInf - x) / xTau`` (1/ms)."""
        inf, tau = self.rates(V, cai)
        return (inf - np.asarray(state, dtype=float)) / tau

    def step(self, state, V, dt: float, cai=None) -> np.ndarray:
        """Advance the gates by ``dt`` ms with NEURON's ``cnexp`` scheme.

        ``cnexp`` integrates ``x' = (xInf - x)/xTau`` exactly for constant
        coefficients: ``x(t+dt) = xInf + (x(t) - xInf) * exp(-dt/xTau)``.
        """
        inf, tau = self.rates(V, cai)
        return inf + (np.asarray(state, dtype=float) - inf) * np.exp(-dt / tau)

    # ------------------------------------------------------------- breakpoint
    def open_probability(self, state) -> np.ndarray:
        """``prod(x_i ^ p_i)`` -- conductance normalised by ``gbar``."""
        state = np.asarray(state, dtype=float)
        po = np.ones(state.shape[1:], dtype=float)
        for i, p in enumerate(self.powers):
            po = po * state[i] ** p
        return po

    def conductance(self, state) -> np.ndarray:
        """Conductance density S/cm^2."""
        return self.gbar * self.open_probability(state)

    def current(self, state, V) -> np.ndarray:
        """Ionic current density mA/cm^2 (positive = outward, NEURON sign)."""
        return self.conductance(state) * (np.asarray(V, dtype=float) - self.erev)

    # ------------------------------------------------------------ description
    def describe(self) -> dict:
        return {
            "name": self.name,
            "ion": self.ion,
            "gates": [{"name": g.name, "power": g.power} for g in self.gates],
            "gbar": self.gbar,
            "erev": self.erev,
            "tau_scale": self.tau_scale,
            "calcium_dependent": self.calcium_dependent,
            "reference": self.reference,
        }
