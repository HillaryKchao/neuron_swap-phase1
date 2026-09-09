"""Voltage-clamp simulation of isolated channels.

Under an ideal voltage clamp the membrane potential *is* the command
waveform, so the channel ODEs decouple from the membrane equation and each
gate can be advanced with NEURON's exact ``cnexp`` update.  The recorded
quantities per sweep and time point are the gating variables, the
conductance and the ionic current -- i.e. exactly what a hardware channel
block must reproduce.

:func:`simulate_clamp` can carry several channels at once (all under the
same command voltage) together with a :class:`CaDynamics_E2` pool fed by the
calcium currents; this is how the calcium-gated ``SK_E2`` is driven by
``Ca_HVA``/``Ca_LVAst`` rather than by a calcium clamp, and it is the seed
of the Phase-4 whole-compartment model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..channels.base import CA_I_DEFAULT, CA_O_DEFAULT, CELSIUS, GatingChannel, nernst
from ..channels.ca_dynamics import CaDynamics_E2
from .waveforms import VoltageClampProtocol


@dataclass
class ClampResult:
    """Recorded response of one channel to a :class:`VoltageClampProtocol`."""

    channel: dict  # channel.describe()
    protocol: VoltageClampProtocol
    gate_names: tuple[str, ...]
    states: np.ndarray  # (n_gates, n_sweeps, n_t)
    g: np.ndarray  # (n_sweeps, n_t) S/cm^2
    I: np.ndarray  # (n_sweeps, n_t) mA/cm^2
    cai: np.ndarray | None = None  # (n_sweeps, n_t) mM, when calcium was tracked
    extras: dict = field(default_factory=dict)

    # convenience -----------------------------------------------------------
    @property
    def name(self) -> str:
        return self.channel["name"]

    @property
    def t(self) -> np.ndarray:
        return self.protocol.t

    @property
    def V(self) -> np.ndarray:
        return self.protocol.V

    @property
    def dt(self) -> float:
        return self.protocol.dt

    def gate(self, name: str) -> np.ndarray:
        return self.states[self.gate_names.index(name)]

    def feature_names(self, include_current: bool = True) -> list[str]:
        names = list(self.gate_names)
        if self.cai is not None:
            names.append("cai")
        if include_current:
            names.append("I")
        return names

    def features(self, include_current: bool = True) -> np.ndarray:
        """Per-sweep feature time series ``(n_sweeps, n_t, n_features)``:
        the gating variables (and ``cai`` if tracked) followed by the current."""
        cols = [self.states[i] for i in range(len(self.gate_names))]
        if self.cai is not None:
            cols.append(self.cai)
        if include_current:
            cols.append(self.I)
        return np.stack(cols, axis=-1)

    # analysis helpers --------------------------------------------------------
    def epoch(self, name: str) -> slice:
        return self.protocol.epoch_slice(name)

    def peak_current(self, epoch: str = "step") -> np.ndarray:
        """Current of largest magnitude in an epoch, per sweep (signed)."""
        seg = self.I[:, self.epoch(epoch)]
        idx = np.argmax(np.abs(seg), axis=1)
        return seg[np.arange(seg.shape[0]), idx]

    def time_to_peak(self, epoch: str = "step") -> np.ndarray:
        seg = self.I[:, self.epoch(epoch)]
        return np.argmax(np.abs(seg), axis=1) * self.dt

    def steady_state_current(self, epoch: str = "step", tail_fraction: float = 0.05) -> np.ndarray:
        """Mean current over the last ``tail_fraction`` of an epoch, per sweep."""
        seg = self.I[:, self.epoch(epoch)]
        n = max(1, int(round(seg.shape[1] * tail_fraction)))
        return seg[:, -n:].mean(axis=1)

    # serialisation -----------------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "channel": self.channel,
            "protocol": self.protocol.describe(),
            "gate_names": list(self.gate_names),
        }
        return d

    def save(self, path, dtype=np.float32) -> None:
        """Write a compressed ``.npz`` archive (``t, V, states, g, I[, cai]``
        plus a JSON ``meta`` string).  ``dtype`` controls the stored precision
        of the waveforms (``float32`` is ample for waveform comparison)."""
        import json

        arrays = {
            "t": self.t.astype(dtype),
            "V": self.V.astype(dtype),
            "states": self.states.astype(dtype),
            "g": self.g.astype(dtype),
            "I": self.I.astype(dtype),
            "meta": json.dumps(self.to_dict()),
        }
        if self.cai is not None:
            arrays["cai"] = self.cai.astype(dtype)
        if self.protocol.cai is not None:
            arrays["cai_command"] = self.protocol.cai.astype(dtype)
        for k, v in self.extras.items():
            arrays[f"extra_{k}"] = np.asarray(v).astype(dtype)
        np.savez_compressed(path, **arrays)


@dataclass
class ClampRun:
    """Result of :func:`simulate_clamp`: one :class:`ClampResult` per channel
    plus the total ionic current and, if a pool was present, the calcium
    concentration."""

    protocol: VoltageClampProtocol
    results: dict[str, ClampResult]
    I_total: np.ndarray
    cai: np.ndarray | None = None

    def __getitem__(self, name: str) -> ClampResult:
        return self.results[name]


def _initial_state(channel: GatingChannel, v0, cai0, init) -> np.ndarray:
    if init is None:
        return channel.steady_state(v0, cai0)
    init = np.asarray(init, dtype=float)
    if init.ndim == 1:
        init = np.broadcast_to(init[:, None], (channel.n_gates,) + np.shape(v0)).copy()
    return init


def simulate_clamp(
    channels: Mapping[str, GatingChannel] | Sequence[GatingChannel],
    protocol: VoltageClampProtocol,
    ca_pool: CaDynamics_E2 | None = None,
    cai0: float | None = None,
    init: Mapping[str, np.ndarray] | None = None,
    record_every: int = 1,
    nernst_ca: bool | None = None,
    cao: float = CA_O_DEFAULT,
    celsius: float = CELSIUS,
) -> ClampRun:
    """Clamp a set of channels to ``protocol`` and record their responses.

    Parameters
    ----------
    channels
        Channels sharing the same membrane patch.  A sequence is keyed by
        ``channel.name``.
    protocol
        Command voltage (and optionally a calcium clamp, ``protocol.cai``).
    ca_pool
        If given, intracellular calcium is integrated from the total calcium
        current of the ``ion == "ca"`` channels and fed to calcium-dependent
        channels.  Ignored when the protocol clamps calcium directly.
    cai0
        Initial calcium concentration (mM).  Defaults to the pool steady
        state for the initial calcium current, or to NEURON's default
        ``5e-5`` mM when no pool is present and calcium is not clamped.
    init
        Optional initial gate states per channel (``(n_gates,)`` or
        ``(n_gates, n_sweeps)``); default is the steady state at ``V[:, 0]``
        (NEURON ``INITIAL``).
    record_every
        Keep every ``record_every``-th time point in the outputs (the
        integration itself always uses ``protocol.dt``).
    nernst_ca
        Recompute the calcium reversal potential from ``cai`` by the Nernst
        equation at every step, as NEURON does once a mechanism writes
        ``cai``.  Defaults to ``True`` whenever calcium is tracked (pool or
        clamp) and ``False`` otherwise (fixed ``channel.erev``).
    cao, celsius
        Extracellular calcium (mM) and temperature used by the Nernst update.
    """
    if not isinstance(channels, Mapping):
        channels = {ch.name: ch for ch in channels}
    init = init or {}
    V = protocol.V
    n_sweeps, n_t = V.shape
    dt = protocol.dt
    ca_channels = [name for name, ch in channels.items() if ch.ion == "ca"]
    calcium_clamped = protocol.cai is not None
    track_ca = calcium_clamped or ca_pool is not None or any(
        ch.calcium_dependent for ch in channels.values()
    )
    if nernst_ca is None:
        nernst_ca = calcium_clamped or ca_pool is not None
    nernst_ca = bool(nernst_ca and ca_channels)

    # ---- initial conditions ------------------------------------------------
    v0 = V[:, 0]
    if calcium_clamped:
        cai = protocol.cai[:, 0].copy()
    elif ca_pool is not None:
        if cai0 is None:
            # steady state of the pool at the holding potential
            ica0 = np.zeros(n_sweeps)
            for name in ca_channels:
                ch = channels[name]
                ica0 += ch.current(ch.steady_state(v0), v0)
            cai = np.asarray(ca_pool.steady_state(ica0), dtype=float).copy()
        else:
            cai = np.full(n_sweeps, float(cai0))
    else:
        cai = np.full(n_sweeps, CA_I_DEFAULT if cai0 is None else cai0)

    def erev_of(ch: GatingChannel, cai_now: np.ndarray) -> np.ndarray | float:
        if nernst_ca and ch.ion == "ca":
            return nernst(2, cai_now, cao, celsius)
        return ch.erev

    states = {name: _initial_state(ch, v0, cai, init.get(name)) for name, ch in channels.items()}

    # ---- recording buffers ---------------------------------------------------
    rec_idx = np.arange(0, n_t, record_every)
    n_rec = len(rec_idx)
    rec_states = {name: np.empty((ch.n_gates, n_sweeps, n_rec)) for name, ch in channels.items()}
    rec_g = {name: np.empty((n_sweeps, n_rec)) for name in channels}
    rec_I = {name: np.empty((n_sweeps, n_rec)) for name in channels}
    rec_cai = np.empty((n_sweeps, n_rec)) if track_ca else None
    rec_total = np.zeros((n_sweeps, n_rec))
    rec_erev = {name: np.empty((n_sweeps, n_rec)) for name in ca_channels} if nernst_ca else {}

    # ---- time stepping -------------------------------------------------------
    k = 0
    for n in range(n_t):
        v = V[:, n]
        if calcium_clamped:
            cai = protocol.cai[:, n]
        ica = np.zeros(n_sweeps)
        record = (n % record_every) == 0
        for name, ch in channels.items():
            g = ch.conductance(states[name])
            erev = erev_of(ch, cai)
            I = g * (v - erev)
            if ch.ion == "ca":
                ica += I
            if record:
                rec_states[name][:, :, k] = states[name]
                rec_g[name][:, k] = g
                rec_I[name][:, k] = I
                rec_total[:, k] += I
                if name in rec_erev:
                    rec_erev[name][:, k] = erev
        if record:
            if track_ca:
                rec_cai[:, k] = cai
            k += 1
        if n == n_t - 1:
            break
        # advance gates (cnexp) with the voltage / calcium of this step
        for name, ch in channels.items():
            states[name] = ch.step(states[name], v, dt, cai if ch.calcium_dependent else None)
        if ca_pool is not None and not calcium_clamped:
            cai = ca_pool.step(cai, ica, dt)

    results = {}
    for name, ch in channels.items():
        results[name] = ClampResult(
            channel=ch.describe(),
            protocol=protocol if record_every == 1 else _decimate(protocol, record_every),
            gate_names=ch.gate_names,
            states=rec_states[name],
            g=rec_g[name],
            I=rec_I[name],
            cai=rec_cai if (ch.calcium_dependent and rec_cai is not None) else None,
            extras={"erev": rec_erev[name]} if name in rec_erev else {},
        )
    return ClampRun(protocol=results[next(iter(results))].protocol, results=results, I_total=rec_total, cai=rec_cai)


def _decimate(protocol: VoltageClampProtocol, every: int) -> VoltageClampProtocol:
    return VoltageClampProtocol(
        name=protocol.name,
        dt=protocol.dt * every,
        V=protocol.V[:, ::every],
        sweep_values=protocol.sweep_values,
        sweep_label=protocol.sweep_label,
        epochs=protocol.epochs,
        description=protocol.description,
        cai=None if protocol.cai is None else protocol.cai[:, ::every],
    )


def run_voltage_clamp(
    channel: GatingChannel,
    protocol: VoltageClampProtocol,
    ca_pool: CaDynamics_E2 | None = None,
    ca_sources: Sequence[GatingChannel] = (),
    **kwargs,
) -> ClampResult:
    """Clamp a single channel and return its :class:`ClampResult`.

    For a calcium-dependent channel either use a protocol with a calcium
    clamp (:func:`~neuron_swap.protocols.waveforms.calcium_steps`) or pass a
    ``ca_pool`` plus the calcium ``ca_sources`` channels that feed it.
    """
    channels = {channel.name: channel}
    for src in ca_sources:
        channels[src.name] = src
    run = simulate_clamp(channels, protocol, ca_pool=ca_pool, **kwargs)
    result = run[channel.name]
    if ca_sources:
        result.extras["ica_sources"] = np.stack([run[s.name].I for s in ca_sources])
    return result
