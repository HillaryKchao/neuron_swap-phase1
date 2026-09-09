"""Voltage-clamp command waveforms.

A :class:`VoltageClampProtocol` is a family of *sweeps*: an array of command
voltages ``V[sweep, t]`` sampled on the simulation time grid
``t = 0, dt, ..., duration``.  Each builder returns a protocol with a
specified time step and a specified (or defaulted) number of points, as
requested in ``src/conductances/README.md``.

Four classic step families isolate different aspects of a channel:

* :func:`activation_steps` -- depolarising steps from a hyperpolarised
  holding potential: activation kinetics, peak (transient) and steady-state
  current versus voltage (I-V curve).
* :func:`steady_state_inactivation` -- long pre-pulses followed by a fixed
  test pulse: availability (h-infinity) versus pre-pulse voltage.
* :func:`deactivation_tails` -- a brief activating pulse followed by
  repolarising steps: tail currents and closing kinetics.
* :func:`ramp` -- a slow voltage ramp: quasi steady-state I-V.

For DCA the step protocols are strongly non-stationary, so
:func:`ou_noise` produces a long, stationary, coloured-noise command
(Ornstein-Uhlenbeck process) that samples the whole voltage range
continuously; it is the primary input for the Step 1.3 signatures.

:func:`calcium_steps` is the analogue of ``activation_steps`` for the
calcium-gated SK_E2 channel: the calcium concentration is clamped instead of
(in addition to) the voltage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..channels.base import DT_DEFAULT

Epoch = tuple[float, float]


@dataclass
class VoltageClampProtocol:
    """A family of voltage-clamp sweeps on a common time grid."""

    name: str
    dt: float
    V: np.ndarray  # (n_sweeps, n_t) command voltage, mV
    sweep_values: np.ndarray  # (n_sweeps,) the quantity varied across sweeps
    sweep_label: str = "sweep"
    epochs: dict[str, Epoch] = field(default_factory=dict)  # name -> (t0, t1) ms
    description: str = ""
    cai: np.ndarray | None = None  # (n_sweeps, n_t) clamped [Ca]_i, mM

    def __post_init__(self) -> None:
        self.V = np.atleast_2d(np.asarray(self.V, dtype=float))
        self.sweep_values = np.asarray(self.sweep_values, dtype=float)
        if self.sweep_values.shape != (self.V.shape[0],):
            raise ValueError("sweep_values must have one entry per sweep")
        if self.cai is not None:
            self.cai = np.broadcast_to(np.asarray(self.cai, dtype=float), self.V.shape).copy()

    @property
    def n_sweeps(self) -> int:
        return self.V.shape[0]

    @property
    def n_t(self) -> int:
        return self.V.shape[1]

    @property
    def duration(self) -> float:
        return (self.n_t - 1) * self.dt

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n_t) * self.dt

    def epoch_mask(self, name: str) -> np.ndarray:
        t0, t1 = self.epochs[name]
        t = self.t
        return (t >= t0) & (t < t1)

    def epoch_slice(self, name: str) -> slice:
        t0, t1 = self.epochs[name]
        return slice(_index(t0, self.dt), _index(t1, self.dt))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dt": self.dt,
            "n_t": self.n_t,
            "n_sweeps": self.n_sweeps,
            "duration_ms": self.duration,
            "sweep_label": self.sweep_label,
            "sweep_values": self.sweep_values.tolist(),
            "epochs": {k: list(v) for k, v in self.epochs.items()},
            "description": self.description,
            "calcium_clamp": self.cai is not None,
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _index(t_ms: float, dt: float) -> int:
    return int(round(t_ms / dt))


def _n_points(duration_ms: float, dt: float) -> int:
    return _index(duration_ms, dt) + 1


def _piecewise(segments: Sequence[tuple[float, float]], dt: float) -> tuple[np.ndarray, dict[str, Epoch]]:
    """Build one sweep from ``[(duration_ms, value), ...]``.

    Returns the waveform and the epoch table ``{"seg0": (t0, t1), ...}``.
    Sample ``n`` (time ``n*dt``) takes the value of the segment containing
    ``n*dt``; the final sample (``t == duration``) repeats the last segment.
    """
    total = sum(d for d, _ in segments)
    n_t = _n_points(total, dt)
    out = np.empty(n_t)
    epochs: dict[str, Epoch] = {}
    t0 = 0.0
    for k, (dur, value) in enumerate(segments):
        i0, i1 = _index(t0, dt), _index(t0 + dur, dt)
        out[i0:i1] = value
        epochs[f"seg{k}"] = (t0, t0 + dur)
        t0 += dur
    out[-1] = segments[-1][1]
    return out, epochs


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


# --------------------------------------------------------------------------
# step families
# --------------------------------------------------------------------------
def activation_steps(
    hold: float = -90.0,
    steps: Iterable[float] = range(-80, 60, 10),
    t_pre: float = 20.0,
    t_step: float = 100.0,
    t_post: float = 50.0,
    dt: float = DT_DEFAULT,
) -> VoltageClampProtocol:
    """Hold at ``hold``, step to each of ``steps`` for ``t_step`` ms, return."""
    steps = _as_array(steps)
    V = []
    for vs in steps:
        wave, ep = _piecewise([(t_pre, hold), (t_step, vs), (t_post, hold)], dt)
        V.append(wave)
    epochs = {"pre": ep["seg0"], "step": ep["seg1"], "post": ep["seg2"]}
    return VoltageClampProtocol(
        name="activation",
        dt=dt,
        V=np.stack(V),
        sweep_values=steps,
        sweep_label="step voltage (mV)",
        epochs=epochs,
        description=f"hold {hold} mV, {t_step} ms steps",
    )


def steady_state_inactivation(
    hold: float = -90.0,
    prepulses: Iterable[float] = range(-120, 10, 10),
    test: float = 0.0,
    t_pre: float = 20.0,
    t_prepulse: float = 500.0,
    t_test: float = 50.0,
    t_post: float = 50.0,
    dt: float = DT_DEFAULT,
) -> VoltageClampProtocol:
    """Conditioning pre-pulse to each of ``prepulses`` then a fixed test pulse."""
    prepulses = _as_array(prepulses)
    V = []
    for vp in prepulses:
        wave, ep = _piecewise([(t_pre, hold), (t_prepulse, vp), (t_test, test), (t_post, hold)], dt)
        V.append(wave)
    epochs = {"pre": ep["seg0"], "prepulse": ep["seg1"], "test": ep["seg2"], "post": ep["seg3"]}
    return VoltageClampProtocol(
        name="inactivation",
        dt=dt,
        V=np.stack(V),
        sweep_values=prepulses,
        sweep_label="pre-pulse voltage (mV)",
        epochs=epochs,
        description=f"{t_prepulse} ms pre-pulse then test at {test} mV",
    )


def deactivation_tails(
    hold: float = -90.0,
    activate: float = 20.0,
    tails: Iterable[float] = range(-120, -20, 10),
    t_pre: float = 20.0,
    t_activate: float = 5.0,
    t_tail: float = 50.0,
    t_post: float = 20.0,
    dt: float = DT_DEFAULT,
) -> VoltageClampProtocol:
    """Brief activating pulse to ``activate`` then repolarise to each tail voltage."""
    tails = _as_array(tails)
    V = []
    for vt in tails:
        wave, ep = _piecewise([(t_pre, hold), (t_activate, activate), (t_tail, vt), (t_post, hold)], dt)
        V.append(wave)
    epochs = {"pre": ep["seg0"], "activate": ep["seg1"], "tail": ep["seg2"], "post": ep["seg3"]}
    return VoltageClampProtocol(
        name="deactivation",
        dt=dt,
        V=np.stack(V),
        sweep_values=tails,
        sweep_label="tail voltage (mV)",
        epochs=epochs,
        description=f"{t_activate} ms at {activate} mV then {t_tail} ms tails",
    )


def ramp(
    hold: float = -90.0,
    v_start: float = -120.0,
    v_end: float = 60.0,
    t_pre: float = 20.0,
    t_ramp: float = 500.0,
    t_post: float = 20.0,
    dt: float = DT_DEFAULT,
) -> VoltageClampProtocol:
    """A single linear ramp from ``v_start`` to ``v_end``."""
    n_pre, n_ramp, n_post = _index(t_pre, dt), _index(t_ramp, dt), _index(t_post, dt)
    wave = np.concatenate(
        [
            np.full(n_pre, hold),
            np.linspace(v_start, v_end, n_ramp, endpoint=False),
            np.full(n_post + 1, hold),
        ]
    )
    epochs = {
        "pre": (0.0, t_pre),
        "ramp": (t_pre, t_pre + t_ramp),
        "post": (t_pre + t_ramp, t_pre + t_ramp + t_post),
    }
    return VoltageClampProtocol(
        name="ramp",
        dt=dt,
        V=wave[None, :],
        sweep_values=np.array([(v_end - v_start) / t_ramp]),
        sweep_label="ramp rate (mV/ms)",
        epochs=epochs,
        description=f"ramp {v_start} to {v_end} mV in {t_ramp} ms",
    )


def ou_noise(
    mean: float = -50.0,
    sd: float = 20.0,
    tau: float = 5.0,
    duration: float = 2000.0,
    n_sweeps: int = 1,
    clip: tuple[float, float] = (-120.0, 60.0),
    seed: int | None = 0,
    dt: float = DT_DEFAULT,
) -> VoltageClampProtocol:
    """Stationary Ornstein-Uhlenbeck command voltage.

    ``dV = -(V - mean)/tau dt + sd * sqrt(2 dt / tau) dW`` sampled exactly on
    the grid.  ``tau`` (ms) sets the correlation time; ``sd`` the stationary
    standard deviation.  Each sweep is an independent realisation.
    """
    rng = np.random.default_rng(seed)
    n_t = _n_points(duration, dt)
    a = np.exp(-dt / tau)
    noise_sd = sd * np.sqrt(1 - a * a)
    V = np.empty((n_sweeps, n_t))
    V[:, 0] = mean + sd * rng.standard_normal(n_sweeps)
    xi = rng.standard_normal((n_sweeps, n_t - 1))
    for n in range(1, n_t):
        V[:, n] = mean + a * (V[:, n - 1] - mean) + noise_sd * xi[:, n - 1]
    np.clip(V, clip[0], clip[1], out=V)
    return VoltageClampProtocol(
        name="ou_noise",
        dt=dt,
        V=V,
        sweep_values=np.arange(n_sweeps, dtype=float),
        sweep_label="realisation",
        epochs={"all": (0.0, duration)},
        description=f"OU noise mean {mean} mV, sd {sd} mV, tau {tau} ms, seed {seed}",
    )


def calcium_steps(
    v_hold: float = -50.0,
    cai_hold: float = 5e-5,
    steps: Iterable[float] = np.logspace(-5, -2, 13),
    t_pre: float = 5.0,
    t_step: float = 10.0,
    t_post: float = 10.0,
    dt: float = DT_DEFAULT,
) -> VoltageClampProtocol:
    """Calcium-concentration steps (mM) at a fixed voltage, for SK_E2."""
    steps = _as_array(steps)
    cai = []
    for c in steps:
        wave, ep = _piecewise([(t_pre, cai_hold), (t_step, c), (t_post, cai_hold)], dt)
        cai.append(wave)
    cai = np.stack(cai)
    epochs = {"pre": ep["seg0"], "step": ep["seg1"], "post": ep["seg2"]}
    return VoltageClampProtocol(
        name="calcium_steps",
        dt=dt,
        V=np.full_like(cai, v_hold),
        sweep_values=steps,
        sweep_label="[Ca]_i step (mM)",
        epochs=epochs,
        description=f"[Ca]_i steps at {v_hold} mV",
        cai=cai,
    )


# --------------------------------------------------------------------------
# per-channel defaults
# --------------------------------------------------------------------------
#: Channel-specific overrides so that each protocol actually exercises the
#: channel (Ih opens on hyperpolarisation, T-type calcium needs a
#: hyperpolarised hold to de-inactivate, K_Pst inactivates over seconds...).
_OVERRIDES: dict[str, dict[str, dict]] = {
    "Ih": {
        "activation": dict(hold=-40.0, steps=range(-140, -30, 10), t_step=1500.0, t_post=300.0),
        "inactivation": dict(hold=-40.0, prepulses=range(-140, -30, 10), test=-120.0, t_prepulse=1500.0, t_test=500.0),
        "deactivation": dict(hold=-40.0, activate=-120.0, t_activate=1000.0, tails=range(-100, -20, 10), t_tail=500.0),
    },
    "Ca_LVAst": {
        "activation": dict(hold=-100.0, steps=range(-90, 40, 10), t_step=200.0),
        "inactivation": dict(hold=-100.0, prepulses=range(-120, 0, 10), test=-20.0, t_prepulse=1000.0, t_test=100.0),
        "deactivation": dict(hold=-100.0, activate=-20.0, t_activate=20.0),
    },
    "K_Pst": {
        "activation": dict(t_step=500.0, t_post=200.0),
        "inactivation": dict(prepulses=range(-120, 10, 10), test=20.0, t_prepulse=3000.0, t_test=200.0),
        "deactivation": dict(activate=20.0, t_activate=20.0, t_tail=200.0),
    },
    "K_Tst": {
        "inactivation": dict(test=20.0, t_prepulse=500.0, t_test=100.0),
    },
    "Im": {
        "activation": dict(hold=-80.0, steps=range(-70, 50, 10), t_step=500.0, t_post=200.0),
        "deactivation": dict(activate=20.0, t_activate=200.0, t_tail=300.0),
    },
    "SKv3_1": {
        "activation": dict(steps=range(-60, 70, 10)),
    },
    "Nap_Et2": {
        "activation": dict(t_step=300.0),
        "inactivation": dict(prepulses=range(-120, 10, 10), test=-20.0, t_prepulse=3000.0, t_test=100.0),
    },
    "Ca_HVA": {
        "activation": dict(steps=range(-60, 70, 10), t_step=300.0),
        "inactivation": dict(test=10.0, t_prepulse=2000.0, t_test=100.0),
    },
}


def default_protocols(channel_name: str, dt: float = DT_DEFAULT, ou_seed: int = 0) -> dict[str, VoltageClampProtocol]:
    """The standard Phase-1 protocol set for a channel, keyed by protocol name."""
    ov = _OVERRIDES.get(channel_name, {})
    protocols = {
        "activation": activation_steps(dt=dt, **ov.get("activation", {})),
        "inactivation": steady_state_inactivation(dt=dt, **ov.get("inactivation", {})),
        "deactivation": deactivation_tails(dt=dt, **ov.get("deactivation", {})),
        "ramp": ramp(dt=dt, **ov.get("ramp", {})),
        "ou_noise": ou_noise(dt=dt, seed=ou_seed, **ov.get("ou_noise", {})),
    }
    if channel_name == "SK_E2":
        protocols["calcium_steps"] = calcium_steps(dt=dt)
    return protocols
