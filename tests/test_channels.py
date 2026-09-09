"""Golden-model channel tests: parity with the .mod equations, integrator
exactness, and API invariants."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neuron_swap.channels import CHANNELS, CaDynamics_E2, get_channel
from neuron_swap.channels.base import QT

V_GRID = np.linspace(-120.0, 60.0, 181)  # includes every singular point of the guards
CA_GRID = np.logspace(-8, -1, 30)

# ---------------------------------------------------------------------------
# Independent scalar transcription of PROCEDURE rates() of every .mod file.
# Written separately (math.exp, scalar) so that a transcription error in the
# vectorised implementation cannot be masked by the same error here.
# ---------------------------------------------------------------------------
exp = math.exp


def _nat(v, va, vh):
    if v == va:
        v = v + 0.0001
    mA = (0.182 * (v - va)) / (1 - exp(-(v - va) / 6))
    mB = (0.124 * (-v + va)) / (1 - exp(-(-v + va) / 6))
    if v == vh:
        v = v + 0.0001
    hA = (-0.015 * (v - vh)) / (1 - exp((v - vh) / 6))
    hB = (-0.015 * (-v + vh)) / (1 - exp((-v + vh) / 6))
    return {"m": (mA / (mA + mB), 1 / (mA + mB) / QT), "h": (hA / (hA + hB), 1 / (hA + hB) / QT)}


def ref_NaTa_t(v):
    return _nat(v, -38.0, -66.0)


def ref_NaTs2_t(v):
    return _nat(v, -32.0, -60.0)


def ref_Nap_Et2(v):
    mInf = 1.0 / (1 + exp((v + 52.6) / -4.6))
    if v == -38:
        v = v + 0.0001
    mA = (0.182 * (v + 38)) / (1 - exp(-(v + 38) / 6))
    mB = (0.124 * (-v - 38)) / (1 - exp(-(-v - 38) / 6))
    mTau = 6 * (1 / (mA + mB)) / QT
    if v == -17:
        v = v + 0.0001
    if v == -64.4:
        v = v + 0.0001
    hInf = 1.0 / (1 + exp((v + 48.8) / 10))
    hA = -2.88e-6 * (v + 17) / (1 - exp((v + 17) / 4.63))
    hB = 6.94e-6 * (v + 64.4) / (1 - exp(-(v + 64.4) / 2.63))
    return {"m": (mInf, mTau), "h": (hInf, 1 / (hA + hB) / QT)}


def ref_K_Tst(v):
    v = v + 10
    return {
        "m": (1 / (1 + exp(-v / 19)), (0.34 + 0.92 * exp(-(((v + 71) / 59) ** 2))) / QT),
        "h": (1 / (1 + exp((v + 66) / 10)), (8 + 49 * exp(-(((v + 73) / 23) ** 2))) / QT),
    }


def ref_K_Pst(v):
    v = v + 10
    mInf = 1 / (1 + exp(-(v + 1) / 12))
    if v < -50:
        mTau = (1.25 + 175.03 * exp(0.026 * v)) / QT
    else:
        mTau = (1.25 + 13 * exp(-0.026 * v)) / QT
    hInf = 1 / (1 + exp((v + 54) / 11))
    hTau = (360 + (1010 + 24 * (v + 55)) * exp(-(((v + 75) / 48) ** 2))) / QT
    return {"m": (mInf, mTau), "h": (hInf, hTau)}


def ref_SKv3_1(v):
    return {"m": (1 / (1 + exp((v - 18.7) / -9.7)), 4.0 / (1 + exp((v + 46.56) / -44.14)))}


def ref_Im(v):
    a = 3.3e-3 * exp(0.1 * (v + 35))
    b = 3.3e-3 * exp(-0.1 * (v + 35))
    return {"m": (a / (a + b), 1 / (a + b) / QT)}


def ref_Ih(v):
    if v == -154.9:
        v = v + 0.0001
    a = 0.001 * 6.43 * (v + 154.9) / (exp((v + 154.9) / 11.9) - 1)
    b = 0.001 * 193 * exp(v / 33.1)
    return {"m": (a / (a + b), 1 / (a + b))}


def ref_Ca_HVA(v):
    if v == -27:
        v = v + 0.0001
    mA = (0.055 * (-27 - v)) / (exp((-27 - v) / 3.8) - 1)
    mB = 0.94 * exp((-75 - v) / 17)
    hA = 0.000457 * exp((-13 - v) / 50)
    hB = 0.0065 / (exp((-v - 15) / 28) + 1)
    return {"m": (mA / (mA + mB), 1 / (mA + mB)), "h": (hA / (hA + hB), 1 / (hA + hB))}


def ref_Ca_LVAst(v):
    v = v + 10
    return {
        "m": (1 / (1 + exp((v + 30) / -6)), (5 + 20 / (1 + exp((v + 25) / 5))) / QT),
        "h": (1 / (1 + exp((v + 80) / 6.4)), (20 + 50 / (1 + exp((v + 40) / 7))) / QT),
    }


def ref_SK_E2(ca):
    if ca < 1e-7:
        ca = ca + 1e-7
    return {"z": (1 / (1 + (0.00043 / ca) ** 4.8), 1.0)}


REFS = {
    "NaTa_t": ref_NaTa_t,
    "NaTs2_t": ref_NaTs2_t,
    "Nap_Et2": ref_Nap_Et2,
    "K_Tst": ref_K_Tst,
    "K_Pst": ref_K_Pst,
    "SKv3_1": ref_SKv3_1,
    "Im": ref_Im,
    "Ih": ref_Ih,
    "Ca_HVA": ref_Ca_HVA,
    "Ca_LVAst": ref_Ca_LVAst,
}

VOLTAGE_CHANNELS = [n for n in CHANNELS if not CHANNELS[n].calcium_dependent]


@pytest.mark.parametrize("name", VOLTAGE_CHANNELS)
def test_rates_match_independent_transcription(name):
    ch = get_channel(name)
    inf, tau = ch.rates(V_GRID)
    for j, v in enumerate(V_GRID):
        ref = REFS[name](float(v))
        for i, gate in enumerate(ch.gate_names):
            r_inf, r_tau = ref[gate]
            assert inf[i, j] == pytest.approx(r_inf, rel=1e-12, abs=1e-15), (name, gate, v)
            assert tau[i, j] == pytest.approx(r_tau, rel=1e-12), (name, gate, v)


def test_sk_e2_rates_match_transcription():
    ch = get_channel("SK_E2")
    inf, tau = ch.rates(np.full_like(CA_GRID, -50.0), cai=CA_GRID)
    for j, ca in enumerate(CA_GRID):
        r_inf, r_tau = ref_SK_E2(float(ca))["z"]
        assert inf[0, j] == pytest.approx(r_inf, rel=1e-12)
        assert tau[0, j] == pytest.approx(r_tau)


@pytest.mark.parametrize("name", VOLTAGE_CHANNELS)
def test_rates_are_well_behaved(name):
    """inf in [0, 1], tau finite and positive, also exactly at the guarded
    singular voltages of the .mod files."""
    ch = get_channel(name)
    v = np.concatenate([V_GRID, [-38.0, -66.0, -32.0, -60.0, -17.0, -64.4, -154.9, -27.0]])
    inf, tau = ch.rates(v)
    assert np.all(np.isfinite(inf)) and np.all(np.isfinite(tau))
    assert np.all(inf >= 0) and np.all(inf <= 1)
    assert np.all(tau > 0)


@pytest.mark.parametrize("name", VOLTAGE_CHANNELS)
def test_mod_parameters_match(name, mod_text):
    """gbar default and gate exponents agree with the .mod text."""
    import re

    ch = get_channel(name)
    txt = mod_text[name]
    m = re.search(r"g\w+bar\s*=\s*([0-9.eE+-]+)", txt)
    assert float(m.group(1)) == ch.default_gbar
    # conductance expression, e.g. "gNaTa_tbar*m*m*m*h" or "gK_Tstbar*(m^4)*h"
    g = re.search(r"g\w*\s*=\s*g\w+bar\s*\*\s*(.+)", txt).group(1).strip()
    for gate, p in zip(ch.gate_names, ch.powers):
        count = g.count(gate) if f"{gate}^" not in g else int(re.search(rf"{gate}\^(\d)", g).group(1))
        assert count == p, (name, gate, g)


@pytest.mark.parametrize("name", list(CHANNELS))
def test_steady_state_is_fixed_point(name):
    ch = get_channel(name)
    kw = {"cai": 1e-3} if ch.calcium_dependent else {}
    for v in (-80.0, -40.0, 0.0):
        s = ch.steady_state(v, **kw)
        assert np.allclose(ch.derivatives(s, v, **kw), 0.0, atol=1e-14)
        assert np.allclose(ch.step(s, v, 0.025, **kw), s)


@pytest.mark.parametrize("name", list(CHANNELS))
def test_cnexp_matches_ode_solution(name):
    """The cnexp update is the exact solution for a voltage step, so a
    sequence of small steps must agree with an adaptive ODE solver."""
    ch = get_channel(name)
    kw = {"cai": 5e-3} if ch.calcium_dependent else {}
    v0, v1, dt, n = -90.0, 0.0, 0.025, 400
    s = ch.steady_state(v0, cai=5e-5 if ch.calcium_dependent else None)
    traj = [s]
    for _ in range(n):
        s = ch.step(s, v1, dt, **kw)
        traj.append(s)
    traj = np.array(traj)

    sol = solve_ivp(
        lambda t, y: ch.derivatives(y, v1, **kw),
        (0, n * dt),
        traj[0],
        t_eval=np.arange(n + 1) * dt,
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.allclose(traj, sol.y.T, atol=1e-7)


@pytest.mark.parametrize("name", VOLTAGE_CHANNELS)
def test_tau_scale(name):
    ch = get_channel(name)
    fast = get_channel(name, tau_scale=0.1)
    inf, tau = ch.rates(V_GRID)
    inf2, tau2 = fast.rates(V_GRID)
    assert np.allclose(inf, inf2)
    assert np.allclose(tau2, 0.1 * tau)
    # accelerated model reaches the same state in 1/10 of the time
    s0 = ch.steady_state(-90.0)
    assert np.allclose(ch.step(s0, 0.0, 1.0), fast.step(s0, 0.0, 0.1))


def test_current_sign_and_vectorisation():
    na = get_channel("NaTa_t")
    k = get_channel("SKv3_1")
    v = np.array([-70.0, 0.0, 40.0])
    s_na = na.steady_state(v)
    s_k = k.steady_state(v)
    assert s_na.shape == (2, 3) and s_k.shape == (1, 3)
    i_na = na.current(s_na, v)
    i_k = k.current(s_k, v)
    assert i_na.shape == (3,)
    assert i_na[1] < 0 < i_k[1]  # inward Na, outward K at 0 mV
    assert i_na[2] < 0 and i_k[2] > 0
    # conductance scales with gbar
    na2 = get_channel("NaTa_t", gbar=2 * na.gbar)
    assert np.allclose(na2.current(s_na, v), 2 * i_na)


def test_sk_e2_requires_calcium():
    ch = get_channel("SK_E2")
    with pytest.raises(ValueError):
        ch.rates(-50.0)
    inf, _ = ch.rates(-50.0, cai=np.array([1e-5, 4.3e-4, 1e-2]))
    assert inf[0, 0] < 0.05 and inf[0, 1] == pytest.approx(0.5) and inf[0, 2] > 0.99


def test_ca_dynamics_exact_step_and_steady_state():
    pool = CaDynamics_E2(gamma=0.000609, decay=210.485284)
    ica = -0.05  # mA/cm^2 inward
    css = pool.steady_state(ica)
    assert css > pool.min_cai
    assert pool.derivative(css, ica) == pytest.approx(0.0, abs=1e-18)
    # exact update vs ODE solver
    dt, n = 0.025, 2000
    c = pool.min_cai
    traj = [c]
    for _ in range(n):
        c = pool.step(c, ica, dt)
        traj.append(c)
    sol = solve_ivp(lambda t, y: pool.derivative(y, ica), (0, n * dt), [pool.min_cai], t_eval=np.arange(n + 1) * dt, rtol=1e-10, atol=1e-14)
    assert np.allclose(traj, sol.y[0], rtol=1e-8, atol=1e-12)
    # zero current relaxes to minCai
    assert pool.steady_state(0.0) == pytest.approx(pool.min_cai)
    # time scaling: faster pool with the same steady state
    fast = CaDynamics_E2(gamma=0.000609, decay=210.485284, tau_scale=0.01)
    assert fast.steady_state(ica) == pytest.approx(css)
    assert fast.step(pool.min_cai, ica, 0.01) == pytest.approx(pool.step(pool.min_cai, ica, 1.0))


def test_registry():
    assert len(CHANNELS) == 11
    with pytest.raises(KeyError):
        get_channel("KdShu2007")
    for name, cls in CHANNELS.items():
        assert cls.name == name
        assert cls.ion in {"na", "k", "ca", "hcn"}
