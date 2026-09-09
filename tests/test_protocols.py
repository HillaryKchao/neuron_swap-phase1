"""Voltage-clamp protocol builders and simulator tests."""

import numpy as np
import pytest

from neuron_swap.channels import CHANNELS, CaDynamics_E2, get_channel
from neuron_swap.channels.l5pc import ca_dynamics_for_region, representative_channel
from neuron_swap.protocols import (
    activation_steps,
    calcium_steps,
    deactivation_tails,
    default_protocols,
    ou_noise,
    ramp,
    run_voltage_clamp,
    simulate_clamp,
    steady_state_inactivation,
)


def test_activation_steps_grid_and_epochs():
    p = activation_steps(hold=-90, steps=[-40, 0, 40], t_pre=10, t_step=20, t_post=5, dt=0.025)
    assert p.V.shape == (3, int(round(35 / 0.025)) + 1)
    assert p.t[-1] == pytest.approx(35.0)
    assert p.dt == 0.025
    assert np.all(p.V[:, p.epoch_mask("pre")] == -90)
    assert np.allclose(p.V[:, p.epoch_mask("step")], np.array([[-40], [0], [40]]))
    assert np.all(p.V[:, p.epoch_mask("post")] == -90)
    assert p.V[1, p.epoch_slice("step")].size == int(round(20 / 0.025))


def test_other_builders_shapes():
    dt = 0.05
    p = steady_state_inactivation(prepulses=[-100, -50], t_pre=5, t_prepulse=50, t_test=10, t_post=5, dt=dt)
    assert p.n_sweeps == 2 and p.duration == pytest.approx(70)
    assert set(p.epochs) == {"pre", "prepulse", "test", "post"}
    p = deactivation_tails(tails=[-100, -80, -60], dt=dt)
    assert p.n_sweeps == 3 and np.all(p.V[:, p.epoch_mask("activate")] == 20)
    p = ramp(v_start=-100, v_end=50, t_pre=5, t_ramp=100, t_post=5, dt=dt)
    r = p.V[0, p.epoch_slice("ramp")]
    assert r[0] == -100 and r[-1] < 50 and np.all(np.diff(r) > 0)
    p = calcium_steps(steps=[1e-5, 1e-3], dt=dt)
    assert p.cai is not None and p.cai.shape == p.V.shape
    assert np.all(p.V == -50.0)


def test_ou_noise_is_stationary_and_reproducible():
    p = ou_noise(mean=-50, sd=20, tau=5, duration=5000, n_sweeps=2, seed=3, dt=0.025)
    q = ou_noise(mean=-50, sd=20, tau=5, duration=5000, n_sweeps=2, seed=3, dt=0.025)
    assert np.array_equal(p.V, q.V)
    v = p.V.ravel()
    assert abs(v.mean() + 50) < 3
    assert abs(v.std() - 20) < 3
    # autocorrelation at lag tau is ~ e^-1
    lag = int(5 / 0.025)
    x = p.V[0] - p.V[0].mean()
    ac = np.dot(x[:-lag], x[lag:]) / np.dot(x, x)
    assert abs(ac - np.exp(-1)) < 0.1
    assert v.min() >= -120 and v.max() <= 60


@pytest.mark.parametrize("name", list(CHANNELS))
def test_default_protocols_exist(name):
    ps = default_protocols(name, dt=0.1)
    assert {"activation", "inactivation", "deactivation", "ramp", "ou_noise"} <= set(ps)
    if name == "SK_E2":
        assert "calcium_steps" in ps
    for p in ps.values():
        assert p.dt == 0.1 and np.all(np.isfinite(p.V))


def test_sodium_activation_gives_inward_transient():
    ch = representative_channel("NaTa_t")
    p = activation_steps(hold=-90, steps=[-20, 0, 20], t_step=20, dt=0.025)
    r = run_voltage_clamp(ch, p)
    assert r.states.shape == (2, 3, p.n_t)
    assert r.I.shape == (3, p.n_t)
    # holding state is the steady state
    assert np.allclose(r.states[:, :, 0], ch.steady_state(np.full(3, -90.0)))
    peak = r.peak_current("step")
    assert np.all(peak < 0)
    ttp = r.time_to_peak("step")
    assert np.all(ttp > 0) and np.all(ttp < 2.0)  # sub-ms activation at 34 C
    # inactivates: steady state much smaller than peak
    assert np.all(np.abs(r.steady_state_current("step")) < 0.05 * np.abs(peak))


def test_potassium_activation_is_outward_and_non_inactivating():
    ch = representative_channel("SKv3_1")
    p = activation_steps(hold=-90, steps=[0, 40], t_step=50, dt=0.025)
    r = run_voltage_clamp(ch, p)
    assert np.all(r.steady_state_current("step") > 0)
    assert np.allclose(r.steady_state_current("step"), r.peak_current("step"), rtol=0.02)


def test_ih_activates_on_hyperpolarisation():
    ch = get_channel("Ih")
    p = activation_steps(hold=-40, steps=[-120, -80], t_step=1000, dt=0.05)
    r = run_voltage_clamp(ch, p)
    m_end = r.gate("m")[:, r.epoch("step")][:, -1]
    assert m_end[0] > m_end[1] > 0.0
    assert np.all(r.steady_state_current("step") < 0)  # inward below ehcn


def test_sk_e2_calcium_clamp_and_pool_drive():
    sk = representative_channel("SK_E2")
    p = calcium_steps(steps=[1e-5, 4.3e-4, 1e-2], t_step=10, dt=0.025)
    r = run_voltage_clamp(sk, p)
    z_end = r.gate("z")[:, r.epoch("step")][:, -1]
    assert z_end[1] == pytest.approx(0.5, abs=1e-3)
    assert r.cai is not None and np.allclose(r.cai, p.cai)
    assert r.I[2, r.epoch("step")][-1] > 0  # outward K current at -50 mV

    # driven by Ca_HVA + Ca_LVAst through the somatic calcium pool
    pool = ca_dynamics_for_region("somatic")
    p2 = activation_steps(hold=-80, steps=[0.0], t_step=200, t_post=300, dt=0.025)
    r2 = run_voltage_clamp(
        sk,
        p2,
        ca_pool=pool,
        ca_sources=[representative_channel("Ca_HVA"), representative_channel("Ca_LVAst")],
    )
    assert r2.cai is not None
    cai_step = r2.cai[0, r2.epoch("step")]
    assert cai_step[-1] > cai_step[0]  # calcium accumulates during the step
    assert r2.I[0, r2.epoch("step")][-1] > r2.I[0, 0]  # SK current follows
    assert "ica_sources" in r2.extras and r2.extras["ica_sources"].shape[0] == 2


def test_simulate_clamp_multi_channel_total_current():
    chans = [get_channel("NaTa_t"), get_channel("SKv3_1")]
    p = activation_steps(steps=[0.0], t_step=10, dt=0.025)
    run = simulate_clamp(chans, p)
    assert set(run.results) == {"NaTa_t", "SKv3_1"}
    assert np.allclose(run.I_total, run["NaTa_t"].I + run["SKv3_1"].I)


def test_record_every_decimates_outputs():
    ch = get_channel("Im")
    p = activation_steps(steps=[0.0, 20.0], t_step=10, dt=0.025)
    r = run_voltage_clamp(ch, p, record_every=4)
    assert r.dt == pytest.approx(0.1)
    assert r.I.shape[1] == (p.n_t + 3) // 4
    full = run_voltage_clamp(ch, p)
    assert np.allclose(r.I, full.I[:, ::4])


def test_save_roundtrip(tmp_path):
    ch = get_channel("K_Tst")
    r = run_voltage_clamp(ch, activation_steps(steps=[0.0], t_step=5, dt=0.025))
    r.save(tmp_path / "k_tst.npz")
    d = np.load(tmp_path / "k_tst.npz")
    assert np.allclose(d["I"], r.I, rtol=1e-6) and d["states"].shape == r.states.shape
    assert d["I"].dtype == np.float32
    r.save(tmp_path / "k_tst64.npz", dtype=np.float64)
    assert np.array_equal(np.load(tmp_path / "k_tst64.npz")["I"], r.I)
    import json

    meta = json.loads(str(d["meta"]))
    assert meta["channel"]["name"] == "K_Tst" and meta["protocol"]["name"] == "activation"
