"""DCA signature tests (require the BouchardLab ``dca`` package)."""

import numpy as np
import pytest

from conftest import have_dca

pytestmark = [pytest.mark.dca, pytest.mark.skipif(not have_dca(), reason="dca not installed")]


def _ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    return x


def test_pi_of_ar1_matches_theory():
    """For a Gaussian AR(1) process with T=1 windows, PI = -0.5 log(1 - phi^2)."""
    from neuron_swap.dca import pi_curve

    phi = 0.8
    x = _ar1(200_000, phi)[:, None]
    pi = pi_curve(x, [1, 2])
    expected = -0.5 * np.log(1 - phi**2)
    assert pi[1] == pytest.approx(expected, rel=0.03)
    # AR(1) is Markov: a longer window adds no information
    assert pi[2] == pytest.approx(expected, rel=0.03)


def test_input_output_mi_detects_driven_response():
    from neuron_swap.dca.signatures import input_output_mi

    rng = np.random.default_rng(1)
    u = _ar1(50_000, 0.9, seed=2)[:, None]
    x_driven = 0.7 * np.roll(u, 1) + 0.3 * rng.standard_normal(u.shape)
    x_indep = rng.standard_normal(u.shape)
    assert input_output_mi(x_driven, u, T=3) > 0.5
    assert input_output_mi(x_indep, u, T=3) < 0.05


def test_signature_on_channel_response():
    from neuron_swap.channels.l5pc import representative_channel
    from neuron_swap.dca import ChannelSignature, compare_signatures, compute_signature, load_signature
    from neuron_swap.protocols import ou_noise, run_voltage_clamp

    ch = representative_channel("NaTa_t")
    r = run_voltage_clamp(ch, ou_noise(duration=1500, seed=0, dt=0.025))
    sig = compute_signature(r, T_values=(1, 2, 5), bin_ms=0.5, seed=0)
    assert isinstance(sig, ChannelSignature)
    assert sig.feature_names == ["m", "h", "I"]
    assert sig.n_samples == 3000
    # PI is non-negative and non-decreasing in T for the full response
    pis = [sig.pi_curve[T] for T in (1, 2, 5)]
    assert pis[0] >= 0 and pis[-1] >= pis[0] - 1e-6
    # each single feature carries at most the full-response PI
    for name, c in sig.pi_per_feature.items():
        assert c[5] <= sig.pi_curve[5] + 1e-6
    # DCA: d = n_features recovers the full PI, d = 1 <= that
    assert sig.dca[3]["pi"] == pytest.approx(sig.pi_curve[5], rel=1e-3, abs=1e-3)
    assert sig.dca[1]["pi"] <= sig.dca[3]["pi"] + 1e-6
    assert np.asarray(sig.dca[1]["coef"]).shape == (3, 1)
    # the clamp voltage carries information about the response
    assert sig.io_mi_curve[5] > 0.1
    if sig.iodca:
        assert np.asarray(sig.iodca["coef"]).shape == (3, 1)


def test_signature_roundtrip_and_compare(tmp_path):
    from neuron_swap.channels import get_channel
    from neuron_swap.dca import compare_signatures, compute_signature, load_signature
    from neuron_swap.protocols import ou_noise, run_voltage_clamp

    ch = get_channel("SKv3_1")
    r = run_voltage_clamp(ch, ou_noise(duration=800, seed=1, dt=0.025))
    sig = compute_signature(r, T_values=(1, 4), bin_ms=1.0, io=False)
    sig.save(tmp_path / "sig.json")
    back = load_signature(tmp_path / "sig.json")
    assert back.pi_curve == sig.pi_curve and back.feature_names == sig.feature_names
    diff = compare_signatures(sig, back)
    assert all(v == 0 for v in diff["pi_abs_diff"].values())
    assert all(abs(v) < 1e-6 for v in diff["dca_angle_deg"].values())

    # a time-scaled channel sampled at the equivalently scaled bin has the same signature
    fast = get_channel("SKv3_1", tau_scale=0.5)
    p = ou_noise(duration=800, seed=1, dt=0.025)
    p_fast = type(p)(name=p.name, dt=p.dt * 0.5, V=p.V, sweep_values=p.sweep_values, epochs={"all": (0, 400)})
    r_fast = run_voltage_clamp(fast, p_fast)
    sig_fast = compute_signature(r_fast, T_values=(1, 4), bin_ms=0.5, io=False)
    diff = compare_signatures(sig, sig_fast)
    assert max(diff["pi_rel_diff"].values()) < 1e-6
