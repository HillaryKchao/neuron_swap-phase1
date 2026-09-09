"""Ground-truth parity: golden model vs NEURON running the actual L5PC .mod
files under a single-electrode voltage clamp.

NEURON's fixed-step scheme is staggered (the membrane potential and the
gating states are advanced half a step apart) and the SEClamp has a finite
series resistance, so agreement is checked at the waveform level: gating
variables to 1e-3 absolute and the current to a fraction of its peak, after
allowing for a one-step offset.
"""

import numpy as np
import pytest

from conftest import have_neuron

from neuron_swap.channels import CHANNELS, get_channel
from neuron_swap.protocols import activation_steps, run_voltage_clamp

pytestmark = [pytest.mark.neuron, pytest.mark.skipif(not have_neuron(), reason="NEURON not available")]

DT = 0.025
STEPS = {
    "NaTa_t": (-90.0, [-40.0, 0.0, 30.0], 20.0),
    "NaTs2_t": (-90.0, [-40.0, 0.0, 30.0], 20.0),
    "Nap_Et2": (-90.0, [-50.0, -20.0, 20.0], 100.0),
    "K_Tst": (-90.0, [-20.0, 20.0, 50.0], 100.0),
    "K_Pst": (-90.0, [-20.0, 20.0, 50.0], 300.0),
    "SKv3_1": (-90.0, [-20.0, 20.0, 50.0], 30.0),
    "Im": (-80.0, [-40.0, 0.0, 40.0], 300.0),
    "Ih": (-40.0, [-120.0, -90.0, -60.0], 800.0),
    "Ca_HVA": (-90.0, [-20.0, 10.0, 40.0], 200.0),
    "Ca_LVAst": (-100.0, [-60.0, -30.0, 0.0], 150.0),
}


def _run_neuron(nrn, name, hold, step, t_pre, t_step, t_post, cai=None):
    h = nrn
    sec = h.Section(name=f"sec_{name}")
    sec.L = sec.diam = 20.0
    sec.nseg = 1
    sec.cm = 1e-3  # negligible capacitive transient under clamp
    sec.insert(name)
    if cai is not None:
        # a concentration that no mechanism writes is a plain parameter in
        # NEURON (not reset by finitialize), so set it on the section
        sec.cai = cai
    clamp = h.SEClamp(sec(0.5))
    clamp.rs = 1e-4  # MOhm
    clamp.dur1, clamp.amp1 = t_pre, hold
    clamp.dur2, clamp.amp2 = t_step, step
    clamp.dur3, clamp.amp3 = t_post, hold

    mech = getattr(sec(0.5), name)
    gates = [g.name for g in CHANNELS[name].gates]
    ion_cur = {"na": "ina", "k": "ik", "ca": "ica", "hcn": "ihcn"}[CHANNELS[name].ion]
    rec = {g: h.Vector().record(getattr(mech, f"_ref_{g}")) for g in gates}
    if ion_cur == "ihcn":
        rec_i = h.Vector().record(mech._ref_ihcn)
    else:
        rec_i = h.Vector().record(getattr(sec(0.5), f"_ref_{ion_cur}"))
    rec_v = h.Vector().record(sec(0.5)._ref_v)
    rec_t = h.Vector().record(h._ref_t)

    h.celsius = 34.0
    h.dt = DT
    h.steps_per_ms = 1.0 / DT
    h.tstop = t_pre + t_step + t_post
    h.v_init = hold  # h.run() re-initialises from v_init
    h.finitialize(hold)
    ion = CHANNELS[name].ion
    erev = None if ion == "hcn" else getattr(sec, f"e{ion}")
    h.run()
    out = {g: np.array(rec[g]) for g in gates}
    out["I"] = np.array(rec_i)
    out["v"] = np.array(rec_v)
    out["t"] = np.array(rec_t)
    out["erev"] = erev
    out["gbar"] = getattr(mech, f"g{name}bar")
    del clamp, sec
    return out


def _best_shift_error(a, b, max_shift=1):
    """Min over small shifts of max |a - b|."""
    best = np.inf
    for s in range(0, max_shift + 1):
        n = min(len(a), len(b)) - s
        best = min(best, np.max(np.abs(a[s : s + n] - b[:n])), np.max(np.abs(a[:n] - b[s : s + n])))
    return best


def _interior_mask(V, pad=2):
    """False within ``pad`` samples of a command-voltage transition.

    NEURON's SEClamp applies a new level one sample after the command changes
    while the gating states are already aligned, so the current at the edge
    sample differs by the full driving-force jump; it is not a model error.
    """
    mask = np.ones(len(V), dtype=bool)
    for e in np.flatnonzero(np.diff(V) != 0):
        mask[max(0, e - pad) : e + pad + 2] = False
    return mask


def _current_error(ref_I, I, V):
    """Max |ref - model| away from step edges, normalised by the peak.

    NEURON evaluates BREAKPOINT (the current) before advancing the states in
    each step and records afterwards, so its recorded current lags its
    recorded gates by one sample; the model's ``I[n]`` corresponds to
    NEURON's ``I[n+1]``.  Both alignments are tried and the better is used.
    """
    n = min(len(ref_I), len(I), len(V))
    scale = max(np.max(np.abs(ref_I[:n])), 1e-12)
    best = np.inf
    for s in (0, 1):
        m = _interior_mask(V[: n - s])
        best = min(best, np.max(np.abs(ref_I[s:n][m] - I[: n - s][m])) / scale)
    return best


@pytest.mark.parametrize("name", list(STEPS))
def test_voltage_gated_channel_matches_neuron(nrn, name):
    hold, steps, t_step = STEPS[name]
    t_pre, t_post = 10.0, 20.0
    for step in steps:
        ref = _run_neuron(nrn, name, hold, step, t_pre, t_step, t_post)
        ch = get_channel(name, gbar=ref["gbar"], erev=ref["erev"] if ref["erev"] is not None else None)
        proto = activation_steps(hold=hold, steps=[step], t_pre=t_pre, t_step=t_step, t_post=t_post, dt=DT)
        res = run_voltage_clamp(ch, proto)
        n = min(len(ref["t"]), proto.n_t)
        # the clamp holds the command voltage (NEURON applies a step one
        # sample after the command changes, hence the shift allowance)
        assert _best_shift_error(ref["v"][:n], proto.V[0, :n]) < 0.5
        for gi, g in enumerate(ch.gate_names):
            err = _best_shift_error(ref[g][:n], res.states[gi, 0, :n])
            assert err < 2e-3, (name, step, g, err)
        err = _current_error(ref["I"], res.I[0], proto.V[0])
        assert err < 1e-2, (name, step, err)
        # steady-state current at the end of the step agrees closely
        scale = max(np.max(np.abs(ref["I"])), 1e-12)
        k = int((t_pre + t_step) / DT) - 5
        assert abs(ref["I"][k + 1] - res.I[0, k]) < 1e-3 * scale + 1e-12, (name, step)


def test_sk_e2_matches_neuron_at_fixed_calcium(nrn):
    """SK_E2 at clamped calcium; NEURON keeps cai at cai0 when nothing writes it."""
    name = "SK_E2"
    for cai in (1e-4, 4.3e-4, 2e-3):
        ref = _run_neuron(nrn, name, -80.0, 0.0, 5.0, 10.0, 5.0, cai=cai)
        ch = get_channel(name, gbar=ref["gbar"], erev=ref["erev"])
        proto = activation_steps(hold=-80.0, steps=[0.0], t_pre=5.0, t_step=10.0, t_post=5.0, dt=DT)
        proto.cai = np.full_like(proto.V, cai)
        res = run_voltage_clamp(ch, proto)
        n = min(len(ref["t"]), proto.n_t)
        assert _best_shift_error(ref["z"][:n], res.states[0, 0, :n]) < 1e-3
        assert _current_error(ref["I"], res.I[0], proto.V[0]) < 1e-2
        # the z gate really reflects the clamped calcium (zInf at 4.3e-4 is 0.5)
        if cai == 4.3e-4:
            assert abs(ref["z"][-1] - 0.5) < 1e-3


def test_ca_dynamics_matches_neuron(nrn):
    """CaDynamics_E2 driven by Ca_HVA during a depolarising step."""
    from neuron_swap.channels import CaDynamics_E2
    from neuron_swap.protocols import simulate_clamp

    h = nrn
    sec = h.Section(name="sec_cad")
    sec.L = sec.diam = 20.0
    sec.cm = 1e-3
    sec.insert("Ca_HVA")
    sec.insert("CaDynamics_E2")
    sec.gCa_HVAbar_Ca_HVA = 0.000994
    sec.gamma_CaDynamics_E2 = 0.000609
    sec.decay_CaDynamics_E2 = 210.485284
    clamp = h.SEClamp(sec(0.5))
    clamp.rs = 1e-4
    clamp.dur1, clamp.amp1 = 10.0, -80.0
    clamp.dur2, clamp.amp2 = 100.0, 10.0
    clamp.dur3, clamp.amp3 = 200.0, -80.0
    rec_cai = h.Vector().record(sec(0.5)._ref_cai)
    rec_eca = h.Vector().record(sec(0.5)._ref_eca)
    rec_ica = h.Vector().record(sec(0.5)._ref_ica)
    h.celsius = 34.0
    h.dt = DT
    h.tstop = 310.0
    h.v_init = -80.0
    h.finitialize(-80.0)
    cai0 = sec.cai
    h.run()
    cai_ref = np.array(rec_cai)
    eca_ref = np.array(rec_eca)
    ica_ref = np.array(rec_ica)

    # NEURON recomputes eca from cai by the Nernst equation at every step
    # once a mechanism writes cai; the simulator does the same (nernst_ca).
    pool = CaDynamics_E2(gamma=0.000609, decay=210.485284)
    ca = get_channel("Ca_HVA", gbar=0.000994)
    proto = activation_steps(hold=-80.0, steps=[10.0], t_pre=10.0, t_step=100.0, t_post=200.0, dt=DT)
    run = simulate_clamp([ca], proto, ca_pool=pool, cai0=cai0, cao=sec.cao, celsius=34.0)
    n = min(len(cai_ref), proto.n_t)
    assert np.max(np.abs(eca_ref - eca_ref[0])) > 5.0  # eca really does move
    erev = run["Ca_HVA"].extras["erev"][0]
    assert np.max(np.abs(erev[:n] - eca_ref[:n])) < 0.5
    rel = np.max(np.abs(run.cai[0, :n] - cai_ref[:n])) / np.max(cai_ref)
    assert rel < 3e-2, rel
    assert _current_error(ica_ref, run["Ca_HVA"].I[0], proto.V[0]) < 2e-2
    del clamp, sec
