# Phase 1 reference waveforms

Voltage-clamp recordings of the golden model for the 11 L5PC ion channels
plus the calcium pool. One folder per channel: NaTa_t, NaTs2_t, Nap_Et2,
K_Tst, K_Pst, SKv3_1, SK_E2, Im, Ih, Ca_HVA, Ca_LVAst.

## What the model is

A Python transcription of the BBP L5PC `.mod` files (Hay et al. 2011,
`cADpyr232_L5_TTPC1`), integrated with NEURON's exact cnexp update at
dt = 0.025 ms, 34 °C, E_Na = 50 mV, E_K = −85 mV, E_HCN = −45 mV, E_Ca from
the Nernst equation. Each channel uses the maximal conductance of one
representative L5PC region:

| region | channels |
|---|---|
| axon | NaTa_t, Nap_Et2, K_Tst, K_Pst, SKv3_1, Ca_LVAst |
| soma | NaTs2_t, SK_E2, Ca_HVA |
| apical | Im |
| basal | Ih |

The model was validated against NEURON running the same `.mod` files:
gates agree within 2e-3, currents within 1 % of peak.

## Files per channel

| file | protocol |
|---|---|
| `activation.npz` | hold, depolarising steps (hyperpolarising for Ih) |
| `inactivation.npz` | long prepulses then a fixed test step |
| `deactivation.npz` | open the channel, then step to tail voltages |
| `ramp.npz` | slow voltage ramp |
| `ou_noise.npz` | Ornstein–Uhlenbeck random voltage, mean −50 mV, sd 20 mV, tau 5 ms, 20 s × 2 realisations, seed 0 |

SK_E2 additionally has `calcium_steps.npz`: [Ca]ᵢ stepped at −50 mV.

## Arrays in each npz

| key | shape | contents |
|---|---|---|
| `t` | `(n_t,)` | time, ms |
| `V` | `(n_sweeps, n_t)` | command voltage, mV |
| `states` | `(n_gates, n_sweeps, n_t)` | gating variables, 0–1, in the order given by `meta["gate_names"]` |
| `g` | `(n_sweeps, n_t)` | conductance, S/cm² |
| `I` | `(n_sweeps, n_t)` | current, mA/cm² (NEURON sign convention: inward negative, so Na and Ca currents are negative, K currents positive) |
| `meta` | JSON string | channel name, ion, gate names and exponents, gbar, reversal potential, tau_scale, protocol name, dt, epoch times (pre / step / post), sweep values and their label |

SK_E2 files also carry:

| key | shape | contents |
|---|---|---|
| `cai` | `(n_sweeps, n_t)` | [Ca]ᵢ, mM, from the CaDynamics_E2 pool driven by Ca_HVA + Ca_LVAst |
| `extra_ica_sources` | `(2, n_sweeps, n_t)` | the Ca_HVA and Ca_LVAst currents that fed the pool |

`calcium_steps.npz` has `cai_command` instead, the clamped [Ca]ᵢ.

## Loading

```python
import numpy as np, json

d = np.load("results/phase1/NaTa_t/activation.npz")
meta = json.loads(str(d["meta"]))
t, V, I, states = d["t"], d["V"], d["I"], d["states"]
```

## Things to be aware of

- Arrays are float32.
- The simulation ran at 0.025 ms but the step/ramp protocols are stored
  every 0.1 ms and the noise protocols every 0.5 ms. If you need full
  resolution or float64, run `scripts/run_phase1.py --record-dt 0.025` and
  save with `dtype=np.float64`; it regenerates everything in a few minutes.
- Currents are per unit area. Multiply by membrane area to get absolute
  current, or compare `I / gbar` to remove the conductance scale entirely.
- `tau_scale = 1` in all files (biological time). The same script can
  produce time-scaled references once the hardware acceleration factor is
  chosen.
- The current at a given sample is computed from the gate states at that
  same sample, before the states are advanced. NEURON's recorded current
  lags its recorded gates by one sample, so do not be surprised by a
  one-sample offset if you compare against NEURON directly.
- The DCA signature JSON files next to the npz files are the intended
  parity metric for later phases; they were computed from `ou_noise.npz`.
