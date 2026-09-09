# Conductance scripts (legacy reference)

> **Status (Phase 1):** these scripts are superseded by the installable
> golden model in `src/neuron_swap/channels`, which is transcribed from the
> L5PC `.mod` files and validated against NEURON (see `docs/phase1.md`).
> They are kept unchanged as the reference the current `src/cpp` port was
> made from.  Discrepancies with the `.mod` files found while building the
> golden model:
>
> * `K_Pst.py` uses `^` (bitwise XOR) instead of `**` in `hTau`, which raises
>   a `TypeError` on floats.
> * `SK_E2.py`: `current()` reads an undefined `ca` and computes the driving
>   force as `E_K` instead of `V - E_K`.
> * `Nap_Et2.py`: `current()` has no `dt` / cnexp update unlike the other
>   modules, and the voltage-shift epsilon is 0.001 instead of 0.0001.
> * `Ih.py` docstring is titled "Im".
> * There is no `CaDynamics_E2` model although `SK_E2` needs `cai`.
> * `Ca.py` duplicates `Ca_HVA`; `KdShu2007` and `StochKV` are not part of
>   the L5PC mechanism set.
> * `current()` advances the gating state internally without returning it,
>   so a caller cannot integrate consistently across calls.

Each script provides functions for updating the conductances of a single neuron channel. These Python scripts should be used as a reference for the hardware implmentations to be developed in this repository.

## Test Suite (In Progress)

1. An automated test-suite will be designed to produce current waveforms for a stereotyped voltage-clamp experiment. The tests should be designed to produce a specified number of points with a specified time step, with reasonable defaults provided for both.
2. Once hardware models are available, a testing environment will be developed that to automatically compare the simulated hardware against the ideal dynamics with some reasonable shifting and scaling of waveforms as needed. While Python simulations can be conducted with full floating-point precision, the digitized hardware models will be at arbitrarily low precision, and spice models will have scaled dyanmics and voltage ranges. As a result, the test-suite should not perform a point-by-point comparison, but instead should determine suitable voltage and temporal scaling to compare the maximum correlation between the two waveforms.

## High-level Synthesis (HLS)

In tandem with the test-suite development, the dynamic equations should be translated to C++ for HLS to automate the conversion to digitized models. HLS should be used to generate synthesized RTL, and OpenROAD will be used for automated RTL-to-GDS and testbench simmulations of a given implementation.
