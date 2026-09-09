# C++ conductance translation

This directory contains a direct C++ translation of the Python conductance scripts in `src/conductances`.

The code is intentionally simple and syntax-oriented:

- Each Python module has a matching `.hpp` and `.cpp` file.
- Each file exposes uniquely named functions based on the source module name.
- [ConductanceTypes.hpp](ConductanceTypes.hpp) is the host-side header for pure-C++ simulation.
- [ConductanceTypes_hls.hpp](ConductanceTypes_hls.hpp) is the HLS-side header for Bambu-oriented builds.
- `StochKV` exposes a host-side sampler in the pure-C++ header and a synthesizable sampler in the HLS header.
- Include one of the conductance type headers before [StochKV.hpp](StochKV.hpp) to select the desired behavior.

## Bambu HLS next steps

The next synthesis-oriented step is to make these modules acceptable to Bambu in a Docker-based flow.

Recommended workflow:

1. Build a small Docker image that contains Bambu, a C++ compiler, and any helper tools needed for simulation.
2. Replace `scalar_t` in [ConductanceTypes_hls.hpp](ConductanceTypes_hls.hpp) with a Bambu-friendly fixed-point type, then tune bit widths per channel.
3. Keep one top-level function per conductance so Bambu can synthesize each channel independently.
4. Add a simple command-line or testbench entry point for each channel before generating RTL.

## Recommended validation plan

Do not compare outputs point-by-point. Use waveform-level checks against the Python equations instead.

Suggested validation steps:

1. Run a stereotyped voltage-clamp protocol in Python and generate reference current waveforms.
2. Run the same stimulus through the C++ implementation with the same time step and initial state.
3. Normalize or align the waveforms by allowing small time shifts and amplitude scaling where the hardware model requires it.
4. Compare the maximum correlation, peak timing, steady-state level, and gross waveform shape rather than every sample.
5. For stochastic models, compare distributions or ensemble statistics instead of a single trace.

These checks should be added after the synthesis-ready model is agreed on, not before the fixed-point interface is finalized.

## Stochastic sampling caveat

The caveat is not that the whole StochKV path is inherently unsynthesizable. The issue is that the standard C++ random facilities used in the pure-C++ workflow, such as `std::mt19937` and `std::binomial_distribution`, are not a good match for Bambu HLS.

For synthesis, there are three practical options:

1. Keep the stochastic sampling outside the synthesized block and pass the sampled transition counts into `StochKV_current`.
2. Use a small synthesizable PRNG in the HLS header, such as the xorshift-style helper currently provided in [ConductanceTypes_hls.hpp](ConductanceTypes_hls.hpp), and generate the transition counts inside the HLS model.
3. Replace the probabilistic update entirely with a deterministic approximation if the hardware model does not need explicit stochastic behavior.

The current layout is meant to support both workflows:

- [ConductanceTypes.hpp](ConductanceTypes.hpp) uses standard-library RNG support for host-side simulation.
- [ConductanceTypes_hls.hpp](ConductanceTypes_hls.hpp) uses a lightweight, synthesizable integer-based sampler intended for Bambu-oriented builds.

So the caveat is mainly about implementation choice, not about the stochastic conductance being impossible to synthesize. If the final HDL should preserve randomness, the RNG must come from a synthesizable source. If you only need functional comparison against the Python model, it is often better to keep the randomness external and treat the synthesized block as deterministic given its inputs.