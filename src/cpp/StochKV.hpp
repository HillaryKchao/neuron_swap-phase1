#pragma once

// Include a conductance types header before this file.
// Use ConductanceTypes.hpp for pure C++ simulation or ConductanceTypes_hls.hpp
// for Bambu-oriented synthesis builds.

scalar_t StochKV_derivatives(scalar_t n, scalar_t V, scalar_t celsius);
scalar_t StochKV_current(scalar_t n, scalar_t V, scalar_t celsius, scalar_t E_K, scalar_t area, StochKVTransitions transitions, scalar_t dt = 0.025);