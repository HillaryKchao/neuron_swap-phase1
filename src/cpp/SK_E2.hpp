#pragma once

#include "ConductanceTypes.hpp"

scalar_t SK_E2_derivatives(scalar_t z, scalar_t ca);
scalar_t SK_E2_current(scalar_t z, scalar_t ca, scalar_t V, scalar_t E_K, scalar_t dt = 0.025);