#pragma once

#include "ConductanceTypes.hpp"

scalar_t SKv3_1_derivatives(scalar_t m, scalar_t V);
scalar_t SKv3_1_current(scalar_t m, scalar_t V, scalar_t E_K, scalar_t dt = 0.025);