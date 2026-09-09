#pragma once

#include "ConductanceTypes.hpp"

scalar_t Ih_derivatives(scalar_t m, scalar_t V);
scalar_t Ih_current(scalar_t m, scalar_t V, scalar_t dt = 0.025);