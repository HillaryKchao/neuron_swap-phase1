#pragma once

#include "ConductanceTypes.hpp"
#include <utility>

std::pair<scalar_t, scalar_t> Ca_HVA_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V);
scalar_t Ca_HVA_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Ca, scalar_t dt = 0.025);