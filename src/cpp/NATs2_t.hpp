#pragma once

#include "ConductanceTypes.hpp"
#include <utility>

std::pair<scalar_t, scalar_t> NATs2_t_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V);
scalar_t NATs2_t_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Na, scalar_t dt = 0.025);