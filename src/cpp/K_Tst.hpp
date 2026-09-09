#pragma once

#include "ConductanceTypes.hpp"
#include <utility>

std::pair<scalar_t, scalar_t> K_Tst_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V);
scalar_t K_Tst_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_K, scalar_t dt = 0.025);