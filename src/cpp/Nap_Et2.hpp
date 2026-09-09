#pragma once

#include "ConductanceTypes.hpp"
#include <utility>

std::pair<scalar_t, scalar_t> Nap_Et2_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V);
scalar_t Nap_Et2_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Na);