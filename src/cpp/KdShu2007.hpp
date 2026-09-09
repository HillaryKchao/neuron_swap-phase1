#pragma once

#include "ConductanceTypes.hpp"
#include <utility>

std::pair<scalar_t, scalar_t> KdShu2007_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V);
scalar_t KdShu2007_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t dt = 0.025);