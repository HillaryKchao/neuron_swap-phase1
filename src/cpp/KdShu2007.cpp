#include "KdShu2007.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t VHALF_M = -43.0;
const scalar_t K_M = 8.0;
const scalar_t VHALF_H = -67.0;
const scalar_t K_H = 7.3;
const scalar_t M_TAU = 0.6;
const scalar_t H_TAU = 1500.0;
const scalar_t gKBar = 0.1;
const scalar_t e_K = -100.0;
}

std::pair<scalar_t, scalar_t> KdShu2007_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    const scalar_t mInf = 1.0 - 1.0 / (1.0 + std::pow(e, ((V - VHALF_M) / K_M)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((V - VHALF_H) / K_H)));

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / M_TAU;
    const scalar_t dh_dt = (hInf - h) / H_TAU;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t KdShu2007_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t dt) {
    const scalar_t mInf = 1.0 - 1.0 / (1.0 + std::pow(e, ((V - VHALF_M) / K_M)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((V - VHALF_H) / K_H)));

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / M_TAU));
    h = hInf + (h - hInf) * std::pow(e, (-dt / H_TAU));

    const scalar_t I_K = gKBar * m * h * (V - e_K);
    return I_K;
}