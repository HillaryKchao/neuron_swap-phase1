#include "Nap_Et2.hpp"

#include <cmath>

namespace {
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gNap_Et2bar = 0.00001;
}

std::pair<scalar_t, scalar_t> Nap_Et2_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, ((V + 52.6) / -4.6)));

    if (V == -38) {
        V += 0.0001;
    }

    const scalar_t mAlpha = (0.182 * (V + 38)) / (1 - (std::pow(e, (-(V + 38) / 6))));
    const scalar_t mBeta = -(0.124 * (V + 38)) / (1 - (std::pow(e, ((V + 38) / 6))));
    const scalar_t mTau = 6.0 * (1.0 / (mAlpha + mBeta)) / QT;

    if (V == -17 || V == -64.4) {
        V += 0.001;
    }

    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, ((V + 48.8) / 10.0)));
    const scalar_t hAlpha = -2.88e-6 * (V + 17) / (1 - std::pow(e, ((V + 17) / 4.63)));
    const scalar_t hBeta = 6.94e-6 * (V + 64.4) / (1 - std::pow(e, (-(V + 64.4) / 2.63)));
    const scalar_t hTau = (1.0 / (hAlpha + hBeta)) / QT;

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t Nap_Et2_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Na) {
    scalar_t m = state.first;
    scalar_t h = state.second;

    const scalar_t gNap_Et2 = gNap_Et2bar * m * m * m * h;
    const scalar_t I_Na = gNap_Et2 * (V - E_Na);
    return I_Na;
}