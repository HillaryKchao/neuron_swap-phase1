#include "Ca.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gCa_bar = 0.00001;
}

std::pair<scalar_t, scalar_t> Ca_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    if (V == -27) {
        V += 0.0001;
    }

    const scalar_t mAlpha = -(0.055 * (V + 27)) / (std::pow(e, (-(V + 27) / 3.8)) - 1);
    const scalar_t mBeta = 0.94 * std::pow(e, (-(V + 75) / 17));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = 1.0 / (mAlpha + mBeta);

    const scalar_t hAlpha = 0.000457 * std::pow(e, (-(V + 13) / 50));
    const scalar_t hBeta = 0.0065 / (std::pow(e, (-(V + 15) / 28)) + 1);
    const scalar_t hInf = hAlpha / (hAlpha + hBeta);
    const scalar_t hTau = 1.0 / (hAlpha + hBeta);

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t Ca_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Ca, scalar_t dt) {
    if (V == -27) {
        V += 0.0001;
    }

    const scalar_t mAlpha = -(0.055 * (V + 27)) / (std::pow(e, (-(V + 27) / 3.8)) - 1);
    const scalar_t mBeta = 0.94 * std::pow(e, (-(V + 75) / 17));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = 1.0 / (mAlpha + mBeta);

    const scalar_t hAlpha = 0.000457 * std::pow(e, (-(V + 13) / 50));
    const scalar_t hBeta = 0.0065 / (std::pow(e, (-(V + 15) / 28)) + 1);
    const scalar_t hInf = hAlpha / (hAlpha + hBeta);
    const scalar_t hTau = 1.0 / (hAlpha + hBeta);

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    h = hInf + (h - hInf) * std::pow(e, (-dt / hTau));

    const scalar_t gCa = gCa_bar * m * m * h;
    const scalar_t I_Ca = gCa * (V - E_Ca);
    return I_Ca;
}