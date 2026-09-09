#include "NATs2_t.hpp"

#include <cmath>

namespace {
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gNaTs2_tbar = 0.00001;
}

std::pair<scalar_t, scalar_t> NATs2_t_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    if (V == -32 || V == -60) {
        V += 0.0001;
    }

    const scalar_t mAlpha = (0.182 * (V + 32)) / (1 - std::pow(e, (-(V + 32) / 6)));
    const scalar_t mBeta = -(0.124 * (V + 32)) / (1 - std::pow(e, ((V + 32) / 6)));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = (1.0 / (mAlpha + mBeta)) / QT;

    const scalar_t hAlpha = -(0.015 * (V + 60)) / (1 - std::pow(e, ((V + 60) / 6)));
    const scalar_t hBeta = (0.015 * (V + 60)) / (1 - std::pow(e, (-(V + 60) / 6)));
    const scalar_t hInf = hAlpha / (hAlpha + hBeta);
    const scalar_t hTau = (1.0 / (hAlpha + hBeta)) / QT;

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t NATs2_t_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_Na, scalar_t dt) {
    if (V == -32 || V == -60) {
        V += 0.0001;
    }

    const scalar_t mAlpha = (0.182 * (V + 32)) / (1 - std::pow(e, (-(V + 32) / 6)));
    const scalar_t mBeta = -(0.124 * (V + 32)) / (1 - std::pow(e, ((V + 32) / 6)));
    const scalar_t mInf = mAlpha / (mAlpha + mBeta);
    const scalar_t mTau = (1.0 / (mAlpha + mBeta)) / QT;

    const scalar_t hAlpha = -(0.015 * (V + 60)) / (1 - std::pow(e, ((V + 60) / 6)));
    const scalar_t hBeta = (0.015 * (V + 60)) / (1 - std::pow(e, (-(V + 60) / 6)));
    const scalar_t hInf = hAlpha / (hAlpha + hBeta);
    const scalar_t hTau = (1.0 / (hAlpha + hBeta)) / QT;

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    h = hInf + (h - hInf) * std::pow(e, (-dt / hTau));

    const scalar_t gNaTs2_t = gNaTs2_tbar * m * m * m * h;
    const scalar_t I_Na = gNaTs2_t * (V - E_Na);
    return I_Na;
}