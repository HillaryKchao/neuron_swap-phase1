#include "K_Pst.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t QT = std::pow(2.3, (34.0 - 21.0) / 10.0);
const scalar_t gK_Pstbar = 0.00001;
}

std::pair<scalar_t, scalar_t> K_Pst_derivatives(std::pair<scalar_t, scalar_t> state, scalar_t V) {
    const double v = V + 10.0;

    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, (-(v + 1.0) / 12.0)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, (-(v + 54.0) / -11.0)));

    const scalar_t mTau = (v < -50.0) ? ((1.25 + 175.03 * std::pow(e, (0.026 * v))) / QT)
                                     : ((1.25 + 13.0 * std::pow(e, (-v * 0.026))) / QT);
    const scalar_t hTau = (360.0 + (1010.0 + 24.0 * (v + 55.0)) * std::pow(e, (-std::pow((v + 75.0) / 48.0, 2.0)))) / QT;

    const scalar_t m = state.first;
    const scalar_t h = state.second;

    const scalar_t dm_dt = (mInf - m) / mTau;
    const scalar_t dh_dt = (hInf - h) / hTau;

    return std::make_pair(dm_dt, dh_dt);
}

scalar_t K_Pst_current(std::pair<scalar_t, scalar_t> state, scalar_t V, scalar_t E_K, scalar_t dt) {
    const double v = V + 10.0;

    const scalar_t mInf = 1.0 / (1.0 + std::pow(e, (-(v + 1.0) / 12.0)));
    const scalar_t hInf = 1.0 / (1.0 + std::pow(e, (-(v + 54.0) / -11.0)));

    const scalar_t mTau = (v < -50.0) ? ((1.25 + 175.03 * std::pow(e, (0.026 * v))) / QT)
                                     : ((1.25 + 13.0 * std::pow(e, (-v * 0.026))) / QT);
    const scalar_t hTau = (360.0 + (1010.0 + 24.0 * (v + 55.0)) * std::pow(e, (-std::pow((v + 75.0) / 48.0, 2.0)))) / QT;

    scalar_t m = state.first;
    scalar_t h = state.second;

    m = mInf + (m - mInf) * std::pow(e, (-dt / mTau));
    h = hInf + (h - hInf) * std::pow(e, (-dt / hTau));

    const scalar_t gK_PST = gK_Pstbar * m * m * h;
    const scalar_t I_K = gK_PST * (V - E_K);
    return I_K;
}