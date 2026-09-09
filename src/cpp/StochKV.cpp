#include "ConductanceTypes.hpp"
#include "StochKV.hpp"

#include <algorithm>
#include <cmath>

namespace {
const scalar_t THA = -40.0;
const scalar_t QA = 9.0;
const scalar_t RA = 0.02;
const scalar_t RB = 0.002;
const scalar_t Q10 = 2.3;
const scalar_t TEMP = 23.0;
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t gK_bar = 0.75;
const scalar_t gamma = 30.0;
}

scalar_t StochKV_derivatives(scalar_t n, scalar_t V, scalar_t celsius) {
    const scalar_t tadj = std::pow(Q10, ((celsius - TEMP) / 10.0));

    scalar_t a = 0.0;
    scalar_t b = 0.0;
    if (std::fabs(V - THA) > 1e-6) {
        a = RA * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
        b = -RB * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
    } else {
        a = RA * QA;
        b = RB * QA;
    }
    a *= tadj;
    b *= tadj;

    const scalar_t nInf = a / (a + b);
    const scalar_t nTau = 1.0 / (a + b);
    return (nInf - n) / nTau;
}

scalar_t StochKV_current(scalar_t n, scalar_t V, scalar_t celsius, scalar_t E_K, scalar_t area, StochKVTransitions transitions, scalar_t dt) {
    const scalar_t tadj = std::pow(Q10, ((celsius - TEMP) / 10.0));

    scalar_t a = 0.0;
    scalar_t b = 0.0;
    if (std::fabs(V - THA) > 1e-6) {
        a = RA * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
        b = -RB * (V - THA) / (1.0 - std::pow(e, ((THA - V) / QA)));
    } else {
        a = RA * QA;
        b = RB * QA;
    }
    a *= tadj;
    b *= tadj;

    const scalar_t eta = gK_bar / gamma;
    const scalar_t scale_dens = gamma / area;
    const int N = static_cast<int>(std::floor(eta * area + 0.5));
    const int N1 = static_cast<int>(std::floor(n * N + 0.5));
    const int N0 = N - N1;

    const int N0_new = std::max(0, N0 - transitions.n0_to_n1 + transitions.n1_to_n0);
    const int N1_new = (N0 + N1) - N0_new;

    const scalar_t gk = static_cast<scalar_t>(N1_new) * scale_dens * tadj;
    const scalar_t I_K = 1e-4 * gk * (V - E_K);
    return I_K;
}