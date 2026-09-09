#include "SK_E2.hpp"

#include <cmath>

namespace {
const scalar_t e = 2.718281828459045235360287471352;
const scalar_t Z_TAU = 1.0;
const scalar_t CA_FLOOR = 1e-7;
const scalar_t gSK_E2bar = 0.000001;
}

scalar_t SK_E2_derivatives(scalar_t z, scalar_t ca) {
    if (ca < CA_FLOOR) {
        ca = ca + CA_FLOOR;
    }

    const scalar_t zInf = 1.0 / (1.0 + std::pow((0.00043 / ca), 4.8));
    return (zInf - z) / Z_TAU;
}

scalar_t SK_E2_current(scalar_t z, scalar_t ca, scalar_t V, scalar_t E_K, scalar_t dt) {
    if (ca < CA_FLOOR) {
        ca = ca + CA_FLOOR;
    }

    const scalar_t zInf = 1.0 / (1.0 + std::pow((0.00043 / ca), 4.8));

    z = zInf + (z - zInf) * std::pow(e, (-dt / Z_TAU));

    const scalar_t gSK_E2 = gSK_E2bar * z;
    const scalar_t I_K = gSK_E2 * E_K;
    return I_K;
}