"""Nap_Et2: persistent sodium current.

Transcribed from ``Nap_Et2.mod`` (Reference: Magistretti & Alonso 1999;
mTau is the NaT time constant times 6).
"""

import numpy as np

from .base import E_NA, QT, GateSpec, GatingChannel, _guard


class Nap_Et2(GatingChannel):
    name = "Nap_Et2"
    ion = "na"
    gates = (GateSpec("m", 3), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_NA
    reference = "Magistretti & Alonso 1999"

    def _rates(self, v, cai):
        mInf = 1.0 / (1 + np.exp((v - -52.6) / -4.6))
        v = _guard(v, -38.0)
        mAlpha = (0.182 * (v - -38)) / (1 - np.exp(-(v - -38) / 6))
        mBeta = (0.124 * (-v - 38)) / (1 - np.exp(-(-v - 38) / 6))
        mTau = 6 * (1 / (mAlpha + mBeta)) / QT

        v = _guard(v, -17.0)
        v = _guard(v, -64.4)
        hInf = 1.0 / (1 + np.exp((v - -48.8) / 10))
        hAlpha = -2.88e-6 * (v + 17) / (1 - np.exp((v + 17) / 4.63))
        hBeta = 6.94e-6 * (v + 64.4) / (1 - np.exp(-(v + 64.4) / 2.63))
        hTau = (1 / (hAlpha + hBeta)) / QT
        return [(mInf, mTau), (hInf, hTau)]
