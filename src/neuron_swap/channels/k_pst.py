"""K_Pst: persistent (slowly inactivating) component of the potassium current.

Transcribed from ``K_Pst.mod`` (Reference: Korngreen and Sakmann 2000;
shifted by -10 mV for the junction potential).
"""

import numpy as np

from .base import E_K, QT, GateSpec, GatingChannel


class K_Pst(GatingChannel):
    name = "K_Pst"
    ion = "k"
    gates = (GateSpec("m", 2), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_K
    reference = "Korngreen and Sakmann 2000"

    def _rates(self, v, cai):
        v = v + 10
        mInf = 1 / (1 + np.exp(-(v + 1) / 12))
        mTau_low = (1.25 + 175.03 * np.exp(-v * -0.026)) / QT
        mTau_high = (1.25 + 13 * np.exp(-v * 0.026)) / QT
        mTau = np.where(v < -50, mTau_low, mTau_high)
        hInf = 1 / (1 + np.exp(-(v + 54) / -11))
        hTau = (360 + (1010 + 24 * (v + 55)) * np.exp(-(((v + 75) / 48) ** 2))) / QT
        return [(mInf, mTau), (hInf, hTau)]
