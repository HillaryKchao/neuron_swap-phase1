"""K_Tst: transient (A-type) component of the potassium current.

Transcribed from ``K_Tst.mod`` (Reference: Korngreen and Sakmann 2000;
shifted by -10 mV for the junction potential).
"""

import numpy as np

from .base import E_K, QT, GateSpec, GatingChannel


class K_Tst(GatingChannel):
    name = "K_Tst"
    ion = "k"
    gates = (GateSpec("m", 4), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_K
    reference = "Korngreen and Sakmann 2000"

    def _rates(self, v, cai):
        v = v + 10
        mInf = 1 / (1 + np.exp(-(v + 0) / 19))
        mTau = (0.34 + 0.92 * np.exp(-(((v + 71) / 59) ** 2))) / QT
        hInf = 1 / (1 + np.exp(-(v + 66) / -10))
        hTau = (8 + 49 * np.exp(-(((v + 73) / 23) ** 2))) / QT
        return [(mInf, mTau), (hInf, hTau)]
