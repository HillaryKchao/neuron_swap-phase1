"""Ca_LVAst: low-voltage-activated (T-type) calcium current.

Transcribed from ``Ca_LVAst.mod`` (Reference: Avery and Johnston 1996, tau
from Randall 1997; shifted by -10 mV for the junction potential).
"""

import numpy as np

from .base import E_CA, QT, GateSpec, GatingChannel


class Ca_LVAst(GatingChannel):
    name = "Ca_LVAst"
    ion = "ca"
    gates = (GateSpec("m", 2), GateSpec("h", 1))
    default_gbar = 0.00001
    default_erev = E_CA
    reference = "Avery and Johnston 1996; Randall 1997"

    def _rates(self, v, cai):
        v = v + 10
        mInf = 1.0000 / (1 + np.exp((v - -30.000) / -6))
        mTau = (5.0000 + 20.0000 / (1 + np.exp((v - -25.000) / 5))) / QT
        hInf = 1.0000 / (1 + np.exp((v - -80.000) / 6.4))
        hTau = (20.0000 + 50.0000 / (1 + np.exp((v - -40.000) / 7))) / QT
        return [(mInf, mTau), (hInf, hTau)]
