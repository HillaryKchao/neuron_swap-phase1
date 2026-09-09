"""SK_E2: small-conductance calcium-activated potassium current.

Transcribed from ``SK_E2.mod`` (Reference: Kohler et al. 1996).  The single
gate ``z`` depends on the intracellular calcium concentration ``cai`` (mM)
rather than on voltage, with a fixed 1 ms time constant.
"""

import numpy as np

from .base import E_K, GateSpec, GatingChannel


class SK_E2(GatingChannel):
    name = "SK_E2"
    ion = "k"
    gates = (GateSpec("z", 1),)
    default_gbar = 0.000001
    default_erev = E_K
    reference = "Kohler et al. 1996"
    calcium_dependent = True

    #: zTau (ms) PARAMETER of the .mod file
    z_tau = 1.0

    def _rates(self, v, cai):
        ca = np.where(cai < 1e-7, cai + 1e-07, cai)
        zInf = 1 / (1 + (0.00043 / ca) ** 4.8)
        zTau = np.full_like(zInf, self.z_tau)
        return [(zInf, zTau)]
