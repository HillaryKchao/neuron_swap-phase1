"""L5PC operating point: the per-region parameters of ``biophysics.hoc``
(BBP ``cADpyr232_L5_TTPC1``, Hay et al. 2011 optimisation).

Conductance densities are in S/cm^2.  The apical ``Ih`` density is
distance dependent in the original model; the value recorded here is the
proximal (distance 0) value and :func:`apical_ih_gbar` gives the full rule.
"""

from __future__ import annotations

import math

from . import CHANNELS, CaDynamics_E2, GatingChannel

#: Passive and ionic parameters common to all sections.
PASSIVE = {
    "e_pas": -75.0,  # mV
    "g_pas": 3e-5,  # S/cm^2
    "Ra": 100.0,  # ohm cm
    "cm": 1.0,  # uF/cm^2 (2.0 in apical and basal dendrites)
    "ena": 50.0,  # mV
    "ek": -85.0,  # mV
    "celsius": 34.0,
    "v_init": -65.0,
    "dt": 0.025,  # ms
}

#: Maximal conductance densities per region (gbar, S/cm^2).
GBAR = {
    "somatic": {
        "NaTs2_t": 0.983955,
        "SKv3_1": 0.303472,
        "SK_E2": 0.008407,
        "Ca_HVA": 0.000994,
        "Ca_LVAst": 0.000333,
        "Ih": 0.000080,
    },
    "axonal": {
        "NaTa_t": 3.137968,
        "Nap_Et2": 0.006827,
        "K_Tst": 0.089259,
        "K_Pst": 0.973538,
        "SKv3_1": 1.021945,
        "SK_E2": 0.007104,
        "Ca_HVA": 0.000990,
        "Ca_LVAst": 0.008752,
    },
    "apical": {
        "NaTs2_t": 0.026145,
        "SKv3_1": 0.004226,
        "Im": 0.000143,
        "Ih": 0.000080,  # proximal value; see apical_ih_gbar
    },
    "basal": {
        "Ih": 0.000080,
    },
}

#: CaDynamics_E2 parameters per region (gamma is unitless, decay in ms).
CA_DYNAMICS = {
    "somatic": {"gamma": 0.000609, "decay": 210.485284},
    "axonal": {"gamma": 0.002910, "decay": 287.198731},
}

#: The region in which each channel is expressed with its largest density;
#: used as the "representative" operating point for single-channel studies.
REPRESENTATIVE_REGION = {
    "NaTa_t": "axonal",
    "NaTs2_t": "somatic",
    "Nap_Et2": "axonal",
    "K_Tst": "axonal",
    "K_Pst": "axonal",
    "SKv3_1": "axonal",
    "SK_E2": "somatic",
    "Im": "apical",
    "Ih": "basal",
    "Ca_HVA": "somatic",
    "Ca_LVAst": "axonal",
}


def apical_ih_gbar(distance_um: float) -> float:
    """Apical ``gIhbar`` as a function of path distance from the soma (um)."""
    return (-0.869600 + 2.087000 * math.exp(distance_um * 0.003100)) * 0.000080


def channel_for_region(name: str, region: str, **kwargs) -> GatingChannel:
    """Instantiate ``name`` with the gbar it has in ``region``."""
    gbar = GBAR[region][name]
    return CHANNELS[name](gbar=gbar, **kwargs)


def representative_channel(name: str, **kwargs) -> GatingChannel:
    """Instantiate ``name`` at its representative L5PC operating point."""
    return channel_for_region(name, REPRESENTATIVE_REGION[name], **kwargs)


def ca_dynamics_for_region(region: str, **kwargs) -> CaDynamics_E2:
    return CaDynamics_E2(**CA_DYNAMICS[region], **kwargs)


def region_channels(region: str, **kwargs) -> dict[str, GatingChannel]:
    """All channels of a region at their fitted densities (for Phase 4)."""
    return {name: CHANNELS[name](gbar=g, **kwargs) for name, g in GBAR[region].items()}
