"""Golden-model channel registry.

The eleven conductances and the calcium pool of the BBP L5PC model
(``cADpyr232_L5_TTPC1``), each transcribed from its NEURON ``.mod`` file.
"""

from __future__ import annotations

from .base import (
    CELSIUS,
    DT_DEFAULT,
    E_CA,
    E_HCN,
    E_K,
    E_NA,
    FARADAY,
    QT,
    GateSpec,
    GatingChannel,
    nernst,
)
from .ca_dynamics import CaDynamics_E2
from .ca_hva import Ca_HVA
from .ca_lvast import Ca_LVAst
from .ih import Ih
from .im import Im
from .k_pst import K_Pst
from .k_tst import K_Tst
from .nap_et2 import Nap_Et2
from .nata_t import NaTa_t
from .nats2_t import NaTs2_t
from .sk_e2 import SK_E2
from .skv3_1 import SKv3_1

#: All L5PC channel classes keyed by mechanism (SUFFIX) name, in the order of
#: the biophysics file.
CHANNELS: dict[str, type[GatingChannel]] = {
    cls.name: cls
    for cls in (
        NaTa_t,
        NaTs2_t,
        Nap_Et2,
        K_Tst,
        K_Pst,
        SKv3_1,
        SK_E2,
        Im,
        Ih,
        Ca_HVA,
        Ca_LVAst,
    )
}

CHANNEL_NAMES: tuple[str, ...] = tuple(CHANNELS)


def get_channel(name: str, **kwargs) -> GatingChannel:
    """Instantiate a channel by mechanism name, e.g. ``get_channel("NaTa_t")``."""
    try:
        cls = CHANNELS[name]
    except KeyError as exc:
        raise KeyError(f"unknown channel {name!r}; choose from {CHANNEL_NAMES}") from exc
    return cls(**kwargs)


__all__ = [
    "CHANNELS",
    "CHANNEL_NAMES",
    "get_channel",
    "GatingChannel",
    "GateSpec",
    "CaDynamics_E2",
    "NaTa_t",
    "NaTs2_t",
    "Nap_Et2",
    "K_Tst",
    "K_Pst",
    "SKv3_1",
    "SK_E2",
    "Im",
    "Ih",
    "Ca_HVA",
    "Ca_LVAst",
    "nernst",
    "CELSIUS",
    "DT_DEFAULT",
    "E_NA",
    "E_K",
    "E_CA",
    "E_HCN",
    "FARADAY",
    "QT",
]
