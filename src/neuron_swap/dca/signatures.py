"""Dynamical signatures of a channel from its voltage-clamp response.

The "signature" of a channel is the set of DCA statistics of its response
time series that Phase 2/3 must reproduce (after time/voltage scaling) for
the hardware to be declared dynamically equivalent:

1. **Predictive information (PI) curve** -- the Gaussian PI between the past
   and future windows (``T`` bins each) of the full response
   ``X = [gates..., (cai), I]``.  PI is invariant to invertible linear maps
   of ``X`` so it does not depend on units or on ``gbar``.
2. **Per-component PI** -- the same for each gating variable and for the
   current alone: the "functional relevance of individual dynamical
   components" of the README.
3. **DCA projections** -- the ``d``-dimensional subspace of ``X`` with
   maximal PI (``d = 1..n_features``) and its PI.
4. **Input/output information** -- the mutual information between a window
   of the clamp command ``U = V`` and the *future* window of ``X``
   (externally-aware DCA of the ``sdca`` package, ``external_input=True``),
   plus the ioDCA projection that maximises it.

The computations are delegated to the lab packages
`DynamicalComponentsAnalysis <https://github.com/BouchardLab/DynamicalComponentsAnalysis>`_
(``dca``) and ``ioDCA`` (the local ``sdca`` checkout).  ``dca`` is required;
``ioDCA`` is optional and the input/output parts are skipped without it.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

try:  # required
    from dca import DynamicalComponentsAnalysis
    from dca.cov_util import (
        calc_cov_from_cross_cov_mats,
        calc_cross_cov_mats_from_data,
        calc_pi_from_cross_cov_mats,
    )

    HAVE_DCA = True
except ImportError:  # pragma: no cover - exercised only without the package
    HAVE_DCA = False

try:  # optional
    from ioDCA.sdca_gaussian_exogenous import SupervisedDCA
    from ioDCA.sdca_utils_gaussian_exogenous import calc_pi_from_cov as _iodca_pi_from_cov

    HAVE_IODCA = True
except ImportError:  # pragma: no cover
    HAVE_IODCA = False

from ..protocols.simulate import ClampResult

_INSTALL_HINT = (
    "BouchardLab DCA is required: pip install "
    "'git+https://github.com/BouchardLab/DynamicalComponentsAnalysis' "
    "(note: the PyPI package named 'dca' is unrelated)."
)


def _require_dca() -> None:
    if not HAVE_DCA:
        raise ImportError(_INSTALL_HINT)


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
def _bin_average(x: np.ndarray, stride: int) -> np.ndarray:
    """Average consecutive ``stride`` samples along axis 1 of ``(n_sweeps, n_t, ...)``."""
    if stride == 1:
        return x
    n = (x.shape[1] // stride) * stride
    x = x[:, :n]
    shape = (x.shape[0], n // stride, stride) + x.shape[2:]
    return x.reshape(shape).mean(axis=2)


def build_timeseries(
    result: ClampResult,
    bin_ms: float = 0.5,
    include_current: bool = True,
    zscore: bool = True,
    min_std: float = 1e-9,
    discard_ms: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict]:
    """Turn a :class:`ClampResult` into DCA-ready trials.

    Returns ``(X_trials, U_trials, feature_names, info)`` where each trial is
    one sweep: ``X`` has shape ``(n_bins, n_features)`` (gates, optional
    ``cai``, current) and ``U`` shape ``(n_bins, 1)`` (command voltage).
    Samples are averaged into ``bin_ms`` bins.  Features are z-scored with
    statistics pooled over sweeps; features with (near-)zero variance are
    dropped and reported in ``info["dropped"]``.
    """
    stride = max(1, int(round(bin_ms / result.dt)))
    n_skip = int(round(discard_ms / result.dt))
    feats = _bin_average(result.features(include_current)[:, n_skip:], stride)
    names = result.feature_names(include_current)
    U = _bin_average(result.V[:, n_skip:, None], stride)

    pooled = feats.reshape(-1, feats.shape[-1])
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    keep = std > min_std
    dropped = [n for n, k in zip(names, keep) if not k]
    feats = feats[..., keep]
    names = [n for n, k in zip(names, keep) if k]
    if zscore:
        feats = (feats - mean[keep]) / std[keep]
        U = (U - U.mean()) / max(U.std(), min_std)
    info = {
        "bin_ms": stride * result.dt,
        "stride": stride,
        "n_sweeps": feats.shape[0],
        "n_bins_per_sweep": feats.shape[1],
        "dropped": dropped,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return list(feats), list(U), names, info


def _concat(trials: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(trials), axis=0)


# ---------------------------------------------------------------------------
# predictive information
# ---------------------------------------------------------------------------
def cross_cov_mats(X, T: int, jitter: float = 1e-8) -> np.ndarray:
    """Cross-covariance matrices for lags ``0 .. 2T-1`` (past + future windows
    of ``T`` bins) with a small ridge on the zero-lag covariance."""
    _require_dca()
    X_in = list(X) if isinstance(X, (list, tuple)) else X
    ccm = np.asarray(calc_cross_cov_mats_from_data(X_in, 2 * T))
    if jitter:
        ccm = ccm.copy()
        ccm[0] += jitter * np.trace(ccm[0]) / ccm.shape[1] * np.eye(ccm.shape[1])
    return ccm


def predictive_information(X, T: int, proj: np.ndarray | None = None, jitter: float = 1e-8) -> float:
    """Gaussian PI (nats) between the past and future ``T``-bin windows of ``X``."""
    ccm = cross_cov_mats(X, T, jitter)
    return float(calc_pi_from_cross_cov_mats(ccm, proj=proj))


def pi_curve(X, T_values: Iterable[int], jitter: float = 1e-8) -> dict[int, float]:
    return {int(T): predictive_information(X, int(T), jitter=jitter) for T in T_values}


def fit_dca(X, d: int, T: int, seed: int = 0, jitter: float = 1e-8, n_init: int = 1) -> tuple[np.ndarray, float]:
    """Fit DCA: the ``d``-dim projection of ``X`` with maximal PI at window ``T``.

    Returns ``(coef (n_features, d), PI)``.
    """
    _require_dca()
    model = DynamicalComponentsAnalysis(d=d, T=T, n_init=n_init, rng_or_seed=seed)
    X_in = list(X) if isinstance(X, (list, tuple)) else X
    model.estimate_data_statistics(X_in)
    if jitter:
        import torch

        n = model.cross_covs.shape[1]
        ridge = jitter * float(torch.trace(model.cross_covs[0])) / n
        model.cross_covs[0] += ridge * torch.eye(n, dtype=model.cross_covs.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_projection()
    coef = np.asarray(model.coef_)
    return coef, float(model.score())


# ---------------------------------------------------------------------------
# input / output information (externally aware DCA)
# ---------------------------------------------------------------------------
def _shifted_XU(X: np.ndarray, U: np.ndarray, T_shift: int) -> np.ndarray:
    """``[X(t+T_shift), U(t)]`` -- the future response next to the past input
    (the ``external_input=True`` concatenation of ``ioDCA``)."""
    if T_shift == 0:
        return np.concatenate([X, U], axis=-1)
    return np.concatenate([X[T_shift:], U[:-T_shift]], axis=-1)


def input_output_mi(X, U, T: int, T_shift: int | None = None, jitter: float = 1e-8) -> float:
    """Mutual information (nats) between a ``T``-bin window of the input ``U``
    and the ``T``-bin window of the response ``X`` shifted ``T_shift`` bins
    into the future (default ``T_shift = T``)."""
    _require_dca()
    if T_shift is None:
        T_shift = T
    Xs = list(X) if isinstance(X, (list, tuple)) else [X]
    Us = list(U) if isinstance(U, (list, tuple)) else [U]
    XU = [_shifted_XU(x, u, T_shift) for x, u in zip(Xs, Us)]
    N, M = Xs[0].shape[-1], Us[0].shape[-1]
    ccm = np.asarray(calc_cross_cov_mats_from_data(XU, T))
    if jitter:
        ccm = ccm.copy()
        ccm[0] += jitter * np.trace(ccm[0]) / ccm.shape[1] * np.eye(ccm.shape[1])
    cov = np.asarray(calc_cov_from_cross_cov_mats(ccm))
    if HAVE_IODCA:
        return float(_iodca_pi_from_cov(cov, N=N, M=M))
    return float(_mi_from_cov(cov, N, M, T))


def _mi_from_cov(cov: np.ndarray, N: int, M: int, T: int) -> float:
    """Numpy fallback of ``ioDCA.sdca_utils_gaussian_exogenous.calc_pi_from_cov``."""
    nm = N + M
    xi = np.concatenate([np.arange(i * nm, i * nm + N) for i in range(T)])
    ui = np.concatenate([np.arange(i * nm + N, (i + 1) * nm) for i in range(T)])
    lx = np.linalg.slogdet(cov[np.ix_(xi, xi)])[1]
    lu = np.linalg.slogdet(cov[np.ix_(ui, ui)])[1]
    lxu = np.linalg.slogdet(cov)[1]
    return 0.5 * lx + 0.5 * lu - 0.5 * lxu


def input_output_mi_curve(X, U, T_values: Iterable[int], jitter: float = 1e-8) -> dict[int, float]:
    return {int(T): input_output_mi(X, U, int(T), jitter=jitter) for T in T_values}


def fit_iodca(X, U, d: int, T: int, T_shift: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit externally-aware (supervised) DCA from the ``sdca`` package.

    Finds the ``d``-dim projection ``V`` of the response and the projection
    ``W`` of the input that maximise the Gaussian MI between the past input
    window and the future response window.  Returns ``(V, W, MI)``.
    """
    if not HAVE_IODCA:
        raise ImportError("ioDCA (the sdca checkout) is required for fit_iodca")
    Xc = _concat(X) if isinstance(X, (list, tuple)) else X
    Uc = _concat(U) if isinstance(U, (list, tuple)) else U
    model = SupervisedDCA(
        d=d,
        d_exo=Uc.shape[-1],
        T=T,
        T_shift=T if T_shift is None else T_shift,
        external_input=True,
        rng_or_seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xc, Uc)
    return np.asarray(model.coef_), np.asarray(model.coef_exogenous_), float(model.score())


# ---------------------------------------------------------------------------
# signature container
# ---------------------------------------------------------------------------
@dataclass
class ChannelSignature:
    channel: str
    protocol: str
    feature_names: list[str]
    bin_ms: float
    n_samples: int
    T_values: list[int]
    pi_curve: dict[int, float]
    pi_per_feature: dict[str, dict[int, float]]
    dca: dict[int, dict]  # d -> {"T": T, "pi": float, "coef": list}
    io_mi_curve: dict[int, float] = field(default_factory=dict)
    iodca: dict = field(default_factory=dict)  # {"T","d","mi","coef","coef_exo"}
    info: dict = field(default_factory=dict)

    # -- serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> dict:
        """Scalar highlights for tables."""
        T_max = max(self.T_values)
        out = {
            "channel": self.channel,
            "protocol": self.protocol,
            "n_features": len(self.feature_names),
            f"PI(T={T_max})": self.pi_curve[T_max],
            "PI_current_only": self.pi_per_feature.get("I", {}).get(T_max, float("nan")),
        }
        if self.dca:
            out["DCA_d1_PI"] = self.dca[1]["pi"]
            out["DCA_d1_leading_feature"] = self.feature_names[int(np.argmax(np.abs(self.dca[1]["coef"])))]
        if self.io_mi_curve:
            out[f"IO_MI(T={T_max})"] = self.io_mi_curve[T_max]
        return out


def load_signature(path) -> ChannelSignature:
    with open(path) as f:
        d = json.load(f)
    d["pi_curve"] = {int(k): v for k, v in d["pi_curve"].items()}
    d["pi_per_feature"] = {n: {int(k): v for k, v in c.items()} for n, c in d["pi_per_feature"].items()}
    d["dca"] = {int(k): v for k, v in d["dca"].items()}
    d["io_mi_curve"] = {int(k): v for k, v in d.get("io_mi_curve", {}).items()}
    return ChannelSignature(**d)


def compute_signature(
    result: ClampResult,
    T_values: Sequence[int] = (1, 2, 5, 10, 20),
    T_dca: int | None = None,
    d_values: Sequence[int] | None = None,
    bin_ms: float = 0.5,
    include_current: bool = True,
    io: bool = True,
    seed: int = 0,
    discard_ms: float = 0.0,
) -> ChannelSignature:
    """Compute the full :class:`ChannelSignature` of a clamp result."""
    _require_dca()
    X, U, names, info = build_timeseries(result, bin_ms=bin_ms, include_current=include_current, discard_ms=discard_ms)
    n_feat = len(names)
    T_values = [int(t) for t in T_values]
    if T_dca is None:
        T_dca = max(T_values)
    if d_values is None:
        d_values = range(1, n_feat + 1)

    curve = pi_curve(X, T_values)
    per_feature = {n: pi_curve([x[:, [i]] for x in X], T_values) for i, n in enumerate(names)}
    dca_fits: dict[int, dict] = {}
    for d in d_values:
        if d > n_feat:
            continue
        coef, pi = fit_dca(X, d, T_dca, seed=seed)
        dca_fits[int(d)] = {"T": T_dca, "pi": pi, "coef": coef.tolist()}

    io_curve: dict[int, float] = {}
    iodca: dict = {}
    if io:
        io_curve = input_output_mi_curve(X, U, T_values)
        if HAVE_IODCA:
            V, W, mi = fit_iodca(X, U, d=1, T=T_dca, seed=seed)
            iodca = {"T": T_dca, "d": 1, "mi": mi, "coef": V.tolist(), "coef_exo": W.tolist()}
        else:
            warnings.warn("ioDCA not importable; skipping the ioDCA projection fit")

    return ChannelSignature(
        channel=result.name,
        protocol=result.protocol.name,
        feature_names=names,
        bin_ms=info["bin_ms"],
        n_samples=int(sum(len(x) for x in X)),
        T_values=T_values,
        pi_curve=curve,
        pi_per_feature=per_feature,
        dca=dca_fits,
        io_mi_curve=io_curve,
        iodca=iodca,
        info=info,
    )


# ---------------------------------------------------------------------------
# comparing signatures (used for hardware parity in later phases)
# ---------------------------------------------------------------------------
def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between the column spaces of ``A`` and ``B``."""
    qa, _ = np.linalg.qr(np.asarray(A, dtype=float))
    qb, _ = np.linalg.qr(np.asarray(B, dtype=float))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def compare_signatures(ref: ChannelSignature, other: ChannelSignature) -> dict:
    """Distances between two signatures of the same channel/features.

    * ``pi_abs_diff`` / ``pi_rel_diff`` per ``T`` (full-response PI)
    * ``io_mi_abs_diff`` per ``T``
    * ``dca_angle_deg`` per ``d``: largest principal angle between the DCA
      subspaces (0 = identical dynamics subspace)
    """
    if ref.feature_names != other.feature_names:
        raise ValueError(f"feature mismatch: {ref.feature_names} vs {other.feature_names}")
    out: dict = {"pi_abs_diff": {}, "pi_rel_diff": {}, "io_mi_abs_diff": {}, "dca_angle_deg": {}}
    for T in ref.pi_curve:
        if T in other.pi_curve:
            a, b = ref.pi_curve[T], other.pi_curve[T]
            out["pi_abs_diff"][T] = abs(a - b)
            out["pi_rel_diff"][T] = abs(a - b) / max(abs(a), 1e-12)
    for T in ref.io_mi_curve:
        if T in other.io_mi_curve:
            out["io_mi_abs_diff"][T] = abs(ref.io_mi_curve[T] - other.io_mi_curve[T])
    for d in ref.dca:
        if d in other.dca:
            ang = principal_angles(np.asarray(ref.dca[d]["coef"]), np.asarray(other.dca[d]["coef"]))
            out["dca_angle_deg"][d] = float(np.degrees(ang.max()))
    return out
