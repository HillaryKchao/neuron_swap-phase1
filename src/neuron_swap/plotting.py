"""Matplotlib figures for clamp results and signatures (optional dependency)."""

from __future__ import annotations

import numpy as np

from .protocols.simulate import ClampResult


def _mpl():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_clamp(result: ClampResult, max_sweeps: int = 12, title: str | None = None):
    """Command voltage, gating variables and current for every sweep."""
    plt = _mpl()
    n_gates = len(result.gate_names)
    rows = 2 + n_gates + (1 if result.cai is not None else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(9, 1.8 * rows), sharex=True)
    t = result.t
    n_sw = result.V.shape[0]
    idx = np.linspace(0, n_sw - 1, min(n_sw, max_sweeps)).round().astype(int)
    cmap = plt.get_cmap("viridis")
    for k, s in enumerate(idx):
        c = cmap(k / max(1, len(idx) - 1))
        lab = f"{result.protocol.sweep_values[s]:g}"
        axes[0].plot(t, result.V[s], color=c, lw=0.8, label=lab)
        for gi, gname in enumerate(result.gate_names):
            axes[1 + gi].plot(t, result.states[gi, s], color=c, lw=0.8)
        r = 1 + n_gates
        if result.cai is not None:
            axes[r].plot(t, result.cai[s], color=c, lw=0.8)
            r += 1
        axes[r].plot(t, result.I, color=c, lw=0.8) if False else axes[r].plot(t, result.I[s], color=c, lw=0.8)
    axes[0].set_ylabel("V (mV)")
    axes[0].legend(fontsize=6, ncol=4, title=result.protocol.sweep_label, title_fontsize=6, loc="upper right")
    for gi, gname in enumerate(result.gate_names):
        axes[1 + gi].set_ylabel(gname)
        axes[1 + gi].set_ylim(-0.05, 1.05)
    r = 1 + n_gates
    if result.cai is not None:
        axes[r].set_ylabel("[Ca]i (mM)")
        axes[r].set_yscale("log")
        r += 1
    axes[r].set_ylabel("I (mA/cm²)")
    axes[-1].set_xlabel("t (ms)")
    fig.suptitle(title or f"{result.name} - {result.protocol.name}: {result.protocol.description}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_iv(result: ClampResult, epoch: str = "step"):
    """Peak and steady-state current versus the sweep variable."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = result.protocol.sweep_values
    ax.plot(x, result.peak_current(epoch), "o-", label="peak")
    ax.plot(x, result.steady_state_current(epoch), "s--", label="steady state")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(result.protocol.sweep_label)
    if result.protocol.cai is not None:
        ax.set_xscale("log")
    ax.set_ylabel("I (mA/cm²)")
    ax.set_title(f"{result.name} - {result.protocol.name}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_signature(sig, title: str | None = None):
    """PI curves, per-feature PI, DCA loadings and input/output MI."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    T = sig.T_values
    ax = axes[0]
    ax.plot(T, [sig.pi_curve[t] for t in T], "k-o", label="all features")
    for name, curve in sig.pi_per_feature.items():
        ax.plot(T, [curve[t] for t in T], "--", marker=".", label=name)
    if sig.io_mi_curve:
        ax.plot(T, [sig.io_mi_curve[t] for t in T], ":", marker="^", color="C3", label="MI(V → response)")
    ax.set_xlabel(f"window T (bins of {sig.bin_ms:g} ms)")
    ax.set_ylabel("PI / MI (nats)")
    ax.set_xscale("log")
    ax.legend(fontsize=7)
    ax.set_title("predictive information", fontsize=9)

    ax = axes[1]
    ds = sorted(sig.dca)
    ax.plot(ds, [sig.dca[d]["pi"] for d in ds], "o-")
    ax.set_xlabel("DCA dimension d")
    ax.set_ylabel(f"PI (nats), T={sig.dca[ds[0]]['T'] if ds else '-'}")
    ax.set_title("DCA subspace PI", fontsize=9)

    ax = axes[2]
    if ds:
        coef = np.asarray(sig.dca[1]["coef"]).ravel()
        ax.bar(sig.feature_names, coef, label="DCA d=1")
    if sig.iodca:
        v = np.asarray(sig.iodca["coef"]).ravel()
        ax.bar(sig.feature_names, v, alpha=0.5, label="ioDCA d=1")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("loading")
    ax.legend(fontsize=7)
    ax.set_title("leading component loadings", fontsize=9)
    fig.suptitle(title or f"{sig.channel} signature ({sig.protocol})", fontsize=10)
    fig.tight_layout()
    return fig
