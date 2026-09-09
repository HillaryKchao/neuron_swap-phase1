#!/usr/bin/env python
"""Run Phase 1 end to end: golden-model voltage clamp of every L5PC channel,
waveform export, figures and DCA signatures.

Usage (from the repository root, inside the ``neuron-swap`` env)::

    python scripts/run_phase1.py --out results/phase1
    python scripts/run_phase1.py --channels NaTa_t Ih --no-dca --quick

Outputs, per channel ``<out>/<channel>/``:

* ``<protocol>.npz``      recorded t, V, gates, g, I (and cai)
* ``<protocol>.png``      waveform figure, ``<protocol>_iv.png`` I-V summary
* ``signature_ou.json``   DCA signature from the OU-noise clamp (primary)
* ``signature_steps.json`` DCA signature from the pooled step protocols
* ``signature.png``

and ``<out>/summary.md`` / ``summary.json`` with the headline numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuron_swap.channels import CHANNEL_NAMES  # noqa: E402
from neuron_swap.channels.l5pc import (  # noqa: E402
    REPRESENTATIVE_REGION,
    ca_dynamics_for_region,
    representative_channel,
)
from neuron_swap.protocols import default_protocols, run_voltage_clamp  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/phase1", help="output directory")
    ap.add_argument("--channels", nargs="*", default=list(CHANNEL_NAMES))
    ap.add_argument("--dt", type=float, default=0.025, help="integration step (ms)")
    ap.add_argument("--record-dt", type=float, default=0.1, help="sampling of the archived step-protocol waveforms (ms)")
    ap.add_argument("--bin-ms", type=float, default=0.5, help="DCA sampling interval (ms); the OU clamp is recorded at this rate")
    ap.add_argument("--T", type=int, nargs="*", default=[1, 2, 5, 10, 20, 40], help="DCA window lengths (bins)")
    ap.add_argument("--ou-duration", type=float, default=20000.0, help="OU clamp length (ms)")
    ap.add_argument("--ou-sweeps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dca", action="store_true", help="skip the DCA signatures")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--quick", action="store_true", help="short OU clamp, coarser dt (smoke test)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.ou_duration = min(args.ou_duration, 2000.0)
        args.dt = max(args.dt, 0.05)
        args.T = [t for t in args.T if t <= 10]

    do_dca = not args.no_dca
    if do_dca:
        try:
            from neuron_swap.dca import compute_signature
        except ImportError as exc:
            print(f"[warn] DCA disabled: {exc}")
            do_dca = False
    if not args.no_plots:
        from neuron_swap.plotting import plot_clamp, plot_iv, plot_signature

        import matplotlib.pyplot as plt

    summary = []
    for name in args.channels:
        t0 = time.time()
        region = REPRESENTATIVE_REGION[name]
        ch = representative_channel(name)
        cdir = out / name
        cdir.mkdir(exist_ok=True)
        print(f"== {name} (gbar {ch.gbar:g} S/cm2, {region}, erev {ch.erev:.1f} mV)")

        protocols = default_protocols(name, dt=args.dt, ou_seed=args.seed)
        # longer, multi-sweep OU clamp for the signatures
        from neuron_swap.protocols import ou_noise

        protocols["ou_noise"] = ou_noise(duration=args.ou_duration, n_sweeps=args.ou_sweeps, seed=args.seed, dt=args.dt)

        # SK_E2 has no voltage dependence of its own: drive its calcium from
        # the somatic Ca channels through the somatic calcium pool.
        clamp_kwargs = {}
        if ch.calcium_dependent:
            clamp_kwargs = dict(
                ca_pool=ca_dynamics_for_region(region),
                ca_sources=[representative_channel("Ca_HVA"), representative_channel("Ca_LVAst")],
            )

        results = {}
        rec_steps = max(1, int(round(args.record_dt / args.dt)))
        rec_ou = max(1, int(round(args.bin_ms / args.dt)))
        for pname, proto in protocols.items():
            kw = {} if proto.cai is not None else clamp_kwargs
            res = run_voltage_clamp(ch, proto, record_every=rec_ou if pname == "ou_noise" else rec_steps, **kw)
            results[pname] = res
            res.save(cdir / f"{pname}.npz")
            if not args.no_plots:
                fig = plot_clamp(res)
                fig.savefig(cdir / f"{pname}.png", dpi=120)
                plt.close(fig)
                if "step" in proto.epochs:
                    fig = plot_iv(res, "step")
                    fig.savefig(cdir / f"{pname}_iv.png", dpi=120)
                    plt.close(fig)
            print(f"   {pname:14s} {proto.n_sweeps:2d} sweeps x {proto.n_t:7d} pts (recorded {res.I.shape[1]})  "
                  f"peak |I| {np.max(np.abs(res.I)):.3e} mA/cm2")

        row = {
            "channel": name,
            "region": region,
            "gbar": ch.gbar,
            "erev": ch.erev,
            "gates": list(ch.gate_names),
            "peak_I_activation": float(np.max(np.abs(results["activation"].I))),
        }
        if do_dca:
            # the OU clamp is already recorded at bin_ms, so no further binning
            sig = compute_signature(results["ou_noise"], T_values=args.T, bin_ms=results["ou_noise"].dt, seed=args.seed, discard_ms=50.0)
            sig.save(cdir / "signature_ou.json")
            steps = results["activation"]
            sig_steps = compute_signature(steps, T_values=args.T, bin_ms=args.bin_ms, seed=args.seed)
            sig_steps.save(cdir / "signature_steps.json")
            if not args.no_plots:
                fig = plot_signature(sig)
                fig.savefig(cdir / "signature.png", dpi=120)
                plt.close(fig)
            row.update(sig.summary())
            Tm = max(args.T)
            row["pi_per_feature"] = {k: v[Tm] for k, v in sig.pi_per_feature.items()}
            print(f"   signature: PI(T={Tm})={sig.pi_curve[Tm]:.3f} nats, "
                  f"per feature {{{', '.join(f'{k}: {v[Tm]:.3f}' for k, v in sig.pi_per_feature.items())}}}, "
                  f"IO MI={sig.io_mi_curve.get(Tm, float('nan')):.3f}")
        row["seconds"] = round(time.time() - t0, 1)
        summary.append(row)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _write_markdown(out / "summary.md", summary, args)
    print(f"\nWrote {out}/summary.md")


def _write_markdown(path: Path, rows: list[dict], args) -> None:
    lines = ["# Phase 1 summary", ""]
    lines.append(f"dt = {args.dt} ms (archived at {args.record_dt} ms), DCA sampling = {args.bin_ms} ms, "
                 f"T = {args.T} samples, OU clamp {args.ou_duration:g} ms x {args.ou_sweeps} sweeps, seed {args.seed}")
    lines.append("")
    hdr = ["channel", "region", "gbar (S/cm²)", "gates", "peak |I| act. (mA/cm²)"]
    has_dca = any("pi_per_feature" in r for r in rows)
    if has_dca:
        Tm = max(args.T)
        hdr += [f"PI(T={Tm})", "PI per feature", "DCA d=1 leading", f"IO MI(T={Tm})"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in rows:
        cells = [r["channel"], r["region"], f"{r['gbar']:.4g}", ",".join(r["gates"]), f"{r['peak_I_activation']:.3g}"]
        if has_dca and "pi_per_feature" in r:
            Tm = max(args.T)
            cells += [
                f"{r[f'PI(T={Tm})']:.3f}",
                ", ".join(f"{k} {v:.2f}" for k, v in r["pi_per_feature"].items()),
                r.get("DCA_d1_leading_feature", ""),
                f"{r.get(f'IO_MI(T={Tm})', float('nan')):.3f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
