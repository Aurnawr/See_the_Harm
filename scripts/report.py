import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plots and a markdown summary for one stage of probes.py, including the pass/fail table for the
# decision criteria in docs/analysis_plan.md. No overall verdict is computed.

COLORS = {"img": "#2a6fdb", "last": "#d9480f"}


def decode(m, **kw):
    q = m[m["analysis"] == kw.pop("analysis", "decode")]
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.sort_values("layer")


def plot_curves(m, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    panels = [("Within prompt A (all safe)", dict(prompt_train="A", prompt_eval="A", subset="all")),
              ("Cross prompt A to B (all safe)", dict(prompt_train="A", prompt_eval="B", subset="all")),
              ("Unsafe vs topic-matched safe (A)", dict(prompt_train="A", prompt_eval="A", subset="unsafe_vs_matched"))]
    low = m[(m["analysis"] == "lowlevel") & (m["subset"] == "all")]["auroc"]
    for ax, (title, kw) in zip(axes, panels):
        for pos in ["img", "last"]:
            for probe, ls in [("lr", "-"), ("dim", "--")]:
                q = decode(m, position=pos, probe=probe, **kw)
                ax.plot(q["layer"], q["auroc"], ls, color=COLORS[pos], label=f"{pos} {probe}")
                if probe == "lr":
                    ax.fill_between(q["layer"], q["auroc_lo"], q["auroc_hi"], color=COLORS[pos], alpha=0.15)
        null = m[(m["analysis"] == "perm_null") & (m["position"] == "last") & (m["probe"] == "dim")].sort_values("layer")
        ax.plot(null["layer"], null["null_p95"], ":", color="gray", label="perm 95th (last, dim)")
        if len(low):
            ax.axhline(low.iloc[0], color="black", lw=0.8, label="low-level baseline")
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel("layer (0 = vision encoder output)")
    axes[0].set_ylabel("test AUROC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "decodability.png"), dpi=150)
    plt.close(fig)


def plot_directions(dirs, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for pos in ["img", "last"]:
        q = dirs[dirs["position"] == pos].sort_values("layer")
        ax.plot(q["layer"], q["split_half_cos"], "-", color=COLORS[pos], label=f"{pos} split-half")
        ax.plot(q["layer"], q["cos_dim_A_B"], "--", color=COLORS[pos], label=f"{pos} prompt A vs B")
        ax.plot(q["layer"], q["cos_dim_lr"], ":", color=COLORS[pos], label=f"{pos} DiM vs LR")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.set_title("DiM direction stability")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "directions.png"), dpi=150)
    plt.close(fig)


def best_layer(m):
    within = decode(m, prompt_train="A", prompt_eval="A", subset="all", position="last", probe="lr").set_index("layer")
    return int(within["auroc"].idxmax())


def criteria(m, dirs, L):
    within = decode(m, prompt_train="A", prompt_eval="A", subset="all", position="last", probe="lr").set_index("layer")
    w = within.loc[L]
    vision = decode(m, prompt_train="A", prompt_eval="A", subset="all", position="img", probe="lr").set_index("layer").loc[0, "auroc"]
    # LR permutations only run at a few layers; elsewhere fall back to the DiM null
    nulls = m[(m["analysis"] == "perm_null") & (m["position"] == "last")]
    lr_null = nulls[nulls["probe"] == "lr"].set_index("layer")["null_p95"]
    null_src = "LR" if L in lr_null.index else "DiM"
    null_L = lr_null.loc[L] if L in lr_null.index else nulls[nulls["probe"] == "dim"].set_index("layer").loc[L, "null_p95"]
    low = m[(m["analysis"] == "lowlevel") & (m["subset"] == "all")]["auroc"].iloc[0]
    cross = decode(m, prompt_train="A", prompt_eval="B", subset="all", position="last", probe="lr").set_index("layer").loc[L, "auroc"]
    matched = decode(m, prompt_train="A", prompt_eval="A", subset="unsafe_vs_matched", position="last", probe="lr").set_index("layer").loc[L, "auroc"]
    loco = decode(m, analysis="loco", position="last", probe="lr")
    loco_L = loco[loco["layer"] == L]["auroc"]
    dim = decode(m, prompt_train="A", prompt_eval="A", subset="all", position="last", probe="dim").set_index("layer").loc[L, "auroc"]
    dr = dirs[(dirs["position"] == "last") & (dirs["layer"] == L)].iloc[0]
    rows = [
        ("Decodable beyond vision encoder",
         f"LR AUROC {w['auroc']:.3f} [{w['auroc_lo']:.3f}, {w['auroc_hi']:.3f}]; perm p95 {null_L:.3f} ({null_src}); vision-encoder {vision:.3f}",
         w["auroc_lo"] > 0.75 and w["auroc"] > null_L and w["auroc"] > vision + 0.05),
        ("Not a low-level artefact", f"low-level AUROC {low:.3f}", low < 0.65 or low <= w["auroc"] - 0.10),
        ("Text-independent", f"cross-prompt {cross:.3f} vs within {w['auroc']:.3f}", abs(cross - w["auroc"]) <= 0.05),
        ("Harm, not topic",
         f"unsafe vs matched {matched:.3f}; mean LOCO {loco_L.mean():.3f} over {len(loco_L)} categories",
         matched >= 0.70 and len(loco_L) > 0 and loco_L.mean() >= 0.70),
        ("Single direction",
         f"DiM {dim:.3f} vs LR {w['auroc']:.3f}; split-half cos {dr['split_half_cos']:.3f}; A vs B cos {dr['cos_dim_A_B']:.3f}",
         abs(dim - w["auroc"]) <= 0.05 and dr["split_half_cos"] >= 0.8 and dr["cos_dim_A_B"] >= 0.8),
    ]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "final"], required=True)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    out = os.path.join(args.results, args.stage)
    m = pd.read_csv(os.path.join(out, "metrics.csv"))
    dirs = pd.read_csv(os.path.join(out, "directions.csv"))
    plot_curves(m, out)
    plot_directions(dirs, out)

    # L* is always chosen on the pilot stage, so the test split plays no part in picking it
    pilot = pd.read_csv(os.path.join(args.results, "pilot", "metrics.csv"))
    L = best_layer(pilot)
    rows = criteria(m, dirs, L)
    lines = [f"# RQ1 results ({args.stage})", "",
             f"L* = layer {L} (highest LR AUROC, last token, prompt A, chosen on the pilot stage).", "",
             "| Criterion | Numbers | Pass |", "|---|---|---|"]
    lines += [f"| {name} | {nums} | {'yes' if ok else 'no'} |" for name, nums, ok in rows]
    loco = decode(m, analysis="loco", position="last", probe="lr")
    loco = loco[loco["layer"] == L][["subset", "auroc", "auroc_lo", "auroc_hi", "n_pos", "n_neg"]]
    loco_table = loco.to_markdown(index=False, floatfmt=".3f") if len(loco) else "No category has >= 8 images per class here."
    lines += ["", "## Leave-one-category-out at L* (last token, LR)", "", loco_table,
              "", "![decodability](decodability.png)", "", "![directions](directions.png)", ""]
    if args.stage == "pilot":
        lines.insert(2, "Pilot numbers come from 5-fold CV on the pilot split only. They are for debugging, not for conclusions.\n")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
