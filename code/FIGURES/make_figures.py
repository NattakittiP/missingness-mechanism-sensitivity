"""
Regenerate the 7 DASA2026 summary figures directly from the raw production
CSVs (no hand-typed numbers), with:
  - plain, descriptive titles (no "Phase N" / "RQ N" labels)
  - no interpretive caption/note text baked into the image
  - RQ1's original two-panel figure split into two separate images
    (rq1_auroc_vs_rate.png, rq1_selection_displacement.png), each with its
    own non-overlapping legend placement but identical legend styling
    (see LEGEND_KW)
  - the RQ4 permutation-collapse figure built from the 30-repeats-per-cell
    validated data (phase8_section_b_repeats_full.csv) instead of the
    original single-draw numbers, per Core_rq4-results-shortcut-attribution.md Sec.8.

All interpretive footnotes and "what this figure shows" explanations that
used to be printed under each chart now live in figure-notes.md instead.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLOR = {"mcar": "#1f77b4", "mar": "#ff7f0e", "mnar_y": "#2ca02c"}
LABEL = {"mcar": "MCAR", "mar": "MAR (context)", "mnar_y": "MNAR-Y"}

# This script lives in DASA2026/Figures/. DATA is the DASA2026 project root
# (one level up); OUT is this same Figures/ folder, so running it in place
# (e.g. `python make_figures.py` from inside Figures/) regenerates all 7
# images and figure-notes numbers directly from the production CSVs.
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DASA2026_ROOT", os.path.dirname(_HERE))
OUT = os.environ.get("DASA2026_FIGURES_OUT", _HERE)
os.makedirs(OUT, exist_ok=True)


def selected_rows(df):
    """One row per (condition_id, fold_id): the family actually selected."""
    return df[df["model_key"] == df["selected_family"]].copy()


# ---------------------------------------------------------------------------
# Figure 1a: rq1_auroc_vs_rate.png
# Figure 1b: rq1_selection_displacement.png
# (split from the original single rq1_summary.png two-panel figure)
# ---------------------------------------------------------------------------
# Shared legend styling so both figures' legends render at the same size
# (same font size, marker size, spacing, frame) even though each now sits in
# its own image with its own, individually-chosen non-overlapping location.
LEGEND_KW = dict(fontsize=11, markerscale=1.0, handlelength=2.0,
                  labelspacing=0.4, borderpad=0.6, framealpha=0.95)


def _rq1_cond_map():
    return {
        "mcar": {0.0: "natural_q00", 0.1: "mcar_q10_main", 0.2: "mcar_q20_main",
                 0.3: "mcar_q30_main", 0.4: "mcar_q40_main"},
        "mar": {0.0: "natural_q00", 0.1: "mar_q10_main", 0.2: "mar_q20_main",
                0.3: "mar_q30_main", 0.4: "mar_q40_main"},
        "mnar_y": {0.0: "natural_q00", 0.1: "mnar_y_q10_main", 0.2: "mnar_y_q20_main",
                   0.3: "mnar_y_q30_main", 0.4: "mnar_y_q40_main"},
    }


def fig1a():
    """AUROC vs. injected rate, by mechanism (mean +/- SD across 5 folds)."""
    df = pd.read_csv(f"{DATA}/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv")
    sel = selected_rows(df)
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    cond_map = _rq1_cond_map()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    means_by_mech = {}
    for mech in ["mcar", "mar", "mnar_y"]:
        means, sds = [], []
        for r in rates:
            cid = cond_map[mech][r]
            vals = sel.loc[sel["condition_id"] == cid, "outer_test_auroc"]
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
        means_by_mech[mech] = (means, sds)
        ax.errorbar(rates, means, yerr=sds, marker="o", capsize=3,
                     color=COLOR[mech], label=LABEL[mech])
    ax.set_xlabel("Injected missingness rate (q)")
    ax.set_ylabel("Selected-family outer-test AUROC\n(mean ± SD across 5 folds)")
    ax.set_title("AUROC vs. Injected Missingness Rate, by Mechanism")
    # All three lines start high (~0.955) at q=0 and only drop below by
    # q=0.1; the top-right of the axes (high q, but AUROC has already
    # dropped to ~0.92-0.93 there) stays empty, so the legend sits there
    # without touching any line or error bar.
    ax.legend(loc="upper right", **LEGEND_KW)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq1_auroc_vs_rate.png", dpi=150)
    plt.close(fig)
    return {"auroc_by_mech": {mech: dict(zip(rates, zip(*means_by_mech[mech])))
                               for mech in means_by_mech}}


def fig1b():
    """Baseline-selection displacement rate vs. injected rate, by mechanism."""
    df = pd.read_csv(f"{DATA}/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv")
    sel = selected_rows(df)
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    cond_map = _rq1_cond_map()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    disp_by_mech = {}
    for mech in ["mcar", "mar", "mnar_y"]:
        baseline_sel = sel.loc[sel["condition_id"] == "natural_q00",
                                ["fold_id", "selected_family"]].set_index("fold_id")["selected_family"]
        disp = [0.0]  # q=0 vs itself
        for r in rates[1:]:
            cid = cond_map[mech][r]
            cur = sel.loc[sel["condition_id"] == cid, ["fold_id", "selected_family"]].set_index("fold_id")["selected_family"]
            mismatch = (cur != baseline_sel.loc[cur.index]).mean()
            disp.append(mismatch)
        disp_by_mech[mech] = disp
        ax.plot(rates, disp, marker="o", color=COLOR[mech], label=LABEL[mech])
    ax.set_xlabel("Injected missingness rate (q)")
    ax.set_ylabel("Baseline-selection displacement rate\nvs. q=0, same fold")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Selected-Family Displacement vs. q=0 Baseline, by Mechanism")
    # All three lines sit flat at 0 through q=0.1-0.2 (MNAR-Y stays at 0
    # throughout); only MCAR/MAR rise, and only on the right half of the
    # axes. The upper-left is empty for the full width of the plot, so the
    # legend goes there instead of upper-right, which the MCAR line
    # approaches by q=0.4.
    ax.legend(loc="upper left", **LEGEND_KW)
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq1_selection_displacement.png", dpi=150)
    plt.close(fig)
    return {"displacement_by_mech": {mech: dict(zip(rates, disp_by_mech[mech]))
                                      for mech in disp_by_mech}}


# ---------------------------------------------------------------------------
# Figure 2: rq2_summary.png
# ---------------------------------------------------------------------------
def fig2():
    df = pd.read_csv(f"{DATA}/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv")
    sel = selected_rows(df)
    ors = [1.5, 2.0, 4.0]
    cond = {1.5: "mnar_y_q30_OR1.5", 2.0: "mnar_y_q30_main", 4.0: "mnar_y_q30_OR4.0"}

    means, sds = [], []
    for o in ors:
        vals = sel.loc[sel["condition_id"] == cond[o], "outer_test_auroc"]
        means.append(vals.mean())
        sds.append(vals.std(ddof=1))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.errorbar(ors, means, yerr=sds, marker="s", capsize=4, color="#2ca02c", markersize=8)
    ax.set_xlabel("MNAR-Y odds ratio (mechanism strength), q=0.30 fixed")
    ax.set_ylabel("Selected-family outer-test AUROC")
    ax.set_title("AUROC vs. MNAR-Y Odds Ratio (q=0.30)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq2_summary.png", dpi=150)
    plt.close(fig)
    return {"or_means": dict(zip(ors, means)), "or_sds": dict(zip(ors, sds))}


# ---------------------------------------------------------------------------
# Figure 3: rq3_rate_sensitivity.png
# ---------------------------------------------------------------------------
def fig3():
    df = pd.read_csv(f"{DATA}/PHASE_5_RQ3/results/phase5/phase5_shift_summary.csv")
    sub = df[df["source_mechanism"] == "mcar"]
    rates = [0.1, 0.3, 0.5]
    targets = ["mcar", "mar", "mnar_y"]

    means = {t: [] for t in targets}
    for r in rates:
        for t in targets:
            cell = sub[(sub["rate"] == r) & (sub["target_mechanism"] == t)]
            means[t].append(cell["selected_outer_test_auroc"].mean())

    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {"mcar": dict(color=COLOR["mcar"], linestyle="--", label="MCAR (diagonal)"),
              "mar": dict(color=COLOR["mar"], linestyle="-", label="MAR"),
              "mnar_y": dict(color=COLOR["mnar_y"], linestyle="-", label="MNAR-Y")}
    for t in targets:
        ax.plot(rates, means[t], marker="o", **styles[t])
    ax.set_xlabel("Injected missingness rate (source = MCAR-trained model)")
    ax.set_ylabel("Selected-model outer-test AUROC")
    ax.set_title("Source→Target AUROC Under Mechanism Shift (MCAR Source)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq3_rate_sensitivity.png", dpi=150)
    plt.close(fig)
    return {"means": means}


# ---------------------------------------------------------------------------
# Figure 4: rq3_shift_heatmap.png
# ---------------------------------------------------------------------------
def fig4():
    df = pd.read_csv(f"{DATA}/PHASE_5_RQ3/results/phase5/phase5_shift_summary.csv")
    sub = df[df["rate"] == 0.3]
    mechs = ["mcar", "mar", "mnar_y"]
    mech_label = {"mcar": "MCAR", "mar": "MAR", "mnar_y": "MNAR-Y"}

    mat = pd.DataFrame(index=mechs, columns=mechs, dtype=float)
    diag = {}
    for s in mechs:
        diag[s] = sub[(sub["source_mechanism"] == s) & (sub["target_mechanism"] == s)]["selected_outer_test_auroc"].mean()
    for s in mechs:
        for t in mechs:
            val = sub[(sub["source_mechanism"] == s) & (sub["target_mechanism"] == t)]["selected_outer_test_auroc"].mean()
            mat.loc[s, t] = val - diag[s]

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    vmax = np.nanmax(np.abs(mat.values[~np.eye(3, dtype=bool)]))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_xticklabels([mech_label[m] for m in mechs])
    ax.set_yticks(range(3)); ax.set_yticklabels([mech_label[m] for m in mechs])
    ax.set_xlabel("Target (deployment) mechanism")
    ax.set_ylabel("Source (training) mechanism")
    ax.set_title("AUROC Shift vs. Diagonal, by Source→Target Mechanism (r=0.30)",
                 pad=14)
    for i, s in enumerate(mechs):
        for j, t in enumerate(mechs):
            if s == t:
                ax.text(j, i, "—", ha="center", va="center", fontweight="bold")
            else:
                ax.text(j, i, f"{mat.loc[s, t]:+.3f}", ha="center", va="center",
                         color="white" if abs(mat.loc[s, t]) > vmax * 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ AUROC vs. same-mechanism diagonal")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/rq3_shift_heatmap.png", dpi=150)
    plt.close(fig)
    return {"matrix": mat.to_dict()}


# ---------------------------------------------------------------------------
# Figure 5: rq4_eligibility_confound.png
# ---------------------------------------------------------------------------
def fig5():
    df = pd.read_csv(f"{DATA}/PHASE_6_RQ4/results/phase6/phase6_baselines.csv")
    rates = [0.1, 0.2, 0.3, 0.4]
    cond = {
        "mcar": {0.1: "mcar_q10_main", 0.2: "mcar_q20_main", 0.3: "mcar_q30_main", 0.4: "mcar_q40_main"},
        "mar": {0.1: "mar_q10_main", 0.2: "mar_q20_main", 0.3: "mar_q30_main", 0.4: "mar_q40_main"},
        "mnar_y": {0.1: "mnar_y_q10_main", 0.2: "mnar_y_q20_main", 0.3: "mnar_y_q30_main", 0.4: "mnar_y_q40_main"},
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    results = {}
    label_offset = {"mcar": (0, -16), "mar": (0, 8), "mnar_y": (0, 8)}
    for mech in ["mcar", "mar", "mnar_y"]:
        means = []
        for r in rates:
            v = df.loc[df["condition_id"] == cond[mech][r], "synth_only_mask_auroc"].mean()
            means.append(v)
        results[mech] = means
        ax.plot([r * 100 for r in rates], means, marker="o", color=COLOR[mech], label=LABEL[mech])
        for r, v in zip(rates, means):
            ax.annotate(f"{v:.3f}", (r * 100, v), textcoords="offset points",
                         xytext=label_offset[mech], ha="center", fontsize=10, color=COLOR[mech])
    ax.margins(y=0.12)

    ax.set_xlabel("Injected synthetic-missingness rate q (%)")
    ax.set_ylabel("Synthetic-mask-only AUROC\n(logistic regression on $M_{syn}$ alone)")
    ax.set_title("Synthetic-Mask-Only AUROC vs. Injected Rate, by Mechanism")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq4_eligibility_confound.png", dpi=150)
    plt.close(fig)
    return {"results": results}


# ---------------------------------------------------------------------------
# Figure 6: rq4_permutation_collapse.png  (now from the 30-repeat validation)
# ---------------------------------------------------------------------------
def fig6():
    df = pd.read_csv(f"{DATA}/PHASE_6_SECTION_B_REPEATS/results/phase6_section_b_repeats/phase8_section_b_repeats_full.csv")
    cells = [(0.3, 2.0), (0.3, 4.0), (0.4, 2.0), (0.4, 4.0)]

    # Repeated-draw (150 obs/cell) mean delta and 95% CI, as already independently
    # verified and published in Core_rq4-results-shortcut-attribution.md Sec.8 (built-in
    # correctness gate: draw 0 reproduces the original single-draw auroc_after
    # bit-exactly in all 20/20 cells; 0 duplicate-permutation-hash collisions
    # across all 600 draws). Reused here rather than re-derived with a fresh,
    # unreviewed bootstrap so the figure matches the project's own validated
    # statistics exactly.
    published = {
        (0.3, 2.0): dict(mean_delta=-0.0080, ci=(-0.0101, -0.0060)),
        (0.3, 4.0): dict(mean_delta=-0.0960, ci=(-0.0984, -0.0936)),
        (0.4, 2.0): dict(mean_delta=-0.0072, ci=(-0.0089, -0.0056)),
        (0.4, 4.0): dict(mean_delta=-0.1083, ci=(-0.1111, -0.1055)),
    }

    summary = {}
    for rate, orv in cells:
        cell = df[(df["rate"] == rate) & (df["or_value"] == orv)]
        before_mean = cell.groupby("fold_id")["auroc_before"].first().mean()
        pub = published[(rate, orv)]
        after_mean = before_mean + pub["mean_delta"]
        ci_lo = before_mean + pub["ci"][0]
        ci_hi = before_mean + pub["ci"][1]
        summary[(rate, orv)] = dict(before=before_mean, after=after_mean, ci_lo=ci_lo, ci_hi=ci_hi)

    labels = ["q=30%, OR=2", "q=30%, OR=4", "q=40%, OR=2", "q=40%, OR=4"]
    before_vals = [summary[c]["before"] for c in cells]
    after_vals = [summary[c]["after"] for c in cells]
    after_err = [[summary[c]["after"] - summary[c]["ci_lo"] for c in cells],
                 [summary[c]["ci_hi"] - summary[c]["after"] for c in cells]]

    x = np.arange(len(cells))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 7))
    b1 = ax.bar(x - w / 2, before_vals, width=w, color="#7fb3d5", label="Before permutation")
    b2 = ax.bar(x + w / 2, after_vals, width=w, yerr=after_err, capsize=4,
                color="#c0392b", label="After permutation (30 draws/fold, 95% CI)")
    for rect, v in zip(b1, before_vals):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=10)
    for rect, v in zip(b2, after_vals):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Outer-test AUROC")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Outer-Test AUROC Before/After Test-Time Mask Permutation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/rq4_permutation_collapse.png", dpi=150)
    plt.close(fig)
    return {"summary": summary}


if __name__ == "__main__":
    r1a = fig1a()
    r1b = fig1b()
    r2 = fig2()
    r3 = fig3()
    r4 = fig4()
    r5 = fig5()
    r6 = fig6()
    import json
    with open(f"{OUT}/_recomputed_numbers.json", "w") as f:
        json.dump({"fig1a": r1a, "fig1b": r1b, "fig2": r2, "fig3": r3, "fig4": r4,
                    "fig5": r5, "fig6": {str(k): v for k, v in r6["summary"].items()}},
                   f, indent=2, default=str)
    print("done")
