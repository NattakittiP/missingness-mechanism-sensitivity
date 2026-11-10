"""
Build the three two-panel manuscript figures (Manuscript_Plan F-1 to F-3) from
the result tables, drawn natively at IEEE double-column width:

  fig1_selection_vs_discrimination  (a) selected-family AUROC vs q
                                    (b) displaced folds (of 5) vs q
  fig2_mechanism_shift              (a) q = 0.30 source -> target AUROC change
                                    (b) shift change for all six source -> target pairs at
                                        q = 0.10, 0.30, 0.50 (Results/R1_permutation_rate_shift)
                                    (b) MCAR-trained model on three targets, q = 0.10/0.30/0.50
  fig3_mask_attribution             (a) injected-mask-only AUROC vs q
                                    (b) AUROC before / after test-time mask permutation

Panels carry (a)/(b) labels only (no titles); rates are fractions q on every
axis; captions live in the manuscript. Every plotted value is recomputed here
and checked against make_figures.py's _recomputed_numbers.json, so the
manuscript panels and the single-panel figures cannot drift apart.

Run from this folder:  python make_manuscript_figures.py
Outputs: ./manuscript/fig{1,2,3}_*.pdf and .png, ./manuscript/_manuscript_numbers.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import COLOR, DATA, _rq1_cond_map, selected_rows

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("DASA2026_FIGURES_OUT", HERE), "manuscript")
os.makedirs(OUT, exist_ok=True)

LABEL = {"mcar": "MCAR", "mar": "MAR", "mnar_y": "MNAR-Y"}
MECHS = ["mcar", "mar", "mnar_y"]
WIDTH = 7.16  # IEEE \textwidth in inches
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "lines.markersize": 4, "savefig.dpi": 300, "pdf.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
# small horizontal dodge so overlapping series stay visible (MAR and MNAR-Y are both 0 in F-1b)
DODGE = {"mcar": -0.006, "mar": 0.0, "mnar_y": 0.006}
RATE_SHIFT = os.environ.get("DASA2026_RATE_SHIFT", os.path.join(DATA, "..", "..", "Results", "R1_permutation_rate_shift",
                                                                 "rate_shift_eval.csv"))


def panel_label(ax, s):
    ax.text(-0.20, 1.03, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig1():
    sel = selected_rows(pd.read_csv(f"{DATA}/PHASE_4_RQ1_RQ2/results/phase4/phase4_full_table.csv"))
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    cmap = _rq1_cond_map()
    base = sel.loc[sel["condition_id"] == "natural_q00"].set_index("fold_id")["selected_family"]
    auroc, disp = {}, {}
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.4))
    for m in MECHS:
        mu, sd, k = [], [], []
        for r in rates:
            v = sel.loc[sel["condition_id"] == cmap[m][r]]
            mu.append(v["outer_test_auroc"].mean())
            sd.append(v["outer_test_auroc"].std(ddof=1))
            cur = v.set_index("fold_id")["selected_family"]
            k.append(int((cur != base.loc[cur.index]).sum()))
        auroc[m], disp[m] = (mu, sd), k
        x = np.array(rates) + DODGE[m]
        a.errorbar(x, mu, yerr=sd, marker="o", capsize=2, color=COLOR[m], label=LABEL[m])
        b.plot(x, k, marker="o", color=COLOR[m], label=LABEL[m])
    a.set_xlabel("Injected rate $q$"); a.set_ylabel("Selected-family AUROC")
    b.set_xlabel("Injected rate $q$"); b.set_ylabel("Displaced folds (of 5)")
    b.set_ylim(-0.2, 5.2); b.set_yticks(range(6))
    for ax in (a, b):
        ax.set_xticks(rates)
    panel_label(a, "(a)"); panel_label(b, "(b)")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig1_selection_vs_discrimination")
    return {"auroc_mean_sd": {m: dict(zip(rates, zip(*auroc[m]))) for m in MECHS},
            "displaced_folds": {m: dict(zip(rates, disp[m])) for m in MECHS}}


def fig2():
    df = pd.read_csv(f"{DATA}/PHASE_5_RQ3/results/phase5/phase5_shift_summary.csv")
    q30 = df[df["rate"] == 0.3]
    cell = lambda s, t: q30[(q30["source_mechanism"] == s) & (q30["target_mechanism"] == t)]["selected_outer_test_auroc"].mean()
    mat = np.array([[cell(s, t) - cell(s, s) for t in MECHS] for s in MECHS])
    src = df[df["source_mechanism"] == "mcar"]
    rates = [0.1, 0.3, 0.5]
    tgt = {t: [src[(src["rate"] == r) & (src["target_mechanism"] == t)]["selected_outer_test_auroc"].mean()
               for r in rates] for t in MECHS}

    rs = pd.read_csv(RATE_SHIFT)
    rs["sel"] = rs.apply(lambda r: r[f"auroc_{r.selected_family}"], axis=1)
    rs["src"], rs["tgt"] = rs.source.str.rsplit("_q", n=1).str[0], rs.target.str.rsplit("_q", n=1).str[0]
    own = rs[rs.source == rs.target].set_index(["rate", "src", "fold_id"]).sel
    rs["delta"] = rs.sel.values - own.loc[list(zip(rs.rate, rs.src, rs.fold_id))].values
    shift = {(s, t): [rs[(rs.src == s) & (rs.tgt == t) & (rs.rate == r)].delta.mean() for r in rates]
             for s in MECHS for t in MECHS if s != t}
    for (s, t), v in shift.items():  # the q = 0.30 column is the matrix of panel (a)
        assert abs(v[1] - mat[MECHS.index(s), MECHS.index(t)]) < 0.03, ("rate file vs Phase 5", s, t, v[1])

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5), gridspec_kw={"width_ratios": [1, 1.15]})
    off = mat[~np.eye(3, dtype=bool)]
    vmax = np.abs(off).max()
    im = a.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    a.set_xticks(range(3)); a.set_xticklabels([LABEL[m] for m in MECHS])
    a.set_yticks(range(3)); a.set_yticklabels([LABEL[m] for m in MECHS])
    a.set_xlabel("Target (deployment) mechanism"); a.set_ylabel("Source (training) mechanism")
    for i in range(3):
        for j in range(3):
            txt = "—" if i == j else f"{mat[i, j]:+.4f}"
            a.text(j, i, txt, ha="center", va="center", fontsize=7,
                   color="white" if i != j and abs(mat[i, j]) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.set_label("ΔAUROC vs. source diagonal", fontsize=7); cb.ax.tick_params(labelsize=6)
    style = {"mcar": ":", "mar": "--", "mnar_y": "-"}  # line style = target, colour = source
    for (s, t), v in shift.items():
        b.plot(rates, v, marker="o", color=COLOR[s], linestyle=style[t], label=f"{LABEL[s]}→{LABEL[t]}")
    b.axhline(0, color="0.6", linewidth=0.6)
    b.set_xticks(rates); b.set_xlabel("Injected rate $q$"); b.set_ylabel("ΔAUROC vs. source diagonal")
    b.legend(frameon=False, fontsize=6, ncol=2)
    panel_label(a, "(a)"); panel_label(b, "(b)")
    fig.tight_layout()
    save(fig, "fig2_mechanism_shift")
    return {"shift_q30": {s: dict(zip(MECHS, mat[i])) for i, s in enumerate(MECHS)},
            "mcar_source_auroc": {t: dict(zip(rates, tgt[t])) for t in MECHS},
            "shift_by_rate": {f"{s}->{t}": dict(zip(rates, v)) for (s, t), v in shift.items()}}


def fig3():
    bl = pd.read_csv(f"{DATA}/PHASE_6_RQ4/results/phase6/phase6_baselines.csv")
    rates = [0.1, 0.2, 0.3, 0.4]
    cmap = _rq1_cond_map()
    synth = {m: [bl.loc[bl["condition_id"] == cmap[m][r], "synth_only_mask_auroc"].mean() for r in rates] for m in MECHS}

    rep = pd.read_csv(f"{DATA}/PHASE_6_SECTION_B_REPEATS/results/phase6_section_b_repeats/phase8_section_b_repeats_full.csv")
    st = pd.read_csv(f"{DATA}/PHASE_STATS_LAYER/phase_stats_layer_v2_results.csv").set_index("test_id")
    cells = [(0.3, 2.0), (0.3, 4.0), (0.4, 2.0), (0.4, 4.0)]
    perm = {}
    for q, o in cells:
        c = rep[(rep["rate"] == q) & (rep["or_value"] == o)]
        row = st.loc[f"RQ4.repeats.rate{q}_OR{o}"]
        assert len(c) == 150 and abs(c["delta_auroc"].mean() - float(row["statistic"])) < 1e-9, (q, o)
        before = c.groupby("fold_id")["auroc_before"].first().mean()
        perm[(q, o)] = dict(before=before, after=before + float(row["statistic"]),
                            lo=before + float(row["ci_low"]), hi=before + float(row["ci_high"]))

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    for m in MECHS:
        a.plot(rates, synth[m], marker="o", color=COLOR[m], label=LABEL[m])
    a.set_xticks(rates); a.set_xlabel("Injected rate $q$"); a.set_ylabel("Injected-mask-only\nAUROC")
    a.legend(frameon=False)
    x = np.arange(len(cells)); w = 0.38
    bef = [perm[c]["before"] for c in cells]; aft = [perm[c]["after"] for c in cells]
    err = [[perm[c]["after"] - perm[c]["lo"] for c in cells], [perm[c]["hi"] - perm[c]["after"] for c in cells]]
    b.bar(x - w / 2, bef, w, color="#7fb3d5", label="before")
    b.bar(x + w / 2, aft, w, yerr=err, capsize=2, color="#c0392b", label="after (150 draws, 95% interval)")
    for xi, v in zip(x - w / 2, bef):
        b.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=5.5)
    for xi, v in zip(x + w / 2, aft):  # inside the bar so it cannot collide with the neighbouring "before" label
        b.text(xi, v - 0.006, f"{v:.4f}", ha="center", va="top", fontsize=5.5, color="white")
    b.set_xticks(x); b.set_xticklabels([f"q={q:.2f}\nOR {o:g}" for q, o in cells])
    b.set_ylim(0.80, 1.0); b.set_ylabel("Outer-test AUROC")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    panel_label(a, "(a)"); panel_label(b, "(b)")
    fig.tight_layout()
    save(fig, "fig3_mask_attribution")
    return {"synth_only_mask_auroc": {m: dict(zip(rates, synth[m])) for m in MECHS},
            "permutation": {str(k): v for k, v in perm.items()}}


def check_against_single_panel(r1, r2, r3):
    """Assert the composites plot the same values as make_figures.py (its last run)."""
    ref = json.load(open(os.path.join(HERE, "_recomputed_numbers.json")))
    close = lambda u, v: abs(float(u) - float(v)) < 1e-12
    for m in MECHS:
        for q, (mu, _) in r1["auroc_mean_sd"][m].items():
            assert close(mu, ref["fig1a"]["auroc_by_mech"][m][str(q)][0]), ("fig1a", m, q)
        for q, k in r1["displaced_folds"][m].items():
            assert close(k / 5, ref["fig1b"]["displacement_by_mech"][m][str(q)]), ("fig1b", m, q)
        for t in MECHS:  # make_figures stores the matrix column-major: {target: {source: value}}
            assert close(r2["shift_q30"][m][t], ref["fig4"]["matrix"][t][m]), ("fig4", m, t)
        for q, v in r2["mcar_source_auroc"][m].items():
            assert close(v, ref["fig3"]["means"][m][[0.1, 0.3, 0.5].index(q)]), ("fig3", m, q)
        for i, v in enumerate(r3["synth_only_mask_auroc"][m].values()):
            assert close(v, ref["fig5"]["results"][m][i]), ("fig5", m, i)
    for k, v in r3["permutation"].items():
        assert close(v["after"], ref["fig6"][k]["after"]) and close(v["before"], ref["fig6"][k]["before"]), ("fig6", k)


if __name__ == "__main__":
    r1, r2, r3 = fig1(), fig2(), fig3()
    check_against_single_panel(r1, r2, r3)
    with open(os.path.join(OUT, "_manuscript_numbers.json"), "w") as f:
        json.dump({"fig1": r1, "fig2": r2, "fig3": r3}, f, indent=2, default=str)
    print("done: 3 figures written to", OUT, "; values match _recomputed_numbers.json")
