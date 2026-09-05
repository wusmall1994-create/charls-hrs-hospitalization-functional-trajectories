from __future__ import annotations

from config import OUTPUT_DIR as OUT

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "legend.frameon": False,
})

COLORS = {
    "CHARLS": "#2F6B9A",
    "HRS": "#D9822B",
    "Fixed-effect pooled": "#4D4D4D",
    "First observed hospitalization": "#9B3A4A",
    "Never hospitalized": "#7A7A7A",
}
MARKERS = {"CHARLS": "o", "HRS": "s", "Fixed-effect pooled": "D"}
TERM_X = {"pre2": -2, "reference": -1, "index": 0, "post1": 1, "post2": 2}
TERM_LABELS = ["≤−2", "−1\n(ref.)", "0\n(index)", "+1", "≥+2"]


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.07, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def save_pub(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def trajectory_panel(ax, data: pd.DataFrame, domain: str, title: str) -> None:
    z = data[data["domain"].eq(domain)].copy()
    for cohort in ["CHARLS", "HRS"]:
        g = z[z["cohort"].eq(cohort)].copy()
        reference = pd.DataFrame([{
            "term": "reference",
            "standardized_estimate": 0.0,
            "standardized_ci_low": 0.0,
            "standardized_ci_high": 0.0,
        }])
        g = pd.concat([g, reference], ignore_index=True)
        g["x"] = g["term"].map(TERM_X)
        g = g.sort_values("x")
        ax.plot(g["x"], g["standardized_estimate"], color=COLORS[cohort], lw=1.2, marker=MARKERS[cohort], ms=4, label=cohort)
        ax.errorbar(
            g["x"], g["standardized_estimate"],
            yerr=[g["standardized_estimate"] - g["standardized_ci_low"], g["standardized_ci_high"] - g["standardized_estimate"]],
            fmt="none", ecolor=COLORS[cohort], elinewidth=0.8, capsize=2,
        )
    ax.axhline(0, color="#B3B3B3", lw=0.7)
    ax.axvline(-0.5, color="#B3B3B3", lw=0.7, ls="--")
    ax.set_xticks([-2, -1, 0, 1, 2], TERM_LABELS)
    ax.set_xlabel("Survey waves relative to first observed hospitalization")
    ax.set_ylabel("Difference, baseline SD units")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E8E8E8", lw=0.5)


def synthesis_panel(ax, multidomain: pd.DataFrame, synthesis: pd.DataFrame, term: str, title: str) -> None:
    rows = []
    domains = [("Basic ADL (5 items)", "adl5", "Basic ADL"), ("Instrumental ADL (4 items)", "iadl4", "Instrumental ADL")]
    for domain, outcome, display in domains:
        for cohort in ["CHARLS", "HRS"]:
            row = multidomain[(multidomain["domain"].eq(domain)) & (multidomain["cohort"].eq(cohort)) & (multidomain["term"].eq(term))].iloc[0]
            rows.append((f"{display}: {cohort}", cohort, row.standardized_estimate, row.standardized_ci_low, row.standardized_ci_high))
        row = synthesis[(synthesis["outcome"].eq(outcome)) & (synthesis["term"].eq(term))].iloc[0]
        rows.append((f"{display}: pooled", "Fixed-effect pooled", row.fixed_effect_standardized_estimate, row.fixed_ci_low, row.fixed_ci_high))
    y = np.arange(len(rows))[::-1]
    for ypos, (label, cohort, estimate, low, high) in zip(y, rows):
        ax.errorbar(estimate, ypos, xerr=[[estimate - low], [high - estimate]], fmt=MARKERS[cohort], color=COLORS[cohort], ms=4, elinewidth=0.9, capsize=2)
    ax.axvline(0, color="#8C8C8C", lw=0.7)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlabel("Difference, baseline SD units")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E8E8E8", lw=0.5)


def figure1() -> None:
    data = pd.read_csv(OUT / "Table_2_multidomain_event_study.csv")
    synthesis = pd.read_csv(OUT / "Table_9_multidomain_two_cohort_synthesis.csv")
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.48, wspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    trajectory_panel(axes[0], data, "Basic ADL (5 items)", "Basic activities of daily living")
    trajectory_panel(axes[1], data, "Instrumental ADL (4 items)", "Instrumental activities of daily living")
    axes[0].legend(loc="upper left", ncol=2, handlelength=1.5)
    synthesis_panel(axes[2], data, synthesis, "post1", "One wave after hospitalization")
    synthesis_panel(axes[3], data, synthesis, "post2", "Two or more waves after hospitalization")
    for label, ax in zip("abcd", axes):
        panel_label(ax, label)
    fig.suptitle("Hospitalization-associated multidomain functional trajectories in CHARLS and HRS", x=0.02, ha="left", fontsize=10, fontweight="bold")
    save_pub(fig, "Figure_1_CHARLS_HRS_multidomain_replication")
    data.to_csv(OUT / "Source_Data_Figure_1_upgraded.csv", index=False, encoding="utf-8-sig")


def figure2() -> None:
    data = pd.read_csv(OUT / "Table_4_effect_modification_interactions.csv")
    data = data[data["term"].isin(["post1", "post2"])].copy()
    order = ["age_60_plus", "female", "baseline_adl_difficulty", "multimorbidity"]
    display = {
        "age_60_plus": "Age ≥60 years",
        "female": "Women vs men",
        "baseline_adl_difficulty": "Baseline ADL difficulty",
        "multimorbidity": "≥2 chronic conditions",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    offsets = {"CHARLS": 0.20, "HRS": 0.0, "Fixed-effect pooled": -0.20}
    for ax, term, title in zip(axes, ["post1", "post2"], ["One wave after hospitalization", "Two or more waves after hospitalization"]):
        for i, modifier in enumerate(order):
            ypos = len(order) - 1 - i
            for cohort in ["CHARLS", "HRS", "Fixed-effect pooled"]:
                row = data[(data["modifier"].eq(modifier)) & (data["cohort"].eq(cohort)) & (data["term"].eq(term))]
                if row.empty:
                    continue
                row = row.iloc[0]
                estimate = row.standardized_interaction_estimate
                low = row.standardized_ci_low
                high = row.standardized_ci_high
                ax.errorbar(estimate, ypos + offsets[cohort], xerr=[[estimate - low], [high - estimate]], fmt=MARKERS[cohort], color=COLORS[cohort], ms=4, elinewidth=0.9, capsize=2, label=cohort if i == 0 else None)
        ax.axvline(0, color="#8C8C8C", lw=0.7)
        ax.set_yticks(range(len(order)), [display[x] for x in order[::-1]])
        ax.set_xlabel("Interaction difference, baseline SD units")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E8E8E8", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.61, 0.01))
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle("Formal effect-modification tests for basic ADL trajectories", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.25, top=0.82, wspace=0.25)
    save_pub(fig, "Figure_2_CHARLS_HRS_effect_modification")
    data.to_csv(OUT / "Source_Data_Figure_2_upgraded.csv", index=False, encoding="utf-8-sig")


def figure_iadl_items() -> None:
    data = pd.read_csv(OUT / "Table_S2_IADL_item_analyses.csv")
    data = data[data["term"].isin(["post1", "post2"])].copy()
    order = ["money", "medications", "shopping", "meals"]
    display = {"money": "Managing money", "medications": "Taking medications", "shopping": "Shopping", "meals": "Preparing meals"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    offsets = {"CHARLS": 0.12, "HRS": -0.12}
    for ax, term, title in zip(axes, ["post1", "post2"], ["One wave after hospitalization", "Two or more waves after hospitalization"]):
        for i, item in enumerate(order):
            ypos = len(order) - 1 - i
            for cohort in ["CHARLS", "HRS"]:
                row = data[(data["item"].eq(item)) & (data["cohort"].eq(cohort)) & (data["term"].eq(term))].iloc[0]
                ax.errorbar(row.estimate, ypos + offsets[cohort], xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], fmt=MARKERS[cohort], color=COLORS[cohort], ms=4, elinewidth=0.9, capsize=2, label=cohort if i == 0 else None)
        ax.axvline(0, color="#8C8C8C", lw=0.7)
        ax.set_yticks(range(len(order)), [display[x] for x in order[::-1]])
        ax.set_xlabel("Risk difference")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E8E8E8", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.58, 0.01))
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle("Item-specific instrumental ADL trajectories", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.25, top=0.80, wspace=0.25)
    save_pub(fig, "Figure_S2_CHARLS_HRS_IADL_items")
    data.to_csv(OUT / "Source_Data_Figure_S2_IADL_items.csv", index=False, encoding="utf-8-sig")


def figure_robustness() -> None:
    data = pd.read_csv(OUT / "Table_3_sensitivity_analyses_upgraded.csv")
    labels = {
        "main": "Primary",
        "never_controls": "Never-hospitalized controls",
        "complete_panel": "Complete panel",
        "survey_weighted": "Survey weighted",
        "ipcw": "Observation weighted",
        "baseline_independent": "Baseline independent",
        "multiple_imputation_items": "Multiple item imputation",
    }
    selected = data[data["sensitivity"].isin(labels) & data["term"].isin(["post1", "post2"])].copy()
    selected["outcome"] = selected["outcome"].replace({"adl5_mi": "adl5", "iadl4_mi": "iadl4"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharey=True)
    for ax, (outcome, domain), term, title in zip(
        axes.ravel(),
        [("adl5", "Basic ADL"), ("adl5", "Basic ADL"), ("iadl4", "Instrumental ADL"), ("iadl4", "Instrumental ADL")],
        ["post1", "post2", "post1", "post2"],
        ["One wave after", "Two or more waves after", "One wave after", "Two or more waves after"],
    ):
        z = selected[(selected["outcome"].eq(outcome)) & (selected["term"].eq(term))]
        order = list(labels)
        offsets = {"CHARLS": 0.12, "HRS": -0.12}
        for i, analysis in enumerate(order):
            ypos = len(order) - 1 - i
            for cohort in ["CHARLS", "HRS"]:
                row = z[(z["sensitivity"].eq(analysis)) & (z["cohort"].eq(cohort))]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.errorbar(row.estimate, ypos + offsets[cohort], xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]], fmt=MARKERS[cohort], color=COLORS[cohort], ms=3.7, elinewidth=0.8, capsize=1.8, label=cohort if i == 0 else None)
        ax.axvline(0, color="#8C8C8C", lw=0.7)
        ax.set_yticks(range(len(order)), [labels[x] for x in order[::-1]])
        ax.set_xlabel("Difference in difficulty count")
        ax.set_title(f"{domain}: {title}", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E8E8E8", lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.62, 0.01))
    for label, ax in zip("abcd", axes.ravel()):
        panel_label(ax, label)
    fig.suptitle("Robustness of post-hospitalization functional trajectories", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.15, top=0.88, hspace=0.45, wspace=0.28)
    save_pub(fig, "Figure_S3_CHARLS_HRS_robustness")
    selected.to_csv(OUT / "Source_Data_Figure_S3.csv", index=False, encoding="utf-8-sig")


def figure_competing() -> None:
    data = pd.read_csv(OUT / "Table_S5_HRS_competing_outcomes.csv")
    groups = data[data["group"].isin(["First observed hospitalization", "Never hospitalized"])]
    differences = data[data["group"].eq("Hospitalization minus never-hospitalized")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, outcome, title in zip(axes, ["cif_adl", "cif_death"], ["Incident ADL difficulty", "Death"]):
        for group in ["First observed hospitalization", "Never hospitalized"]:
            z = groups[groups["group"].eq(group)]
            ax.plot(z["relative_wave"], 100 * z[outcome], marker="o", lw=1.2, color=COLORS[group], label=group)
        ax.set_xticks([1, 2], ["+1", "+2"])
        ax.set_xlabel("Waves after the 2014 index wave")
        ax.set_ylabel("Cumulative incidence (%)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E8E8E8", lw=0.5)
        last = differences[differences["relative_wave"].eq(2)].iloc[0]
        low = 100 * last[f"{outcome}_ci_low"]
        high = 100 * last[f"{outcome}_ci_high"]
        value = 100 * last[outcome]
        ax.text(0.04, 0.94, f"Difference at +2: {value:.1f}%\n95% CI {low:.1f} to {high:.1f}", transform=ax.transAxes, va="top", fontsize=6.5)
    axes[0].legend(loc="lower right", fontsize=6.3)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle("HRS competing outcomes after the 2014 index wave", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.78, wspace=0.30)
    save_pub(fig, "Figure_S4_HRS_competing_outcomes")
    data.to_csv(OUT / "Source_Data_Figure_S4.csv", index=False, encoding="utf-8-sig")


def figure_flow() -> None:
    flow = pd.read_csv(OUT / "cohort_flow.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0))
    for ax, cohort in zip(axes, ["CHARLS", "HRS"]):
        z = flow[flow["cohort"].eq(cohort)].copy()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(cohort, fontsize=9, fontweight="bold", pad=8)
        baseline = z.iloc[0]
        baseline_box = FancyBboxPatch((0.08, 0.81), 0.84, 0.12, boxstyle="round,pad=0.012,rounding_size=0.012", edgecolor=COLORS[cohort], facecolor="#F7FAFC", linewidth=1.0)
        ax.add_patch(baseline_box)
        ax.text(0.5, 0.87, f"Eligible baseline participants aged ≥50\nwithout interval hospitalization\nn = {int(baseline['n']):,}", ha="center", va="center", fontsize=6.6)
        ax.text(0.5, 0.775, "Mutually exclusive timing groups", ha="center", va="center", fontsize=6.2, color="#666666")
        categories = z.iloc[1:].reset_index(drop=True)
        positions = [(0.04, 0.59), (0.54, 0.59), (0.04, 0.31), (0.54, 0.31), (0.29, 0.07)]
        for i, (_, row) in enumerate(categories.iterrows()):
            x, y = positions[i]
            stage = row["stage"].replace("First-observed hospitalization in ", "First observed hospitalization\n")
            stage = stage.replace("No hospitalization observed through ", "No hospitalization observed\nthrough ")
            box = FancyBboxPatch((x, y), 0.42, 0.14, boxstyle="round,pad=0.010,rounding_size=0.010", edgecolor=COLORS[cohort], facecolor="#F7FAFC", linewidth=0.85)
            ax.add_patch(box)
            ax.text(x + 0.21, y + 0.07, f"{stage}\nn = {int(row['n']):,}", ha="center", va="center", fontsize=6.1)
        panel_label(ax, "a" if cohort == "CHARLS" else "b")
    fig.suptitle("Cohort construction and timing of first observed hospitalization", x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.88, wspace=0.20)
    save_pub(fig, "Figure_S5_CHARLS_HRS_cohort_flow")
    flow.to_csv(OUT / "Source_Data_Figure_S5.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    figure1()
    figure2()
    figure_iadl_items()
    figure_robustness()
    figure_competing()
    figure_flow()
    print("Created six upgraded figure bundles in SVG, PDF, PNG, and 600-dpi TIFF.")


if __name__ == "__main__":
    main()
