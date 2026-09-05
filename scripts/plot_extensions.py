from __future__ import annotations

from config import OUTPUT_DIR as OUT

import matplotlib as mpl
import matplotlib.pyplot as plt
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

COLORS = {"CHARLS": "#2F6B9A", "HRS": "#D9822B"}
MARKERS = {"CHARLS": "o", "HRS": "s"}


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.07, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def save_pub(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def age_dose_response() -> None:
    data = pd.read_csv(OUT / "Table_S11_ADL_age_spline_predictions.csv")
    linear = pd.read_csv(OUT / "Table_S10_ADL_age_linear_interactions.csv")
    data = data[data["term"].isin(["post1", "post2"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    for ax, term, title in zip(
        axes,
        ["post1", "post2"],
        ["One wave after hospitalization", "Two or more waves after hospitalization"],
    ):
        for cohort in ["CHARLS", "HRS"]:
            z = data[(data["term"].eq(term)) & (data["cohort"].eq(cohort))].sort_values("age_years")
            ax.plot(
                z["age_years"], z["standardized_estimate"],
                color=COLORS[cohort], lw=1.3,
                label=cohort,
            )
            ax.fill_between(
                z["age_years"].to_numpy(float),
                z["standardized_ci_low"].to_numpy(float),
                z["standardized_ci_high"].to_numpy(float),
                color=COLORS[cohort], alpha=0.13, linewidth=0,
            )
            marked = z[z["age_years"].isin([55, 65, 75, 85])]
            ax.plot(
                marked["age_years"], marked["standardized_estimate"],
                linestyle="none", marker=MARKERS[cohort], color=COLORS[cohort], ms=4,
            )
            slope = linear[(linear["cohort"].eq(cohort)) & (linear["term"].eq(term))].iloc[0]
            ax.text(
                0.03,
                0.96 if cohort == "CHARLS" else 0.86,
                f"{cohort}: +{slope.standardized_age_interaction_per_10y:.2f} SD per 10 y",
                color=COLORS[cohort], transform=ax.transAxes, va="top", fontsize=6.2,
            )
        ax.axhline(0, color="#8C8C8C", lw=0.7)
        ax.set_xticks([55, 65, 75, 85])
        ax.set_xlim(50, 90)
        ax.set_xlabel("Baseline age, years")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#E8E8E8", lw=0.5)
    axes[0].set_ylabel("ADL difference, baseline SD units")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.01))
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle(
        "Age gradients in post-hospitalization basic functional change",
        x=0.02, ha="left", fontsize=10, fontweight="bold",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.80, wspace=0.18)
    save_pub(fig, "Figure_2_CHARLS_HRS_age_dose_response")


def iadl_context_gaps() -> None:
    data = pd.read_csv(OUT / "Table_S14_IADL_context_cross_cohort_differences.csv")
    data = data[data["term"].isin(["post1", "post2"])].copy()
    labels = {
        ("high_school_plus", 0): "Lower education",
        ("high_school_plus", 1): "Higher education",
        ("married_partnered", 0): "Not partnered",
        ("married_partnered", 1): "Married/partnered",
        ("living_alone", 0): "Not living alone",
        ("living_alone", 1): "Living alone",
    }
    order = list(labels)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    for ax, term, title in zip(
        axes,
        ["post1", "post2"],
        ["One wave after hospitalization", "Two or more waves after hospitalization"],
    ):
        rows = []
        for modifier, value in order:
            row = data[
                data["modifier"].eq(modifier)
                & data["modifier_value"].eq(value)
                & data["term"].eq(term)
            ].iloc[0]
            rows.append(row)
        y = np.arange(len(rows))[::-1]
        for ypos, row, key in zip(y, rows, order):
            estimate, low, high = (
                row.HRS_minus_CHARLS_standardized,
                row.ci_low,
                row.ci_high,
            )
            significant = row.q_difference_bh < 0.05
            color = "#9B3A4A" if significant else "#5E6C76"
            ax.errorbar(
                estimate, ypos,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o", color=color, ms=4, elinewidth=0.9, capsize=2,
            )
        ax.axvline(0, color="#8C8C8C", lw=0.7)
        ax.set_yticks(y, [labels[key] for key in order])
        ax.set_xlabel("HRS minus CHARLS IADL difference, baseline SD units")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E8E8E8", lw=0.5)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.suptitle(
        "Exploratory social-context stratification of cross-cohort IADL differences",
        x=0.02, ha="left", fontsize=10, fontweight="bold",
    )
    fig.text(
        0.62, 0.03,
        "Red: BH-adjusted q < 0.05 across 12 post-event stratum comparisons",
        ha="center", fontsize=6.2, color="#6A6A6A",
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.80, wspace=0.20)
    save_pub(fig, "Figure_S7_CHARLS_HRS_IADL_context")


if __name__ == "__main__":
    age_dose_response()
    iadl_context_gaps()
