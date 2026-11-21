"""Figures for the ISI 47011 revision.

Drawn at their printed size (ISI template: column 247.5 pt, text width
520.6 pt) so that every label prints at Times New Roman 10 pt, as the editor
requested.  Each figure is written as PNG (600 dpi, placed in the Word file)
plus SVG with live text and PDF, the editable versions sent to the journal.

Usage:  python src/analysis/make_isi_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision" / "figures"
COL_IN = 247.5 / 72
FULL_IN = 520.6 / 72
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# Reference categorical slots 1-3 (validated: CVD dE 9.2, normal-vision dE 27.6);
# marker shapes add a second encoding for grayscale print.
SERIES = {"PSO": ("#2a78d6", "o"), "GWO": ("#eb6834", "s"), "DE": ("#1baf7a", "^")}
VARIANTS = ["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
NL = "\n"

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "mathtext.fontset": "stix",
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def fig1_pipeline():
    steps = [
        ["Phase 1: fixed benchmark of seven", "GGUF variants and Pareto screening"],
        ["Phase 2: per-variant search", "over (n$_t$, n$_b$, n$_p$, k) by PSO,", "GWO, DE, random and grid"],
        ["Verification: repeated re-measurement", "of each reported optimum"],
        ["Phase 3: cross-variant comparison on a", "common scale; selected configuration"],
    ]
    line_h, pad, gap = 4.6, 3.4, 4.2
    heights = [pad + line_h * len(t) for t in steps]
    repo_h = pad + 2 * line_h
    total = sum(heights) + gap * len(steps) + repo_h + 2
    fig = plt.figure(figsize=(COL_IN, COL_IN * total / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")
    x0, w = 2.5, 69
    top = total - 1
    centers = []
    for i, (lines, h) in enumerate(zip(steps, heights)):
        y0 = top - h
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="#f2f1ed", ec=INK2, lw=0.7))
        ax.text(x0 + w / 2, y0 + h / 2, NL.join(lines), ha="center", va="center", linespacing=1.15)
        centers.append(y0 + h / 2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y0 - 0.3), (x0 + w / 2, y0 - gap + 0.3),
                                         arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK2))
        top = y0 - gap
    # Hardware measurement loop of Phase 2.
    pc = centers[1]
    bx = x0 + w + 4.5
    bw = 100 - bx - 1
    ax.add_patch(FancyBboxPatch((bx, pc - 8.5), bw, 17, boxstyle="round,pad=0,rounding_size=1.5",
                                fc="white", ec=INK2, lw=0.7))
    ax.text(bx + bw / 2, pc, NL.join(["Raspberry", "Pi 5:", "L, Q, M, E"]), ha="center", va="center",
            linespacing=1.1)
    ax.add_patch(FancyArrowPatch((x0 + w + 0.3, pc + 2.5), (bx - 0.3, pc + 2.5), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=INK2))
    ax.add_patch(FancyArrowPatch((bx - 0.3, pc - 2.5), (x0 + w + 0.3, pc - 2.5), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=INK2))
    # Configuration repository (records written by every stage).
    ax.add_patch(FancyBboxPatch((x0, 1), 99 - x0, repo_h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc="white", ec=INK2, lw=0.7, ls=(0, (3, 2))))
    ax.text((x0 + 99) / 2, 1 + repo_h / 2,
            NL.join(["Configuration repository: configurations, raw metrics,",
                     "ranges, seeds, verification evidence"]),
            ha="center", va="center", linespacing=1.15)
    save(fig, "fig1_pipeline")


def fig2_convergence():
    fig, axes = plt.subplots(2, 4, figsize=(FULL_IN, 3.55))
    for ax, v in zip(axes.flat, VARIANTS):
        rows = list(csv.DictReader(open(ROOT / "results" / "phase2_runs" / v / "convergence.csv")))
        it = [int(r["iteration"]) for r in rows]
        for alg, (col, mk) in SERIES.items():
            ax.plot(it, [float(r[alg]) for r in rows], color=col, lw=1.5, marker=mk, ms=3.5,
                    markeredgecolor="white", markeredgewidth=0.4, label=alg)
        ax.set_title(v, pad=3)
        ax.set_xticks(it if len(it) <= 5 else it[::2])
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2, pad=2)
        ax.locator_params(axis="y", nbins=3)
    leg_ax = axes.flat[-1]
    leg_ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    leg_ax.legend(h, lab, loc="center", frameon=False, title="Median of two runs", handlelength=2.2)
    fig.supxlabel("Iteration", y=0.02)
    fig.supylabel("Best cost so far", x=0.004)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.13, wspace=0.45, hspace=0.62)
    save(fig, "fig2_convergence")


if __name__ == "__main__":
    fig1_pipeline()
    fig2_convergence()
    print("figures written to", OUT)
