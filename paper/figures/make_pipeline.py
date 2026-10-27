"""Draw the pipeline schematic of ``paper/paper.md`` (``Software design``).

Run from anywhere with the package installed::

    python paper/figures/make_pipeline.py

It writes ``pipeline.pdf`` and ``pipeline.png`` (300 dpi) next to this script. The PNG is what
``paper.md`` includes; JOSS' toolchain renders both. Every box below names a code path that
exists on ``dev``; if one is renamed, the entry here is the one to change. The boxes are not
generated from the code, so nothing but this docstring and review keeps them in step.

The figure is drawn at 16 cm, the width ``paper.md`` gives it, with 8 pt text, so what is
checked here is what the reader sees.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from erotica import viz

HERE = Path(__file__).resolve().parent
WIDTH_IN = 16.0 / 2.54  # the width of the included figure
HEIGHT_IN = 4.16
FONT_PT = 8.0

BLUE, GREEN, ORANGE, PINK = viz.OKABE_ITO  # stages, outputs, calibration, validation
INK = "#222222"


def tint(hex_color: str, amount: float = 0.82) -> str:
    """Blend a color toward white; the box fill, so the text stays dark on it."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(*(round(c + (255 - c) * amount) for c in (r, g, b)))


def box(ax, x, y, w, h, title, body, color, *, dashed=False, title_dy=0.0):
    """A rounded box with a bold title line on top and a body below it; returns its geometry."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=tint(color),
            edgecolor=color,
            linewidth=1.3,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.09 + title_dy,
        title,
        ha="center",
        va="top",
        fontsize=FONT_PT + 0.5,
        fontweight="bold",
        color=INK,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 0.33 + title_dy,
            body,
            ha="center",
            va="top",
            fontsize=FONT_PT,
            color=INK,
            linespacing=1.25,
        )
    return x, y, w, h


def arrow(ax, start, end, *, style="-|>", color=INK, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            linewidth=1.2,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def build():
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    ax.set_xlim(0, WIDTH_IN)
    ax.set_ylim(0, HEIGHT_IN)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    m = 0.03  # outer margin
    gap = 0.27
    col_w = (WIDTH_IN - 2 * m - 3 * gap) / 4
    col_x = [m + i * (col_w + gap) for i in range(4)]

    # Row 1: the four stages.
    row1_h, row1_y = 1.3, HEIGHT_IN - 1.3 - m
    stages = [
        ("Gaia catalog", "DataLoader,\nDataPreprocessor\n(zero point, cuts)"),
        (
            "Membership",
            "HDBSCAN pseudo-\nprobability sweep\n(Clustering), Bayesian\nPM and parallax\nrefinement",
        ),
        (
            "Structure",
            "King, EFF and corona\nprofiles, unbinned\npoint process\n(erotica.analysis.\nstructure)",
        ),
        (
            "Isochrone",
            "IsochroneFitterKW:\nunbinned per-star\nlikelihood on MIST\nisochrones, sampled\nwith nutpie",
        ),
    ]
    for x, (title, body) in zip(col_x, stages, strict=True):
        box(ax, x, row1_y, col_w, row1_h, title, body, BLUE)
    mid1 = row1_y + row1_h / 2
    for i in range(3):
        arrow(ax, (col_x[i] + col_w, mid1), (col_x[i + 1], mid1))

    # Row 2: calibration check (left) and outputs (right).
    row2_h, row2_y = 1.1, row1_y - 0.4 - 1.1
    cal_x, cal_w = col_x[0], col_x[1] + 0.9 * col_w - col_x[0]
    box(
        ax,
        cal_x,
        row2_y,
        cal_w,
        row2_h,
        "Calibration check (erotica.calibration)",
        "reliability diagram, Hosmer\u2013Lemeshow,\n"
        "Brier score, expected and maximum\n"
        "calibration error; isotonic and\n"
        "Platt recalibration",
        ORANGE,
    )
    out_x = col_x[2]
    out_w = col_x[3] + col_w - out_x
    box(
        ax,
        out_x,
        row2_y,
        out_w,
        row2_h,
        "Outputs",
        "ArviZ InferenceData for every fit;\n"
        "provenance sidecar: git commit, seeds,\n"
        "dependency versions, input checksums",
        GREEN,
    )
    # Membership probabilities go to the calibration check; the fits go to the outputs.
    arrow(
        ax,
        (col_x[1] + 0.35 * col_w, row1_y),
        (col_x[1] + 0.35 * col_w, row2_y + row2_h),
        color=ORANGE,
    )
    for c in (col_x[2], col_x[3]):
        arrow(ax, (c + col_w / 2, row1_y), (c + col_w / 2, row2_y + row2_h), color=GREEN)
    arrow(
        ax,
        (col_x[1] + 0.85 * col_w, row1_y),
        (out_x, row2_y + 0.6 * row2_h),
        color=GREEN,
        rad=0.0,
    )

    # Row 3: the validation layer, as a dashed band that exercises everything above it.
    band_h, band_y = 1.0, m
    box(
        ax,
        m,
        band_y,
        WIDTH_IN - 2 * m,
        band_h,
        "Validation layer (exercises every stage above)",
        "",
        PINK,
        dashed=True,
    )
    inner_w = (WIDTH_IN - 2 * m - 4 * 0.12) / 3
    items = [
        "Synthetic injection\nand recovery\n(tools/validation)",
        "Mutation audit of\nthe test suite",
        "Slow tests: NUTS fit\nagainst a committed run",
    ]
    for i, text in enumerate(items):
        x = m + 0.12 + i * (inner_w + 0.12)
        ax.add_patch(
            FancyBboxPatch(
                (x, band_y + 0.1),
                inner_w,
                band_h - 0.46,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor="white",
                edgecolor=PINK,
                linewidth=1.0,
            )
        )
        ax.text(
            x + inner_w / 2,
            band_y + 0.1 + (band_h - 0.46) / 2,
            text,
            ha="center",
            va="center",
            fontsize=FONT_PT,
            color=INK,
            linespacing=1.2,
        )
    return fig, ax


def main() -> None:
    with viz.aanda_style({"font.size": FONT_PT, "font.family": "sans-serif"}):
        fig, ax = build()
        n_problems = len(viz.check_layout(fig))
        pdf, preview = viz.save_figure(fig, HERE / "pipeline.pdf")
    os.replace(preview, HERE / "pipeline.png")
    print(f"wrote {pdf.name} and pipeline.png; check_layout problems: {n_problems}")


if __name__ == "__main__":
    main()
