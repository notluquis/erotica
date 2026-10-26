"""Tests for erotica.viz: A&A widths against aa.cls, the style context, and the layout check.

Oracles:
* widths — the 184 mm text width and 4 mm column gap that ``aa.cls`` passes to ``geometry``, and,
  independently, the ``\\textwidth=523.5307pt`` that LaTeX reports in the log of an A&A build;
* layout — each check is exercised by a figure built to violate exactly that rule, next to a clean
  figure that must pass (a check that cannot fail looks the same as one that passes).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from erotica import viz  # noqa: E402

PT_PER_IN = 72.27  # TeX points


def test_widths_match_aa_cls_and_the_latex_log():
    assert viz.AANDA_TEXT_IN == pytest.approx(184 / 25.4, rel=1e-12)
    assert viz.AANDA_COLUMN_IN == pytest.approx(90 / 25.4, rel=1e-12)
    # \textwidth as LaTeX reports it for an aa.cls document, and the column it implies.
    assert viz.AANDA_TEXT_IN * PT_PER_IN == pytest.approx(523.5307, abs=1e-3)
    assert viz.AANDA_COLUMN_IN * PT_PER_IN == pytest.approx(256.0748, abs=1e-3)


def test_figsize():
    assert viz.figsize("column", 0.5) == pytest.approx(
        (viz.AANDA_COLUMN_IN, viz.AANDA_COLUMN_IN / 2)
    )
    assert viz.figsize("text") == pytest.approx((viz.AANDA_TEXT_IN, 0.75 * viz.AANDA_TEXT_IN))
    assert viz.figsize(2.0, 1.0) == (2.0, 2.0)
    with pytest.raises(ValueError):
        viz.figsize("page")
    with pytest.raises(ValueError):
        viz.figsize("column", 0)


def test_style_is_scoped_and_nothing_defaults_below_seven_points():
    before = mpl.rcParams["font.size"]
    with viz.aanda_style({"font.size": 10}):
        assert mpl.rcParams["font.size"] == 10
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert [c["color"] for c in mpl.rcParams["axes.prop_cycle"]] == list(viz.OKABE_ITO)
    assert mpl.rcParams["font.size"] == before
    sizes = [
        v for k, v in viz.AANDA_RC.items() if k.endswith("size") and isinstance(v, (int, float))
    ]
    assert sizes and min(sizes) >= 7


def _base(legend_loc="upper right", legend_size=7.5, left=None):
    fig, ax = plt.subplots(figsize=viz.figsize("column", 0.85))
    ax.scatter([0.04, 0.5, 0.9], [0.03, 0.5, 0.9], label="data")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("marginal fraction beyond the limit (%)")
    ax.legend(loc=legend_loc, fontsize=legend_size, borderaxespad=0.0)
    if left is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=left)
    return fig


def test_clean_figure_passes():
    fig = _base()
    assert viz.check_layout(fig) == []
    plt.close(fig)


def test_legend_over_a_marker_is_flagged():
    fig = _base(legend_loc="lower left")  # sits on the point at (0.04, 0.03)
    problems = viz.check_layout(fig)
    assert any(p.startswith("OVERLAP: legend") for p in problems), problems
    plt.close(fig)


def test_clipped_axis_label_is_flagged():
    fig = _base(left=0.02)
    problems = viz.check_layout(fig)
    assert any(p.startswith("CLIPPED") for p in problems), problems
    plt.close(fig)


def test_small_text_is_flagged_and_threshold_is_configurable():
    fig = _base(legend_size=6.5)
    assert any(p.startswith("FONTSIZE") for p in viz.check_layout(fig))
    assert viz.check_layout(fig, min_fontsize=6.0) == []
    plt.close(fig)


def test_text_on_a_dense_cloud_is_not_an_overlap_but_on_a_point_is():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=viz.figsize("column"))
    ax.scatter(rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000), s=1)  # posterior-like cloud
    ax.text(0.5, 0.5, "cloud label", ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    assert viz.check_layout(fig) == []
    ax.plot([0.5], [0.5], "o")  # a single datum under the same label
    problems = viz.check_layout(fig)
    assert any(p.startswith("OVERLAP: text 'cloud label'") for p in problems), problems
    assert viz.check_layout(fig, allow=("cloud label",)) == []
    plt.close(fig)
