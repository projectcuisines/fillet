"""Shared styling for the presentation versions of the report's figures: vplot's colour scheme, wide 16:9 proportions, no grids, legends below the panels.

These are deliberately separate objects from the figures in the report, not
replacements for them.  A figure read at arm's length on paper and a figure
read from the back of a room want different things: the report's versions
carry every qualification in the legend text, which is right there and wrong
on a slide.  Here the labels are cut to the identity of the series and the
qualifications move to what the speaker says.

Colour comes from `vplot`, the VPLanet plotting package, so that these figures
match the palette the consortium's own tooling uses.  Its five hues happen to
be exactly the number of codes, and scored with the same all-pairs validator
used for the report's palette (`explorations/validateFigurePalette.py`) they
clear both separation floors comfortably: worst-pair colour-vision-deficiency
separation 10.5 and worst-pair normal-vision separation 16.7, against floors
of 6 and 15.  Contrast against white is 2.50:1, below the 3:1 guideline, which
is why every series is also named in a legend rather than identified by colour
alone.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# vplot is imported for its colour constants only.  Importing it also
# REPLACES `matplotlib.figure.Figure` and `plt.figure` with a subclass that
# forces `tight_layout()` on every draw (see vplot/figure.py, "HACK: Override
# Figure"), which silently discards the explicit `subplots_adjust` and
# `add_axes` placements these figures depend on.  The originals are captured
# first and put back immediately, so we take the palette and not the layout
# engine.
import matplotlib.figure as _oMplFigure
_oFigureClass = _oMplFigure.Figure
_fnFigure = plt.figure
from vplot import colors as oVplotColors   # noqa: E402
_oMplFigure.Figure = _oFigureClass
plt.figure = _fnFigure


# vplot's five hues, one per participating code.  The order preserves the
# report's own assignment as closely as vplot's palette allows.
COLOR = {
    "avalon": oVplotColors.dark_blue,
    "hextor": oVplotColors.orange,
    "poise": oVplotColors.pale_blue,
    "ops": oVplotColors.purple,
    "shields_bitz": oVplotColors.red,
}
LABEL = {
    "avalon": "AVALON",
    "hextor": "HEXTOR",
    "poise": "VPLanet",
    "ops": "OPS",
    "shields_bitz": "Shields-Bitz",
}

INK = "#101010"
INK_MUTED = "#6b6b6b"
SURFACE = "#ffffff"

# A 16:9 slide is 13.33 x 7.5 inches.  A two-panel figure that leaves room for
# a title sits at about this size.
SLIDE_WIDE = (12.0, 5.6)


def fvApplyStyle():
    """vplot's own matplotlib style, with grids off and marks made for a screen."""
    sStyle = os.path.join(os.path.dirname(oVplotColors.__file__),
                          "style", "vplot.mplstyle")
    if os.path.exists(sStyle):
        plt.style.use(sStyle)
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "legend.frameon": False,
        "lines.linewidth": 2.4,
        "pdf.fonttype": 42,
    })


def fvTidy(oAx):
    """Recessive axes, no grid, ticks pointing out."""
    oAx.grid(False)
    oAx.tick_params(direction="out", length=4, width=0.9)
    for sSide in ("left", "bottom"):
        oAx.spines[sSide].set_linewidth(0.9)


def fvLegendBelow(oAx, iCols=1, fPad=-0.16):
    """The report's Figure 2 format: each panel's key sits under its own axes."""
    oAx.legend(loc="upper center", bbox_to_anchor=(0.5, fPad), ncol=iCols,
               handlelength=2.2, borderaxespad=0.0, columnspacing=1.4)


def fvSaveSlideFigure(oFig, sPath):
    """Write the vector file and a PNG beside it, both sized for a slide."""
    oFig.savefig(sPath)
    if sPath.lower().endswith(".pdf"):
        oFig.savefig(sPath[:-4] + ".png", dpi=220)
