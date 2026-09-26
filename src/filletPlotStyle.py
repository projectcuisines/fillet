"""Shared matplotlib styling for the FILLET code-comparison figures.

Colour comes from `vplot`, the VPLanet plotting package, so that these figures
match the palette the consortium's own tooling already uses. Its five hues are
exactly the number of participating codes, and scored on the all-pairs
pairlist by `explorations/validateFigurePalette.py` they clear both separation
floors comfortably: worst-pair colour-vision-deficiency separation 10.5 and
worst-pair normal-vision separation 16.7, against floors of 6 and 15. That is
better on both counts than the reference categorical palette this report used
previously, which reached 9.2 and 16.3 and needed an achromatic fifth mark
because no chromatic slot was left.

Contrast against a white surface is 2.50:1, below the 3:1 guideline, so every
series is additionally named in a legend and in the accompanying tables;
identity is never carried by colour alone.

Branch, version and convention distinctions are carried by line style and
marker, never by an extra hue, so that a model keeps one colour everywhere.
Grids are off: with at most five series and direct labelling, the gridlines
were decoration rather than aid.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# vplot's five hues, one per participating code.
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
from vplot import colors as _oVplot   # noqa: E402
_oMplFigure.Figure = _oFigureClass
plt.figure = _fnFigure

COLOR = {
    "avalon": _oVplot.dark_blue,
    "hextor": _oVplot.orange,
    "poise": _oVplot.pale_blue,
    "ops": _oVplot.purple,
    "shields_bitz": _oVplot.red,
}
LABEL = {
    "avalon": "AVALON",
    "hextor": "HEXTOR",
    "poise": "VPLanet-POISE",
    "ops": "OPS",
    "shields_bitz": "Shields-Bitz",
}
# Text tokens; series colour is never used for text.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#ffffff"

# Line styles for within-model distinctions.
STYLE_WARM = "-"
STYLE_COLD = "--"
STYLE_ARCHIVE = "-"
STYLE_CURRENT = ":"


def fvApplyStyle():
    """Install the figure defaults: recessive axes, thin marks, no chart junk."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": "#d6d5d0",
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": "#ebeae5",
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "lines.markersize": 3.2,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def fvLegendBelow(oAx, iCols=1, fPad=-0.17, fSize=7.2):
    """Put a panel's key under its own axes, out of the data's way."""
    oAx.legend(loc="upper center", bbox_to_anchor=(0.5, fPad), ncol=iCols,
               fontsize=fSize, handlelength=2.2, borderaxespad=0.0,
               columnspacing=1.3)


def fvTidy(oAx, bTopRight=True):
    """Drop the top and right spines so the data, not the frame, carries the eye."""
    oAx.grid(False)
    if bTopRight:
        oAx.spines["top"].set_visible(False)
        oAx.spines["right"].set_visible(False)


def fvLatitudeAxis(oAx):
    oAx.set_xlim(-90, 90)
    oAx.set_xticks([-90, -60, -30, 0, 30, 60, 90])


def fvSaveFigure(oFig, sPath):
    """Write the figure as the named vector file and as a PNG beside it.

    The report embeds the PDF; the PNG is what gets pasted into a message or
    a slide, and generating it here rather than by a separate conversion keeps
    the two from drifting apart.
    """
    oFig.savefig(sPath)
    if sPath.lower().endswith(".pdf"):
        oFig.savefig(sPath[:-4] + ".png", dpi=200)
