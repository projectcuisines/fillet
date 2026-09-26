"""Plot the Experiment 1 and Experiment 2 climate-state maps of the three models in the instellation-obliquity plane."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK_MUTED,  # noqa: E402
                             fvApplyStyle, fvTidy, fvSaveFigure)

import matplotlib as mpl   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402

# Climate state is an ORDERED, four-level quantity (ice-free -> snowball), so
# it takes a single-hue sequential ramp rather than four categorical hues.
STATE_ORDER = ["ice_free", "ice_cap", "ice_belt", "snowball"]
STATE_LABEL = ["Ice free", "Ice caps", "Ice belt", "Snowball"]
# Sequential: one hue, light to dark, blended from vplot's dark blue.
STATE_RAMP = ["#e3e4fa", "#a5abf0", "#5f68e5", "#1321d8"]


def faGrid(dictM):
    aInst = np.array(dictM["aInst"])
    aObl = np.array(dictM["aObl"])
    aState = np.array([STATE_ORDER.index(s) if s in STATE_ORDER else np.nan
                       for s in dictM["aStates"]], dtype=float)
    aUI = np.unique(np.round(aInst, 4))
    aUO = np.unique(np.round(aObl, 2))
    aG = np.full((len(aUI), len(aUO)), np.nan)
    for fI, fO, fS in zip(aInst, aObl, aState):
        i = int(np.argmin(np.abs(aUI - fI)))
        j = int(np.argmin(np.abs(aUO - fO)))
        aG[i, j] = fS
    return aUI, aUO, aG


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--states-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    d = json.load(open(oArgs.states_json))
    fvApplyStyle()
    oCmap = mpl.colors.ListedColormap(STATE_RAMP)
    oNorm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], oCmap.N)

    listModels = ["avalon", "hextor", "poise", "ops"]
    oFig, aAx = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True)

    for iRow, sExp in enumerate(("exp1", "exp2")):
        for iCol, sM in enumerate(listModels):
            oAx = aAx[iRow][iCol]
            if sM not in d[sExp]:
                oAx.set_visible(False)
                continue
            aUI, aUO, aG = faGrid(d[sExp][sM])
            oAx.pcolormesh(aUO, aUI, aG, cmap=oCmap, norm=oNorm,
                           shading="nearest")
            c = d[sExp][sM]["dictStateCounts"]
            oAx.set_title(LABEL[sM], fontsize=8.5, pad=11)
            oAx.text(0.5, 1.012,
                     "snowball in %d of %d cases" % (c["snowball"], sum(c.values())),
                     transform=oAx.transAxes, ha="center", va="bottom",
                     fontsize=6.4, color=INK_MUTED)
            # OPS's Experiment 2 extends below the protocol's instellation
            # floor; the branch-order violations are confined to those extra
            # cases, so the grid extent is the thing worth marking, not the
            # labelling, which its own source tree confirms is correct.
            if sM == "ops" and sExp == "exp2":
                oAx.text(0.03, 0.97,
                         "grid extends below the\nprotocol's $1.05\\,S_\\oplus$ floor",
                         transform=oAx.transAxes, ha="left", va="top",
                         fontsize=6.0, color=COLOR["shields_bitz"])
            oAx.set_xticks([0, 30, 60, 90])
            oAx.grid(False)
            if iCol == 0:
                oAx.set_ylabel(r"Instellation ($S_\oplus$)")
            if iRow == 1:
                oAx.set_xlabel(r"Obliquity ($^\circ$)")
            fvTidy(oAx)

    aAx[0][0].text(-0.72, 0.5, "A   Experiment 1\n(warm start)",
                   transform=aAx[0][0].transAxes, rotation=90, va="center",
                   ha="center", fontsize=8.5, fontweight="bold")
    aAx[1][0].text(-0.72, 0.5, "B   Experiment 2\n(cold start)",
                   transform=aAx[1][0].transAxes, rotation=90, va="center",
                   ha="center", fontsize=8.5, fontweight="bold")

    oFig.subplots_adjust(left=0.20, right=0.88, top=0.90, bottom=0.11,
                         hspace=0.46, wspace=0.30)
    oCax = oFig.add_axes([0.90, 0.20, 0.018, 0.55])
    oCb = oFig.colorbar(mpl.cm.ScalarMappable(norm=oNorm, cmap=oCmap), cax=oCax)
    oCb.set_ticks([0, 1, 2, 3])
    oCb.set_ticklabels(STATE_LABEL)
    oCb.ax.tick_params(labelsize=7.0, length=0)
    oCb.outline.set_visible(False)

    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
