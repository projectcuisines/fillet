"""Plot the Experiment 3 and Experiment 4 hysteresis loops of the five models, marking where a model fails to reach an end state inside the protocol range."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK_MUTED, INK_SECONDARY,   # noqa: E402
                             STYLE_WARM, STYLE_COLD, fvApplyStyle, fvTidy,
                             fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--bifurcation-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    d = json.load(open(oArgs.bifurcation_json))
    fvApplyStyle()
    listModels = ["avalon", "hextor", "poise", "ops", "shields_bitz"]

    oFig, aAx = plt.subplots(2, 5, figsize=(8.6, 4.5))

    for iCol, sM in enumerate(listModels):
        # --- Experiment 3 --------------------------------------------------
        oAx = aAx[0][iCol]
        dM = d["dictExp3"].get(sM, {})
        for sB, sStyle, sLab in (("warm", STYLE_WARM, "warm start"),
                                 ("cold", STYLE_COLD, "cold start")):
            if sB not in dM:
                continue
            oAx.plot(dM[sB]["aX"], dM[sB]["aTglob"], color=COLOR[sM],
                     linestyle=sStyle, marker="o", markersize=2.4,
                     label=sLab if iCol == 0 else None)
        oAx.set_title(LABEL[sM], fontsize=8.5, pad=11)
        # A width measured where the two branches agree almost everywhere is
        # the grid spacing, not the physics, so it is not shown.
        dDist = dM.get("dictBranchDistinctness") or {}
        bSameRun = (dDist.get("bComparable")
                    and dDist["iIdenticalCases"] >= 0.9 * dDist["iCases"])
        if bSameRun:
            oAx.text(0.5, 1.012,
                     "branches agree in %d of %d cases"
                     % (dDist["iIdenticalCases"], dDist["iCases"]),
                     transform=oAx.transAxes, ha="center", va="bottom",
                     fontsize=6.4, color=COLOR["shields_bitz"])
        elif "warm" in dM and "cold" not in dM:
            oAx.text(0.5, 1.012, "warm branch only", transform=oAx.transAxes,
                     ha="center", va="bottom", fontsize=6.4, color=INK_MUTED)
        elif "dictHysteresis" in dM:
            h = dM["dictHysteresis"]
            sTxt = ("bistable width %s" % ("%.3f $S_\\oplus$" % h["fBistableWidthInst"]
                                           if h["fBistableWidthInst"] else "n/a"))
            oAx.text(0.5, 1.012, sTxt, transform=oAx.transAxes, ha="center",
                     va="bottom", fontsize=6.4, color=INK_MUTED)
            if h["fGlaciationInst"]:
                oAx.axvspan(h["fGlaciationInst"], h["fDeglaciationInst"],
                            color="#f2f1ec", zorder=0)
        if not bSameRun and not dM.get("dictBranchGrid", {}).get("bSameGrid", True):
            oAx.text(0.04, 0.93, "branches on\ndifferent grids",
                     transform=oAx.transAxes, fontsize=6.4, color=COLOR["shields_bitz"],
                     va="top")
        oAx.set(xlabel=r"Instellation ($S_\oplus$)", xlim=(0.76, 1.52),
                ylim=(200, 375))
        if iCol == 0:
            oAx.set_ylabel(r"$T_{\rm glob}$ (K)")
            # Two line-style entries only, so the key stays inside the panel:
            # moving it below costs a row of height and buys nothing.
            oAx.legend(loc="upper left", fontsize=7.0, handlelength=3.2)

        fvTidy(oAx)

        # --- Experiment 4 --------------------------------------------------
        oAx = aAx[1][iCol]
        dM = d["dictExp4"].get(sM, {})
        for sB, sStyle, sLab in (("warm", STYLE_WARM, "warm start"),
                                 ("cold", STYLE_COLD, "cold start")):
            if sB not in dM:
                continue
            oAx.semilogx(dM[sB]["aX"], dM[sB]["aTglob"], color=COLOR[sM],
                         linestyle=sStyle, marker="o", markersize=2.4,
                         label=sLab if iCol == 0 else None)
        if "warm" not in dM and "cold" not in dM:
            oAx.text(0.5, 0.5, "no CO$_2$ term\nin the model",
                     transform=oAx.transAxes, ha="center", va="center",
                     fontsize=7.4, color=INK_MUTED)
        dDist4 = dM.get("dictBranchDistinctness") or {}
        if dDist4.get("bBranchesIdentical"):
            oAx.text(0.5, 1.012, "cold file copies the warm file",
                     transform=oAx.transAxes, ha="center", va="bottom",
                     fontsize=6.0, color=COLOR["shields_bitz"])
        elif "dictHysteresis" in dM:
            h = dM["dictHysteresis"]
            if h["fGlaciationCO2ppm"] and h["fDeglaciationCO2ppm"]:
                oAx.axvspan(h["fGlaciationCO2ppm"], h["fDeglaciationCO2ppm"],
                            color="#f2f1ec", zorder=0)
                oAx.text(0.5, 1.012,
                         "bistable width %.2f decades" % h["fBistableWidthDecades"],
                         transform=oAx.transAxes, ha="center", va="bottom",
                         fontsize=6.4, color=INK_MUTED)
            else:
                oAx.text(0.5, 1.012, "loop not closed in range",
                         transform=oAx.transAxes, ha="center", va="bottom",
                         fontsize=6.4, color=COLOR["shields_bitz"])
        # Set the scale explicitly: a panel with no series never reaches
        # semilogx above and would otherwise be drawn on a linear axis.
        oAx.set_xscale("log")
        oAx.set(xlabel=r"CO$_2$ (ppm)", xlim=(0.7, 1.5e5), ylim=(200, 375))
        if iCol == 0:
            oAx.set_ylabel(r"$T_{\rm glob}$ (K)")
        fvTidy(oAx)

    aAx[0][0].text(-0.60, 0.5, "A   Experiment 3\n(instellation)",
                   transform=aAx[0][0].transAxes, rotation=90, va="center",
                   ha="center", fontsize=8.5, fontweight="bold")
    aAx[1][0].text(-0.60, 0.5, r"B   Experiment 4" + "\n" + r"(CO$_2$)",
                   transform=aAx[1][0].transAxes, rotation=90, va="center",
                   ha="center", fontsize=8.5, fontweight="bold")

    oFig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.11,
                         hspace=0.55, wspace=0.32)

    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
