"""Plot the Benchmark 1 latitudinal profiles of the three models together with the global energy budget each of them balances at that temperature."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK_MUTED, INK_SECONDARY,   # noqa: E402
                             fvApplyStyle, fvTidy, fvLatitudeAxis,
                             fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402

# Observed pre-industrial / present-day Earth reference values.
OBS_BOND_ALBEDO = 0.294
OBS_OLR = 239.7


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--benchmark-json", required=True)
    oParser.add_argument("--benchmark", default="ben1")
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listB = json.load(open(oArgs.benchmark_json))["listBenchmarks"]
    dictB = {(d["sModel"], d["sBenchmark"]): d for d in listB}
    fvApplyStyle()

    oFig = plt.figure(figsize=(7.6, 4.6))
    aAx = [oFig.add_subplot(2, 3, i + 1) for i in range(6)]

    listModels = ["avalon", "hextor", "poise", "ops", "shields_bitz"]
    sB = oArgs.benchmark

    for sM in listModels:
        d = dictB[(sM, sB)]
        aLat = np.array(d["aLat"])
        aAx[0].plot(aLat, d["aTsurf"], color=COLOR[sM],
                    label="%s (%d)" % (LABEL[sM].replace("VPLanet-", ""),
                                       d["iNumLatitudes"]))
        aAx[1].plot(aLat, d["aAsurf"], color=COLOR[sM])
        aAx[2].plot(aLat, d["aATOA"], color=COLOR[sM])
        aAx[3].plot(aLat, d["aOLR"], color=COLOR[sM])
        aAx[4].plot(aLat, np.array(d["aInsolation"]) * (1 - np.array(d["aATOA"])),
                    color=COLOR[sM])

    for i, (sLab, sTitle) in enumerate([
            ("Surface temperature (K)", "A   Surface temperature"),
            ("Surface albedo", "B   Surface albedo"),
            ("Planetary albedo", "C   Top-of-atmosphere albedo"),
            (r"OLR (W m$^{-2}$)", "D   Outgoing longwave"),
            (r"Absorbed shortwave (W m$^{-2}$)", "E   Absorbed shortwave")]):
        aAx[i].set(xlabel="Latitude (deg)", ylabel=sLab, title=sTitle)
        fvLatitudeAxis(aAx[i])
        fvTidy(aAx[i])
    # One key for the whole figure, under the panels, rather than a box
    # sitting inside panel A.
    oHandles, oLabels = aAx[0].get_legend_handles_labels()
    oFig.legend(oHandles, oLabels, loc="lower center", ncol=5, fontsize=7.2,
                frameon=False, handlelength=2.2,
                bbox_to_anchor=(0.5, -0.015))
    aAx[0].set_ylim(245, 305)
    aAx[2].axhline(OBS_BOND_ALBEDO, color=INK_MUTED, linestyle=":", linewidth=0.9)
    # The dotted reference is named in the caption: with five series there is
    # no space inside the axes for a label that does not sit on a curve.

    # --- panel F: the global budget each model balances --------------------
    oF = aAx[5]
    aX = np.arange(len(listModels))
    aASR = [dictB[(m, sB)]["fAbsorbedShortwave"] for m in listModels]
    aOLR = [dictB[(m, sB)]["fOLRAreaWeighted"] for m in listModels]
    aAlb = [dictB[(m, sB)]["fATOAInsolationWeighted"] for m in listModels]
    for i, sM in enumerate(listModels):
        oF.bar(i - 0.18, aASR[i], width=0.34, color=COLOR[sM], alpha=0.95,
               edgecolor="white", linewidth=1.2)
        oF.bar(i + 0.18, aOLR[i], width=0.34, color=COLOR[sM], alpha=0.42,
               edgecolor="white", linewidth=1.2)
        oF.text(i - 0.18, aASR[i] + 1.5, "%.0f" % aASR[i], ha="center",
                fontsize=6.2, color=INK_SECONDARY)
        oF.text(i + 0.18, aOLR[i] + 1.5, "%.0f" % aOLR[i], ha="center",
                fontsize=6.2, color=INK_SECONDARY)
    oF.axhline(OBS_OLR, color=INK_MUTED, linestyle=":", linewidth=0.9)
    # The Earth reference is named in the caption; a label here collides
    # with the bar values.
    oF.set(xticks=aX,
           xticklabels=[LABEL[m].replace("VPLanet-", "").replace("Shields-Bitz", "S-B")
                        for m in listModels],
           ylabel=r"Global mean flux (W m$^{-2}$)", ylim=(220, 284),
           title="F   Global energy budget")
    oF.tick_params(axis="x", labelsize=6.2, rotation=24)
    for oLab in oF.get_xticklabels():
        oLab.set_ha("right")
    oF.set_xlabel("solid bar: absorbed shortwave      pale bar: outgoing longwave",
                  fontsize=6.4, color=INK_MUTED)
    oF.grid(axis="x", visible=False)
    fvTidy(oF)

    oFig.tight_layout(h_pad=1.4, w_pad=1.1, rect=(0, 0.045, 1, 1))
    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
