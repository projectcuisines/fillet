"""Plot how much of the reported ice-line spread comes from the definition rather than from the simulated temperature field."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK, INK_MUTED, INK_SECONDARY,  # noqa: E402
                             fvApplyStyle, fvTidy, fvLegendBelow,
                             fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402

DEFS = [("263.15/interpolate", r"$-10\,^\circ$C, interp."),
        ("263.15/snapToNode", r"$-10\,^\circ$C, snapped"),
        ("271.15/interpolate", r"$-2\,^\circ$C, interp."),
        ("271.15/snapToNode", r"$-2\,^\circ$C, snapped"),
        ("273.15/interpolate", r"$0\,^\circ$C, interp."),
        ("273.15/snapToNode", r"$0\,^\circ$C, snapped")]

# The definition each model actually used in the archived submission.
NATIVE = {"avalon": "263.15/snapToNode",
          "hextor": "263.15/interpolate",
          "poise": "273.15/snapToNode",
          "ops": "263.15/interpolate",
          "shields_bitz": "271.15/interpolate"}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--harmonized-json", required=True)
    oParser.add_argument("--drift-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    d = json.load(open(oArgs.harmonized_json))
    fvApplyStyle()
    dictBench = {(b["sModel"], b["sBenchmark"]): b
                 for b in d["listBenchmarkEdges"]}

    oFig, aAx = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # --- panel A: Benchmark 1 ice edge under six definitions ---------------
    oA = aAx[0]
    aX = np.arange(len(DEFS))
    for sM in ("avalon", "hextor", "poise", "ops", "shields_bitz"):
        b = dictBench[(sM, "ben1")]
        aY = [b["dictEdges"][k] for k, _ in DEFS]
        oA.plot(aX, aY, color=COLOR[sM], marker="o",
                label="%s (%d)" % (LABEL[sM].replace("VPLanet-", ""),
                                    b["iNumLatitudes"]))
        iNat = [k for k, _ in DEFS].index(NATIVE[sM])
        oA.plot([iNat], [aY[iNat]], marker="o", markersize=9,
                markerfacecolor="none", markeredgecolor=COLOR[sM],
                markeredgewidth=1.6)
    oA.set(xticks=aX, xticklabels=[s for _, s in DEFS],
           ylabel=r"NH ice-edge latitude ($^\circ$)",
           title="A   Benchmark 1 ice edge, six definitions",
           ylim=(42, 80))
    oA.tick_params(axis="x", labelsize=6.4, rotation=32)
    oA.grid(axis="x", visible=False)
    fvLegendBelow(oA, iCols=3, fPad=-0.42, fSize=6.8)
    oA.text(0.98, 0.96, "ringed: the definition the model used",
            transform=oA.transAxes, fontsize=6.4, color=INK_MUTED, ha="right",
            va="top")
    fvTidy(oA)

    # --- panel B: Experiment 1 sensitivity ---------------------------------
    oB = aAx[1]
    listS = d["listDefinitionSensitivity"]
    aX2 = np.arange(len(DEFS))
    fW = 0.34
    for j, sM in enumerate(("hextor", "avalon")):
        aShift = []
        for sKey, _ in DEFS:
            r = [x for x in listS if x["sModel"] == sM
                 and x["sDefinition"] == sKey][0]
            aShift.append(r["fMeanShiftVsReference_deg"] or 0.0)
        oB.bar(aX2 + (j - 0.5) * fW, aShift, width=fW * 0.88, color=COLOR[sM],
               edgecolor="white", linewidth=1.0, label=LABEL[sM])
        for sKey, fV, iX in zip([k for k, _ in DEFS], aShift, aX2):
            r = [x for x in listS if x["sModel"] == sM
                 and x["sDefinition"] == sKey][0]
            if r["iStateChangesVsReference"]:
                oB.text(iX + (j - 0.5) * fW, fV - 0.55,
                        "%d" % r["iStateChangesVsReference"], ha="center",
                        va="top", fontsize=6.2, color=INK_SECONDARY)
    oB.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    oB.set(xticks=aX2, xticklabels=[s for _, s in DEFS],
           ylabel="Mean shift of the NH ice edge  " + r"($^\circ$)",
           title="B   Experiment 1: definition sensitivity", ylim=(-13.5, 7.5))
    oB.tick_params(axis="x", labelsize=6.4, rotation=32)
    oB.grid(axis="x", visible=False)
    fvLegendBelow(oB, iCols=2, fPad=-0.42, fSize=7.0)
    oB.text(0.02, 0.05, "numerals: cases (of 190) changing state",
            transform=oB.transAxes, fontsize=6.4, color=INK_MUTED)
    fvTidy(oB)

    oFig.tight_layout(w_pad=1.2, rect=(0, 0.02, 1, 1))
    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
