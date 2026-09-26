"""Plot the drift between each code's archived FILLET submission and its current repository HEAD, and the forensic recovery of HEXTOR's post-submission CO2-coordinate correction."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK, INK_MUTED, INK_SECONDARY,  # noqa: E402
                             fvApplyStyle, fvTidy,
                             fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402

ROWS = [("hextor", "ben1", "Ben 1"), ("hextor", "ben2", "Ben 2"),
        ("hextor", "ben3", "Ben 3"), ("hextor", "exp1", "Exp 1"),
        ("hextor", "exp1a", "Exp 1a"), ("hextor", "exp2", "Exp 2"),
        ("hextor", "exp2a", "Exp 2a"),
        ("avalon", "ben1", "Ben 1"), ("avalon", "ben2", "Ben 2"),
        ("avalon", "ben3", "Ben 3"), ("avalon", "exp1", "Exp 1"),
        ("avalon", "exp2", "Exp 2"), ("avalon", "exp2a", "Exp 2a"),
        ("avalon", "exp3_cooling", "Exp 3 warm"),
        ("avalon", "exp3_warming", "Exp 3 cold"),
        ("avalon", "exp4_cooling", "Exp 4 warm"),
        ("avalon", "exp4_warming", "Exp 4 cold")]


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--drift-json", required=True)
    oParser.add_argument("--pifactor-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dD = {(c["sModel"], c["sExperiment"]): c
          for c in json.load(open(oArgs.drift_json))["listComparisons"]}
    dP = json.load(open(oArgs.pifactor_json))
    fvApplyStyle()

    oFig, aAx = plt.subplots(1, 2, figsize=(7.2, 3.6),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    # --- panel A: drift per experiment -------------------------------------
    oA = aAx[0]
    listUse = [(m, e, lab) for m, e, lab in ROWS if (m, e) in dD]
    aY = np.arange(len(listUse))[::-1]
    for iY, (sM, sE, sLab) in zip(aY, listUse):
        c = dD[(sM, sE)]
        oA.barh(iY, c["fRmsDeltaTglob"], height=0.62, color=COLOR[sM],
                edgecolor="white", linewidth=1.0)
        if c["iStateChanges"]:
            oA.text(c["fRmsDeltaTglob"] + 0.8, iY,
                    "%d/%d states change"
                    % (c["iStateChanges"], c["iMatchedCases"]),
                    va="center", fontsize=6.2, color=INK_SECONDARY)
    oA.set(yticks=aY, yticklabels=["%s  %s" % (LABEL[m].split("-")[-1][:6], lab)
                                   for m, e, lab in listUse],
           xlabel=r"RMS change in $T_{\rm glob}$ (K)  " +
                  "between the submitted and the current code",
           title="A   Drift since submission", xlim=(0, 62))
    oA.tick_params(axis="y", labelsize=6.6)
    oA.grid(axis="y", visible=False)
    fvTidy(oA)

    # --- panel B: the pi/2 CO2-coordinate recovery -------------------------
    oB = aAx[1]
    listBench = dP["listBenchmarks"]
    aX = np.arange(len(listBench))
    aBest = [b["fBestFitCO2ScaleFactor"] for b in listBench]
    oB.bar(aX, aBest, width=0.5, color=COLOR["hextor"], edgecolor="white",
           linewidth=1.2)
    oB.axhline(dP["fPiOverTwo"], color=INK, linewidth=1.4, linestyle="--")
    oB.text(-0.42, dP["fPiOverTwo"] + 0.03,
            r"$\pi/2$", ha="left", fontsize=7.6, color=INK)
    for i, b in enumerate(listBench):
        oB.text(i, b["fBestFitCO2ScaleFactor"] - 0.06,
                "%.2f" % b["fBestFitCO2ScaleFactor"], ha="center", va="top",
                fontsize=7.0, color="white")
        oB.text(i, 0.09, "%.2f $\\rightarrow$ %.2f"
                % (b["fRmsAtScaleOne"], b["fRmsAtPiOverTwo"]),
                ha="center", fontsize=6.6, color=INK_SECONDARY)
    oB.set(xticks=aX,
           xticklabels=[b["sBenchmark"].replace("ben", "Benchmark ")
                        for b in listBench],
           ylabel=r"Best-fit CO$_2$ scale factor", ylim=(0, 1.78),
           title=r"B   Recovering HEXTOR's CO$_2$-coordinate error")
    oB.tick_params(axis="x", labelsize=7.0)
    oB.grid(axis="x", visible=False)
    oB.set_xlabel(r"numerals in the bars: best-fit scale;  below: RMS OLR"
                  "\n"
                  r"residual (W m$^{-2}$) at scale 1 $\rightarrow$ at scale $\pi/2$",
                  fontsize=6.4, color=INK_MUTED)
    fvTidy(oB)

    oFig.tight_layout(w_pad=1.4)
    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
