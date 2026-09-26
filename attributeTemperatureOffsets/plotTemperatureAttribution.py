"""Plot the Shapley attribution of the pairwise Benchmark 1 and Benchmark 2 temperature differences to instellation, albedo, longwave intercept and Planck slope."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, INK_MUTED, INK_SECONDARY, INK,   # noqa: E402
                             fvApplyStyle, fvTidy,
                             fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402

# Factors are not model identities, so they take neutral inks plus the two
# structural poles of the palette rather than the categorical model hues.
FACTOR_STYLE = [
    ("fAlbedo", "albedo  " + r"$\alpha$", COLOR["avalon"]),
    ("fA", "longwave intercept  " + r"$A$", COLOR["hextor"]),
    ("fB", "Planck slope  " + r"$B$", COLOR["ops"]),
    ("fS0", "instellation  " + r"$S$", INK_MUTED),
]
# OPS is absent by design: the archived ATOA column cannot close its
# annual-mean energy budget, so Equation (1) does not reproduce its own
# temperature and a decomposition of its differences would not sum to
# anything meaningful.  The omission is stated in the caption, not hidden.
PAIR_LABEL = {
    ("avalon", "hextor"): "AVALON\n" + r"$\rightarrow$ HEXTOR",
    ("avalon", "poise"): "AVALON\n" + r"$\rightarrow$ POISE",
    ("avalon", "shields_bitz"): "AVALON\n" + r"$\rightarrow$ S-B",
    ("hextor", "poise"): "HEXTOR\n" + r"$\rightarrow$ POISE",
    ("hextor", "shields_bitz"): "HEXTOR\n" + r"$\rightarrow$ S-B",
    ("poise", "shields_bitz"): "POISE\n" + r"$\rightarrow$ S-B",
}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--attribution-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listB = json.load(open(oArgs.attribution_json))["listBenchmarks"]
    dictB = {d["sBenchmark"]: d for d in listB}
    fvApplyStyle()

    oFig, aAx = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)

    for iPanel, (sBench, sTitle) in enumerate(
            [("ben1", "A   Benchmark 1: each model tuned to 288 K"),
             ("ben2", "B   Benchmark 2: common parameters, no tuning")]):
        oAx = aAx[iPanel]
        listPairs = [p for p in dictB[sBench]["listPairs"]
                     if (p["sModelA"], p["sModelB"]) in PAIR_LABEL]
        aX = np.arange(len(listPairs))
        fW = 0.19
        for j, (sKey, sLab, sCol) in enumerate(FACTOR_STYLE):
            aV = [p["dictShapley"][sKey] for p in listPairs]
            oAx.bar(aX + (j - 1.5) * fW, aV, width=fW * 0.88, color=sCol,
                    edgecolor="white", linewidth=1.0,
                    label=sLab if iPanel == 0 else None)
        for i, p in enumerate(listPairs):
            oAx.plot([i - 2.2 * fW, i + 2.2 * fW],
                     [p["fDeltaTZeroD"]] * 2, color=INK, linewidth=1.6,
                     solid_capstyle="butt")
            oAx.text(i, p["fDeltaTZeroD"] + (0.9 if p["fDeltaTZeroD"] >= 0 else -2.1),
                     "net %+.1f K" % p["fDeltaTZeroD"], ha="center",
                     fontsize=6.6, color=INK)
        oAx.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
        oAx.set(xticks=aX,
                xticklabels=[PAIR_LABEL[(p["sModelA"], p["sModelB"])]
                             for p in listPairs],
                title=sTitle, ylim=(-18.0, 19.0))
        oAx.tick_params(axis="x", labelsize=7.0)
        oAx.grid(axis="x", visible=False)
        fvTidy(oAx)

    aAx[0].set_ylabel("Contribution to " + r"$\Delta T_{\rm glob}$" + " (K)")
    # With six pairs per panel there is no clear space inside the axes, so
    # the key sits below the figure.
    oHandles, oLabels = aAx[0].get_legend_handles_labels()
    oFig.legend(oHandles, oLabels, loc="lower center", ncol=4, fontsize=7.0,
                frameon=False, bbox_to_anchor=(0.5, -0.01))
    aAx[1].text(1.0, 17.2,
                "black bar: net zero-dimensional difference",
                fontsize=6.6, color=INK_MUTED, ha="center")

    oFig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
