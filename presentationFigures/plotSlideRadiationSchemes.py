"""Presentation version of the report's Figure 1, the longwave parameterisations: wide for a 16:9 slide, legends below the panels, no grids, vplot colours.

Same data and same reimplemented schemes as the report's figure.  What changes
is the reading distance.  Series labels are cut to the model name and, where
the distinction matters, the one word that separates two of its options; the
shaded-range annotations that the report spells out in the panel are dropped,
because on a slide they are the speaker's line rather than the figure's.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slidePlotStyle import (COLOR, LABEL, INK_MUTED, SLIDE_WIDE,   # noqa: E402
                            fvApplyStyle, fvTidy, fvLegendBelow,
                            fvSaveSlideFigure)

import matplotlib.pyplot as plt   # noqa: E402

SIGMA = 5.670374419e-8


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--radiation-json", required=True)
    oParser.add_argument("--co2-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictR = json.load(open(oArgs.radiation_json))
    dictC = json.load(open(oArgs.co2_json))
    aT = np.array(dictR["aTemperatureK"])
    dictS = dictR["dictSurfaces"]["280"]

    fvApplyStyle()
    oFig, aAx = plt.subplots(1, 2, figsize=SLIDE_WIDE)

    # --- A: OLR against temperature at the protocol CO2 -------------------
    oA = aAx[0]
    for sKey, sModel, sLab in (("avalon", "avalon", None),
                               ("hextor", "hextor", None),
                               ("poise_fixed", "poise", "VPLanet"),
                               ("ops", "ops", None),
                               ("shields_bitz", "shields_bitz", None)):
        oA.plot(aT, dictS[sKey], color=COLOR[sModel],
                label=sLab or LABEL[sModel])
    oA.plot(aT, dictS["poise_wk97"], color=COLOR["poise"], linestyle="--",
            label="VPLanet (WK97)")
    oA.plot(aT, SIGMA * aT ** 4, color=INK_MUTED, linestyle=":", linewidth=1.4,
            label="blackbody")
    oA.set(xlabel="Surface temperature (K)", ylabel=r"OLR (W m$^{-2}$)",
           xlim=(200, 330), ylim=(60, 380),
           title=r"A   Longwave at 280 ppm CO$_2$")
    fvTidy(oA)
    fvLegendBelow(oA, iCols=3)

    # --- B: OLR against CO2 at 288 K --------------------------------------
    oB = aAx[1]
    aC = np.array(dictC["aCO2ppm"])
    oB.semilogx(aC, dictC["aOLRAvalon"], color=COLOR["avalon"],
                label=LABEL["avalon"])
    oB.semilogx(aC, dictC["aOLRHextor"], color=COLOR["hextor"],
                label=LABEL["hextor"])
    oB.semilogx(aC, dictC["aOLRPoiseWK97"], color=COLOR["poise"],
                linestyle="--", label="VPLanet (WK97)")
    oB.semilogx(aC, dictC["aOLRPoiseHM16"], color=COLOR["poise"],
                linestyle="-.", label="VPLanet (HM16)")
    oB.semilogx(aC, dictC["aOLROps"], color=COLOR["ops"], label=LABEL["ops"])
    fSB = dictS["shields_bitz"][int(np.argmin(np.abs(aT - 288.0)))]
    oB.semilogx(aC, np.full(len(aC), fSB), color=COLOR["shields_bitz"],
                label="Shields-Bitz (no CO$_2$)")
    # The one annotation worth keeping: HEXTOR's table floor is the point of
    # the panel, and a shaded span says it without a sentence.
    fFloor = dictC["dictHextorClamp"]["fTableMinCO2ppm"]
    oB.axvspan(1.0, fFloor, color=COLOR["hextor"], alpha=0.10, zorder=0)
    oB.set(xlabel=r"CO$_2$ mixing ratio (ppm)", ylabel=r"OLR (W m$^{-2}$)",
           xlim=(1.0, 1e5), ylim=(180, 290),
           title=r"B   CO$_2$ dependence at $T$ = 288 K")
    fvTidy(oB)
    fvLegendBelow(oB, iCols=3)

    oFig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.30,
                         wspace=0.22)
    fvSaveSlideFigure(oFig, oArgs.out)
    print("wrote %s (and .png)" % oArgs.out)


if __name__ == "__main__":
    main()
