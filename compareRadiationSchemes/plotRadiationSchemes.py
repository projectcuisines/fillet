"""Plot the four models' outgoing-longwave-radiation parameterisations against temperature and against CO2, marking the ranges over which each is being extrapolated."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK_MUTED, INK_SECONDARY,  # noqa: E402
                             fvApplyStyle, fvTidy, fvLegendBelow, fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402

SIGMA = 5.670367e-8


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--radiation-json", required=True)
    oParser.add_argument("--co2-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictR = json.load(open(oArgs.radiation_json))
    dictC = json.load(open(oArgs.co2_json))
    fvApplyStyle()

    aT = np.array(dictR["aTemperatureK"])
    dictS = dictR["dictSurfaces"]["280"]

    oFig, aAx = plt.subplots(1, 2, figsize=(7.2, 4.0))

    # --- panel A: OLR against temperature at the protocol CO2 --------------
    oA = aAx[0]
    oA.plot(aT, dictS["avalon"], color=COLOR["avalon"],
            label=LABEL["avalon"])
    oA.plot(aT, dictS["hextor"], color=COLOR["hextor"],
            label=LABEL["hextor"])
    oA.plot(aT, dictS["poise_fixed"], color=COLOR["poise"],
            label=LABEL["poise"])
    oA.plot(aT, dictS["poise_wk97"], color=COLOR["poise"], linestyle="--",
            label="%s (WK97)" % LABEL["poise"])
    oA.plot(aT, dictS["ops"], color=COLOR["ops"],
            label=LABEL["ops"])
    oA.plot(aT, dictS["shields_bitz"], color=COLOR["shields_bitz"],
            label=LABEL["shields_bitz"])
    oA.plot(aT, SIGMA * aT ** 4, color=INK_MUTED, linestyle=":", linewidth=1.0,
            label=r"blackbody $\sigma T^4$")
    oA.axvspan(190, 370, color="#f6f5f1", zorder=0, linewidth=0)
    oA.set(xlabel="Surface temperature (K)",
           ylabel=r"OLR (W m$^{-2}$)", xlim=(200, 330), ylim=(60, 380),
           title=r"A   Longwave scheme at 280 ppm CO$_2$")
    fvLegendBelow(oA, iCols=2, fPad=-0.20, fSize=7.0)
    fvTidy(oA)

    # --- panel B: OLR against CO2 at 288 K ---------------------------------
    oB = aAx[1]
    aC = np.array(dictC["aCO2ppm"])
    oB.semilogx(aC, dictC["aOLRAvalon"], color=COLOR["avalon"],
                label=LABEL["avalon"])
    oB.semilogx(aC, dictC["aOLRHextor"], color=COLOR["hextor"],
                label=LABEL["hextor"])
    oB.semilogx(aC, dictC["aOLRPoiseWK97"], color=COLOR["poise"], linestyle="--",
                label="%s (WK97)" % LABEL["poise"])
    oB.semilogx(aC, dictC["aOLRPoiseHM16"], color=COLOR["poise"], linestyle="-.",
                label="%s (HM16)" % LABEL["poise"])
    oB.semilogx(aC, dictC["aOLROps"], color=COLOR["ops"],
                label=LABEL["ops"])
    # Shields-Bitz has no CO2 term at all, so this line is flat by
    # construction rather than by clamping, which is the point of showing it.
    fSB = dictR["dictSurfaces"]["280"]["shields_bitz"][
        int(np.argmin(np.abs(aT - 288.0)))]
    oB.semilogx(aC, np.full(len(aC), fSB), color=COLOR["shields_bitz"],
                label="%s (no CO$_2$)" % LABEL["shields_bitz"])
    fFloor = dictC["dictHextorClamp"]["fTableMinCO2ppm"]
    oB.axvspan(1.0, fFloor, color="#fbeee8", zorder=0)
    oB.axvline(fFloor, color=COLOR["hextor"], linewidth=0.8, linestyle=":")
    oB.text(1.3, 168, "shaded: HEXTOR clamped below its %.0f ppm table floor"
            % fFloor, color=COLOR["hextor"], fontsize=6.5, ha="left",
            va="bottom")
    oB.axhline(SIGMA * 288.0 ** 4, color=INK_MUTED, linestyle=":", linewidth=1.0)
    oB.text(1.3, SIGMA * 288.0 ** 4 + 6, r"blackbody $\sigma T^4$ at 288 K",
            color=INK_MUTED, fontsize=6.5, ha="left")
    oB.set(xlabel=r"CO$_2$ mixing ratio (ppm)", ylabel=r"OLR (W m$^{-2}$)",
           ylim=(160, 430),
           title=r"B   CO$_2$ dependence at $T$ = 288 K")
    fvLegendBelow(oB, iCols=2, fPad=-0.20, fSize=7.0)
    fvTidy(oB)

    oFig.tight_layout()
    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
