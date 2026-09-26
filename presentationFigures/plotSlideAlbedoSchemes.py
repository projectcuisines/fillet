"""Presentation version of the report's Figure 2, the albedo parameterisations: wide for a 16:9 slide, legends below the panels, no grids, vplot colours.

Same data and same reimplemented schemes as the report's figure, with the
labels cut to what a listener needs.  Panel A drops the report's threshold
guide lines and its AVALON archived/current pair, keeping one curve per code
so that the thing the panel exists to show -- that three schemes step and two
ramp, and that the steps are not the same height -- survives at reading
distance.  Panel B keeps only the ice-free curves, since the glaciated ones
make the same point twice.
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


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--albedo-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    d = json.load(open(oArgs.albedo_json))
    aT = np.array(d["aTemperatureK"])
    dz = d["dictZenithDependence"]
    aZ = np.array(dz["aZenithDeg"])

    fvApplyStyle()
    oFig, aAx = plt.subplots(1, 2, figsize=SLIDE_WIDE)

    # --- A: albedo against temperature ------------------------------------
    oA = aAx[0]
    # AVALON and VPLanet coincide exactly wherever both sit at the protocol's
    # prescribed 0.60 ice albedo, so the wider line is drawn underneath and
    # shows as a halo around the narrower one.  Linewidth rather than dashes,
    # because dashes already mean "a different scheme option" in Figure 1.
    for sKey, sModel, fLw, iZ in (("aAlbedoAvalon_Tice0C", "avalon", 4.2, 2),
                                  ("aPlanetaryAlbedoHextor", "hextor", 2.4, 3),
                                  ("aAlbedoPoise_zenith45", "poise", 2.0, 4),
                                  ("aPlanetaryAlbedoOps", "ops", 2.4, 3),
                                  ("aAlbedoShieldsBitz", "shields_bitz", 2.4, 3)):
        oA.plot(aT, d[sKey], color=COLOR[sModel], label=LABEL[sModel],
                linewidth=fLw, zorder=iZ)
    oA.set(xlabel="Surface temperature (K)", ylabel="Planetary albedo",
           xlim=(238, 302), ylim=(0.15, 0.65),
           title="A   Ice\u2013albedo transition")
    fvTidy(oA)
    fvLegendBelow(oA, iCols=3)

    # --- B: zenith-angle dependence, ice-free -----------------------------
    oB = aAx[1]
    for sKey, sModel in (("aAvalonAnyState", "avalon"),
                         ("aHextorPlanetaryIceFree", "hextor"),
                         ("aPoiseIceFree", "poise"),
                         ("aOpsPlanetaryIceFree", "ops")):
        oB.plot(aZ, dz[sKey], color=COLOR[sModel], marker="o", markersize=5,
                label=LABEL[sModel])
    aSB = np.array(d["aAlbedoShieldsBitz"])
    oB.plot(aZ, np.full(len(aZ), aSB[-1]), color=COLOR["shields_bitz"],
            marker="o", markersize=5, label=LABEL["shields_bitz"])
    oB.set(xlabel="Solar zenith angle (degrees)", ylabel="Planetary albedo",
           xlim=(-3, 93), ylim=(0.15, 0.65), xticks=[0, 30, 60, 90],
           title="B   Zenith dependence, ice-free")
    fvTidy(oB)
    fvLegendBelow(oB, iCols=3)

    oFig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.30,
                         wspace=0.22)
    fvSaveSlideFigure(oFig, oArgs.out)
    print("wrote %s (and .png)" % oArgs.out)


if __name__ == "__main__":
    main()
