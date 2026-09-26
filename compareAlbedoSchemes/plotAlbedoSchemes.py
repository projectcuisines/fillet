"""Plot the four models' albedo parameterisations against temperature and against solar zenith angle."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from filletPlotStyle import (COLOR, LABEL, INK_MUTED,  # noqa: E402
                             fvApplyStyle, fvTidy, fvSaveFigure)

import matplotlib.pyplot as plt   # noqa: E402


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--albedo-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    d = json.load(open(oArgs.albedo_json))
    fvApplyStyle()
    aT = np.array(d["aTemperatureK"])

    oFig, aAx = plt.subplots(1, 2, figsize=(7.2, 4.1))

    oA = aAx[0]
    oA.plot(aT, d["aAlbedoAvalon_Tice0C"], color=COLOR["avalon"],
            label=r"%s, $T_{\rm ice}$ = 0 $^\circ$C (current source)" % LABEL["avalon"])
    oA.plot(aT, d["aAlbedoAvalon_TiceM10C"], color=COLOR["avalon"], linestyle="--",
            label=r"%s, $T_{\rm ice}$ = $-$10 $^\circ$C (as submitted)" % LABEL["avalon"])
    oA.plot(aT, d["aAlbedoPoise_zenith45"], color=COLOR["poise"],
            label=r"%s, $z$ = 45$^\circ$" % LABEL["poise"])
    oA.plot(aT, d["aSurfaceAlbedoHextor"], color=COLOR["hextor"], linestyle=":",
            label="%s, surface (fractional ice)" % LABEL["hextor"])
    oA.plot(aT, d["aPlanetaryAlbedoHextor"], color=COLOR["hextor"],
            label=r"%s, planetary, $z$ = 45$^\circ$" % LABEL["hextor"])
    oA.plot(aT, d["aSurfaceAlbedoOps"], color=COLOR["ops"], linestyle=":",
            label="%s, surface (34 K ice ramp, clouds)" % LABEL["ops"])
    oA.plot(aT, d["aPlanetaryAlbedoOps"], color=COLOR["ops"],
            label=r"%s, planetary, $z$ = 45$^\circ$" % LABEL["ops"])
    # Shields-Bitz has no shortwave atmosphere, so one curve serves for both
    # the surface and the top of the atmosphere.
    oA.plot(aT, d["aAlbedoShieldsBitz"], color=COLOR["shields_bitz"],
            label="%s, surface = planetary" % LABEL["shields_bitz"])
    for fThr, sLab, fY, sHa in ((263.15, r"$-10\,^\circ$C", 0.163, "center"),
                                (271.15, r"$-2\,^\circ$C", 0.196, "right"),
                                (273.15, r"$0\,^\circ$C", 0.163, "left")):
        oA.axvline(fThr, color=INK_MUTED, linewidth=0.6, linestyle=":")
        oA.text(fThr + (0.4 if sHa == "left" else (-0.4 if sHa == "right" else 0.0)),
                fY, sLab, color=INK_MUTED, fontsize=6.0, ha=sHa, va="bottom")
    oA.set(xlabel="Surface temperature (K)", ylabel="Albedo",
           xlim=(238, 302), ylim=(0.15, 0.72),
           title="A   Ice-albedo transition")
    oA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
              fontsize=6.2, ncol=1, handlelength=2.4)
    fvTidy(oA)

    oB = aAx[1]
    dz = d["dictZenithDependence"]
    aZ = np.array(dz["aZenithDeg"])
    oB.plot(aZ, dz["aAvalonAnyState"], color=COLOR["avalon"], marker="o",
            label="%s (no zenith dependence)" % LABEL["avalon"])
    oB.plot(aZ, dz["aPoiseIceFree"], color=COLOR["poise"], marker="s",
            label=r"%s, $\alpha_{\rm surf}$ = 0.225" % LABEL["poise"])
    oB.plot(aZ, dz["aHextorPlanetaryIceFree"], color=COLOR["hextor"], marker="^",
            label=r"%s, $\alpha_{\rm surf}$ = 0.225" % LABEL["hextor"])
    oB.plot(aZ, dz["aHextorPlanetaryGlaciated"], color=COLOR["hextor"],
            marker="^", linestyle="--",
            label=r"%s, $\alpha_{\rm surf}$ = 0.60" % LABEL["hextor"])
    oB.plot(aZ, dz["aOpsPlanetaryIceFree"], color=COLOR["ops"], marker="D",
            label=r"%s, $\alpha_{\rm surf}$ = 0.225" % LABEL["ops"])
    oB.plot(aZ, dz["aOpsPlanetaryGlaciated"], color=COLOR["ops"], marker="D",
            linestyle="--", label=r"%s, $\alpha_{\rm surf}$ = 0.60" % LABEL["ops"])
    aSB = np.asarray(d["aAlbedoShieldsBitz"])
    oB.plot(aZ, np.full(len(aZ), aSB[-1]), color=COLOR["shields_bitz"],
            marker="o", label="%s (no zenith dependence)" % LABEL["shields_bitz"])
    oB.plot(aZ, np.full(len(aZ), aSB[0]), color=COLOR["shields_bitz"],
            marker="o", linestyle="--",
            label="%s, glaciated" % LABEL["shields_bitz"])
    for fZ in (0.0, 30.0, 60.0, 90.0):
        oB.axvline(fZ, color=INK_MUTED, linewidth=0.5, linestyle=":")
    oB.text(2, 0.77, "dotted: HEXTOR's four zenith nodes;\nOPS resolves the axis with 19",
            color=INK_MUTED, fontsize=6.2, ha="left", va="top")
    oB.set(xlabel=r"Solar zenith angle (degrees)", ylabel="Planetary albedo",
           xlim=(-3, 93), ylim=(0.10, 0.78), xticks=[0, 15, 30, 45, 60, 75, 90],
           title="B   Zenith-angle dependence, ice-free and glaciated")
    oB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17),
              fontsize=6.0, ncol=1, handlelength=2.4)
    fvTidy(oB)

    oFig.subplots_adjust(bottom=0.46, wspace=0.26, top=0.93)
    fvSaveFigure(oFig, oArgs.out)
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
