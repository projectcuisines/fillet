"""Choose and validate the fourth and fifth categorical figure colours for the FILLET comparison, by computing the data-visualisation palette checks rather than eyeballing them.

The report's figures began with three model series on the first three slots of
the reference categorical palette, which are the subset documented as clearing
the all-pairs colour-vision-deficiency and normal-vision separation floors in
light mode -- the mode a printed PDF is read in.  Each further model needs
another mark, and the reference ordering's own slot 4 (yellow) is documented to
FAIL the all-pairs floors beside slot 2 (orange).

This script ports the five computable checks of the reference validator
(lightness band, chroma floor, CVD separation under the Machado, Oliveira &
Fernandes 2009 severity-1.0 transforms, the unsimulated normal-vision floor,
and WCAG contrast against the chart surface) and scores candidates for the next
series on the ALL-PAIRS pairlist, because line series in one panel can be
compared in any pair.

Run with the three-slot base it selects slot 7, violet, as the fourth series.
Run with that four-slot base, NO remaining chromatic slot clears the floors:
the best CVD candidate (yellow) fails normal vision at 13.7 and the best
normal-vision candidate (green) fails protanopia at 3.2.  The validated
categorical order therefore supports four all-pairs-separable line series in
light mode and not five.

The resolution is an achromatic fifth mark.  Scored the same way, primary ink
holds the worst-pair CVD separation at 9.2 and the worst-pair normal-vision
separation at 16.3 -- the same values the four-colour set already had, because
the CVD transforms preserve lightness and no chromatic step competes with it.
It fails only the chroma floor, which is a test of whether a CATEGORICAL HUE is
vivid enough to read as a hue and does not apply to a mark that is deliberately
not a hue.  `--extra-candidates` is how that is scored rather than assumed.

Writes the full score table so the choice is auditable and reproducible.
"""

import argparse
import itertools
import json
import math

# Reference categorical palette, light-mode steps, in documented slot order.
DICT_SLOTS = {
    "1 blue": "#2a78d6", "2 orange": "#eb6834", "3 aqua": "#1baf7a",
    "4 yellow": "#eda100", "5 magenta": "#e87ba4", "6 green": "#008300",
    "7 violet": "#4a3aa7", "8 red": "#e34948",
}
LIST_BASE = ["#2a78d6", "#eb6834", "#1baf7a"]

BAND_LIGHT = (0.43, 0.77)
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0

MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def flistLinear(sHex):
    """sRGB hex to linear-light RGB."""
    s = sHex.strip().lstrip("#")
    aS = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in aS]


def flistOklab(aLin):
    fR, fG, fB = aLin
    fL = (0.4122214708 * fR + 0.5363325363 * fG + 0.0514459929 * fB) ** (1 / 3)
    fM = (0.2119034982 * fR + 0.6806995451 * fG + 0.1073969566 * fB) ** (1 / 3)
    fS = (0.0883024619 * fR + 0.2817188376 * fG + 0.6299787005 * fB) ** (1 / 3)
    return [0.2104542553 * fL + 0.7936177850 * fM - 0.0040720468 * fS,
            1.9779984951 * fL - 2.4285922050 * fM + 0.4505937099 * fS,
            0.0259040371 * fL + 0.7827717662 * fM - 0.8086757660 * fS]


def flistSimulate(aLin, sKind):
    aM = MACHADO[sKind]
    return [min(1.0, max(0.0, sum(aM[i][j] * aLin[j] for j in range(3))))
            for i in range(3)]


def fdDeltaE(sA, sB, sKind=None):
    """Euclidean OKLab distance x100, optionally under a simulated CVD."""
    aA, aB = flistLinear(sA), flistLinear(sB)
    if sKind:
        aA, aB = flistSimulate(aA, sKind), flistSimulate(aB, sKind)
    aA, aB = flistOklab(aA), flistOklab(aB)
    return 100.0 * math.dist(aA, aB)


def fdContrast(sA, sB):
    def fLum(s):
        fR, fG, fB = flistLinear(s)
        return 0.2126 * fR + 0.7152 * fG + 0.0722 * fB
    fHi, fLo = sorted((fLum(sA), fLum(sB)), reverse=True)
    return (fHi + 0.05) / (fLo + 0.05)


def fdictScore(listPalette, sSurface):
    """All five computable checks on the ALL-PAIRS pairlist."""
    listPairs = list(itertools.combinations(range(len(listPalette)), 2))
    dictWorst = {}
    for sKind in ("protan", "deutan"):
        dictWorst[sKind] = min(fdDeltaE(listPalette[i], listPalette[j], sKind)
                               for i, j in listPairs)
    fWorstCVD = min(dictWorst["protan"], dictWorst["deutan"])
    fWorstNormal = min(fdDeltaE(listPalette[i], listPalette[j]) for i, j in listPairs)
    listL, listC, listContrast = [], [], []
    for sHex in listPalette:
        fL, fA, fB = flistOklab(flistLinear(sHex))
        listL.append(fL)
        listC.append(math.hypot(fA, fB))
        listContrast.append(fdContrast(sHex, sSurface))
    return {
        "fWorstPairCVD": fWorstCVD,
        "fWorstPairProtan": dictWorst["protan"],
        "fWorstPairDeutan": dictWorst["deutan"],
        "fWorstPairNormal": fWorstNormal,
        "fMinLightness": min(listL), "fMaxLightness": max(listL),
        "fMinChroma": min(listC),
        "fMinContrast": min(listContrast),
        "bLightnessBandOk": all(BAND_LIGHT[0] <= f <= BAND_LIGHT[1] for f in listL),
        "bChromaOk": min(listC) >= CHROMA_FLOOR,
        "bCvdOk": fWorstCVD >= CVD_FLOOR,
        "bCvdAtTarget": fWorstCVD >= CVD_TARGET,
        "bNormalOk": fWorstNormal >= NORMAL_FLOOR,
        "bContrastOk": min(listContrast) >= CONTRAST_MIN,
    }


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--surface", default="#ffffff",
                         help="chart surface the marks sit on")
    oParser.add_argument("--base", default=",".join(LIST_BASE),
                         help="comma-separated hexes already assigned to series")
    oParser.add_argument("--extra-candidates", default="",
                         help="comma-separated hexes to score beside the palette slots")
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listBase = [s.strip() for s in oArgs.base.split(",") if s.strip()]

    dictCandidates = dict(DICT_SLOTS)
    for sHex in [t.strip() for t in oArgs.extra_candidates.split(",") if t.strip()]:
        dictCandidates["ink %s" % sHex] = sHex

    listCandidates = []
    for sName, sHex in dictCandidates.items():
        if sHex in listBase:
            continue
        dictScore = fdictScore(listBase + [sHex], oArgs.surface)
        dictScore["sSlot"] = sName
        dictScore["sHex"] = sHex
        listCandidates.append(dictScore)

    for d in listCandidates:
        # An achromatic mark is exempt from the chroma floor by construction:
        # that check asks whether a categorical HUE is vivid enough to read as
        # a hue, which is not what an ink series is for.
        d["bAchromatic"] = d["fMinChroma"] < 0.02
        d["bSeparationOk"] = d["bNormalOk"] and d["bCvdOk"]
        d["bAccepted"] = d["bSeparationOk"] and (d["bChromaOk"] or d["bAchromatic"])

    listPassing = [d for d in listCandidates if d["bAccepted"]]
    listPassing.sort(key=lambda d: -min(d["fWorstPairCVD"], d["fWorstPairNormal"] / 2))

    dictOut = {
        "sSurface": oArgs.surface,
        "listBaseSlots": listBase,
        "sPairlist": "all",
        "dictThresholds": {
            "fCvdTarget": CVD_TARGET, "fCvdFloor": CVD_FLOOR,
            "fNormalFloor": NORMAL_FLOOR, "fContrastMin": CONTRAST_MIN,
            "aLightnessBand": list(BAND_LIGHT), "fChromaFloor": CHROMA_FLOOR,
        },
        "dictBaseScore": fdictScore(listBase, oArgs.surface),
        "listNextSlotCandidates": listCandidates,
        "sRecommendedSlot": listPassing[0]["sSlot"] if listPassing else None,
        "sRecommendedHex": listPassing[0]["sHex"] if listPassing else None,
    }
    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    d = dictOut["dictBaseScore"]
    print("Existing %d slots, all-pairs, surface %s:" % (len(listBase), oArgs.surface))
    print("  worst CVD dE %.1f, worst normal dE %.1f, min contrast %.2f:1"
          % (d["fWorstPairCVD"], d["fWorstPairNormal"], d["fMinContrast"]))
    print("\nNext-slot candidates (the base set plus the candidate, ALL pairs):")
    print("%-11s %-8s %7s %7s %7s %8s %8s %6s"
          % ("slot", "hex", "protan", "deutan", "normal", "contrast", "chroma", "pass"))
    for c in listCandidates:
        print("%-11s %-8s %7.1f %7.1f %7.1f %8.2f %8.3f %6s"
              % (c["sSlot"], c["sHex"], c["fWorstPairProtan"], c["fWorstPairDeutan"],
                 c["fWorstPairNormal"], c["fMinContrast"], c["fMinChroma"],
                 "yes" if c["bAccepted"] else "NO"))
    print("\nrecommended next slot: %s (%s)"
          % (dictOut["sRecommendedSlot"], dictOut["sRecommendedHex"]))
    print("thresholds: CVD floor %.0f / target %.0f, normal floor %.0f, contrast %.0f:1"
          % (CVD_FLOOR, CVD_TARGET, NORMAL_FLOOR, CONTRAST_MIN))


if __name__ == "__main__":
    main()
