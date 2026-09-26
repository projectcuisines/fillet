"""Test whether HEXTOR's "pi/2" lookup-table CO2 coordinate is in fact the CO2/N2 molar-mass ratio 44/28, which OPS's source writes out explicitly.

`hextor/model/radiation/fix_fco2_coordinate.py` records that the stored fCO2
column of the shipped tables was pCO2/(X + pCO2) with X = pi/2 for the Sun
table and X = pi for the 2600 K table, and attributes the constant to "the pi
factor removed from the instellation function".  `testHextorPiFactorHypothesis`
confirms from the archived profiles that the pre-fix table behaved as though
queried at about 1.57 times the intended CO2 abundance, with per-benchmark
best fits of 1.55, 1.59 and 1.62.

OPS is a fourth code of the same Williams--Kasting lineage, and it converts
between the CO2 mixing ratio and its own lookup table's CO2 coordinate with
the molar masses written out (driver.f90):

    fco2  = pco2i/((Pdry/28. + pco2i/44.)*44.)
    pco2r = 44*Pdry*fco2(k)/(28*(1-fco2(k)))

The first of these is algebraically f = pCO2/((44/28) P_dry + pCO2): the SAME
form as HEXTOR's stored coordinate, with X = 44/28 = 1.5714 at a 1 bar
background rather than pi/2 = 1.5708.  The two constants differ by 0.04%, so
no fit to the archived profiles can separate them; what separates them is that
one of them is a quantity a radiative-transfer front end would actually
compute, and it appears in a sibling code with its molar masses visible.

This script quantifies both halves of that argument:

  (a) how far apart the two hypotheses are in the archived HEXTOR residuals,
      and how wide the range of scale factors is that the data cannot
      distinguish, so that the numerical test is correctly reported as
      inconclusive rather than as support for either constant;
  (b) the corresponding statement for OPS's own table, where the 44/28
      factor is not a hypothesis but the code's actual behaviour.

It also records the arithmetic for the 2600 K table, whose X = pi would be
2 x 44/28 = 3.1429 under the molar-mass reading -- a 2 bar background --
which the shipped tables cannot confirm because both carry the same 1 to 11
bar total-pressure axis.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import HextorTable, OpsOlrTable   # noqa: E402

F_MOLAR = 44.0 / 28.0
F_PI_HALF = np.pi / 2.0


def flistSourceEvidence(sDriverPath):
    """The two driver.f90 lines that convert between mixing ratio and table."""
    listOut = []
    for iNum, sLine in enumerate(open(sDriverPath, errors="replace"), 1):
        if sLine.lstrip().startswith("!"):
            continue
        if re.search(r"fco2\(1:nbelts\) = pco2i/|pco2r\(k\) = 44", sLine):
            listOut.append({"iLine": iNum, "sCode": sLine.split("!")[0].strip()})
    return listOut


def fdRmsAtScale(oTab, aT, aOLR, fFCO2, fCloudIR, fScale):
    """RMS residual of the archived OLR against the table queried at fScale*f."""
    aPred = np.array([oTab.fdOLR(float(t), fFCO2 * fScale) for t in aT]) - fCloudIR
    return float(np.sqrt(np.mean((aOLR - aPred) ** 2)))


def fdictDiscriminationBand(oTab, aT, aOLR, fFCO2, fCloudIR, fTolerance):
    """Range of scale factors whose RMS is within fTolerance of the minimum."""
    aScale = np.linspace(1.30, 1.90, 601)
    aRms = np.array([fdRmsAtScale(oTab, aT, aOLR, fFCO2, fCloudIR, float(f))
                     for f in aScale])
    fBest = aRms.min()
    aIn = aScale[aRms <= fBest + fTolerance]
    return {"fBestScale": float(aScale[int(np.argmin(aRms))]),
            "fBestRms": fBest,
            "fScaleLow": float(aIn.min()), "fScaleHigh": float(aIn.max()),
            "fBandWidth": float(aIn.max() - aIn.min())}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--ops-driver", required=True)
    oParser.add_argument("--fco2", type=float, default=2.8e-4)
    oParser.add_argument("--cloudir", type=float, default=-6.3)
    oParser.add_argument("--rms-tolerance", type=float, default=0.05,
                         help="W/m^2 of RMS a fit cannot meaningfully resolve")
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oTab = HextorTable(oArgs.hextor_table)
    oOps = OpsOlrTable(oArgs.ops_olr_table)
    dictLat = json.load(open(oArgs.lat_json))

    listBench = []
    for sBench in ("ben1", "ben2", "ben3"):
        aDat = np.array(dictLat["hextor/%s" % sBench]["aaData"], dtype=float)
        aT, aOLR = aDat[:, 1], aDat[:, 4]
        dictBand = fdictDiscriminationBand(oTab, aT, aOLR, oArgs.fco2,
                                           oArgs.cloudir, oArgs.rms_tolerance)
        fRmsPi = fdRmsAtScale(oTab, aT, aOLR, oArgs.fco2, oArgs.cloudir, F_PI_HALF)
        fRmsMolar = fdRmsAtScale(oTab, aT, aOLR, oArgs.fco2, oArgs.cloudir, F_MOLAR)
        listBench.append({
            "sBenchmark": sBench,
            "fRmsAtPiOverTwo": fRmsPi,
            "fRmsAtMolarMassRatio": fRmsMolar,
            "fRmsDifference": fRmsMolar - fRmsPi,
            "bBothInsideBand": (dictBand["fScaleLow"] <= F_PI_HALF
                                <= dictBand["fScaleHigh"]
                                and dictBand["fScaleLow"] <= F_MOLAR
                                <= dictBand["fScaleHigh"]),
            "dictDiscriminationBand": dictBand,
        })

    # The same constant applied to OPS's own table, where it is not a
    # hypothesis: how much OLR the 0.04% difference between the two constants
    # is worth, against the ~2.3 W/m^2 the factor itself is worth.
    aTGrid = np.array([250.0, 273.15, 288.0, 300.0])
    listOps = []
    for fT in aTGrid:
        fMolar = oOps.faOLR(np.array([fT]), 280e-6, 1.0, F_MOLAR)[0]
        fPi = oOps.faOLR(np.array([fT]), 280e-6, 1.0, F_PI_HALF)[0]
        fUnity = oOps.faOLR(np.array([fT]), 280e-6, 1.0, 1.0)[0]
        listOps.append({
            "fTemperatureK": float(fT),
            "fOLRWithMolarMassFactor": float(fMolar),
            "fOLRWithPiOverTwo": float(fPi),
            "fOLRWithNoFactor": float(fUnity),
            "fDifferenceBetweenConstants": float(fMolar - fPi),
            "fValueOfTheFactorItself": float(fMolar - fUnity),
        })

    dictOut = {
        "fPiOverTwo": F_PI_HALF,
        "fMolarMassRatio": F_MOLAR,
        "fFractionalDifference": float(abs(F_MOLAR - F_PI_HALF) / F_PI_HALF),
        "listHextorBenchmarks": listBench,
        "listOpsTable": listOps,
        "listOpsSourceEvidence": flistSourceEvidence(oArgs.ops_driver),
        "dictTwentySixHundredKelvinTable": {
            "fReportedConstant": float(np.pi),
            "fTwiceMolarMassRatio": 2.0 * F_MOLAR,
            "sNote": "pi = 3.14159 against 2 x 44/28 = 3.14286; under the "
                     "molar-mass reading this is a 2 bar background, which the "
                     "shipped tables cannot confirm because both carry the "
                     "same 1-11 bar total-pressure axis.",
        },
    }
    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("pi/2 = %.6f, 44/28 = %.6f, fractional difference %.2e"
          % (F_PI_HALF, F_MOLAR, dictOut["fFractionalDifference"]))
    print("\nHEXTOR archived benchmarks, current table queried at scale*fCO2:")
    print("%-6s %10s %10s %10s %12s %s"
          % ("bench", "rms@pi/2", "rms@44/28", "difference", "best scale",
             "scales within %.2f W/m2 of best" % oArgs.rms_tolerance))
    for d in listBench:
        b = d["dictDiscriminationBand"]
        print("%-6s %10.4f %10.4f %+10.5f %12.4f  [%.3f, %.3f]  both inside: %s"
              % (d["sBenchmark"], d["fRmsAtPiOverTwo"], d["fRmsAtMolarMassRatio"],
                 d["fRmsDifference"], b["fBestScale"], b["fScaleLow"],
                 b["fScaleHigh"], "yes" if d["bBothInsideBand"] else "no"))
    print("\nThe two constants are not separable by this fit; the evidence that")
    print("selects 44/28 is that OPS computes it explicitly:")
    for d in dictOut["listOpsSourceEvidence"]:
        print("  driver.f90:%-5d %s" % (d["iLine"], d["sCode"]))
    print("\nOPS's own table, OLR at 280 ppm (W/m2):")
    print("%8s %12s %12s %12s %14s %14s"
          % ("T (K)", "at 44/28", "at pi/2", "at pCO2", "44/28 - pi/2", "44/28 - pCO2"))
    for d in listOps:
        print("%8.2f %12.3f %12.3f %12.3f %+14.4f %+14.3f"
              % (d["fTemperatureK"], d["fOLRWithMolarMassFactor"],
                 d["fOLRWithPiOverTwo"], d["fOLRWithNoFactor"],
                 d["fDifferenceBetweenConstants"], d["fValueOfTheFactorItself"]))
    print("\n2600 K table: %s" % dictOut["dictTwentySixHundredKelvinTable"]["sNote"])


if __name__ == "__main__":
    main()
