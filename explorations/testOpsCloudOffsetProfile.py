"""Test whether the latitude-varying cloud offset OPS diagnoses accounts for the residual its archived OLR column leaves against the shipped lookup table.

Porting OPS's longwave to Python and predicting the archived per-latitude OLR
from the archived per-latitude temperature leaves a residual of a few W/m2
that a single additive constant does not remove: regressed against
temperature, the archived column is steeper than the table.  A constant cloud
offset cannot produce that, because it shifts the level and not the slope.

A latitude-varying one can, and the source specifies exactly such a thing.
`driver.f90` diagnoses the offset per belt from a surface convective heat
flux, `cloudir(k) = max(-6.8 ln(F_c/F_E + 1), -10)`, sets it to zero where the
annual mean lies between 263 and 273 K, and subtracts it from the table value:
`ir(k) = ir(k) - cloudir(k)`.  The offset is therefore larger in the warm
tropics than at the cold poles, which steepens OLR against temperature.

This script inverts the relation rather than assuming it.  For each benchmark
it forms

    cloudir_implied(k) = table(T_k, pCO2) - OLR_archived(k),

which is what the offset must have been if the port is otherwise right, and
asks three questions of the result:

1. Does it lie inside the range the source's parameterisation can produce,
   namely [-10, 0] W/m2?
2. Does it vary with temperature in the direction and by the magnitude the
   parameterisation implies, i.e. monotonically more negative as the belt
   warms?
3. Does allowing that variation account for the slope discrepancy, i.e. does
   the residual about a temperature-dependent offset fall to the level of the
   residual about a constant one for a model that has no such term?

A constant-offset control is scored the same way, so the comparison is against
something rather than against nothing.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import OpsOlrTable   # noqa: E402

# The bounds `driver.f90` puts on the diagnosed offset.
CLOUDIR_FLOOR_WM2 = -10.0
CLOUDIR_CEILING_WM2 = 0.0


def fdictImpliedOffset(aTK, aOLR, oTable, fFCO2, fPdryBar):
    """The per-belt offset implied by the archived column, and its properties."""
    aTable = oTable.faOLR(aTK, fFCO2, fPdryBar)
    aImplied = aTable - aOLR
    aFit = np.polyfit(aTK, aImplied, 1)
    aLinear = np.polyval(aFit, aTK)
    return {
        "aImpliedCloudIR": aImplied.tolist(),
        "fMin": float(aImplied.min()), "fMax": float(aImplied.max()),
        "fRange": float(aImplied.max() - aImplied.min()),
        "bInsideSourceBounds": bool(aImplied.min() >= CLOUDIR_FLOOR_WM2 - 0.5
                                    and aImplied.max() <= CLOUDIR_CEILING_WM2 + 0.5),
        "fSlopePerK": float(aFit[0]),
        "fCorrelationWithT": float(np.corrcoef(aTK, aImplied)[0, 1]),
        "fRMSAboutConstantOffset": float(np.std(aImplied)),
        "fRMSAboutLinearInT": float(np.sqrt(np.mean((aImplied - aLinear) ** 2))),
    }


def fdictScaleFit(aTK, aOLR, oTable, fFCO2, fPdryBar):
    """The multiplicative-and-additive fit, for comparison with the offset reading."""
    aTable = oTable.faOLR(aTK, fFCO2, fPdryBar)
    aA = np.vstack([aTable, np.ones(len(aTable))]).T
    aSol, _, _, _ = np.linalg.lstsq(aA, aOLR, rcond=None)
    aPred = aA @ aSol
    return {"fScale": float(aSol[0]), "fOffset": float(aSol[1]),
            "fRMS": float(np.sqrt(np.mean((aOLR - aPred) ** 2))),
            "fPercentSteeper": float(100.0 * (aSol[0] - 1.0))}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--fco2", type=float, default=2.8e-4)
    oParser.add_argument("--pdry-bar", type=float, default=1.0)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oTable = OpsOlrTable(oArgs.ops_olr_table)
    dictLat = json.load(open(oArgs.lat_json))

    listOut = []
    for sBench in ("ben1", "ben2", "ben3"):
        dictProf = dictLat.get("ops/%s" % sBench)
        if dictProf is None:
            continue
        aDat = np.array(dictProf["aaData"], dtype=float)
        aTK, aOLR = aDat[:, 1], aDat[:, 4]
        dictRow = {"sBenchmark": sBench, "iBelts": int(len(aTK))}
        dictRow.update(fdictImpliedOffset(aTK, aOLR, oTable, oArgs.fco2,
                                          oArgs.pdry_bar))
        dictRow["dictScaleFit"] = fdictScaleFit(aTK, aOLR, oTable, oArgs.fco2,
                                                oArgs.pdry_bar)
        dictRow["aTsurfK"] = aTK.tolist()
        listOut.append(dictRow)

    json.dump({"listBenchmarks": listOut,
               "fCloudIRFloorWm2": CLOUDIR_FLOOR_WM2,
               "fCloudIRCeilingWm2": CLOUDIR_CEILING_WM2},
              open(oArgs.out, "w"), indent=1)

    print("Cloud offset implied by the archived OLR column, against the bounds")
    print("`driver.f90` puts on the quantity it diagnoses (%.0f to %.0f W/m2):"
          % (CLOUDIR_FLOOR_WM2, CLOUDIR_CEILING_WM2))
    print("%-6s %6s %8s %8s %8s %9s %9s %8s %8s"
          % ("bench", "belts", "min", "max", "range", "dCIR/dT", "corr(T)",
             "rms/const", "rms/lin"))
    for d in listOut:
        print("%-6s %6d %8.2f %8.2f %8.2f %9.4f %9.3f %8.2f %8.2f%s"
              % (d["sBenchmark"], d["iBelts"], d["fMin"], d["fMax"],
                 d["fRange"], d["fSlopePerK"], d["fCorrelationWithT"],
                 d["fRMSAboutConstantOffset"], d["fRMSAboutLinearInT"],
                 "" if d["bInsideSourceBounds"] else "   OUTSIDE BOUNDS"))
    print("\nFor comparison, the multiplicative fit the same residual admits:")
    for d in listOut:
        f = d["dictScaleFit"]
        print("  %-6s scale %.4f (%.1f%% steeper), offset %+7.2f, rms %.2f W/m2"
              % (d["sBenchmark"], f["fScale"], f["fPercentSteeper"],
                 f["fOffset"], f["fRMS"]))
    print("\nwrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
