"""Validate the reimplementation of the OPS longwave scheme against the archived OPS profiles, and audit the internal consistency of the OPS submission.

Two checks, neither of which uses the other's answer:

1. LONGWAVE PORT VALIDATION.  Predict the archived per-latitude OLR from the
   archived per-latitude temperature through the OPS lookup table, allowing
   one free additive constant for the cloud-infrared offset, which no
   archived file records.  A correct port reproduces the SHAPE of OLR(T) and
   leaves a small residual; the residual is only interpretable against a
   negative control, so the same fit is run with HEXTOR's table, the other
   Williams--Kasting-lineage scheme in the ensemble and the one that shares
   OPS's ice-line convention.  A second fit with a free SLOPE separates two
   ways the port can disagree with the archive: an additive offset is an
   unrecorded configuration choice and says nothing about the scheme, whereas
   a slope departure means the archived OLR responds to temperature
   differently from the table value alone.  `testOpsCloudOffsetProfile.py`
   takes the slope departure further and shows it is the per-belt cloud
   offset the source diagnoses.

2. INTERNAL CONSISTENCY OF THE SUBMISSION.  Case counts against the protocol,
   the CO2 unit actually used in each file, and whether the two branches of
   the hysteresis experiments are in fact distinct runs.

The grid is read and reported, but no longer tested for provenance: the source
tree compiles 36 belts, which is what the archive carries, so there is nothing
left to infer.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import (CO2_COORD_FACTOR,  # noqa: E402
                                         HextorTable, OpsOlrTable)

BENCH_CO2_PPM = 280.0


def faReadLatFile(sPath):
    listRows = []
    for sLine in open(sPath):
        if sLine.strip().startswith("#") or not sLine.strip():
            continue
        listRows.append([float(t) for t in sLine.split()])
    return np.array(listRows, dtype=float)


def fdictFitOffset(aObserved, aPredicted):
    """One free additive constant; report what is left after removing it.

    A second fit, with a free SLOPE as well, separates two ways a port can be
    wrong: a constant offset is an unrecorded configuration choice and carries
    no information about the scheme, whereas a slope departure means the
    archived OLR responds to temperature differently from the table we hold.
    """
    fOffset = float(np.mean(aObserved - aPredicted))
    aResid = aObserved - (aPredicted + fOffset)
    fSlope, fIntercept = np.polyfit(aPredicted, aObserved, 1)
    aResid2 = aObserved - (fSlope * aPredicted + fIntercept)
    return {"fFittedCloudIR": -fOffset,
            "fRms": float(np.sqrt(np.mean(aResid ** 2))),
            "fMaxAbs": float(np.max(np.abs(aResid))),
            "fRangeOfObserved": float(np.ptp(aObserved)),
            "fFittedSlope": float(fSlope),
            "fRmsAfterSlopeFit": float(np.sqrt(np.mean(aResid2 ** 2)))}


def fdictPlanckSlope(aT, aOLRObserved, aOLRPredicted):
    """Effective dOLR/dT of the archive against that of the reimplementation."""
    fObs = float(np.polyfit(aT, aOLRObserved, 1)[0])
    fPre = float(np.polyfit(aT, aOLRPredicted, 1)[0])
    return {"fArchivedPlanckSlope": fObs, "fPortedPlanckSlope": fPre,
            "fRatio": fObs / fPre}


def fdictLongwaveFingerprint(aT, aOLR, oOps, oHextor, fFCO2):
    """Fit each candidate scheme's shape to the archived OLR column."""
    dictOut = {}
    aOpsal = oOps.faOLR(aT, fFCO2)
    dictOut["ops"] = fdictFitOffset(aOLR, aOpsal)
    dictOut["ops"].update(fdictPlanckSlope(aT, aOLR, aOpsal))
    aOpsPlain = oOps.faOLR(aT, fFCO2, 1.0, 1.0)
    dictOut["ops_no_molar_factor"] = fdictFitOffset(aOLR, aOpsPlain)
    aHex = np.array([oHextor.fdOLR(float(t), fFCO2) for t in aT])
    dictOut["hextor"] = fdictFitOffset(aOLR, aHex)
    return dictOut


def fdictGridFit(aLat):
    """Best equal-width and equal-area constructions for the recorded column."""
    iN = len(aLat)
    aEqualWidth = -90.0 + (np.arange(iN) + 0.5) * 180.0 / iN
    aX = -1.0 + (np.arange(iN) + 0.5) * 2.0 / iN
    aEqualArea = np.rad2deg(np.arcsin(aX))
    dictOut = {"iNumLatitudes": iN}
    for sName, aCand in (("equalWidth", aEqualWidth), ("equalArea", aEqualArea)):
        aD = aLat - aCand
        dictOut[sName] = {"fMaxAbsDeviation_deg": float(np.max(np.abs(aD))),
                          "fRmsDeviation_deg": float(np.sqrt(np.mean(aD ** 2)))}
    aNorth = aLat[aLat > 0]
    aSouth = -aLat[aLat < 0][::-1]
    dictOut["fMaxNorthSouthAsymmetry_deg"] = float(np.max(np.abs(aNorth - aSouth)))
    dictOut["iPairsDisagreeing"] = int(np.count_nonzero(aNorth != aSouth))
    dictOut["bRecordedAsIntegers"] = bool(np.all(aLat == np.round(aLat)))
    return dictOut


def flistDataRows(sPath):
    listRows, listComments = [], []
    for sLine in open(sPath):
        if sLine.strip().startswith("#"):
            listComments.append(sLine.strip())
        elif sLine.strip():
            listRows.append(sLine.split())
    return listRows, listComments


def fdictSubmissionAudit(sOpsDir):
    """Case counts, CO2 units and branch distinctness across the ops files."""
    dictProtocolCases = {"ben1": 1, "ben2": 1, "ben3": 1, "exp1": 190,
                         "exp2": 190, "exp3_cold": 57, "exp3_warm": 57,
                         "exp4_cold": 50, "exp4_warm": 50}
    dictOut = {"listFiles": []}
    for sExp, iExpected in dictProtocolCases.items():
        sPath = os.path.join(sOpsDir, sExp, "global_output.dat")
        if not os.path.exists(sPath):
            continue
        listRows, listComments = flistDataRows(sPath)
        aX = np.array([float(r[3]) for r in listRows])
        dictOut["listFiles"].append({
            "sExperiment": sExp,
            "iCases": len(listRows), "iProtocolCases": iExpected,
            "iNumericColumns": len(listRows[0]),
            "sResolvedFormat": "v1.0" if len(listRows[0]) <= 7 else "v1.1",
            "fXCO2Min": float(aX.min()), "fXCO2Max": float(aX.max()),
            "sXCO2Unit": ("mixing ratio" if aX.max() < 1.0 else "ppm"),
        })
    for sCold, sWarm in (("exp3_cold", "exp3_warm"), ("exp4_cold", "exp4_warm")):
        sA = os.path.join(sOpsDir, sCold, "global_output.dat")
        sB = os.path.join(sOpsDir, sWarm, "global_output.dat")
        if not (os.path.exists(sA) and os.path.exists(sB)):
            continue
        aA = np.array([[float(t) for t in r] for r in flistDataRows(sA)[0]])
        aB = np.array([[float(t) for t in r] for r in flistDataRows(sB)[0]])
        bSame = aA.shape == aB.shape and bool(np.array_equal(aA, aB))
        dictRow = {"sPair": "%s / %s" % (sCold, sWarm), "bIdentical": bSame}
        if aA.shape == aB.shape:
            dictRow["iRowsDiffering"] = int(np.count_nonzero(
                np.any(aA != aB, axis=1)))
            dictRow["fMaxTglobDifference_K"] = float(
                np.max(np.abs(aA[:, 4] - aB[:, 4])))
        dictOut.setdefault("listBranchPairs", []).append(dictRow)
    return dictOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--ops-dir", required=True)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oOps = OpsOlrTable(oArgs.ops_olr_table)
    oHextor = HextorTable(oArgs.hextor_table)
    fFCO2 = BENCH_CO2_PPM * 1e-6

    listBench = []
    for sBench in ("ben1", "ben2", "ben3"):
        sPath = os.path.join(oArgs.ops_dir, sBench, "case_0", "lat_output.dat")
        aDat = faReadLatFile(sPath)
        aLat, aT, aOLR = aDat[:, 0], aDat[:, 1], aDat[:, 4]
        aAsurf = aDat[:, 2]
        listBench.append({
            "sBenchmark": sBench,
            "dictLongwaveFingerprint":
                fdictLongwaveFingerprint(aT, aOLR, oOps, oHextor, fFCO2),
            "dictGrid": fdictGridFit(aLat),
            "fTmin": float(aT.min()), "fTmax": float(aT.max()),
        })

    dictOut = {
        "fCO2CoordinateFactor": CO2_COORD_FACTOR,
        "listBenchmarks": listBench,
        "dictSubmissionAudit": fdictSubmissionAudit(oArgs.ops_dir),
    }
    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("Longwave port: archived OPS OLR column predicted from its own")
    print("temperature column, one free additive constant per fit.\n")
    print("%-6s %-24s %8s %8s %10s %8s %10s"
          % ("bench", "candidate scheme", "rms", "max|res|", "fit cloudir",
             "slope", "rms w/slope"))
    for d in listBench:
        for sName, dFit in d["dictLongwaveFingerprint"].items():
            print("%-6s %-24s %8.3f %8.3f %10.2f %8.4f %10.3f"
                  % (d["sBenchmark"], sName, dFit["fRms"], dFit["fMaxAbs"],
                     dFit["fFittedCloudIR"], dFit["fFittedSlope"],
                     dFit["fRmsAfterSlopeFit"]))
        print("")

    print("Effective Planck slope, archived profile against the ported table:")
    for d in listBench:
        p = d["dictLongwaveFingerprint"]["ops"]
        print("  %-5s archive %.3f, port %.3f W/m2/K, ratio %.3f"
              % (d["sBenchmark"], p["fArchivedPlanckSlope"],
                 p["fPortedPlanckSlope"], p["fRatio"]))
    print("")

    print("Latitude grid of the archived profiles:")
    for d in listBench:
        g = d["dictGrid"]
        print("  %-5s %d points; equal-width max dev %.2f deg, equal-area max dev "
              "%.2f deg; %d of %d N/S pairs disagree, worst %.1f deg; integers: %s"
              % (d["sBenchmark"], g["iNumLatitudes"],
                 g["equalWidth"]["fMaxAbsDeviation_deg"],
                 g["equalArea"]["fMaxAbsDeviation_deg"],
                 g["iPairsDisagreeing"], g["iNumLatitudes"] // 2,
                 g["fMaxNorthSouthAsymmetry_deg"], g["bRecordedAsIntegers"]))

    print("\nSubmission audit:")
    print("  %-11s %6s %6s %5s %6s %12s %12s"
          % ("file", "cases", "proto", "cols", "format", "XCO2 min", "XCO2 unit"))
    for d in dictOut["dictSubmissionAudit"]["listFiles"]:
        print("  %-11s %6d %6d %5d %6s %12.4g %12s"
              % (d["sExperiment"], d["iCases"], d["iProtocolCases"],
                 d["iNumericColumns"], d["sResolvedFormat"], d["fXCO2Min"],
                 d["sXCO2Unit"]))
    for d in dictOut["dictSubmissionAudit"].get("listBranchPairs", []):
        print("  %s : identical = %s, rows differing = %s, max dTglob = %s K"
              % (d["sPair"], d["bIdentical"], d.get("iRowsDiffering"),
                 d.get("fMaxTglobDifference_K")))


if __name__ == "__main__":
    main()
