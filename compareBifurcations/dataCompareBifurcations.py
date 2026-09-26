"""Extract the glaciation and deglaciation thresholds and the width of the bistable region from the Experiment 3 (instellation) and Experiment 4 (CO2) hysteresis loops of each model, and test which OLR parameterisation VPLanet-POISE used for Experiment 4.

The bifurcation diagram is the most demanding FILLET product because it
depends on both branches being computed on the same abscissa.  This step
therefore reports, alongside the thresholds:

  * whether the two branches of a model's loop share an instellation grid
    (VPLanet-POISE's do not: its cold-start runs were configured at
     a = 1.02 au and its warm-start runs at a = 1.00 au);
  * the CO2 abundance at which each model's warm branch collapses, in the
    units the protocol asks for;
  * whether the two branches are in fact distinct runs at all, since a
    submission whose warm and cold files are identical reports a bistable
    width of one grid step for a reason that has nothing to do with physics;
  * a hysteresis width derived instead from the Experiment 1 / Experiment 2
    PAIR, which is a warm start and a cold start on the same grid and is
    therefore a second, independent measurement of the same quantity -- the
    only one available for a model whose Experiment 3 branches are degenerate.
    Which of the two files is the warm start is decided from the data rather
    than from the file name, because at least one submission has them
    interchanged (see the branch-order test in step A02);
  * an inversion for the POISE Experiment 4 OLR scheme.  The benchmarks and
    Experiments 1-3 were run with a fixed linearisation (bCalcAB = 0), under
    which CO2 cannot act at all; Experiment 4 must therefore have enabled
    bCalcAB, which in poise.c also switches the top-of-atmosphere albedo
    parameterisation.  The candidate schemes are distinguished by matching
    the archived Tglob(CO2) curve against a zero-dimensional balance.
"""

import argparse
import json
import os
import sys

import numpy as np

# OPS is archived under the directory name `ops`.
LIST_MODELS = ("avalon", "hextor", "poise", "ops", "shields_bitz")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import (faOLRPoiseWK97, faOLRPoiseHM16,  # noqa: E402
                                         faOLRPoiseSMS09, faOLRPoiseFixed)

S_EARTH = 1361.0


def flistShieldsBitzExp3(sSchemePath):
    """The repository's shipped instellation sweep, as Experiment 3 records.

    Shields-Bitz filed no submission, so its Experiment 3 warm branch is the
    sweep `warmstart_sweep` writes at the package defaults -- the same 25% land
    surface the protocol prescribes, started warm and stepped down in
    instellation.  There is no cold branch to pair it with, so no hysteresis
    width is defined for it and none is reported.

    The state is read from the sweep's own ice-edge sentinels, 0 and 90, which
    mean fully glaciated and ice-free.  Those two values are independent of the
    latitude axis the locator uses, so the axis error documented in step A04
    does not reach the classification.
    """
    if not sSchemePath or not os.path.exists(sSchemePath):
        return []
    dictSweep = json.load(open(sSchemePath)).get("dictShippedSweep")
    if not dictSweep:
        return []
    listOut = []
    for fInst, fEdge, fT in zip(dictSweep["aInstellation"],
                                dictSweep["aIceEdgeDeg"],
                                dictSweep["aTglobK"]):
        sState = ("snowball" if fEdge <= 0.0
                  else "ice_free" if fEdge >= 90.0 else "partial")
        listOut.append({"sModel": "shields_bitz", "sExperiment": "exp3_warm",
                        "sBranch": "cooling", "fInst": float(fInst),
                        "fTglob": float(fT), "sStateNH": sState,
                        "fIceLineN": float(fEdge)})
    return listOut


def fdictThresholds(listRecs, sX, sBranch):
    """Locate the transition in a monotone sweep of the control parameter."""
    listS = sorted(listRecs, key=lambda r: r[sX])
    aX = np.array([r[sX] for r in listS])
    aT = np.array([r["fTglob"] for r in listS])
    listState = [r["sStateNH"] for r in listS]
    aSnow = np.array([s == "snowball" for s in listState])
    aFree = np.array([s == "ice_free" for s in listState])

    dictOut = {
        "sBranch": sBranch, "iCases": len(listS),
        "fXMin": float(aX.min()), "fXMax": float(aX.max()),
        "iSnowball": int(np.count_nonzero(aSnow)),
        "iIceFree": int(np.count_nonzero(aFree)),
        "iPartial": int(np.count_nonzero(~(aSnow | aFree))),
        "fMaxSnowballX": float(aX[aSnow].max()) if aSnow.any() else None,
        "fMinNonSnowballX": float(aX[~aSnow].min()) if (~aSnow).any() else None,
        "fMaxTemperatureJump": float(np.abs(np.diff(aT)).max()),
        "fXAtLargestJump": float(0.5 * (aX[int(np.argmax(np.abs(np.diff(aT))))]
                                        + aX[int(np.argmax(np.abs(np.diff(aT)))) + 1])),
        "aX": aX.tolist(), "aTglob": aT.tolist(), "saStates": listState,
    }
    return dictOut


def fdictBranchDistinctness(dictM):
    """Are the two branches distinct runs, or the same numbers filed twice?"""
    if "warm" not in dictM or "cold" not in dictM:
        return None
    aW, aC = np.array(dictM["warm"]["aTglob"]), np.array(dictM["cold"]["aTglob"])
    if aW.shape != aC.shape:
        return {"bComparable": False}
    aD = np.abs(aW - aC)
    return {"bComparable": True,
            "iCases": int(aD.size),
            "iIdenticalCases": int(np.count_nonzero(aD < 1e-9)),
            "fMaxAbsDifference_K": float(aD.max()),
            "bBranchesIdentical": bool(np.all(aD < 1e-9))}


def fdictExp12Hysteresis(listRecs, sModel):
    """Bistable width per obliquity from the Experiment 1 / 2 pair.

    Experiments 1 and 2 are a warm start and a cold start on a shared
    (instellation, obliquity) grid, so their disagreement measures the same
    bistable interval Experiment 3 traces at one obliquity.  Which file is the
    warm start is taken from the data -- the one that is warmer on the matched
    cases -- so a mislabelled submission is measured correctly rather than
    backwards.
    """
    dA = {(round(r["fInst"], 6), round(r["fObl"], 1)): r for r in listRecs
          if r["sModel"] == sModel and r["sExperiment"] == "exp1"}
    dB = {(round(r["fInst"], 6), round(r["fObl"], 1)): r for r in listRecs
          if r["sModel"] == sModel and r["sExperiment"] == "exp2"}
    listK = sorted(set(dA) & set(dB))
    if not listK:
        return None
    fMean = float(np.mean([dA[k]["fTglob"] - dB[k]["fTglob"] for k in listK]))
    dW, dC = (dA, dB) if fMean >= 0 else (dB, dA)
    dictOut = {"iMatchedCases": len(listK),
               "sWarmStartFile": "exp1" if fMean >= 0 else "exp2",
               "bFilesAsLabelled": bool(fMean >= 0),
               "listByObliquity": []}
    for fObl in sorted({k[1] for k in listK}):
        listO = sorted([k for k in listK if k[1] == fObl])
        aInst = np.array([k[0] for k in listO])
        aDiffer = np.array([dW[k]["sStateNH"] != dC[k]["sStateNH"] for k in listO])
        dictRow = {"fObliquityDeg": fObl, "iCases": len(listO),
                   "iBistableCases": int(np.count_nonzero(aDiffer))}
        if aDiffer.any():
            dictRow["fBistableInstMin"] = float(aInst[aDiffer].min())
            dictRow["fBistableInstMax"] = float(aInst[aDiffer].max())
            dictRow["fBistableWidthInst"] = float(aInst[aDiffer].max()
                                                  - aInst[aDiffer].min())
        else:
            dictRow["fBistableWidthInst"] = 0.0
        dictOut["listByObliquity"].append(dictRow)
    aW = np.array([d["fBistableWidthInst"] for d in dictOut["listByObliquity"]])
    aAllInst = np.array(sorted({k[0] for k in listK}))
    # The two experiments only overlap over part of the instellation axis, so a
    # measured width can be censored by the overlap rather than by the physics.
    fOverlap = float(aAllInst.max() - aAllInst.min())
    dictOut["fOverlapWidthInst"] = fOverlap
    dictOut["fMaxBistableWidthInst"] = float(aW.max())
    dictOut["fMeanBistableWidthInst"] = float(aW.mean())
    dictOut["bWidthCensoredByOverlap"] = bool(aW.max() >= fOverlap - 1e-9)
    return dictOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--global-json", required=True)
    oParser.add_argument("--shields-scheme-json", default=None)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listRecs = json.load(open(oArgs.global_json))["listRecords"]
    listRecs = listRecs + flistShieldsBitzExp3(oArgs.shields_scheme_json)
    dictOut = {"dictExp3": {}, "dictExp4": {}}

    # ---- Experiment 3: instellation ---------------------------------------
    for sModel in LIST_MODELS:
        dictM = {}
        if sModel == "avalon":
            listAll = [r for r in listRecs if r["sModel"] == sModel
                       and r["sExperiment"] == "exp3"]
            dictM["warm"] = fdictThresholds(
                [r for r in listAll if r["sBranch"] == "cooling"], "fInst", "warm")
            dictM["cold"] = fdictThresholds(
                [r for r in listAll if r["sBranch"] == "warming"], "fInst", "cold")
        else:
            for sB, sExp in (("warm", "exp3_warm"), ("cold", "exp3_cold")):
                listSub = [r for r in listRecs if r["sModel"] == sModel
                           and r["sExperiment"] == sExp]
                if listSub:
                    dictM[sB] = fdictThresholds(listSub, "fInst", sB)
        if "warm" in dictM and "cold" in dictM:
            aW = np.array(dictM["warm"]["aX"])
            aC = np.array(dictM["cold"]["aX"])
            dictM["dictBranchGrid"] = {
                "bSameGrid": bool(len(aW) == len(aC)
                                  and np.allclose(np.sort(aW), np.sort(aC), atol=1e-4)),
                "fGridRatioColdOverWarm": float(np.median(np.sort(aC)) / np.median(np.sort(aW))),
                "iWarmPoints": int(len(aW)), "iColdPoints": int(len(aC)),
            }
            fGl = dictM["warm"]["fMaxSnowballX"]      # glaciation from the warm side
            fDe = dictM["cold"]["fMinNonSnowballX"]   # deglaciation from the cold side
            dictM["dictHysteresis"] = {
                "fGlaciationInst": fGl, "fDeglaciationInst": fDe,
                "fBistableWidthInst": (fDe - fGl) if (fGl and fDe) else None,
                "fBistableWidthAbsorbedFlux_Wm2":
                    ((fDe - fGl) * S_EARTH / 4.0 * 0.7) if (fGl and fDe) else None,
            }
        dictM["dictBranchDistinctness"] = fdictBranchDistinctness(dictM)
        dictOut["dictExp3"][sModel] = dictM

    # ---- Experiment 4: CO2 -------------------------------------------------
    for sModel in LIST_MODELS:
        dictM = {}
        if sModel == "avalon":
            listAll = [r for r in listRecs if r["sModel"] == sModel
                       and r["sExperiment"] == "exp4"]
            dictM["warm"] = fdictThresholds(
                [r for r in listAll if r["sBranch"] == "cooling"], "fXCO2ppm", "warm")
            dictM["cold"] = fdictThresholds(
                [r for r in listAll if r["sBranch"] == "warming"], "fXCO2ppm", "cold")
        else:
            for sB, sExp in (("warm", "exp4_warm"), ("cold", "exp4_cold")):
                listSub = [r for r in listRecs if r["sModel"] == sModel
                           and r["sExperiment"] == sExp]
                if listSub:
                    dictM[sB] = fdictThresholds(listSub, "fXCO2ppm", sB)
        if "warm" in dictM and "cold" in dictM:
            fGl = dictM["warm"]["fMaxSnowballX"]
            fDe = dictM["cold"]["fMinNonSnowballX"]
            dictM["dictHysteresis"] = {
                "fGlaciationCO2ppm": fGl, "fDeglaciationCO2ppm": fDe,
                "fBistableWidthDecades":
                    float(np.log10(fDe / fGl)) if (fGl and fDe and fGl > 0) else None,
            }
        dictM["dictBranchDistinctness"] = fdictBranchDistinctness(dictM)
        dictOut["dictExp4"][sModel] = dictM

    # ---- Hysteresis from the Experiment 1 / 2 pair ------------------------
    dictOut["dictExp12Hysteresis"] = {}
    for sModel in LIST_MODELS:
        dictH = fdictExp12Hysteresis(listRecs, sModel)
        if dictH:
            dictOut["dictExp12Hysteresis"][sModel] = dictH

    # ---- Which OLR scheme did POISE use for Experiment 4? -----------------
    listP4 = sorted([r for r in listRecs if r["sModel"] == "poise"
                     and r["sExperiment"] == "exp4_warm"], key=lambda r: r["fXCO2ppm"])
    aCO2 = np.array([r["fXCO2ppm"] for r in listP4]) * 1e-6
    aTg = np.array([r["fTglob"] for r in listP4])
    aOLR = np.array([r["fOLRglob"] for r in listP4])
    dictFit = {}
    for sName, fn in (("wk97", faOLRPoiseWK97), ("hm16", faOLRPoiseHM16),
                      ("sms09", lambda t, p: faOLRPoiseSMS09(t)),
                      ("fixed", lambda t, p: faOLRPoiseFixed(t))):
        aPred = np.array([float(fn(np.array([t]), float(p))[0])
                          for t, p in zip(aTg, aCO2)])
        aRes = aOLR - aPred
        dictFit[sName] = {
            "fMeanResidual": float(aRes.mean()),
            "fRmsResidual": float(np.sqrt((aRes ** 2).mean())),
            "fMaxAbsResidual": float(np.abs(aRes).max()),
        }
    dictOut["dictPoiseExp4OLRIdentification"] = {
        "iCases": len(listP4),
        "dictCandidateFits": dictFit,
        "sBestCandidate": min(dictFit, key=lambda k: dictFit[k]["fRmsResidual"]),
        "sNote": ("The reported OLR is a global mean over a non-uniform "
                  "temperature field, so a zero-dimensional evaluation at "
                  "Tglob carries a Jensen bias; the ranking, not the absolute "
                  "residual, is the discriminant."),
    }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f)

    print("=== Experiment 3: instellation hysteresis ===")
    print("%-7s %6s %8s %8s %10s %10s %12s %10s"
          % ("model", "branch", "Smin", "Smax", "snowball", "icefree",
             "maxdT/step", "at S"))
    for sModel, dictM in dictOut["dictExp3"].items():
        for sB in ("warm", "cold"):
            if sB not in dictM:
                continue
            d = dictM[sB]
            print("%-7s %6s %8.4f %8.4f %10d %10d %12.2f %10.4f"
                  % (sModel, sB, d["fXMin"], d["fXMax"], d["iSnowball"],
                     d["iIceFree"], d["fMaxTemperatureJump"], d["fXAtLargestJump"]))
        if "dictHysteresis" in dictM:
            h, g = dictM["dictHysteresis"], dictM["dictBranchGrid"]
            print("        -> glaciation S=%s  deglaciation S=%s  bistable width=%s "
                  "(%s W/m2 absorbed);  branches share a grid: %s (cold/warm=%.4f)"
                  % (_fmt(h["fGlaciationInst"]), _fmt(h["fDeglaciationInst"]),
                     _fmt(h["fBistableWidthInst"]),
                     _fmt(h["fBistableWidthAbsorbedFlux_Wm2"], "%.1f"),
                     g["bSameGrid"], g["fGridRatioColdOverWarm"]))

    print("\n=== Experiment 4: CO2 hysteresis ===")
    print("%-7s %6s %10s %10s %10s %10s %12s"
          % ("model", "branch", "CO2min", "CO2max", "snowball", "icefree", "maxdT/step"))
    for sModel, dictM in dictOut["dictExp4"].items():
        for sB in ("warm", "cold"):
            if sB not in dictM:
                continue
            d = dictM[sB]
            print("%-7s %6s %10.4g %10.4g %10d %10d %12.2f"
                  % (sModel, sB, d["fXMin"], d["fXMax"], d["iSnowball"],
                     d["iIceFree"], d["fMaxTemperatureJump"]))
        if "dictHysteresis" in dictM:
            h = dictM["dictHysteresis"]
            print("        -> glaciation CO2=%s ppm  deglaciation CO2=%s ppm  "
                  "bistable width=%s decades"
                  % (_fmt(h["fGlaciationCO2ppm"], "%.4g"),
                     _fmt(h["fDeglaciationCO2ppm"], "%.4g"),
                     _fmt(h["fBistableWidthDecades"], "%.3f")))

    print("\n=== Are the two branches distinct runs? ===")
    for sExp, sKey in (("Experiment 3", "dictExp3"), ("Experiment 4", "dictExp4")):
        for sModel, dictM in dictOut[sKey].items():
            d = dictM.get("dictBranchDistinctness")
            if not d or not d.get("bComparable"):
                continue
            print("  %-12s %-7s %3d of %3d cases identical, max |dTglob| = %.3f K%s"
                  % (sExp, sModel, d["iIdenticalCases"], d["iCases"],
                     d["fMaxAbsDifference_K"],
                     "   <-- SAME RUN FILED TWICE" if d["bBranchesIdentical"] else ""))

    print("\n=== Hysteresis from the Experiment 1 / 2 pair ===")
    print("  %-7s %-11s %8s %10s %10s %9s %s"
          % ("model", "warm file", "matched", "max width", "mean width",
             "overlap", "note"))
    for sModel, d in dictOut["dictExp12Hysteresis"].items():
        print("  %-7s %-11s %8d %10.4f %10.4f %9.4f %s"
              % (sModel, d["sWarmStartFile"], d["iMatchedCases"],
                 d["fMaxBistableWidthInst"], d["fMeanBistableWidthInst"],
                 d["fOverlapWidthInst"],
                 ("width censored by the overlap"
                  if d["bWidthCensoredByOverlap"] else "")
                 + ("" if d["bFilesAsLabelled"] else "; files interchanged")))

    print("\n=== Which OLR scheme reproduces POISE's Experiment 4? ===")
    d = dictOut["dictPoiseExp4OLRIdentification"]
    for sName, f in d["dictCandidateFits"].items():
        print("  %-6s mean %+9.2f   rms %9.2f   max|.| %9.2f W/m2"
              % (sName, f["fMeanResidual"], f["fRmsResidual"], f["fMaxAbsResidual"]))
    print("  best candidate: %s" % d["sBestCandidate"])


def _fmt(v, s="%.4f"):
    return "--" if v is None else s % v


if __name__ == "__main__":
    main()
