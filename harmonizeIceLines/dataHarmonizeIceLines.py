"""Recompute every model's ice edge from its own latitudinal temperature profile under a common definition, and split the inter-model ice-line disagreement into a definitional part and a physical part.

The three codes report an ice line using three different conventions:

  HEXTOR  annual-mean belt temperature crosses 263.15 K (-10 C), the edge
          located by LINEAR INTERPOLATION between two 10-deg belt centres.
  AVALON  annual-mean band temperature below the `T_ice` threshold; the edge
          is the latitude of the outermost such band -- SNAPPED to a node.
          The archived submission used T_ice = -10 C; the current source
          default is 0 C.
  POISE   annual-mean cell temperature at or below 273.15 K on land and
          271.15 K on ocean; the edge is SNAPPED to a cell latitude.
          (Its own albedo switches at -2 C on both surfaces, so the reported
          land ice line is not the latitude where its albedo changes.)
  OPS   annual-mean belt temperature crosses 263.15 K, located by LINEAR
          INTERPOLATION between belt centres -- the same convention as HEXTOR,
          reached independently by descent from the same ancestor rather than
          by agreement.  OPS's submission is archived under `Results/ops/`,
          so its benchmark edges are recomputed from the archive like the
          others; the locator port is additionally checked against the edge
          the submission itself reports, and against the shipped runs of the
          source snapshot.

This step re-derives the edge from Tsurf(lat) under every (threshold,
locator) combination, so that the part of the disagreement attributable to
the definition can be measured against the part attributable to the
simulated temperature field.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

THRESHOLDS_K = [263.15, 271.15, 273.15]
LOCATORS = ["interpolate", "snapToNode"]


def fdIceEdgeNH(aLat, aT, fThreshold, sLocator):
    """Equatorward edge of northern-hemisphere ice, in degrees.

    Returns 90.0 when the northern hemisphere is ice-free and 0.0 when it is
    glaciated to the equator, matching the FILLET sentinel convention.
    """
    aMaskN = aLat >= 0
    aLatN, aTN = aLat[aMaskN], aT[aMaskN]
    aIce = aTN <= fThreshold
    if not aIce.any():
        return 90.0
    if aIce.all():
        return 0.0
    if sLocator == "snapToNode":
        return float(aLatN[aIce].min())
    # Linear interpolation on the most equatorward crossing of the threshold.
    aIdx = np.where(aIce)[0]
    iFirst = aIdx.min()
    if iFirst == 0:
        return 0.0
    fT1, fT2 = aTN[iFirst - 1], aTN[iFirst]
    fL1, fL2 = aLatN[iFirst - 1], aLatN[iFirst]
    if fT2 == fT1:
        return float(fL2)
    return float(fL1 + (fL2 - fL1) * (fThreshold - fT1) / (fT2 - fT1))


def fdIcePoleward(aLat, aT, fThreshold):
    """Poleward NH edge: 90 if the polemost cell is iced, else its latitude."""
    aMaskN = aLat >= 0
    aLatN, aTN = aLat[aMaskN], aT[aMaskN]
    aIce = aTN <= fThreshold
    if not aIce.any():
        return 90.0
    return 90.0 if aIce[-1] else float(aLatN[aIce].max())


def fsClassify(fNMax, fNMin):
    if fNMax == fNMin:
        return "ice_free"
    if fNMax >= 90.0 and fNMin > 0.0:
        return "ice_cap"
    if fNMax >= 90.0 and fNMin == 0.0:
        return "snowball"
    if fNMax < 90.0 and fNMin == 0.0:
        return "ice_belt"
    return "unclassified"


def faReadLatFile(sPath):
    listRows = []
    for sLine in open(sPath):
        if sLine.strip().startswith("#") or not sLine.strip():
            continue
        listRows.append([float(t) for t in sLine.split()])
    return np.array(listRows, dtype=float)


def fdictHeaderMeta(sPath):
    dictM = {}
    for sLine in open(sPath):
        if not sLine.startswith("#"):
            break
        oM = re.match(r"#\s*(Instellation \(S_earth\)|Obliquity \(degrees\)|"
                      r"XCO2 \(ppm\)|Case number):\s*([-\d.eE+]+)", sLine)
        if oM:
            dictM[oM.group(1)] = float(oM.group(2))
    return dictM


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True,
                         help="benchmark profiles from parseFilletOutputs")
    oParser.add_argument("--hextor-exp1-dir", required=True)
    oParser.add_argument("--avalon-exp1-dir", required=True)
    oParser.add_argument("--ops-runs-json", required=True)
    oParser.add_argument("--shields-benchmarks-json", default=None)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictOut = {}

    # ---- Benchmarks: all three models, archived submissions ---------------
    dictLat = json.load(open(oArgs.lat_json))
    listBench = []
    for sKey, dictProf in sorted(dictLat.items()):
        sModel, sBench = sKey.split("/")
        aDat = np.array(dictProf["aaData"], dtype=float)
        aLat, aT = aDat[:, 0], aDat[:, 1]
        dictEdges = {}
        for fThr in THRESHOLDS_K:
            for sLoc in LOCATORS:
                dictEdges["%.2f/%s" % (fThr, sLoc)] = fdIceEdgeNH(aLat, aT, fThr, sLoc)
        listBench.append({
            "sModel": sModel, "sBenchmark": sBench,
            "iNumLatitudes": int(len(aLat)),
            "fMedianNodeSpacingNearEdge_deg": float(np.median(np.diff(aLat))),
            "dictEdges": dictEdges,
            "fSpreadAcrossDefinitions_deg":
                float(max(dictEdges.values()) - min(dictEdges.values())),
        })
    # ---- Shields-Bitz: the benchmark runs this pipeline produced ---------
    if oArgs.shields_benchmarks_json:
        for dictB in json.load(open(oArgs.shields_benchmarks_json))["listBenchmarks"]:
            aLat = np.array(dictB["aLat"], dtype=float)
            aT = np.array(dictB["aTsurf"], dtype=float)
            dictEdges = {"%.2f/%s" % (fThr, sLoc): fdIceEdgeNH(aLat, aT, fThr, sLoc)
                         for fThr in THRESHOLDS_K for sLoc in LOCATORS}
            listBench.append({
                "sModel": "shields_bitz", "sBenchmark": dictB["sBenchmark"],
                "iNumLatitudes": int(len(aLat)),
                "fMedianNodeSpacingNearEdge_deg": float(np.median(np.diff(aLat))),
                "dictEdges": dictEdges,
                "fSpreadAcrossDefinitions_deg":
                    float(max(dictEdges.values()) - min(dictEdges.values())),
                # What the model's own locator returns, for comparison with the
                # harmonised values: a different quantity on a different axis.
                "fNativeReportedEdge_deg": dictB["fIceEdgePackageDeg"],
            })

    # ---- OPS source snapshot: its own shipped runs ----------------------
    listOps = []
    for dictRun in json.load(open(oArgs.ops_runs_json))["listRuns"]:
        if "aTsurf" not in dictRun or dictRun["sStar"] not in ("Sun", "out"):
            continue
        aLat = np.array(dictRun["aLat"], dtype=float)
        aT = np.array(dictRun["aTsurf"], dtype=float)
        dictEdges = {"%.2f/%s" % (fThr, sLoc): fdIceEdgeNH(aLat, aT, fThr, sLoc)
                     for fThr in THRESHOLDS_K for sLoc in LOCATORS}
        listReported = [f for f in dictRun["dictIceLines"]["listEdgesDeg"] if f >= 0]
        listOps.append({
            "sModel": "ops", "sRun": dictRun["sFile"],
            "bShippedInputDeck": dictRun["bShippedInputDeck"],
            "bProtocolBenchmark": False,
            "iNumLatitudes": int(len(aLat)),
            "sNativeDefinition": "263.15/interpolate",
            "dictEdges": dictEdges,
            "fSpreadAcrossDefinitions_deg":
                float(max(dictEdges.values()) - min(dictEdges.values())),
            "sReportedState": dictRun["dictIceLines"]["sState"],
            "fReportedEdgeNH": listReported[0] if listReported else None,
            "fLocatorPortResidual_deg":
                (dictEdges["263.15/interpolate"] - listReported[0]
                 if listReported else None),
        })
    # Validate the locator against the edge the ops submission itself reports.
    listLocator = []
    for dictB in listBench:
        dictRow = {"sModel": dictB["sModel"], "sBenchmark": dictB["sBenchmark"],
                   "fRecomputedAtNativeDefinition": None}
        sNative = {"avalon": "263.15/snapToNode", "hextor": "263.15/interpolate",
                   "poise": "273.15/snapToNode",
                   "ops": "263.15/interpolate",
                   "shields_bitz": "271.15/interpolate"}.get(dictB["sModel"])
        if sNative:
            dictRow["sNativeDefinition"] = sNative
            dictRow["fRecomputedAtNativeDefinition"] = dictB["dictEdges"][sNative]
        listLocator.append(dictRow)
    dictOut["listLocatorCheck"] = listLocator

    dictOut["listBenchmarkEdges"] = listBench
    dictOut["listSnapshotRunEdges"] = listOps
    dictOut["listOpsRunEdges"] = listOps

    # ---- Experiment 1: HEXTOR and AVALON, per-case profiles ---------------
    dictExp1 = {}
    for sModel, sDir, sPattern in [
            ("hextor", oArgs.hextor_exp1_dir, "lat_output_HEXTOR_exp1_*.dat"),
            ("avalon", oArgs.avalon_exp1_dir, "lat_output_AVALON_exp1_*.dat")]:
        listCases = []
        for sPath in sorted(glob.glob(os.path.join(sDir, sPattern))):
            aDat = faReadLatFile(sPath)
            if aDat.size == 0:
                continue
            aLat, aT = aDat[:, 0], aDat[:, 1]
            dictM = fdictHeaderMeta(sPath)
            dictRow = {
                "sFile": os.path.basename(sPath),
                "fInst": dictM.get("Instellation (S_earth)"),
                "fObl": dictM.get("Obliquity (degrees)"),
                "iCase": int(dictM.get("Case number", -1)),
            }
            for fThr in THRESHOLDS_K:
                for sLoc in LOCATORS:
                    sTag = "%.2f/%s" % (fThr, sLoc)
                    fMin = fdIceEdgeNH(aLat, aT, fThr, sLoc)
                    fMax = fdIcePoleward(aLat, aT, fThr)
                    dictRow["edge_" + sTag] = fMin
                    dictRow["state_" + sTag] = fsClassify(fMax, fMin)
            listCases.append(dictRow)
        dictExp1[sModel] = listCases
    dictOut["dictExperiment1Edges"] = dictExp1

    # ---- Sensitivity summary ----------------------------------------------
    listSens = []
    for sModel, listCases in dictExp1.items():
        sRef = "263.15/interpolate"
        for fThr in THRESHOLDS_K:
            for sLoc in LOCATORS:
                sTag = "%.2f/%s" % (fThr, sLoc)
                aE = np.array([c["edge_" + sTag] for c in listCases])
                aR = np.array([c["edge_" + sRef] for c in listCases])
                aFinite = (aE > 0) & (aE < 90) & (aR > 0) & (aR < 90)
                iStateDiff = sum(1 for c in listCases
                                 if c["state_" + sTag] != c["state_" + sRef])
                listSens.append({
                    "sModel": sModel, "sDefinition": sTag,
                    "iCases": len(listCases),
                    "iIceFree": int(np.count_nonzero(aE == 90.0)),
                    "iSnowball": int(np.count_nonzero(aE == 0.0)),
                    "fMeanEdgeWhereFinite": float(aE[aFinite].mean()) if aFinite.any() else None,
                    "fMeanShiftVsReference_deg":
                        float((aE - aR)[aFinite].mean()) if aFinite.any() else None,
                    "fMaxShiftVsReference_deg":
                        float(np.abs(aE - aR)[aFinite].max()) if aFinite.any() else None,
                    "iStateChangesVsReference": iStateDiff,
                })
    dictOut["listDefinitionSensitivity"] = listSens

    # ---- Definitional versus physical inter-model difference --------------
    # Match HEXTOR and AVALON Experiment 1 cases on (instellation, obliquity)
    # and compare the ice edge under each model's own convention with the edge
    # under one harmonised convention.
    def fdictKey(listCases):
        return {(int(round(c["fInst"] / 0.0125)), round(c["fObl"], 1)): c
                for c in listCases if c["fInst"] is not None}

    dH, dA = fdictKey(dictExp1["hextor"]), fdictKey(dictExp1["avalon"])
    listKeys = sorted(set(dH) & set(dA))
    listPairs = []
    for tKey in listKeys:
        cH, cA = dH[tKey], dA[tKey]
        listPairs.append({
            "fInst": cH["fInst"], "fObl": cH["fObl"],
            # native conventions: HEXTOR -10 C interpolated, AVALON 0 C snapped
            "fEdgeHextorNative": cH["edge_263.15/interpolate"],
            "fEdgeAvalonNative": cA["edge_273.15/snapToNode"],
            # harmonised: both at -10 C with linear interpolation
            "fEdgeHextorHarmonised": cH["edge_263.15/interpolate"],
            "fEdgeAvalonHarmonised": cA["edge_263.15/interpolate"],
            "sStateHextorNative": cH["state_263.15/interpolate"],
            "sStateAvalonNative": cA["state_273.15/snapToNode"],
            "sStateHextorHarmonised": cH["state_263.15/interpolate"],
            "sStateAvalonHarmonised": cA["state_263.15/interpolate"],
        })

    aNative = np.array([[p["fEdgeHextorNative"], p["fEdgeAvalonNative"]]
                        for p in listPairs])
    aHarm = np.array([[p["fEdgeHextorHarmonised"], p["fEdgeAvalonHarmonised"]]
                      for p in listPairs])
    aBothFinite = ((aNative[:, 0] > 0) & (aNative[:, 0] < 90)
                   & (aNative[:, 1] > 0) & (aNative[:, 1] < 90)
                   & (aHarm[:, 1] > 0) & (aHarm[:, 1] < 90))
    dictOut["dictHextorAvalonComparison"] = {
        "iMatchedCases": len(listPairs),
        "iCasesWithFiniteEdgeInBoth": int(np.count_nonzero(aBothFinite)),
        "fMeanAbsDifferenceNative_deg":
            float(np.abs(aNative[aBothFinite, 0] - aNative[aBothFinite, 1]).mean())
            if aBothFinite.any() else None,
        "fMeanAbsDifferenceHarmonised_deg":
            float(np.abs(aHarm[aBothFinite, 0] - aHarm[aBothFinite, 1]).mean())
            if aBothFinite.any() else None,
        "iStateDisagreementsNative":
            sum(1 for p in listPairs
                if p["sStateHextorNative"] != p["sStateAvalonNative"]),
        "iStateDisagreementsHarmonised":
            sum(1 for p in listPairs
                if p["sStateHextorHarmonised"] != p["sStateAvalonHarmonised"]),
        "listPairs": listPairs,
    }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f)

    print("Benchmark ice edges, NH, under six definitions (deg):")
    print("%-7s %-5s %6s %s" % ("model", "bench", "spread",
                                "  ".join("%13s" % ("%.2f/%s" % (t, l[:4]))
                                          for t in THRESHOLDS_K for l in LOCATORS)))
    for d in listBench:
        print("%-7s %-5s %6.2f %s"
              % (d["sModel"], d["sBenchmark"], d["fSpreadAcrossDefinitions_deg"],
                 "  ".join("%13.2f" % d["dictEdges"]["%.2f/%s" % (t, l)]
                           for t in THRESHOLDS_K for l in LOCATORS)))

    print("\nOPS source snapshot, from its own shipped runs (not FILLET cases):")
    print("%-28s %6s %8s %8s %s"
          % ("run", "spread", "native", "reported", "  ".join(
              "%13s" % ("%.2f/%s" % (t, l[:4]))
              for t in THRESHOLDS_K for l in LOCATORS)))
    for d in listOps:
        print("%-28s %6.2f %8.2f %8s %s"
              % (d["sRun"], d["fSpreadAcrossDefinitions_deg"],
                 d["dictEdges"]["263.15/interpolate"],
                 "%.2f" % d["fReportedEdgeNH"] if d["fReportedEdgeNH"] is not None
                 else d["sReportedState"],
                 "  ".join("%13.2f" % d["dictEdges"]["%.2f/%s" % (t, l)]
                           for t in THRESHOLDS_K for l in LOCATORS)))

    print("\nExperiment 1 definition sensitivity (reference: 263.15 K, interpolated):")
    print("%-7s %-22s %6s %6s %6s %10s %10s %8s"
          % ("model", "definition", "cases", "free", "sball", "meanEdge",
             "meanShift", "dStates"))
    for d in listSens:
        print("%-7s %-22s %6d %6d %6d %10s %10s %8d"
              % (d["sModel"], d["sDefinition"], d["iCases"], d["iIceFree"],
                 d["iSnowball"],
                 "%.2f" % d["fMeanEdgeWhereFinite"] if d["fMeanEdgeWhereFinite"] else "-",
                 "%+.2f" % d["fMeanShiftVsReference_deg"] if d["fMeanShiftVsReference_deg"] is not None else "-",
                 d["iStateChangesVsReference"]))

    c = dictOut["dictHextorAvalonComparison"]
    print("\nHEXTOR vs AVALON, Experiment 1 (%d matched cases, %d with a finite edge in both):"
          % (c["iMatchedCases"], c["iCasesWithFiniteEdgeInBoth"]))
    print("  mean |ice-edge difference|, each model's own convention : %s deg"
          % ("%.2f" % c["fMeanAbsDifferenceNative_deg"]
             if c["fMeanAbsDifferenceNative_deg"] else "-"))
    print("  mean |ice-edge difference|, harmonised at -10 C        : %s deg"
          % ("%.2f" % c["fMeanAbsDifferenceHarmonised_deg"]
             if c["fMeanAbsDifferenceHarmonised_deg"] else "-"))
    print("  climate-state disagreements, native     : %d / %d"
          % (c["iStateDisagreementsNative"], c["iMatchedCases"]))
    print("  climate-state disagreements, harmonised : %d / %d"
          % (c["iStateDisagreementsHarmonised"], c["iMatchedCases"]))


if __name__ == "__main__":
    main()
