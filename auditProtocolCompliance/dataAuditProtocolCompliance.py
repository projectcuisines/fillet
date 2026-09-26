"""Audit each model's archived FILLET submission against the v1.0 and v1.1 protocol tables, and test the eight-value ice-edge column ordering against sign constraints rather than assuming it.

Four independent checks are performed:

1. GRID COMPLIANCE -- the instellation / obliquity / semi-major-axis / CO2
   grids actually present in each `global_output.dat`, compared with
   Deitrick+2023 Table 3 and Barnes+2025 Table 3.

2. COLUMN-ORDER FALSIFICATION -- the v1.1 eight-value ice-edge block admits
   two orderings in the wild (protocol order: N-land, N-sea, S-land, S-sea;
   POISE order: N-land, S-land, N-sea, S-sea).  The two are distinguished
   WITHOUT reference to the writer by counting violations of hemisphere sign
   constraints that any physically meaningful assignment must satisfy:
       NH edges in [0, 90],  SH edges in [-90, 0],  max >= min.
   The ordering that produces zero violations is the one on disk.

3. REPORTED-VS-EFFECTIVE DIFFUSION -- HEXTOR rescales its diffusion
   coefficient at runtime (driver.f `diffadj`) but reports the namelist value
   `d0`.  The effective value is recomputed here from the documented formula.
   OPS rescales it too, with a different reference atmosphere, an extra
   rotation-rate factor and a latent-heat factor that makes the coefficient a
   function of the climate state, so both are recomputed and compared.  For
   OPS the recomputation can also be checked directly: its Benchmark 1 run
   prints the coefficient it actually used, against the constant its
   submission files.

4. BRANCH-ORDER CONSISTENCY -- a bistable system started warm cannot settle
   colder than the same system started cold.  Matching Experiment 1 (warm
   start) against Experiment 2 (cold start) on (instellation, obliquity), and
   the warm against the cold branch of Experiments 3 and 4, any case where the
   cold start ends WARMER is either a mislabelled branch, a non-converged run
   or a pair of files that are not the experiment they claim to be.  This is a
   check every submission can be held to without reference to any model's
   source, and it costs nothing to run before filing.  Where the stated
   labelling fails, the SWAPPED labelling is scored the same way, so that a
   mislabelling can be distinguished from a genuinely non-monotonic pair by
   falsification rather than by assumption -- the same method this step
   already applies to the ice-edge column ordering.
"""

import argparse
import json
import os

import numpy as np

# --- Protocol grids (Deitrick+2023 Table 3; Barnes+2025 Table 3) -------------
PROTOCOL = {
    "exp1":  {"inst": (0.80, 1.25, 0.025), "obl": (0, 90, 10), "n": 19 * 10},
    "exp2":  {"inst": (1.05, 1.50, 0.025), "obl": (0, 90, 10), "n": 19 * 10},
    "exp1a": {"axis": (0.875, 1.10, 0.0125), "obl": (0, 90, 10), "n": 19 * 10},
    "exp2a": {"axis": (0.80, 0.975, 0.0125), "obl": (0, 90, 10), "n": 15 * 10},
    "exp3":  {"inst": (0.80, 1.50, 0.0125), "obl": None, "n": 57},
    "exp4_v10": {"xco2": (50.0, 5050.0, 100.0), "n": 51},
    "exp4_v11": {"xco2_log": (1.0, 1.0e5), "n": 50},
}

V11_ORDER = ["NMaxLand", "NMinLand", "NMaxSea", "NMinSea",
             "SMaxLand", "SMinLand", "SMaxSea", "SMinSea"]
POISE_ORDER = ["NMaxLand", "NMinLand", "SMaxLand", "SMinLand",
               "NMaxSea", "NMinSea", "SMaxSea", "SMinSea"]


def fiCountSignViolations(aBlock, listOrder):
    """Count rows whose ice-edge values break hemisphere sign/ordering rules.

    `aBlock` is (nrow, 8) of raw on-disk values; `listOrder` names each column.
    """
    dictCol = {sName: aBlock[:, i] for i, sName in enumerate(listOrder)}
    iViol = 0
    for sSurf in ("Land", "Sea"):
        aNMax, aNMin = dictCol["NMax" + sSurf], dictCol["NMin" + sSurf]
        aSMax, aSMin = dictCol["SMax" + sSurf], dictCol["SMin" + sSurf]
        aBad = ((aNMax < 0) | (aNMax > 90) | (aNMin < 0) | (aNMin > 90)
                | (aSMax > 0) | (aSMax < -90) | (aSMin > 0) | (aSMin < -90)
                | (aNMax < aNMin) | (aSMax < aSMin))
        iViol += int(np.count_nonzero(aBad))
    return iViol


def fdEffectiveHextorDiffusion(fD0, fPg0, fFCO2, fRotRatio=1.0):
    """HEXTOR effective diffusion after `diffadj` (driver.f lines 1607-1621).

    d = d0 * pg0 * (avemol0/avemol)^2 * (hcp/hcp0) * (rot0/rot)^2
    with avemol0 = 28.89 amu, hcp0 = 0.2401, hcp_N2 = 0.2484, hcp_CO2 = 0.2105.
    """
    fPCO2 = fFCO2 * fPg0
    fPN2 = fPg0 - fPCO2
    fAveMol = (28.0 * fPN2 + 44.0 * fPCO2) / fPg0
    fHcp = (0.2484 * fPN2 + 0.2105 * fPCO2) / fPg0
    return fD0 * fPg0 * (28.89 / fAveMol) ** 2 * (fHcp / 0.2401) * fRotRatio ** 2


def fdictGridReport(listRecs, sExp):
    aInst = np.array([r["fInst"] for r in listRecs])
    aObl = np.array([r["fObl"] for r in listRecs])
    aCO2 = np.array([r["fXCO2ppm"] for r in listRecs])
    aUInst, aUObl, aUCO2 = np.unique(np.round(aInst, 6)), np.unique(aObl), np.unique(np.round(aCO2, 6))
    dictOut = {
        "iNumCases": len(listRecs),
        "iNumUniqueInst": len(aUInst),
        "fInstMin": float(aInst.min()), "fInstMax": float(aInst.max()),
        "iNumUniqueObl": len(aUObl),
        "fOblMin": float(aObl.min()), "fOblMax": float(aObl.max()),
        "iNumUniqueXCO2": len(aUCO2),
        "fXCO2Min": float(aCO2.min()), "fXCO2Max": float(aCO2.max()),
    }
    if len(aUInst) > 2:
        dictOut["fInstStep"] = float(np.median(np.diff(aUInst)))
    if len(aUCO2) > 2:
        aRatio = aUCO2[1:] / aUCO2[:-1]
        dictOut["fXCO2LogRatio"] = float(np.median(aRatio))
        dictOut["bXCO2Logarithmic"] = bool(np.ptp(aRatio) < 0.05 * np.median(aRatio))
    return dictOut


def fdEffectiveOpsDiffusion(fD0, fPdryBar, fPCO2Bar, fPH2OBar=0.0,
                            fLatentFactor=1.0, fDayLengthS=86400.0):
    """OPS's run-time rescaling (driver.f90), for comparison with HEXTOR's.

    Same shape as HEXTOR's `diffadj` but referenced to Earth AIR rather than
    to N2, and carrying three terms HEXTOR has no analogue for: an
    (Omega_0/Omega)^2 rotation factor, the water-vapour partial pressure in
    both the pressure ratio and the mean molecular weight, and a latent-heat
    factor that scales the coefficient by the column water content relative to
    Earth's whenever the global mean is below 288 K.  The reference choice is
    why the two codes' corrections differ in sign at Earth-like conditions,
    and the latent-heat factor is why OPS's coefficient is a function of the
    climate state rather than of the atmosphere alone.
    """
    fAveMol0 = 28.965
    fPtot = fPdryBar + fPCO2Bar + fPH2OBar
    fAveMol = (28.965 * fPdryBar + 44.0 * fPCO2Bar + 18.0 * fPH2OBar) / fPtot
    fHcp0, fHcpCO2, fHcpH2O = 0.2401, 0.2105, 0.4450
    fHcp = (fHcp0 * fPdryBar + fPCO2Bar * fHcpCO2 + fPH2OBar * fHcpH2O) / fPtot
    fRot0, fRot = 7.27e-5, (360.0 / fDayLengthS) * (np.pi / 180.0)
    return (fD0 * fPtot / 1.0 * (fAveMol0 / fAveMol) ** 2 * fLatentFactor
            * (fHcp / fHcp0) * (fRot0 / fRot) ** 2)


def flistBranchOrderViolations(listRecs):
    """Cases where a cold start ends warmer than the matching warm start."""
    listOut = []
    listPairs = [("exp1", "exp2", "fInst", "warm start vs cold start", "exp2"),
                 ("exp3_warm", "exp3_cold", "fInst", "Experiment 3 branches", "exp3"),
                 ("exp4_warm", "exp4_cold", "fXCO2ppm", "Experiment 4 branches", None)]
    for sModel in sorted({r["sModel"] for r in listRecs}):
        for sWarm, sCold, sKey, sLabel, sColdGrid in listPairs:
            dW = {(round(r[sKey], 6), round(r["fObl"], 1)): r for r in listRecs
                  if r["sModel"] == sModel and r["sExperiment"] == sWarm}
            dC = {(round(r[sKey], 6), round(r["fObl"], 1)): r for r in listRecs
                  if r["sModel"] == sModel and r["sExperiment"] == sCold}
            listK = sorted(set(dW) & set(dC))
            if not listK:
                continue
            aD = np.array([dW[k]["fTglob"] - dC[k]["fTglob"] for k in listK])

            # A violation inside the protocol grid is a statement about the
            # model; one outside it is a statement about cases the submission
            # added.  The two are counted apart rather than together.
            aInside = np.ones(len(listK), dtype=bool)
            tRange = (PROTOCOL.get(sColdGrid) or {}).get("inst") if sColdGrid else None
            if tRange:
                aAbscissa = np.array([k[0] for k in listK])
                aInside = ((aAbscissa >= tRange[0] - 1e-9)
                           & (aAbscissa <= tRange[1] + 1e-9))
            aViol = aD < -1e-9

            listOut.append({
                "iViolationsInsideProtocolGrid": int(np.count_nonzero(aViol & aInside)),
                "iViolationsOutsideProtocolGrid": int(np.count_nonzero(aViol & ~aInside)),
                "fWorstViolationInsideGrid_K": float(
                    max(0.0, -aD[aViol & aInside].min()) if (aViol & aInside).any() else 0.0),
                "fWorstViolationOutsideGrid_K": float(
                    max(0.0, -aD[aViol & ~aInside].min()) if (aViol & ~aInside).any() else 0.0),
                "sColdBranchProtocolRange": ("%.3f to %.3f" % tRange[:2]
                                             if tRange else "not constrained here"),
                "sModel": sModel, "sComparison": sLabel,
                "iMatchedCases": len(listK),
                "iColdStartWarmer": int(np.count_nonzero(aD < -1e-9)),
                "iIdenticalCases": int(np.count_nonzero(np.abs(aD) < 1e-9)),
                "fWorstColdMinusWarm_K": float(max(0.0, -aD.min())),
                "fMeanWarmMinusCold_K": float(aD.mean()),
                "bBranchesIdentical": bool(np.all(np.abs(aD) < 1e-9)),
                # The same count with the two labels exchanged.  A clean
                # mislabelling shows many violations one way and none the other.
                "iViolationsAsLabelled": int(np.count_nonzero(aD < -1e-9)),
                "fWorstViolationAsLabelled_K": float(max(0.0, -aD.min())),
                "iViolationsIfSwapped": int(np.count_nonzero(aD > 1e-9)),
                "fWorstViolationIfSwapped_K": float(max(0.0, aD.max())),
            })
    return listOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--global-json", required=True)
    oParser.add_argument("--results-dir", required=True)
    oParser.add_argument("--ops-runs-json", default=None)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listRecs = json.load(open(oArgs.global_json))["listRecords"]

    # --- 1. grid compliance -------------------------------------------------
    listGrid = []
    setKeys = sorted({(r["sModel"], r["sExperiment"]) for r in listRecs})
    for sModel, sExp in setKeys:
        listSub = [r for r in listRecs if r["sModel"] == sModel
                   and r["sExperiment"] == sExp]
        dictRep = fdictGridReport(listSub, sExp)
        dictRep.update({"sModel": sModel, "sExperiment": sExp})
        listGrid.append(dictRep)

    # --- 2. column-order falsification -------------------------------------
    # Re-read the raw files so the test is independent of step A01's layout
    # declarations: we only need the eight numeric ice columns in file order.
    import os
    listOrderTest = []
    for sModel, sExp in setKeys:
        sPath = os.path.join(oArgs.results_dir, sModel, sExp, "global_output.dat")
        listRows = []
        for sLine in open(sPath):
            if sLine.strip().startswith("#") or not sLine.strip():
                continue
            listNum = [float(t) for t in sLine.split()
                       if _fbIsNum(t)]
            listRows.append(listNum)
        aRows = np.array(listRows, dtype=float)
        if aRows.shape[1] != 15:
            continue                      # only the 8-value files are ambiguous
        aBlock = aRows[:, 5:13]
        listOrderTest.append({
            "sModel": sModel,
            "sExperiment": sExp,
            "iRows": int(aRows.shape[0]),
            "iViolationsProtocolOrder": fiCountSignViolations(aBlock, V11_ORDER),
            "iViolationsPoiseOrder": fiCountSignViolations(aBlock, POISE_ORDER),
        })

    # --- 3. HEXTOR reported vs effective diffusion -------------------------
    listDiff = []
    for fD0, sCase, fFCO2 in [(0.38, "Ben1 (d0=0.38, 280 ppm)", 2.8e-4),
                              (0.50, "Ben2/3 and Exp1-3 (d0=0.50, 280 ppm)", 2.8e-4),
                              (0.50, "Exp4 low end (d0=0.50, 1 ppm)", 1.0e-6),
                              (0.50, "Exp4 high end (d0=0.50, 1e5 ppm)", 1.0e-1)]:
        fEff = fdEffectiveHextorDiffusion(fD0, 1.0, fFCO2)
        listDiff.append({
            "sCase": sCase, "fD0Reported": fD0,
            "fDEffective": round(fEff, 5),
            "fFractionalOffset": round(fEff / fD0 - 1.0, 5),
        })

    listDiffOps = []
    for fD0, sCase, fPCO2 in [(0.50, "Ben/Exp1-3 (d0=0.50, 280 ppm)", 2.8e-4),
                              (0.50, "Exp4 low end (d0=0.50, 1 ppm)", 1.0e-6),
                              (0.50, "Exp4 high end (d0=0.50, 1e5 ppm)", 1.0e-1)]:
        fEff = fdEffectiveOpsDiffusion(fD0, 1.0, fPCO2)
        listDiffOps.append({
            "sCase": sCase, "fD0Reported": fD0,
            "fDEffective": round(fEff, 5),
            "fFractionalOffset": round(fEff / fD0 - 1.0, 5),
        })

    # What OPS's own Benchmark 1 run reports it used, against what the
    # submission files and against the dry recomputation.  The gap between the
    # dry recomputation and the run is the part carried by the water-vapour
    # and latent-heat terms, which no archived file records.
    dictOpsMeasured = None
    if oArgs.ops_runs_json and os.path.exists(oArgs.ops_runs_json):
        dictRuns = json.load(open(oArgs.ops_runs_json))
        listBenchRun = [r for r in dictRuns["listRuns"]
                        if "Benchmarks/1" in r["sFile"]
                        and r.get("fDiffusionReported")]
        if listBenchRun:
            fMeasured = listBenchRun[0]["fDiffusionReported"]
            fD0 = dictRuns["dictSourceConstants"]["fDiffusionD0"]
            fDry = fdEffectiveOpsDiffusion(fD0, 1.0, 2.8e-4)
            dictOpsMeasured = {
                "sRun": listBenchRun[0]["sFile"],
                "fD0Source": fD0,
                "fDFiledInSubmission": 0.50,
                "fDReportedByRun": fMeasured,
                "fDDryRecomputation": round(fDry, 5),
                "fFractionalOffsetVsFiled": round(fMeasured / 0.50 - 1.0, 5),
                "fImpliedMoistAndLatentFactor": round(fMeasured / fDry, 5),
            }

    listBranch = flistBranchOrderViolations(listRecs)

    dictOut = {
        "listGridCompliance": listGrid,
        "listColumnOrderTest": listOrderTest,
        "listHextorDiffusion": listDiff,
        "listOpsDiffusion": listDiffOps,
        "dictOpsDiffusionMeasured": dictOpsMeasured,
        "listBranchOrderTest": listBranch,
        "dictProtocolGrids": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                  for kk, vv in v.items()}
                              for k, v in PROTOCOL.items()},
    }
    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("grid rows: %d ; order-tested files: %d" % (len(listGrid), len(listOrderTest)))
    for d in listOrderTest:
        print("  %-7s %-10s protocol-order violations=%4d   poise-order violations=%4d"
              % (d["sModel"], d["sExperiment"],
                 d["iViolationsProtocolOrder"], d["iViolationsPoiseOrder"]))
    for d in listDiff:
        print("  HEXTOR %s -> D_eff = %.4f (%+.1f%%)"
              % (d["sCase"], d["fDEffective"], 100 * d["fFractionalOffset"]))
    for d in listDiffOps:
        print("  OPS  %s -> D_eff = %.4f (%+.1f%%)"
              % (d["sCase"], d["fDEffective"], 100 * d["fFractionalOffset"]))
    print("\nBranch-order consistency (a cold start must not end warmer).")
    print("Violations are counted under the labels as filed and under the")
    print("labels exchanged; the correct labelling is the one with fewer:")
    print("  %-7s %-26s %7s %9s %11s %11s %11s"
          % ("model", "comparison", "matched", "identical", "viol. filed",
             "worst (K)", "viol. swap"))
    for d in listBranch:
        print("  %-7s %-26s %7d %9d %11d %11.2f %11d (worst %.2f K)"
              % (d["sModel"], d["sComparison"], d["iMatchedCases"],
                 d["iIdenticalCases"], d["iViolationsAsLabelled"],
                 d["fWorstViolationAsLabelled_K"], d["iViolationsIfSwapped"],
                 d["fWorstViolationIfSwapped_K"]))


def _fbIsNum(sTok):
    try:
        float(sTok)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
