"""Close the global energy budget of each model's FILLET benchmarks from its own latitudinal output, and validate the reimplemented OLR parameterisations against the archived profiles.

Three things are computed per (model, benchmark):

1. AREA-WEIGHTED GLOBAL MEANS of Tsurf, Asurf, ATOA and OLR, recomputed from
   `lat_output.dat` with the correct cos(lat) weighting for each model's own
   grid, so that grids of 18, 90 and 150 cells are compared on equal terms.

2. ENERGY-BUDGET CLOSURE.  In equilibrium a global EBM must satisfy
   <OLR> = (S/4)(1 - <alpha>_insolation-weighted).  Evaluating both sides
   exposes how much of a model's agreement on Tglob is achieved with the
   right energy throughput and how much by compensating errors in albedo
   and OLR.

3. PARAMETERISATION VALIDATION.  The OLR column of each archived profile is
   compared with the OLR that this analysis' reimplementation of that model's
   own scheme predicts from the archived Tsurf column.  Agreement confirms
   the port; a residual is itself diagnostic (e.g. HEXTOR's `cloudir` offset).

OPS is archived under `Results/ops/` and is treated as a fourth submitter
throughout.  Its benchmark files are the older v1.0 product and carry neither
a diffusion nor an OLR column, so its global OLR is recomputed from its own
latitudinal profile rather than read.  Its `cloudir` offset is not recorded in
any archived file, so the port residual is reported twice: against the value
in the source snapshot's input deck, and about the best-fit constant, which is
the quantity that actually measures the SHAPE of the scheme.

The source snapshot's own shipped runs are validated separately at the end.
That test is where the CO2-coordinate convention is decided, because there the
configuration and the output belong to each other: predicting one of those
runs' global OLR with OPS's 44/28 CO2 coordinate factor and without it
separates the two readings, which nothing in the archive can do.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import (CO2_COORD_FACTOR,  # noqa: E402
                                         HextorTable, OpsOlrTable,
                                         faOLRAvalon, faOLRPoiseFixed)

# OPS is filed in the archive under this directory name.
OPS_MODEL = "ops"

S_EARTH = 1361.0

# Instellation and OLR settings actually used for each archived benchmark,
# read from the shipped input decks.
BENCH_SETTINGS = {
    ("avalon", "ben1"): {"fInst": 1.0, "fA": 210.0, "fB": 2.0, "fD": 0.52},
    ("avalon", "ben2"): {"fInst": 1.0, "fA": 210.0, "fB": 2.0, "fD": 0.50},
    ("avalon", "ben3"): {"fInst": 1.0, "fA": 210.0, "fB": 2.0, "fD": 0.50},
    ("poise", "ben1"):  {"fInst": 1.004825, "fA": 203.30, "fB": 2.09, "fD": 0.58},
    ("poise", "ben2"):  {"fInst": 0.999899, "fA": 204.05, "fB": 2.09, "fD": 0.50},
    ("poise", "ben3"):  {"fInst": 0.999899, "fA": 204.05, "fB": 2.09, "fD": 0.50},
    ("hextor", "ben1"): {"fInst": 1.0, "fCloudIR": -6.3, "fD0": 0.38, "fFCO2": 2.8e-4},
    ("hextor", "ben2"): {"fInst": 1.0, "fCloudIR": -6.3, "fD0": 0.50, "fFCO2": 2.8e-4},
    ("hextor", "ben3"): {"fInst": 1.0, "fCloudIR": -6.3, "fD0": 0.50, "fFCO2": 2.8e-4},
    # OPS: instellation 1.00 as filed.  The cloud offset is diagnosed per belt
    # at run time rather than set, so the value here is the source's
    # initialisation and the fitted offset beside it is the measurement.
    ("ops", "ben1"): {"fInst": 1.0, "fCloudIR": -4.0, "fD0": 0.50, "fFCO2": 2.8e-4},
    ("ops", "ben2"): {"fInst": 1.0, "fCloudIR": -4.0, "fD0": 0.50, "fFCO2": 2.8e-4},
    ("ops", "ben3"): {"fInst": 1.0, "fCloudIR": -4.0, "fD0": 0.50, "fFCO2": 2.8e-4},
}


def faCellWeights(aLatDeg):
    """Area weights for a 1-D latitude grid given only the node latitudes.

    Cell edges are placed midway between nodes in sin(lat), with the outermost
    edges at the poles, so the weights sum to one for any of the three grids
    (18 equal-width HEXTOR belts, 90 equal-area AVALON bands, 150 POISE cells)
    without assuming which construction was used.
    """
    aX = np.sin(np.deg2rad(np.asarray(aLatDeg, dtype=float)))
    aEdge = np.empty(len(aX) + 1)
    aEdge[1:-1] = 0.5 * (aX[:-1] + aX[1:])
    aEdge[0], aEdge[-1] = -1.0, 1.0
    aW = np.abs(np.diff(aEdge))
    return aW / aW.sum()


def faAnnualInsolation(aLatDeg, fS0, fOblDeg, iNLambda=720):
    """Annual-mean insolation on a circular orbit (Berger 1978 daily formula)."""
    aPhi = np.deg2rad(np.asarray(aLatDeg, dtype=float))
    fEps = np.deg2rad(fOblDeg)
    aQ = np.zeros_like(aPhi)
    aLam = (np.arange(iNLambda) + 0.5) * 2 * np.pi / iNLambda
    aDec = np.arcsin(np.sin(fEps) * np.sin(aLam))
    for i, fPhi in enumerate(aPhi):
        aT = -np.tan(fPhi) * np.tan(aDec)
        aH0 = np.where(aT <= -1.0, np.pi, np.where(aT >= 1.0, 0.0,
                                                   np.arccos(np.clip(aT, -1, 1))))
        aQ[i] = np.mean(aH0 * np.sin(fPhi) * np.sin(aDec)
                        + np.cos(fPhi) * np.cos(aDec) * np.sin(aH0)) * fS0 / np.pi
    return aQ


def flistOpsReference(dictRuns, oOpsTable, fCloudIR, fPdryBar):
    """Validate the OPS OLR port against every solar run's own converged answer.

    Only solar-host runs are used: the OLR table is shared across host stars
    but the cloud offset is not recorded in the output files.  The three
    `out/Benchmarks/` runs are the ones that matter, because they are the
    FILLET benchmark configurations themselves, run by the model that filed
    the submission -- the port can be held against the model's own answer under
    a deck we can read, which the archive alone never allows.

    The offset is no longer a namelist scalar in this version of OPS: the model
    diagnoses it per belt from a surface convective heat flux,
    `cloudir = max(-6.8 ln(F_c/F_E + 1), -10)`.  `fCloudIR` here is the
    initialisation the source sets before that diagnosis begins, and the
    residual's fitted offset is what actually measures the converged value.
    """
    listOut = []
    for dictRun in dictRuns["listRuns"]:
        bSolar = (dictRun["sStar"] in ("Sun", "out")
                  or "Benchmarks" in dictRun["sFile"])
        if not bSolar or "aTsurf" not in dictRun:
            continue
        aLat = np.array(dictRun["aLat"], dtype=float)
        aT = np.array(dictRun["aTsurf"], dtype=float)
        aW = faCellWeights(aLat)
        fPCO2 = dictRun["fPCO2barReported"]
        fFCO2 = fPCO2 / (fPdryBar + fPCO2)
        dictRow = {
            "sFile": dictRun["sFile"],
            "bShippedInputDeck": dictRun["bShippedInputDeck"],
            "fInstellation": dictRun["fInstellation"],
            "fObliquityDeg": dictRun["fObliquityDeg"],
            "iGeography": dictRun["iGeography"],
            "fTglobReported": dictRun["fTglobReported"],
            "fTglobAreaWeighted": dictRun["fTglobAreaWeighted"],
            "fAlbedoReported": dictRun["fAlbedoReported"],
            "fOLRReported": dictRun["fOLRReported"],
            "fInsolationReported": dictRun["fInsolationReported"],
            "fCO2MixingRatio": fFCO2,
        }
        for sTag, fFactor in (("WithMolarMassFactor", CO2_COORD_FACTOR),
                              ("AsPartialPressure", 1.0)):
            aPred = oOpsTable.faOLR(aT, fFCO2, fPdryBar, fFactor) - fCloudIR
            dictRow["fOLRPredicted" + sTag] = float(np.sum(aW * aPred))
            dictRow["fOLRResidual" + sTag] = float(dictRun["fOLRReported"]
                                                   - np.sum(aW * aPred))
        fASR = dictRun["fInsolationReported"] * (1.0 - dictRun["fAlbedoReported"])
        dictRow["fAbsorbedShortwave"] = float(fASR)
        dictRow["fNetImbalance"] = float(fASR - dictRun["fOLRReported"])
        listOut.append(dictRow)
    return listOut


def flistShieldsBitzRows(dictShields, dictObl):
    """Shields-Bitz benchmark rows, from the runs this pipeline produced.

    Everything the archived submissions supply as a filed column is computed
    here from the run itself, so the quantities mean the same thing.  There is
    no longwave-port residual: the model was executed rather than
    reimplemented, and A + B T is its scheme, so a port check would be
    comparing the scheme against itself.
    """
    listOut = []
    for dictB in dictShields["listBenchmarks"]:
        aLat = np.array(dictB["aLat"], dtype=float)
        aW = faCellWeights(aLat)
        aATOA = np.array(dictB["aATOA"], dtype=float)
        aQ = np.array(dictB["aInsolation"], dtype=float)
        fQmean = float(np.sum(aW * aQ))
        fASR = float(np.sum(aW * aQ * (1.0 - aATOA)))
        listOut.append({
            "sModel": "shields_bitz", "sBenchmark": dictB["sBenchmark"],
            "iNumLatitudes": dictB["iNumLatitudes"],
            "fTglobAreaWeighted": dictB["fTglobK"],
            "fTglobReported": dictB["fTglobK"],
            "fTglobUnweightedMean": float(np.mean(dictB["aTsurf"])),
            "fTglobPreviousOrbitK": dictB.get("fTglobPreviousOrbitK"),
            "fAsurfAreaWeighted": dictB["fAlbedoGlobal"],
            "fATOAAreaWeighted": dictB["fAlbedoGlobal"],
            "fATOAInsolationWeighted": float(1.0 - fASR / fQmean),
            "fOLRAreaWeighted": dictB["fOLRGlobal"],
            "fOLRReported": dictB["fOLRGlobal"],
            "fMeanInsolation": fQmean,
            "fAbsorbedShortwave": fASR,
            "fNetImbalance": float(fASR - dictB["fOLRGlobal"]),
            "fEquatorPoleGradient": float(
                np.array(dictB["aTsurf"])[np.argmin(np.abs(aLat))]
                - 0.5 * (dictB["aTsurf"][0] + dictB["aTsurf"][-1])),
            "fTmin": float(np.min(dictB["aTsurf"])),
            "fTmax": float(np.max(dictB["aTsurf"])),
            "fOLRPortMeanResidual": None,
            "fOLRPortMaxAbsResidual": None,
            "fOLRPortRMSResidual": None,
            "fOLRPortFittedOffset": None,
            "fOLRPortRMSAboutFittedOffset": None,
            "fImpliedCloudIR": None,
            "bRunHere": True,
            "aLat": dictB["aLat"], "aTsurf": dictB["aTsurf"],
            "aAsurf": dictB["aAsurf"], "aATOA": dictB["aATOA"],
            "aOLR": dictB["aOLR"], "aOLRPredicted": dictB["aOLR"],
            "aInsolation": dictB["aInsolation"],
        })
    return listOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--global-json", required=True)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--ops-runs-json", required=True)
    oParser.add_argument("--shields-benchmarks-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictLat = json.load(open(oArgs.lat_json))
    listGlob = json.load(open(oArgs.global_json))["listRecords"]
    oTab = HextorTable(oArgs.hextor_table)
    oOpsTable = OpsOlrTable(oArgs.ops_olr_table)
    dictRuns = json.load(open(oArgs.ops_runs_json))

    dictObl = {"ben1": 23.5, "ben2": 23.5, "ben3": 60.0}

    listOut = []
    for sKey, dictProf in sorted(dictLat.items()):
        sModel, sBench = sKey.split("/")
        aDat = np.array(dictProf["aaData"], dtype=float)
        aLat, aT, aAsurf, aATOA, aOLR = (aDat[:, 0], aDat[:, 1], aDat[:, 2],
                                         aDat[:, 3], aDat[:, 4])
        aW = faCellWeights(aLat)
        dictSet = BENCH_SETTINGS[(sModel, sBench)]
        fS0 = dictSet["fInst"] * S_EARTH
        fObl = dictObl[sBench]
        aQ = faAnnualInsolation(aLat, fS0, fObl)

        fTglob = float(np.sum(aW * aT))
        fAsurfArea = float(np.sum(aW * aAsurf))
        fATOAArea = float(np.sum(aW * aATOA))
        fOLRArea = float(np.sum(aW * aOLR))
        fQmean = float(np.sum(aW * aQ))
        fASR = float(np.sum(aW * aQ * (1.0 - aATOA)))
        fATOAInsolWeighted = 1.0 - fASR / fQmean

        # Reimplemented OLR from the archived temperatures.
        if sModel == "avalon":
            aOLRPred = faOLRAvalon(aT, 280e-6, fA=dictSet["fA"], fB=dictSet["fB"])
        elif sModel == "poise":
            aOLRPred = faOLRPoiseFixed(aT, fA=dictSet["fA"], fB=dictSet["fB"])
        elif sModel == OPS_MODEL:
            aOLRPred = oOpsTable.faOLR(aT, dictSet["fFCO2"]) - dictSet["fCloudIR"]
        else:
            aOLRPred = np.array([oTab.fdOLR(float(t), dictSet["fFCO2"]) for t in aT])
            aOLRPred = aOLRPred - dictSet["fCloudIR"]   # driver.f: ir = ir - cloudir

        aResid = aOLR - aOLRPred
        # The mean residual is whatever additive constant the archived run used
        # that we do not know; the scatter ABOUT it is what tests the shape of
        # the scheme, and is the only part of the residual a wrong scheme
        # cannot absorb.
        fFittedOffset = float(np.mean(aResid))

        # Archived global row, for comparison with the recomputed means.
        listRow = [r for r in listGlob if r["sModel"] == sModel
                   and r["sExperiment"] == sBench]
        dictRow = listRow[0] if listRow else {}

        listOut.append({
            "sModel": sModel, "sBenchmark": sBench,
            "iNumLatitudes": int(len(aLat)),
            "fTglobAreaWeighted": fTglob,
            "fTglobReported": dictRow.get("fTglob"),
            "fTglobUnweightedMean": float(np.mean(aT)),
            "fAsurfAreaWeighted": fAsurfArea,
            "fATOAAreaWeighted": fATOAArea,
            "fATOAInsolationWeighted": float(fATOAInsolWeighted),
            "fOLRAreaWeighted": fOLRArea,
            "fOLRReported": dictRow.get("fOLRglob"),
            "fMeanInsolation": fQmean,
            "fAbsorbedShortwave": fASR,
            "fNetImbalance": float(fASR - fOLRArea),
            "fEquatorPoleGradient": float(aT[np.argmin(np.abs(aLat))]
                                          - 0.5 * (aT[0] + aT[-1])),
            "fTmin": float(aT.min()), "fTmax": float(aT.max()),
            "fOLRPortMeanResidual": float(np.mean(aResid)),
            "fOLRPortMaxAbsResidual": float(np.max(np.abs(aResid))),
            "fOLRPortRMSResidual": float(np.sqrt(np.mean(aResid ** 2))),
            "fOLRPortFittedOffset": fFittedOffset,
            "fOLRPortRMSAboutFittedOffset":
                float(np.sqrt(np.mean((aResid - fFittedOffset) ** 2))),
            "fImpliedCloudIR": (dictSet["fCloudIR"] - fFittedOffset
                                if "fCloudIR" in dictSet else None),
            "aLat": aLat.tolist(), "aTsurf": aT.tolist(),
            "aAsurf": aAsurf.tolist(), "aATOA": aATOA.tolist(),
            "aOLR": aOLR.tolist(), "aOLRPredicted": aOLRPred.tolist(),
            "aInsolation": aQ.tolist(),
        })

    listOut.extend(flistShieldsBitzRows(
        json.load(open(oArgs.shields_benchmarks_json)), dictObl))

    listOps = flistOpsReference(
        dictRuns, oOpsTable,
        fCloudIR=dictRuns["dictSourceConstants"]["fCloudIRInitialWm2"],
        fPdryBar=dictRuns["dictInputDeck"]["Pdry"])

    with open(oArgs.out, "w") as f:
        json.dump({"listBenchmarks": listOut, "listOpsReference": listOps}, f)

    print("%-7s %-5s %4s %8s %8s %8s %8s %8s %8s %8s"
          % ("model", "bench", "nlat", "Tglob", "Tglob_rep", "a_surf",
             "a_TOA", "ASR", "OLR", "imbal"))
    for d in listOut:
        print("%-7s %-5s %4d %8.2f %8s %8.4f %8.4f %8.2f %8.2f %+8.2f"
              % (d["sModel"], d["sBenchmark"], d["iNumLatitudes"],
                 d["fTglobAreaWeighted"],
                 "%.2f" % d["fTglobReported"] if d["fTglobReported"] is not None
                 else "-",
                 d["fAsurfAreaWeighted"], d["fATOAInsolationWeighted"],
                 d["fAbsorbedShortwave"], d["fOLRAreaWeighted"],
                 d["fNetImbalance"]))
    print("\nOLR parameterisation port residuals (archived minus reimplemented).")
    print("`rms about fit` removes the unknown additive constant and is the")
    print("test of the scheme's shape:")
    for d in listOut:
        if d["fOLRPortMeanResidual"] is None:
            print("  %-7s %-5s model executed here, not reimplemented"
                  % (d["sModel"], d["sBenchmark"]))
            continue
        print("  %-7s %-5s mean %+7.3f  rms %6.3f  max|.| %6.3f  "
              "fitted offset %+7.3f  rms about fit %6.3f W/m2"
              % (d["sModel"], d["sBenchmark"], d["fOLRPortMeanResidual"],
                 d["fOLRPortRMSResidual"], d["fOLRPortMaxAbsResidual"],
                 d["fOLRPortFittedOffset"], d["fOLRPortRMSAboutFittedOffset"]))

    print("\nOPS source snapshot: its own shipped runs, not FILLET cases.")
    print("Only the run marked * was produced by the shipped input_ebm.dat; for the")
    print("others `cloudir`, the geography and the source version are unrecorded, so")
    print("their residuals measure the missing deck, not the port.")
    print("%-28s %8s %7s %7s %7s %11s %10s"
          % ("run", "S/Searth", "Tglob", "alpha", "OLR", "resid 44/28", "resid pCO2"))
    for d in listOps:
        print("%-28s %8.4f %7.2f %7.3f %7.1f %+11.2f %+10.2f"
              % (d["sFile"] + (" *" if d["bShippedInputDeck"] else ""),
                 d["fInstellation"], d["fTglobReported"],
                 d["fAlbedoReported"], d["fOLRReported"],
                 d["fOLRResidualWithMolarMassFactor"],
                 d["fOLRResidualAsPartialPressure"]))
    listDeck = [d for d in listOps if d["bShippedInputDeck"]]
    for d in listDeck:
        print("\nPort validation on %s: predicted global OLR %.2f against the model's"
              % (d["sFile"], d["fOLRPredictedWithMolarMassFactor"]))
        print("own %.2f W/m2, a residual of %+.2f W/m2 when the table is queried at"
              % (d["fOLRReported"], d["fOLRResidualWithMolarMassFactor"]))
        print("44/28 times the CO2 partial pressure, and %+.2f W/m2 when it is queried"
              % d["fOLRResidualAsPartialPressure"])
        print("at the partial pressure itself.  Energy budget: ASR %.2f, OLR %.2f,"
              % (d["fAbsorbedShortwave"], d["fOLRReported"]))
        print("imbalance %+.2f W/m2." % d["fNetImbalance"])


if __name__ == "__main__":
    main()
