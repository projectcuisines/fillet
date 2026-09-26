"""Quantify how far each code's current repository HEAD has drifted from the FILLET submission archived in Results/, case by case across every benchmark and experiment.

Two of the three codes ship their own regenerated FILLET outputs alongside the
source (HEXTOR in `fillet/`, AVALON in `experiments/`).  Comparing those with
the archived submission measures, for each experiment, how much of the
inter-model spread reported by FILLET would change if the submissions were
regenerated with today's source.

Cases are matched on (instellation, obliquity, CO2) rather than on case index,
because the two archives do not always order their loops the same way.
"""

import argparse
import json
import os
import re

import numpy as np


def flistRows(sPath):
    listRows = []
    for sLine in open(sPath):
        if sLine.strip().startswith("#") or not sLine.strip():
            continue
        listNum, sTag = [], None
        for sTok in sLine.split():
            try:
                listNum.append(float(sTok))
            except ValueError:
                sTag = sTok
        listRows.append((listNum, sTag))
    return listRows


def fdictIndexByKey(listRows, iInst=1, iObl=2, iCO2=3, iT=4, sBranchFilter=None):
    """Key each row by (inst, obl, CO2) at the coarsest precision any writer uses.

    HEXTOR prints the instellation with a Fortran `f4.2` edit descriptor, so its
    Experiment 1 grid spacing of 0.025 and Experiment 3 spacing of 0.0125 are
    quantised to 0.01 in the file, and the rounding of a half-way value differs
    between the archived run and the current one (0.825 -> 0.83 versus 0.82).
    Matching therefore uses 0.025 bins in instellation, which is unambiguous for
    every grid in the protocol while absorbing that print artefact.
    """
    dictOut = {}
    for listNum, sTag in listRows:
        if sBranchFilter is not None and sTag != sBranchFilter:
            continue
        tKey = (int(round(listNum[iInst] / 0.0125)), round(listNum[iObl], 1),
                round(np.log10(max(listNum[iCO2], 1e-12)), 2))
        dictOut[tKey] = (listNum[iT], listNum)
    return dictOut


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


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--results-dir", required=True)
    oParser.add_argument("--hextor-repo", required=True)
    oParser.add_argument("--avalon-repo", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listCmp = []

    # --- HEXTOR: Results/hextor/<exp> vs <repo>/fillet/<exp> -----------------
    # The repo splits exp3/exp4 by branch inside one file; the archive splits
    # them into _cold/_warm directories.
    listHexPairs = [("ben1", "ben1", None), ("ben2", "ben2", None),
                    ("ben3", "ben3", None), ("exp1", "exp1", None),
                    ("exp1a", "exp1a", None), ("exp2", "exp2", None),
                    ("exp2a", "exp2a", None)]
    for sArc, sRepo, _ in listHexPairs:
        sA = os.path.join(oArgs.results_dir, "hextor", sArc, "global_output.dat")
        sR = os.path.join(oArgs.hextor_repo, "fillet", sRepo,
                          "global_output_HEXTOR_%s.dat" % sRepo)
        if not (os.path.exists(sA) and os.path.exists(sR)):
            continue
        dA = fdictIndexByKey(flistRows(sA))
        dR = fdictIndexByKey(flistRows(sR))
        listCmp.append(fdictCompare("hextor", sArc, dA, dR, 5, 6))

    # --- AVALON: Results/avalon/<exp> vs <repo>/experiments/<exp> ------------
    for sExp in ("ben1", "ben2", "ben3", "exp1", "exp2", "exp2a", "exp3", "exp4"):
        sA = os.path.join(oArgs.results_dir, "avalon", sExp, "global_output.dat")
        sR = os.path.join(oArgs.avalon_repo, "experiments", sExp,
                          "global_output_AVALON_%s.dat" % sExp)
        if not (os.path.exists(sA) and os.path.exists(sR)):
            continue
        for sBranch in ([None] if sExp not in ("exp3", "exp4")
                        else ["cooling", "warming"]):
            dA = fdictIndexByKey(flistRows(sA), sBranchFilter=sBranch)
            dR = fdictIndexByKey(flistRows(sR), sBranchFilter=sBranch)
            sLabel = sExp if sBranch is None else "%s_%s" % (sExp, sBranch)
            listCmp.append(fdictCompare("avalon", sLabel, dA, dR, 5, 6))

    with open(oArgs.out, "w") as f:
        json.dump({"listComparisons": listCmp}, f, indent=1)

    print("%-7s %-14s %6s %8s %8s %8s %8s %6s"
          % ("model", "experiment", "n", "meandT", "maxdT", "rmsdT",
             "max|dIce|", "state"))
    for d in listCmp:
        print("%-7s %-14s %6d %+8.3f %+8.3f %8.3f %8.2f %6d"
              % (d["sModel"], d["sExperiment"], d["iMatchedCases"],
                 d["fMeanDeltaTglob"], d["fMaxAbsDeltaTglob"],
                 d["fRmsDeltaTglob"], d["fMaxAbsDeltaIceEdge"],
                 d["iStateChanges"]))


def fdictCompare(sModel, sExp, dA, dR, iNMaxCol, iNMinCol):
    listKeys = sorted(set(dA) & set(dR))
    aDT, aDIce, iStateChg = [], [], 0
    listChanged = []
    for tKey in listKeys:
        fTa, listA = dA[tKey]
        fTr, listR = dR[tKey]
        aDT.append(fTr - fTa)
        aDIce.append(listR[iNMinCol] - listA[iNMinCol])
        sSa = fsClassify(listA[iNMaxCol], listA[iNMinCol])
        sSr = fsClassify(listR[iNMaxCol], listR[iNMinCol])
        if sSa != sSr:
            iStateChg += 1
            listChanged.append({"tKey": list(tKey), "sArchived": sSa,
                                "sCurrent": sSr,
                                "fTglobArchived": fTa, "fTglobCurrent": fTr})
    aDT = np.array(aDT) if aDT else np.array([np.nan])
    aDIce = np.array(aDIce) if aDIce else np.array([np.nan])
    return {
        "sModel": sModel, "sExperiment": sExp,
        "iArchivedCases": len(dA), "iCurrentCases": len(dR),
        "iMatchedCases": len(listKeys),
        "fMeanDeltaTglob": float(np.nanmean(aDT)),
        "fMaxAbsDeltaTglob": float(aDT[np.nanargmax(np.abs(aDT))]),
        "fRmsDeltaTglob": float(np.sqrt(np.nanmean(aDT ** 2))),
        "fMaxAbsDeltaIceEdge": float(np.nanmax(np.abs(aDIce))),
        "iStateChanges": iStateChg,
        "fStateChangeFraction": (iStateChg / len(listKeys)) if listKeys else None,
        "listStateChanges": listChanged[:40],
    }


if __name__ == "__main__":
    main()
