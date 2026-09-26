"""Test whether HEXTOR's archived exp2a grid is a duplicate of its exp1a grid, and characterise the Experiment 3 instellation spacing of each model."""

import argparse
import json

import numpy as np


def faLoadNumeric(sPath):
    listRows = []
    for sLine in open(sPath):
        if sLine.strip().startswith("#") or not sLine.strip():
            continue
        listRows.append([float(t) for t in sLine.split()])
    return np.array(listRows, dtype=float)


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--results-dir", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictOut = {}

    a1a = faLoadNumeric("%s/hextor/exp1a/global_output.dat" % oArgs.results_dir)
    a2a = faLoadNumeric("%s/hextor/exp2a/global_output.dat" % oArgs.results_dir)
    dictOut["dictHextorExp1aVsExp2a"] = {
        "aShape1a": list(a1a.shape),
        "aShape2a": list(a2a.shape),
        "bIdenticalInstellationGrid": bool(
            a1a.shape == a2a.shape
            and np.allclose(np.unique(a1a[:, 1]), np.unique(a2a[:, 1]))),
        "bIdenticalTglob": bool(a1a.shape == a2a.shape
                                and np.allclose(a1a[:, 4], a2a[:, 4])),
        "fMaxAbsTglobDifference": float(np.max(np.abs(a1a[:, 4] - a2a[:, 4])))
        if a1a.shape == a2a.shape else None,
        "fInstMin1a": float(a1a[:, 1].min()), "fInstMax1a": float(a1a[:, 1].max()),
        "fInstMin2a": float(a2a[:, 1].min()), "fInstMax2a": float(a2a[:, 1].max()),
        "fExpectedExp2aInstMin": 1.0 / 0.975 ** 2,
        "fExpectedExp2aInstMax": 1.0 / 0.800 ** 2,
    }

    dictSpacing = {}
    for sModel, sRel in [("hextor", "exp3_warm"), ("hextor", "exp3_cold"),
                         ("poise", "exp3_warm"), ("poise", "exp3_cold")]:
        aDat = faLoadNumeric("%s/%s/%s/global_output.dat" % (oArgs.results_dir, sModel, sRel))
        aInst = np.unique(np.round(aDat[:, 1], 6))
        aDiff = np.diff(aInst)
        dictSpacing["%s/%s" % (sModel, sRel)] = {
            "iNumPoints": int(len(aInst)),
            "fMin": float(aInst.min()), "fMax": float(aInst.max()),
            "fStepMin": float(aDiff.min()), "fStepMax": float(aDiff.max()),
            "fStepMedian": float(np.median(aDiff)),
            "bUniform": bool(np.ptp(aDiff) < 1e-6),
        }
    # AVALON stores both branches in one file with a trailing text column.
    aAv = []
    for sLine in open("%s/avalon/exp3/global_output.dat" % oArgs.results_dir):
        if sLine.strip().startswith("#") or not sLine.strip():
            continue
        listTok = sLine.split()
        aAv.append((float(listTok[1]), listTok[-1]))
    for sBranch in ("cooling", "warming"):
        aInst = np.unique(np.round([v for v, b in aAv if b == sBranch], 6))
        aDiff = np.diff(aInst)
        dictSpacing["avalon/exp3_%s" % sBranch] = {
            "iNumPoints": int(len(aInst)),
            "fMin": float(aInst.min()), "fMax": float(aInst.max()),
            "fStepMin": float(aDiff.min()), "fStepMax": float(aDiff.max()),
            "fStepMedian": float(np.median(aDiff)),
            "bUniform": bool(np.ptp(aDiff) < 1e-6),
        }
    dictOut["dictExp3Spacing"] = dictSpacing

    # POISE cold/warm instellation-grid offset.
    pw = faLoadNumeric("%s/poise/exp3_warm/global_output.dat" % oArgs.results_dir)
    pc = faLoadNumeric("%s/poise/exp3_cold/global_output.dat" % oArgs.results_dir)
    aRatio = np.sort(pc[:, 1]) / np.sort(pw[:, 1])
    dictOut["dictPoiseExp3BranchOffset"] = {
        "fRatioMean": float(aRatio.mean()),
        "fRatioSpread": float(np.ptp(aRatio)),
        "fImpliedSemiMajorAxisRatio": float(1.0 / np.sqrt(aRatio.mean())),
        "fInferredColdStartSemiMajorAxis_au": float(1.0 / np.sqrt(aRatio.mean())),
    }
    p4 = faLoadNumeric("%s/poise/exp4_cold/global_output.dat" % oArgs.results_dir)
    dictOut["dictPoiseExp4Instellation"] = {
        "fInst": float(np.unique(np.round(p4[:, 1], 6))[0]),
        "fDeficitFromUnity": float(1.0 - np.unique(np.round(p4[:, 1], 6))[0]),
        "fImpliedAbsorbedFluxDeficit_Wm2":
            float((1.0 - np.unique(np.round(p4[:, 1], 6))[0]) * 1361.0 / 4.0 * (1 - 0.3)),
    }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)
    print(json.dumps(dictOut, indent=1))


if __name__ == "__main__":
    main()
