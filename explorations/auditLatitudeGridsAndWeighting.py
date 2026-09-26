"""Reconstruct each model's latitude grid from its archived FILLET profile and quantify how much of the reported global-mean temperature depends on the (unspecified) area-weighting convention rather than on the physics."""

import argparse
import json

import numpy as np


def faEdgeWeights(aLatDeg):
    """Edges midway between nodes in sin(lat), outermost edges at the poles."""
    aX = np.sin(np.deg2rad(aLatDeg))
    aEdge = np.empty(len(aX) + 1)
    aEdge[1:-1] = 0.5 * (aX[:-1] + aX[1:])
    aEdge[0], aEdge[-1] = -1.0, 1.0
    aW = np.abs(np.diff(aEdge))
    return aW / aW.sum()


def faCosWeights(aLatDeg):
    """Naive cos(lat) quadrature, the convention a reader would assume."""
    aW = np.cos(np.deg2rad(aLatDeg))
    return aW / aW.sum()


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--global-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictLat = json.load(open(oArgs.lat_json))
    listGlob = json.load(open(oArgs.global_json))["listRecords"]

    listOut = []
    for sKey, dictProf in sorted(dictLat.items()):
        sModel, sBench = sKey.split("/")
        aDat = np.array(dictProf["aaData"], dtype=float)
        aLat, aT = aDat[:, 0], aDat[:, 1]
        aX = np.sin(np.deg2rad(aLat))
        aDX = np.diff(aX)

        listRow = [r for r in listGlob if r["sModel"] == sModel
                   and r["sExperiment"] == sBench]
        fReported = listRow[0]["fTglob"] if listRow else None

        dictEst = {
            "fSimpleMean": float(np.mean(aT)),
            "fEdgeWeighted": float(np.sum(faEdgeWeights(aLat) * aT)),
            "fCosWeighted": float(np.sum(faCosWeights(aLat) * aT)),
            "fTrapezoidInSinLat": float(np.trapezoid(aT, aX) / (aX[-1] - aX[0])),
        }
        sBest = min(dictEst, key=lambda k: abs(dictEst[k] - fReported))

        listOut.append({
            "sModel": sModel, "sBenchmark": sBench,
            "iNumLatitudes": int(len(aLat)),
            "fFirstLat": float(aLat[0]), "fLastLat": float(aLat[-1]),
            "bNodesAtPoles": bool(abs(abs(aLat[0]) - 90.0) < 1e-6),
            "fDxMin": float(aDX.min()), "fDxMax": float(aDX.max()),
            "bEqualAreaInSinLat": bool(np.ptp(aDX) < 1e-3 * np.median(aDX)),
            "bEqualWidthInLatitude": bool(np.ptp(np.diff(aLat)) < 1e-3 * np.median(np.diff(aLat))),
            "fTglobReported": fReported,
            "dictEstimates": dictEst,
            "sConventionMatchingReport": sBest,
            "fSpreadAcrossConventions_K": float(max(dictEst.values()) - min(dictEst.values())),
            "fReportedMinusBest_K": float(fReported - dictEst[sBest]),
        })

    with open(oArgs.out, "w") as f:
        json.dump({"listGrids": listOut}, f, indent=1)

    print("%-7s %-5s %4s %8s %8s %-9s %-9s %-18s %8s %8s"
          % ("model", "bench", "n", "lat[0]", "lat[-1]", "eq-area", "eq-width",
             "matches", "spread", "resid"))
    for d in listOut:
        print("%-7s %-5s %4d %8.2f %8.2f %-9s %-9s %-18s %8.3f %+8.3f"
              % (d["sModel"], d["sBenchmark"], d["iNumLatitudes"],
                 d["fFirstLat"], d["fLastLat"],
                 d["bEqualAreaInSinLat"], d["bEqualWidthInLatitude"],
                 d["sConventionMatchingReport"],
                 d["fSpreadAcrossConventions_K"], d["fReportedMinusBest_K"]))
    print("\nGlobal-mean temperature under each convention (K):")
    print("%-7s %-5s %12s %12s %12s %12s %12s"
          % ("model", "bench", "reported", "simple", "edge-wtd", "cos-wtd", "trapz"))
    for d in listOut:
        e = d["dictEstimates"]
        print("%-7s %-5s %12.3f %12.3f %12.3f %12.3f %12.3f"
              % (d["sModel"], d["sBenchmark"], d["fTglobReported"],
                 e["fSimpleMean"], e["fEdgeWeighted"], e["fCosWeighted"],
                 e["fTrapezoidInSinLat"]))


if __name__ == "__main__":
    main()
