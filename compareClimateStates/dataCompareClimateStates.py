"""Compare the Experiment 1 and 2 climate-state maps of the three models, locating each model's snowball and ice-free boundaries in the (instellation, obliquity) plane.

Two boundaries are extracted per model and per obliquity:

  S_snow  the largest instellation that still yields a snowball, i.e. the
          deglaciation threshold approached from the cold side (Experiment 2)
          or the glaciation threshold from the warm side (Experiment 1);
  S_free  the smallest instellation that yields an ice-free planet.

Between them lies the band in which the model supports partial ice cover.
The offsets between models in S_snow and S_free are the quantity a downstream
user of an EBM actually inherits, so they are reported in absolute
instellation and converted to an equivalent absorbed-flux offset.
"""

import argparse
import itertools
import json

import numpy as np

# OPS is archived under the directory name `ops`.
LIST_MODELS = ("avalon", "hextor", "poise", "ops")

S_EARTH = 1361.0


def fdictBoundaries(listRecs, sStateKey="sStateNH"):
    """Per-obliquity snowball and ice-free instellation boundaries."""
    dictOut = {}
    aObl = sorted({r["fObl"] for r in listRecs})
    for fObl in aObl:
        listAt = sorted([r for r in listRecs if r["fObl"] == fObl],
                        key=lambda r: r["fInst"])
        aInst = np.array([r["fInst"] for r in listAt])
        listState = [r[sStateKey] for r in listAt]
        aSnow = np.array([s == "snowball" for s in listState])
        aFree = np.array([s == "ice_free" for s in listState])
        aPartial = ~(aSnow | aFree)
        dictOut["%.0f" % fObl] = {
            "fMaxSnowballInst": float(aInst[aSnow].max()) if aSnow.any() else None,
            "fMinIceFreeInst": float(aInst[aFree].min()) if aFree.any() else None,
            "iNumPartial": int(np.count_nonzero(aPartial)),
            "fPartialInstMin": float(aInst[aPartial].min()) if aPartial.any() else None,
            "fPartialInstMax": float(aInst[aPartial].max()) if aPartial.any() else None,
            "saStates": listState,
            "aInst": aInst.tolist(),
        }
    return dictOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--global-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listRecs = json.load(open(oArgs.global_json))["listRecords"]

    # Which of exp1/exp2 is actually the warm start, decided from the data.
    # The census below is keyed by FILE, because that is what the archive
    # contains; this flag records where the file name disagrees with it.
    dictStartLabel = {}
    for sModel in LIST_MODELS:
        dA = {(round(r["fInst"], 6), round(r["fObl"], 1)): r for r in listRecs
              if r["sModel"] == sModel and r["sExperiment"] == "exp1"}
        dB = {(round(r["fInst"], 6), round(r["fObl"], 1)): r for r in listRecs
              if r["sModel"] == sModel and r["sExperiment"] == "exp2"}
        listK = sorted(set(dA) & set(dB))
        if not listK:
            continue
        fMean = float(np.mean([dA[k]["fTglob"] - dB[k]["fTglob"] for k in listK]))
        dictStartLabel[sModel] = {
            "exp1": "warm" if fMean >= 0 else "cold",
            "exp2": "cold" if fMean >= 0 else "warm",
            "bFilesAsLabelled": bool(fMean >= 0),
            "fMeanExp1MinusExp2_K": fMean,
            "iMatchedCases": len(listK),
        }

    dictOut = {"dictStartLabel": dictStartLabel}
    for sExp in ("exp1", "exp2", "exp1a", "exp2a"):
        dictExp = {}
        for sModel in LIST_MODELS:
            listSub = [r for r in listRecs if r["sModel"] == sModel
                       and r["sExperiment"] == sExp]
            if not listSub:
                continue
            # For POISE the land and sea surfaces can disagree; report the sea
            # (ocean) classification, which is the one all three models share.
            dictExp[sModel] = {
                "iCases": len(listSub),
                "dictBoundariesSea": fdictBoundaries(listSub, "sStateNH"),
                "dictBoundariesLand": fdictBoundaries(listSub, "sStateNHLand"),
                "dictStateCounts": {
                    s: sum(1 for r in listSub if r["sStateNH"] == s)
                    for s in ("ice_free", "ice_cap", "ice_belt", "snowball",
                              "unclassified")},
                "dictStateCountsLand": {
                    s: sum(1 for r in listSub if r["sStateNHLand"] == s)
                    for s in ("ice_free", "ice_cap", "ice_belt", "snowball",
                              "unclassified")},
                "fTglobMin": float(min(r["fTglob"] for r in listSub)),
                "fTglobMax": float(max(r["fTglob"] for r in listSub)),
                "aTglob": [r["fTglob"] for r in listSub],
                "aInst": [r["fInst"] for r in listSub],
                "aObl": [r["fObl"] for r in listSub],
                "aStates": [r["sStateNH"] for r in listSub],
                "bLandSeaDistinct": any(r["bIceSurfacesDistinct"] for r in listSub),
                "iLandSeaStateDisagreements":
                    sum(1 for r in listSub if r["sStateNH"] != r["sStateNHLand"]),
                "sInferredStart":
                    dictStartLabel.get(sModel, {}).get(sExp),
                "bFilesAsLabelled":
                    dictStartLabel.get(sModel, {}).get("bFilesAsLabelled", True),
            }
        dictOut[sExp] = dictExp

    # Pairwise temperature comparison on matched (inst, obl) for Experiment 1.
    def fdictKey(listSub):
        return {(int(round(r["fInst"] / 0.0125)), round(r["fObl"], 1)): r
                for r in listSub}

    dictPairs = {}
    for sExp in ("exp1", "exp2"):
        for sA, sB in itertools.combinations(LIST_MODELS, 2):
            dA = fdictKey([r for r in listRecs if r["sModel"] == sA
                           and r["sExperiment"] == sExp])
            dB = fdictKey([r for r in listRecs if r["sModel"] == sB
                           and r["sExperiment"] == sExp])
            listK = sorted(set(dA) & set(dB))
            if not listK:
                continue
            aDT = np.array([dB[k]["fTglob"] - dA[k]["fTglob"] for k in listK])
            iStateDis = sum(1 for k in listK
                            if dA[k]["sStateNH"] != dB[k]["sStateNH"])
            dictPairs["%s:%s-%s" % (sExp, sA, sB)] = {
                "iMatched": len(listK),
                "fMeanDeltaTglob": float(aDT.mean()),
                "fRmsDeltaTglob": float(np.sqrt((aDT ** 2).mean())),
                "fMaxAbsDeltaTglob": float(np.abs(aDT).max()),
                "iStateDisagreements": iStateDis,
                "fStateDisagreementFraction": float(iStateDis / len(listK)),
            }
    dictOut["dictPairwise"] = dictPairs

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f)

    for sExp in ("exp1", "exp2"):
        print("\n=== %s: climate-state census (ocean surface) ===" % sExp)
        print("%-7s %8s %8s %8s %8s %10s %10s"
              % ("model", "free", "cap", "belt", "snowball", "Tmin", "Tmax"))
        for sModel, d in dictOut[sExp].items():
            c = d["dictStateCounts"]
            print("%-7s %8d %8d %8d %8d %10.2f %10.2f"
                  % (sModel, c["ice_free"], c["ice_cap"], c["ice_belt"],
                     c["snowball"], d["fTglobMin"], d["fTglobMax"]))
        print("%-7s land/sea state disagreements: %s"
              % ("poise", dictOut[sExp].get("poise", {}).get(
                  "iLandSeaStateDisagreements")))
        print("  boundaries at obliquity 0 / 30 / 60 / 90 deg "
              "(max snowball inst, min ice-free inst):")
        for sModel, d in dictOut[sExp].items():
            listB = []
            for sO in ("0", "30", "60", "90"):
                b = d["dictBoundariesSea"].get(sO, {})
                listB.append("%s/%s" % (
                    "%.3f" % b["fMaxSnowballInst"] if b.get("fMaxSnowballInst") else "--",
                    "%.3f" % b["fMinIceFreeInst"] if b.get("fMinIceFreeInst") else "--"))
            print("    %-7s %s" % (sModel, "   ".join(listB)))

    print("\n=== pairwise agreement ===")
    print("%-22s %8s %10s %10s %10s %10s"
          % ("pair", "n", "meandT", "rmsdT", "maxdT", "stateDis"))
    for sKey, d in dictPairs.items():
        print("%-22s %8d %+10.2f %10.2f %10.2f %6d (%4.1f%%)"
              % (sKey, d["iMatched"], d["fMeanDeltaTglob"], d["fRmsDeltaTglob"],
                 d["fMaxAbsDeltaTglob"], d["iStateDisagreements"],
                 100 * d["fStateDisagreementFraction"]))


if __name__ == "__main__":
    main()
