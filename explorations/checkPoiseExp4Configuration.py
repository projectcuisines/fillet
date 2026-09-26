"""Test whether VPLanet-POISE's Experiment 4 was run with a different model configuration from its Experiment 3, by comparing the planetary albedo each experiment implies at the same climate state."""

import argparse
import json

import numpy as np

S_EARTH = 1361.0


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--global-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listRecs = json.load(open(oArgs.global_json))["listRecords"]

    def faImpliedAlbedo(sModel, sExp):
        listSub = [r for r in listRecs if r["sModel"] == sModel
                   and r["sExperiment"] == sExp]
        aS = np.array([r["fInst"] for r in listSub]) * S_EARTH
        aO = np.array([r["fOLRglob"] for r in listSub])
        aT = np.array([r["fTglob"] for r in listSub])
        aSt = np.array([r["sStateNH"] for r in listSub])
        return 1.0 - 4.0 * aO / aS, aT, aSt, np.array([r["fXCO2ppm"] for r in listSub])

    dictOut = {}
    for sExp in ("exp3_warm", "exp3_cold", "exp4_warm", "exp4_cold", "ben2"):
        aAlb, aT, aSt, aCO2 = faImpliedAlbedo("poise", sExp)
        aFree = aSt == "ice_free"
        aSnow = aSt == "snowball"
        dictOut[sExp] = {
            "iCases": int(len(aAlb)),
            "fAlbedoIceFreeMean": float(aAlb[aFree].mean()) if aFree.any() else None,
            "iIceFree": int(np.count_nonzero(aFree)),
            "fAlbedoSnowballMean": float(aAlb[aSnow].mean()) if aSnow.any() else None,
            "iSnowball": int(np.count_nonzero(aSnow)),
            "fAlbedoIceFreeSpread": float(np.ptp(aAlb[aFree])) if aFree.any() else None,
            "fAlbedoSnowballSpread": float(np.ptp(aAlb[aSnow])) if aSnow.any() else None,
        }

    # Same test for the other two codes, as a control: their configuration is
    # identical between Experiments 3 and 4, so the implied albedos should agree.
    for sModel, listExp in (("hextor", ("exp3_warm", "exp4_warm",
                                        "exp3_cold", "exp4_cold")),
                            ("avalon", ("exp3", "exp4"))):
        for sExp in listExp:
            aAlb, aT, aSt, aCO2 = faImpliedAlbedo(sModel, sExp)
            aFree = aSt == "ice_free"
            aSnow = aSt == "snowball"
            dictOut["%s/%s" % (sModel, sExp)] = {
                "iCases": int(len(aAlb)),
                "fAlbedoIceFreeMean": float(aAlb[aFree].mean()) if aFree.any() else None,
                "iIceFree": int(np.count_nonzero(aFree)),
                "fAlbedoSnowballMean": float(aAlb[aSnow].mean()) if aSnow.any() else None,
                "iSnowball": int(np.count_nonzero(aSnow)),
            }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("Planetary albedo implied by the reported global energy balance,")
    print("alpha = 1 - 4 <OLR> / S, by experiment and climate state:\n")
    print("%-22s %7s %14s %7s %14s"
          % ("case set", "n(free)", "alpha(ice-free)", "n(snow)", "alpha(snowball)"))
    for sKey, d in dictOut.items():
        print("%-22s %7s %14s %7s %14s"
              % (sKey, d["iIceFree"],
                 "%.4f" % d["fAlbedoIceFreeMean"] if d["fAlbedoIceFreeMean"] else "--",
                 d["iSnowball"],
                 "%.4f" % d["fAlbedoSnowballMean"] if d["fAlbedoSnowballMean"] else "--"))


if __name__ == "__main__":
    main()
