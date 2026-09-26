"""Probe how each model's OLR responds to CO2 across the FILLET v1.1 Experiment 4 range, isolating the HEXTOR lookup-table clamp, the non-monotonic region of the POISE HM16 fit, and the range over which OPS's table is extrapolated rather than clamped."""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "../compareRadiationSchemes")
from dataCompareRadiationSchemes import (CO2_COORD_FACTOR, HextorTable,
                                         OpsOlrTable, faOLRAvalon,
                                         faOLRPoiseHM16, faOLRPoiseWK97)


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--results-dir", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oTab = HextorTable(oArgs.hextor_table)
    oOps = OpsOlrTable(oArgs.ops_olr_table)
    aPPM = np.logspace(0, 5, 50)          # the v1.1 Experiment 4 grid
    fT = 288.0

    aHex = np.array([oTab.fdOLR(fT, p * 1e-6) for p in aPPM])
    aAva = faOLRAvalon(np.full_like(aPPM, fT), 1.0)  # placeholder, replaced below
    aAva = np.array([faOLRAvalon(np.array([fT]), p * 1e-6)[0] for p in aPPM])
    aWK = np.array([faOLRPoiseWK97(np.array([fT]), p * 1e-6)[0] for p in aPPM])
    aHM = np.array([faOLRPoiseHM16(np.array([fT]), p * 1e-6)[0] for p in aPPM])
    aOps = np.array([oOps.faOLR(np.array([fT]), p * 1e-6)[0] for p in aPPM])

    fCO2TableMin_ppm = float(oTab.aFCO2.min() * 1e6)
    aBelow = aPPM < fCO2TableMin_ppm
    dictOut = {
        "aCO2ppm": aPPM.tolist(),
        "aOLRHextor": aHex.tolist(),
        "aOLRAvalon": aAva.tolist(),
        "aOLRPoiseWK97": aWK.tolist(),
        "aOLRPoiseHM16": aHM.tolist(),
        "aOLROps": aOps.tolist(),
        "dictOpsTableFloor": {
            "fTableMinCO2ppmEquivalent":
                float(oOps.aPress[0] / CO2_COORD_FACTOR * 1e6),
            "iGridPointsBelowTableFloor": int(np.count_nonzero(
                aPPM * 1e-6 * CO2_COORD_FACTOR < oOps.aPress[0])),
            "bClampsBelowFloor": False,
            "fOLRSpreadOverExp4Range_Wm2": float(np.ptp(aOps)),
        },
        "dictHextorClamp": {
            "fTableMinCO2ppm": fCO2TableMin_ppm,
            "iGridPointsBelowTableFloor": int(np.count_nonzero(aBelow)),
            "fFractionOfExp4GridClamped": float(np.count_nonzero(aBelow) / len(aPPM)),
            "fOLRSpreadBelowFloor_Wm2": float(np.ptp(aHex[aBelow])) if aBelow.any() else 0.0,
        },
        "dictHM16Monotonicity": {
            "bMonotonicDecreasing": bool(np.all(np.diff(aHM) <= 1e-9)),
            "iSignChangesInSlope": int(np.count_nonzero(np.diff(np.sign(np.diff(aHM))) != 0)),
            "fCO2ppmAtMaxOLR": float(aPPM[int(np.argmax(aHM))]),
            "fMaxOLR": float(aHM.max()), "fMinOLR": float(aHM.min()),
        },
    }

    # Does the archived HEXTOR Experiment 4 output show the clamp?
    listRows = []
    for sLine in open("%s/hextor/exp4_cold/global_output.dat" % oArgs.results_dir):
        if sLine.strip().startswith("#") or not sLine.strip():
            continue
        listRows.append([float(t) for t in sLine.split()])
    aH4 = np.array(listRows)
    aCO2, aTg = aH4[:, 3], aH4[:, 4]
    aMaskLow = aCO2 < fCO2TableMin_ppm
    dictOut["dictHextorExp4Archive"] = {
        "iCasesBelowTableFloor": int(np.count_nonzero(aMaskLow)),
        "fTglobSpreadBelowFloor_K": float(np.ptp(aTg[aMaskLow])) if aMaskLow.any() else None,
        "fTglobSpreadAboveFloor_K": float(np.ptp(aTg[~aMaskLow])),
        "aCO2ppmBelowFloor": aCO2[aMaskLow].tolist(),
        "aTglobBelowFloor": aTg[aMaskLow].tolist(),
    }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("HEXTOR table CO2 floor = %.1f ppm; %d of %d Experiment 4 grid points "
          "lie below it (OLR spread there = %.3f W/m2)"
          % (fCO2TableMin_ppm, dictOut["dictHextorClamp"]["iGridPointsBelowTableFloor"],
             len(aPPM), dictOut["dictHextorClamp"]["fOLRSpreadBelowFloor_Wm2"]))
    d = dictOut["dictHextorExp4Archive"]
    print("archived HEXTOR exp4_cold: %d cases below the floor, Tglob spread there = %s K "
          "(vs %.2f K above the floor)"
          % (d["iCasesBelowTableFloor"], d["fTglobSpreadBelowFloor_K"],
             d["fTglobSpreadAboveFloor_K"]))
    print("HM16 monotonic in CO2: %s ; OLR peaks at %.3g ppm"
          % (dictOut["dictHM16Monotonicity"]["bMonotonicDecreasing"],
             dictOut["dictHM16Monotonicity"]["fCO2ppmAtMaxOLR"]))
    t = dictOut["dictOpsTableFloor"]
    print("OPS table bottom = %.1f ppm equivalent; %d of %d grid points lie below "
          "it, and it EXTRAPOLATES there rather than clamping (OLR range over the "
          "whole Exp. 4 span = %.1f W/m2)"
          % (t["fTableMinCO2ppmEquivalent"], t["iGridPointsBelowTableFloor"],
             len(aPPM), t["fOLRSpreadOverExp4Range_Wm2"]))
    print("\n%10s %10s %10s %10s %10s %10s"
          % ("CO2 ppm", "HEXTOR", "AVALON", "WK97", "HM16", "OPS"))
    for i in range(0, len(aPPM), 7):
        print("%10.3g %10.2f %10.2f %10.2f %10.2f %10.2f"
              % (aPPM[i], aHex[i], aAva[i], aWK[i], aHM[i], aOps[i]))


if __name__ == "__main__":
    main()
