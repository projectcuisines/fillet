"""Test whether the residual between HEXTOR's archived FILLET OLR profiles and the current lookup table is explained by the pi/2 CO2-coordinate error corrected after the submission.

The archived HEXTOR submission predates `fix_fco2_coordinate.py`, which
rewrote the table's CO2 coordinate from pCO2/(pi/2 + pCO2) to the true mixing
ratio pCO2/(1 + pCO2).  Under the pre-fix table a query at the namelist value
f retrieved radiative transfer for an effective CO2 abundance larger by a
factor of about pi/2.  If that is the whole story, evaluating the CURRENT
table at f_eff = (pi/2) f should reproduce the archived profiles.

The test is a one-parameter fit: the CO2 scale factor that minimises the
residual is recovered from the data and compared with pi/2.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import HextorTable   # noqa: E402


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--fco2", type=float, default=2.8e-4)
    oParser.add_argument("--cloudir", type=float, default=-6.3)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oTab = HextorTable(oArgs.hextor_table)
    dictLat = json.load(open(oArgs.lat_json))

    aScale = np.concatenate([np.linspace(0.5, 4.0, 351)])
    dictOut = {"fPiOverTwo": float(np.pi / 2.0), "listBenchmarks": []}

    for sBench in ("ben1", "ben2", "ben3"):
        aDat = np.array(dictLat["hextor/%s" % sBench]["aaData"], dtype=float)
        aT, aOLR = aDat[:, 1], aDat[:, 4]

        aRms = np.empty_like(aScale)
        for i, fS in enumerate(aScale):
            aPred = np.array([oTab.fdOLR(float(t), oArgs.fco2 * fS) for t in aT])
            aPred -= oArgs.cloudir
            aRms[i] = np.sqrt(np.mean((aOLR - aPred) ** 2))
        iBest = int(np.argmin(aRms))

        aPred1 = np.array([oTab.fdOLR(float(t), oArgs.fco2) for t in aT]) - oArgs.cloudir
        aPredPi = np.array([oTab.fdOLR(float(t), oArgs.fco2 * np.pi / 2) for t in aT]) - oArgs.cloudir

        dictOut["listBenchmarks"].append({
            "sBenchmark": sBench,
            "fBestFitCO2ScaleFactor": float(aScale[iBest]),
            "fRmsAtBestFit": float(aRms[iBest]),
            "fRmsAtScaleOne": float(np.sqrt(np.mean((aOLR - aPred1) ** 2))),
            "fRmsAtPiOverTwo": float(np.sqrt(np.mean((aOLR - aPredPi) ** 2))),
            "fMeanResidualAtScaleOne": float(np.mean(aOLR - aPred1)),
            "fMeanResidualAtPiOverTwo": float(np.mean(aOLR - aPredPi)),
        })

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("pi/2 = %.4f\n" % (np.pi / 2))
    print("%-6s %12s %10s %10s %10s %10s"
          % ("bench", "best scale", "rms@best", "rms@1", "rms@pi/2", "mean@pi/2"))
    for d in dictOut["listBenchmarks"]:
        print("%-6s %12.4f %10.3f %10.3f %10.3f %+10.3f"
              % (d["sBenchmark"], d["fBestFitCO2ScaleFactor"], d["fRmsAtBestFit"],
                 d["fRmsAtScaleOne"], d["fRmsAtPiOverTwo"],
                 d["fMeanResidualAtPiOverTwo"]))


if __name__ == "__main__":
    main()
