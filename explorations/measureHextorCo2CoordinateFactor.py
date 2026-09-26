"""Measure the background-pressure factor X in an archived HEXTOR table's CO2 coordinate, f = pCO2/(X + pCO2), from the table's own pressure column."""
import argparse
import json
import numpy as np
import h5py

NCO2 = 92


def fdaImpliedFactors(sTablePath, dMinPressure):
    with h5py.File(sTablePath, "r") as f:
        daOlr = f["olr"][:]
    daPco2 = daOlr[:NCO2, 0] - 1.0
    daFco2 = daOlr[:NCO2, 1]
    baUse = (daOlr[:NCO2, 0] >= dMinPressure) & (daFco2 > 0) & (daFco2 < 1)
    return daPco2[baUse] * (1.0 - daFco2[baUse]) / daFco2[baUse]


def fdictSummarize(daX):
    dMean, dSem = float(daX.mean()), float(daX.std() / np.sqrt(len(daX)))
    return {"iLevels": int(len(daX)), "dMeanX": dMean, "dSemX": dSem,
            "dMinX": float(daX.min()), "dMaxX": float(daX.max()),
            "dPiOverTwo": float(np.pi / 2), "dMolarMassRatio": 44.0 / 28.0,
            "dSigmaFromPiOverTwo": (dMean - np.pi / 2) / dSem,
            "dSigmaFromMolarMassRatio": (dMean - 44.0 / 28.0) / dSem}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="archived HEXTOR Sun lookup table (h5)")
    parser.add_argument("--min-pressure", type=float, default=1.02,
                        help="total pressure (bar) below which the printed pressure is too coarse")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dictResult = fdictSummarize(fdaImpliedFactors(args.table, args.min_pressure))
    with open(args.out, "w") as fh:
        json.dump(dictResult, fh, indent=2)
    print(json.dumps(dictResult, indent=2))
