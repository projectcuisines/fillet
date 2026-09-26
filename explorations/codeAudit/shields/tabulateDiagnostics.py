"""Tabulate the diag_*.json outputs of diagnoseBenchmarkIceLines.py into one compact text table."""
import json
import sys


def fsEdge(o):
    return "free" if o is None else "%.1f" % o[0]


def main(listPaths):
    print("%-15s %-5s %7s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %7s %7s %6s"
          % ("mode", "bench", "Tglob", "dTyr", "sEver", "sHalf", "sPer", "lEver", "lHalf",
             "x263", "x-2C", "pkg", "fW-2", "albU", "albW", "imbal"))
    for sPath in listPaths:
        sMode = sPath.split("diag_")[1].split(".json")[0]
        for sB, d in json.load(open(sPath)).items():
            g = d["diag"]
            print("%-15s %-5s %7.2f %6.0e %6s %6s %6s %6s %6s %6.1f %6.1f %6.1f %6s %7.4f %7.4f %6.3f"
                  % (sMode, sB, g["Tglob"], g["Tglob"] - g["TglobPrevOrbit"],
                     fsEdge(g["seaIceEver"]["N"]), fsEdge(g["seaIceHalfYear"]["N"]),
                     fsEdge(g["seaIcePerennial"]["N"]), fsEdge(g["landSnowEver"]["N"]),
                     fsEdge(g["landSnowHalfYear"]["N"]), g["crossTavg263"], g["crossTavgMinus2C"],
                     g["meanIcelinePackage"],
                     "-" if g["fracIceCellDaysWithWabove_2p013"] is None else "%.2f" % g["fracIceCellDaysWithWabove_2p013"],
                     g["albedoGlobalUnweighted"], g["albedoGlobalInsolWeighted"],
                     g["ASRglob"] - g["OLRglob"]))


if __name__ == "__main__":
    main(sys.argv[1:])
