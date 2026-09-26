"""Inspect HEXTOR v1 Sun lookup tables (current and archived-era) : axes, CO2 coordinate, OLR at key points."""
import sys
import numpy as np
import h5py

NCO2, NTMP = 92, 19


def fdictReadTable(sPath):
    with h5py.File(sPath, "r") as f:
        listKeys = list(f.keys())
        daOlr = f["olr"][:]
        daAlb = f["palb"][:]
    return {"keys": listKeys, "olr": daOlr, "alb": daAlb}


def fnDescribe(sLabel, dictT):
    daOlr, daAlb = dictT["olr"], dictT["alb"]
    print(f"=== {sLabel}: datasets {dictT['keys']} olr{daOlr.shape} palb{daAlb.shape}")
    # flat row per sample; HEXTOR reads columns (Fortran reversed), so rows here
    daFco2 = daOlr[:NCO2, 1]
    daTemp = daOlr[::NCO2, 2]
    daP = daOlr[:NCO2, 0]
    print("  col0 (pressure?) first CO2 block:", daP[:3], "...", daP[-3:])
    print("  fco2 levels n=%d min=%.4e max=%.4e" % (len(daFco2), daFco2.min(), daFco2.max()))
    print("  fco2 levels (ppm):", np.array2string(daFco2 * 1e6, precision=1, max_line_width=200))
    print("  monotonic increasing:", bool(np.all(np.diff(daFco2) > 0)))
    print("  temp levels:", daTemp)
    print("  palb col ranges:", [(daAlb[:, i].min(), daAlb[:, i].max()) for i in range(daAlb.shape[1])])
    daZen = np.unique(daAlb[:, 3]); daSab = np.unique(daAlb[:, 4])
    print("  zenith levels:", daZen, " surfalb levels:", daSab)
    # implied pCO2 given the stored coordinate vs pressure column
    daPco2FromP = daP - 1.0
    print("  pressure-col minus 1 (first 3):", daPco2FromP[:3])
    return daFco2, daTemp


def fnOlrAt(dictT, daFco2, daTemp, dFco2, dT):
    daOlr = -dictT["olr"][:, 3].reshape(NTMP, NCO2)  # stored negated, mW/m2
    iC = np.argmin(np.abs(daFco2 - dFco2)); iT = np.argmin(np.abs(daTemp - dT))
    return daFco2[iC], daTemp[iT], daOlr[iT, iC] / 1000.0


def fnMain(sCurrent, sArchived):
    dictCur = fdictReadTable(sCurrent); dictArc = fdictReadTable(sArchived)
    daFc, daTc = fnDescribe("current", dictCur)
    daFa, daTa = fnDescribe("archived 852fd03", dictArc)
    daRatio = daFc / daFa
    print("ratio current/archived coordinate (first, mid, last):", daRatio[0], daRatio[45], daRatio[-1])
    dX = daFc[0] * (1 - daFa[0]) / (daFa[0] * (1 - daFc[0]))
    print("implied background-pressure factor X (archived f = p/(X+p)):", dX, " pi/2 =", np.pi / 2)
    daPco2 = daFc / (1 - daFc)
    print("pCO2 [bar] implied by current coordinate: min %.3e max %.3e" % (daPco2.min(), daPco2.max()))
    for dQ in (2.8e-4,):
        daPArc = (np.pi / 2) * dQ / (1 - dQ)
        print("archived query fco2=%.1e -> retrieves pCO2 = %.3e bar (%.0f ppm)" % (dQ, daPArc, daPArc * 1e6))
    print("current: table min f = %.3e (%.1f ppm); archived: table min f = %.3e (%.1f ppm)"
          % (daFc[0], daFc[0] * 1e6, daFa[0], daFa[0] * 1e6))
    # Exp4 grid vs archived nearest level
    daGrid = np.logspace(-6, -1, 50)
    iaNear = [int(np.argmin(np.abs(daFa - g) / g)) for g in daGrid]
    print("Exp4 grid (ppm) -> archived nearest CO2 level index:")
    for g, i in zip(daGrid, iaNear):
        print("   %10.1f ppm -> idx %2d (%.1f ppm)%s" % (g * 1e6, i, daFa[i] * 1e6, "  CLAMPED" if g < daFa[0] else ""))


if __name__ == "__main__":
    fnMain(sys.argv[1], sys.argv[2])
