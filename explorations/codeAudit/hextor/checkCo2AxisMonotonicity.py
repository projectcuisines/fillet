"""Report non-monotonic / duplicate CO2 levels in a HEXTOR v1 lookup table and their OLR effect."""
import sys
import numpy as np
import h5py

NCO2, NTMP = 92, 19


def fnMain(sPath):
    with h5py.File(sPath, "r") as f:
        daOlr = f["olr"][:]
    daFco2 = daOlr[:NCO2, 1]
    daPtot = daOlr[:NCO2, 0]
    daFlux = -daOlr[:, 3].reshape(NTMP, NCO2) / 1000.0
    daD = np.diff(daFco2)
    for i in np.where(daD <= 0)[0]:
        print("idx %d->%d: f=%.6e -> %.6e (Ptot %.4f -> %.4f)  OLR@280K %.2f -> %.2f"
              % (i, i + 1, daFco2[i], daFco2[i + 1], daPtot[i], daPtot[i + 1],
                 daFlux[9, i], daFlux[9, i + 1]))
    print("pCO2 [bar] implied (Ptot-1):", np.array2string(daPtot - 1.0, precision=5, max_line_width=160))
    print("OLR [W/m2] at T=190,250,280,300 K for f near 280 ppm:")
    iC = int(np.argmin(np.abs(daFco2 - 2.8e-4)))
    for iT, dT in ((0, 190), (6, 250), (9, 280), (11, 300)):
        print("  T=%d  f=%.3e  OLR=%.2f" % (dT, daFco2[iC], daFlux[iT, iC]))


if __name__ == "__main__":
    fnMain(sys.argv[1])
