"""Emulate radiation.f90 read_table_v1 PALB mapping: find grid cells never filled and -1 sentinel entries."""
import sys
import numpy as np
import h5py

NCO2, NTMP = 92, 19
DA_ZEN = np.array([0., 30., 60., 90.])
DA_SAB = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def fnMain(sPath):
    with h5py.File(sPath, "r") as f:
        daOlr = f["olr"][:]
        daAlb = f["palb"][:]
    daFco2 = daOlr[:NCO2, 1]
    daTemp = daOlr[::NCO2, 2]
    iaFilled = np.zeros((NCO2, NTMP, 4, 5), dtype=int)
    for daRow in daAlb:
        iC = int(np.argmin(np.abs(daFco2 - daRow[1])))
        iT = int(np.argmin(np.abs(daTemp - daRow[2])))
        iZ = int(np.argmin(np.abs(DA_ZEN - daRow[3])))
        iA = int(np.argmin(np.abs(DA_SAB - daRow[4])))
        iaFilled[iC, iT, iZ, iA] += 1
    iaNever = np.argwhere(iaFilled == 0)
    print("cells never assigned: %d ; distinct CO2 indices (1-based): %s"
          % (len(iaNever), sorted(set(int(i) + 1 for i in iaNever[:, 0]))))
    print("cells assigned twice:", int(np.sum(iaFilled == 2)))
    daNeg = daAlb[daAlb[:, 5] < 0]
    print("PALB entries < 0: %d" % len(daNeg))
    if len(daNeg):
        print("  CO2 values (ppm):", np.unique(np.round(daNeg[:, 1] * 1e6, 1))[:20])
        print("  T values:", np.unique(daNeg[:, 2]))
        print("  zenith:", np.unique(daNeg[:, 3]), " surfalb:", np.unique(daNeg[:, 4]))
    daNegO = daOlr[daOlr[:, 3] >= 0]
    print("OLR entries with non-negative stored value (i.e. flux<=0 after flip): %d" % len(daNegO))
    daA = daAlb[:, 5]
    print("PALB at T=280, f~280ppm, surfalb 0.2, zenith 0/30/60/90:")
    iC = int(np.argmin(np.abs(daFco2 - 2.8e-4)))
    for dZ in DA_ZEN:
        bSel = ((np.abs(daAlb[:, 1] - daFco2[iC]) < 1e-12) & (daAlb[:, 2] == 280.) & (daAlb[:, 3] == dZ)
                & (np.abs(daAlb[:, 4] - 0.2) < 1e-9))
        print("   z=%4.0f  palb=%s" % (dZ, daA[bSel]))


if __name__ == "__main__":
    fnMain(sys.argv[1])
