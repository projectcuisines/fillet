"""Test HEXTOR's discrete diffusion operator (driver.f t2prime) for accuracy and global energy conservation.

Usage: python testDiffusionOperator.py <lat_output.dat>  (uses its Tsurf column as a realistic profile)
"""
import sys
import numpy as np
from computeDiffusionAndPoleTransient import ftGrid, fdaT2prime, NBELTS


def fdaArea(daX):
    daLat = np.arcsin(daX[1:-1]); dH = np.pi / (2 * NBELTS)
    return np.abs(np.sin(daLat + dH) - np.sin(daLat - dH)) / 2.0


def fdaWithGhosts(daT):
    return np.concatenate([[daT[0]], daT, [daT[-1]]])


def fnTest(daProfile, sLabel, daX, daDx, daArea, dD):
    da2 = fdaT2prime(fdaWithGhosts(daProfile), daX, daDx)[1:-1]
    dNet = dD * np.sum(daArea * da2)
    dMagn = dD * np.sum(daArea * np.abs(da2))
    print("%-28s global mean D*T'' = %+8.3f W/m2 (area-mean |D*T''| = %.2f W/m2)" % (sLabel, dNet, dMagn))
    return da2


def fnMain(sLatFile):
    daX, daDx = ftGrid(); daArea = fdaArea(daX); dD = 0.5505
    print("sum(area) = %.6f" % daArea.sum())
    daXc = daX[1:-1]
    daP2 = (3 * daXc ** 2 - 1) / 2
    da2 = fnTest(daP2, "P2(x) profile", daX, daDx, daArea, 1.0)
    print("   P2 exact operator -6*P2 vs discrete at belts (85,45,5 N):",
          np.round(-6 * daP2[[17, 13, 9]], 3), np.round(da2[[17, 13, 9]], 3))
    daT = np.loadtxt(sLatFile, comments="#")[:, 1]
    fnTest(daT, "Ben2 regenerated Tsurf", daX, daDx, daArea, dD)
    daStep = np.where(np.abs(np.degrees(np.arcsin(daXc))) > 60, 240.0, 290.0)
    fnTest(daStep, "step profile 240/290 K", daX, daDx, daArea, dD)
    daFlux = []
    for k in range(NBELTS - 1):
        dXm = 0.5 * (daXc[k] + daXc[k + 1])
        daFlux.append(-(1 - dXm ** 2) * (daT[k + 1] - daT[k]) / (daXc[k + 1] - daXc[k]))
    daFlux = np.array([0.0] + daFlux + [0.0])
    daCons = -np.diff(daFlux) / (2 * daArea)
    print("conservative flux-form operator on same Ben2 profile: global mean D*T'' = %+.2e W/m2" % (dD * np.sum(daArea * daCons)))
    print("per-belt D*T'' HEXTOR vs flux-form (Ben2), W/m2:")
    da2H = fdaT2prime(fdaWithGhosts(daT), daX, daDx)[1:-1]
    for k in (0, 1, 4, 8):
        print("   belt %5.1f  HEXTOR %+7.2f  flux-form %+7.2f" % (np.degrees(np.arcsin(daXc[k])), dD * da2H[k], dD * daCons[k]))


if __name__ == "__main__":
    fnMain(sys.argv[1])
