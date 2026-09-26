"""Emulate the archived (852fd03) HEXTOR getOLR/getPALB and compare with correct multilinear interpolation.

Usage: python emulateArchivedLookup.py <archived_table.h5> <current_table.h5>
The archived table has the pi/2 CO2 coordinate; the current one the true mixing ratio.
"""
import sys
import numpy as np
import h5py

NCO2, NTMP = 92, 19
DA_ZEN = np.array([0., 30., 60., 90.])
DA_SAB = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def fdictLoad(sPath):
    with h5py.File(sPath, "r") as f:
        daOlr = f["olr"][:]; daAlb = f["palb"][:]
    daF = daOlr[:NCO2, 1]; daT = daOlr[::NCO2, 2]
    daO = -daOlr[:, 3].reshape(NTMP, NCO2) / 1000.0
    daP = np.full((NCO2, NTMP, 4, 5), np.nan)
    for r in daAlb:
        daP[np.argmin(abs(daF - r[1])), np.argmin(abs(daT - r[2])),
            np.argmin(abs(DA_ZEN - r[3])), np.argmin(abs(DA_SAB - r[4]))] = r[5]
    daP[7] = daP[6]  # duplicate CO2 level: give it the same values (emulates a filled table)
    return {"f": daF, "T": daT, "olr": daO, "palb": daP}


def ftArchivedPair(daLev, dVal):
    """Nearest level by relative difference, plus the neighbour with smaller relative difference."""
    daDiff = np.abs(daLev - dVal) / dVal
    i = int(np.argmin(daDiff)); n = len(daLev)
    if daDiff[i] == 0 or dVal < daLev[0] or dVal > daLev[-1]:
        return i, i
    if i == 0:
        return i, 1
    if i == n - 1:
        return i, n - 2
    return (i, i - 1) if daDiff[i - 1] <= daDiff[i + 1] else (i, i + 1)


def fdArchivedOlr(d, dF, dT):
    ic, _ = ftArchivedPair(d["f"], dF)
    it, jt = ftArchivedPair(d["T"], dT)
    if it == jt:
        return d["olr"][it, ic]
    dFrac = abs(dT - d["T"][it]) / abs(d["T"][jt] - d["T"][it])
    return d["olr"][it, ic] + dFrac * (d["olr"][jt, ic] - d["olr"][it, ic])


def fdArchivedPalb(d, dF, dT, dZ, dA):
    ic, jc = ftArchivedPair(d["f"], dF); it, jt = ftArchivedPair(d["T"], dT)
    iz, jz = ftArchivedPair(DA_ZEN, max(dZ, 1e-9)); ia, ja = ftArchivedPair(DA_SAB, dA)
    daFr = np.zeros(4)
    if it == jt:
        if ic != jc:
            daFr[0] = abs(np.log10(dF) - np.log10(d["f"][ic])) / abs(np.log10(d["f"][jc]) - np.log10(d["f"][ic]))
    else:
        daFr[1] = abs(dT - d["T"][it]) / abs(d["T"][jt] - d["T"][it]); jc = ic
    if iz != jz:
        daFr[0] = 0; daFr[2] = abs(dZ - DA_ZEN[iz]) / abs(DA_ZEN[jz] - DA_ZEN[iz])
    if ia != ja:
        daFr[0] = 0; daFr[3] = abs(dA - DA_SAB[ia]) / abs(DA_SAB[ja] - DA_SAB[ia])
    P = d["palb"]
    p1, p2 = P[ic, it, iz, ia], P[jc, jt, iz, ia]
    p3, p4 = P[ic, it, jz, ja], P[jc, jt, jz, ja]
    a1 = p1 + max(daFr[:2]) * (p2 - p1); a2 = p3 + max(daFr[:2]) * (p4 - p3)
    return a1 + max(daFr[2:]) * (a2 - a1)


def ftBracket(daLev, dVal, bLog=False):
    dV = min(max(dVal, daLev[0]), daLev[-1])
    i = int(np.clip(np.searchsorted(daLev, dV, side="right") - 1, 0, len(daLev) - 2))
    while daLev[i + 1] == daLev[i]:
        i += 1
    x0, x1, xv = (np.log10(daLev[i]), np.log10(daLev[i + 1]), np.log10(dV)) if bLog else (daLev[i], daLev[i + 1], dV)
    return i, (xv - x0) / (x1 - x0)


def fdCorrectOlr(d, dF, dT):
    ic, fc = ftBracket(d["f"], dF, True); it, ft = ftBracket(d["T"], dT)
    O = d["olr"]
    return ((1 - fc) * ((1 - ft) * O[it, ic] + ft * O[it + 1, ic]) + fc * ((1 - ft) * O[it, ic + 1] + ft * O[it + 1, ic + 1]))


def fdCorrectPalb(d, dF, dT, dZ, dA):
    ic, fc = ftBracket(d["f"], dF, True); it, ft = ftBracket(d["T"], dT)
    iz, fz = ftBracket(DA_ZEN, dZ); ia, fa = ftBracket(DA_SAB, dA)
    dS = 0.0
    for a, wa in ((0, 1 - fc), (1, fc)):
        for b, wb in ((0, 1 - ft), (1, ft)):
            for c, wc in ((0, 1 - fz), (1, fz)):
                for e, we in ((0, 1 - fa), (1, fa)):
                    dS += wa * wb * wc * we * d["palb"][ic + a, it + b, iz + c, ia + e]
    return dS


def fnMain(sArc, sCur):
    dA, dC = fdictLoad(sArc), fdictLoad(sCur)
    print("OLR [W/m2]: archived method (query f=2.8e-4 on pi/2 table, nearest CO2) vs correct (280 ppm, bilinear)")
    for dT in (200., 233., 255., 265., 275., 285., 295., 305.):
        dOa = fdArchivedOlr(dA, 2.8e-4, dT); dOc = fdCorrectOlr(dC, 2.8e-4, dT)
        print("  T=%5.1f  archived %7.2f  correct %7.2f  diff %+6.2f" % (dT, dOa, dOc, dOa - dOc))
    print("OLR below table (T=180 K): archived clamps ->", round(fdArchivedOlr(dA, 2.8e-4, 180.), 2),
          " vs T=190 value", round(fdArchivedOlr(dA, 2.8e-4, 190.), 2))
    print("\nPALB: archived method (zenith passed as mu*180/pi, diagonal zen/alb interp) vs correct")
    print("   mu    Tsurf  asurf | zen_arch zen_true | palb_arch palb_correct  diff")
    for dMu, dT, dAs in ((0.65, 300., 0.225), (0.55, 290., 0.225), (0.45, 280., 0.225), (0.35, 270., 0.3),
                         (0.30, 260., 0.53), (0.25, 255., 0.6), (0.15, 250., 0.6), (0.0, 250., 0.6),
                         (0.65, 300., 0.10), (0.5, 290., 0.30)):
        dZa = dMu * 180 / np.pi; dZt = np.degrees(np.arccos(dMu))
        dPa = fdArchivedPalb(dA, 2.8e-4, dT, dZa, dAs); dPc = fdCorrectPalb(dC, 2.8e-4, dT, dZt, dAs)
        print("  %4.2f  %5.1f  %5.3f | %6.1f  %6.1f | %7.3f  %7.3f  %+6.3f" % (dMu, dT, dAs, dZa, dZt, dPa, dPc, dPa - dPc))
    print("\nPALB with the true zenith, isolating the archived interpolation scheme (same table/coordinate):")
    for dZ, dAs in ((45., 0.3), (50., 0.225), (70., 0.5), (20., 0.7), (75., 0.225)):
        dPa = fdArchivedPalb(dC, 2.8e-4, 285., dZ, dAs); dPc = fdCorrectPalb(dC, 2.8e-4, 285., dZ, dAs)
        print("  z=%4.1f a=%5.3f  archived-scheme %6.3f  correct %6.3f  diff %+6.3f" % (dZ, dAs, dPa, dPc, dPa - dPc))


if __name__ == "__main__":
    fnMain(sys.argv[1], sys.argv[2])
