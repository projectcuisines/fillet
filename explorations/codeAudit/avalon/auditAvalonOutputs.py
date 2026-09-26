"""Audit AVALON FILLET outputs (archived submission and regenerated experiments) for internal consistency."""
import glob
import os
import re
import sys
import numpy as np

sArchive = "/workspace/fillet/Results/avalon"
sCurrent = "/workspace/avalon/experiments"
fA, fB, fKelvin = 210.0, 2.0, 273.15


def fdictReadLat(sPath):
    """Read a lat_output file; return header dict and data array."""
    dictHeader = {}
    listRows = []
    for sLine in open(sPath):
        if sLine.startswith("#"):
            match = re.match(r"#\s*([^:]+):\s*(.*)", sLine)
            if match:
                dictHeader[match.group(1).strip()] = match.group(2).strip()
            continue
        if sLine.strip():
            listRows.append([float(s) for s in sLine.split()])
    return dictHeader, np.array(listRows)


def flistReadGlobal(sPath):
    """Read global_output rows as lists of strings."""
    return [s.split() for s in open(sPath) if s.strip() and not s.startswith("#")]


def faAreaWeights(daLat):
    """Exact area weights for nodes given latitudes; cell edges at x-midpoints clipped to [-1,1]."""
    daX = np.sin(np.radians(daLat))
    daEdge = np.concatenate([[-1.0], 0.5 * (daX[1:] + daX[:-1]), [1.0]])
    return np.diff(daEdge) / 2.0


def fdictIceEdges(daT, daX, fTice):
    """Replicate avalon.jl ice_edges (current version) on an annual-mean T profile [K]."""
    baIce = daT - fKelvin < fTice
    iEq = int(np.argmin(np.abs(daX)))
    bAtEq = baIce[iEq]
    n = len(daX)
    listN = [i for i in range(n) if daX[i] > 0 and baIce[i]]
    listS = [i for i in range(n) if daX[i] < 0 and baIce[i]]
    if not listN:
        fNmax = fNmin = 90.0
    else:
        fNmax = 90.0 if max(listN) == n - 1 else np.degrees(np.arcsin(daX[max(listN)]))
        fNmin = 0.0 if bAtEq else np.degrees(np.arcsin(daX[min(listN)]))
    if not listS:
        fSmax = fSmin = -90.0
    else:
        fSmax = 0.0 if bAtEq else np.degrees(np.arcsin(daX[max(listS)]))
        fSmin = -90.0 if min(listS) == 0 else np.degrees(np.arcsin(daX[min(listS)]))
    return fNmax, fNmin, fSmax, fSmin


def fsClassify(fNmax, fNmin, fSmax, fSmin):
    """Classify climate state from NH edges (template convention)."""
    if fNmin == 90.0 and fNmax == 90.0:
        return "icefree"
    if fNmax == 90.0 and fNmin == 0.0:
        return "snowball"
    if fNmax == 90.0:
        return "cap"
    return "belt"


def fnAuditBenchmarks():
    """Area-weighted means, symmetry, and OLR consistency for benchmark lat files."""
    print("=== Benchmarks: reported Tglob vs exact area-weighted mean of Tsurf")
    for sTag in ["ben1", "ben2", "ben3"]:
        for sWhich, sLat, sGlob in [
            ("archive", f"{sArchive}/{sTag}/case_0/lat_output.dat", f"{sArchive}/{sTag}/global_output.dat"),
            ("current", f"{sCurrent}/{sTag}/lat_output_AVALON_{sTag}.dat",
             f"{sCurrent}/{sTag}/global_output_AVALON_{sTag}.dat")]:
            _, da = fdictReadLat(sLat)
            listG = flistReadGlobal(sGlob)[0]
            daW = faAreaWeights(da[:, 0])
            fAreaT = np.sum(daW * da[:, 1])
            fSumN = np.mean(da[:, 1])
            fAreaOLR = np.sum(daW * da[:, 4])
            daOlrLin = fA + fB * (da[:, 1] - fKelvin)
            print(f"{sTag} {sWhich}: n={len(da)} latrange=[{da[0,0]},{da[-1,0]}] Tglob_rep={float(listG[4]):.3f} "
                  f"sum/n={fSumN:.3f} areaW={fAreaT:.3f} diff={float(listG[4])-fAreaT:+.3f} K | "
                  f"OLRglob_rep={float(listG[14]):.3f} areaW_OLR={fAreaOLR:.3f} | "
                  f"max|OLR-(A+B*T)|={np.max(np.abs(da[:,4]-daOlrLin)):.3f} | ATOA==Asurf: {np.allclose(da[:,2],da[:,3])} | "
                  f"minTsurf={da[:,1].min():.2f}, maxTsurf={da[:,1].max():.2f}")
            if sTag != "ben1":
                daAsym = da[:, 1] - da[::-1, 1]
                print(f"    hemispheric asymmetry max|T(phi)-T(-phi)| = {np.max(np.abs(daAsym)):.3f} K "
                      f"at lat {da[np.argmax(np.abs(daAsym)),0]}")


def fnAuditSweep(sTag):
    """Check a regenerated experiment: ice-line recomputation, asymmetry, OLR, state counts."""
    sGlob = f"{sCurrent}/{sTag}/global_output_AVALON_{sTag}.dat"
    listRows = flistReadGlobal(sGlob)
    iBadIce, iAsym, fMaxAsym, fMaxOlr, fMaxTg, iAsymIce = 0, 0, 0.0, 0.0, 0.0, 0
    dictStates = {}
    listAsymCases = []
    for listG in listRows:
        iCase = int(listG[0])
        _, da = fdictReadLat(f"{sCurrent}/{sTag}/lat_output_AVALON_{sTag}_{iCase}.dat")
        daX = np.sin(np.radians(da[:, 0]))
        tEdges = fdictIceEdges(da[:, 1], daX, 0.0)
        tRep = tuple(float(listG[k]) for k in (5, 6, 11, 12))
        # lat file Tsurf is rounded to 0.01 K; tolerate threshold ties
        if not np.allclose(tEdges, tRep, atol=0.02):
            iBadIce += 1
        sState = fsClassify(*tRep)
        dictStates[sState] = dictStates.get(sState, 0) + 1
        fAsym = np.max(np.abs(da[:, 1] - da[::-1, 1]))
        if fAsym > 0.05:
            iAsym += 1
            listAsymCases.append((iCase, listG[1], listG[2], listG[3], round(fAsym, 2), sState))
        if abs(tRep[0] + tRep[3]) > 1e-6 or abs(tRep[1] + tRep[2]) > 1e-6:
            iAsymIce += 1
        fMaxAsym = max(fMaxAsym, fAsym)
        fMaxOlr = max(fMaxOlr, np.max(np.abs(da[:, 4] - (float_Aeff(listG) + fB * (da[:, 1] - fKelvin)))))
        fMaxTg = max(fMaxTg, abs(float(listG[4]) - np.mean(da[:, 1])))
    print(f"--- {sTag}: rows={len(listRows)} states={dictStates} iceline-recompute-mismatch={iBadIce} "
          f"asym(>0.05K)={iAsym} asymIceLines={iAsymIce} maxAsym={fMaxAsym:.2f} K "
          f"max|OLR-(Aeff+B*T)|={fMaxOlr:.3f} max|Tglob-mean(Tsurf)|={fMaxTg:.3f}")
    for tCase in listAsymCases[:12]:
        print("      asym case (case, Inst, Obl, XCO2, maxAsymK, state):", tCase)
    return dictStates


def float_Aeff(listG):
    """OLR intercept with CO2 forcing as in avalon.jl A_eff."""
    return fA - 5.35 * np.log(float(listG[3]) / 280.0)


def fnCompareArchive(sTag):
    """Compare archived and regenerated global outputs row by row."""
    listArc = flistReadGlobal(f"{sArchive}/{sTag}/global_output.dat")
    listCur = flistReadGlobal(f"{sCurrent}/{sTag}/global_output_AVALON_{sTag}.dat")
    iDiffState, fMaxDT = 0, 0.0
    dictArc = {}
    for listA, listC in zip(listArc, listCur):
        assert np.isclose(float(listA[1]), float(listC[1])) and listA[2] == listC[2], (listA[:3], listC[:3])
        sA = fsClassify(*(float(listA[k]) for k in (5, 6, 11, 12)))
        sC = fsClassify(*(float(listC[k]) for k in (5, 6, 11, 12)))
        dictArc[sA] = dictArc.get(sA, 0) + 1
        iDiffState += sA != sC
        fMaxDT = max(fMaxDT, abs(float(listA[4]) - float(listC[4])))
    print(f"--- {sTag}: archive rows={len(listArc)} current rows={len(listCur)} archive states={dictArc} "
          f"cases with different state={iDiffState} max|dTglob|={fMaxDT:.2f} K")


def fnGridChecks():
    """Report grid spacing near poles/ice lines and Exp2a/Exp4 grids."""
    _, da = fdictReadLat(f"{sCurrent}/ben2/lat_output_AVALON_ben2.dat")
    daLat = da[:, 0]
    print("current grid: n=", len(daLat), "polemost node", daLat[-1], "spacing near pole",
          np.round(np.diff(daLat[-4:]), 2), "spacing near 50deg", np.round(np.diff(daLat[65:68]), 2))
    listRows = flistReadGlobal(f"{sArchive}/exp4/global_output.dat")
    daCO2 = np.array(sorted({float(r[3]) for r in listRows}))
    print("archive exp4 XCO2: n=", len(daCO2), "min", daCO2[0], "max", daCO2[-1],
          "log-uniform:", np.allclose(np.diff(np.log10(daCO2)), np.diff(np.log10(daCO2))[0]))
    for sTag in ["exp1", "exp2", "exp2a", "exp3"]:
        listRows = flistReadGlobal(f"{sArchive}/{sTag}/global_output.dat")
        daInst = np.array(sorted({float(r[1]) for r in listRows}))
        daObl = sorted({float(r[2]) for r in listRows})
        print(f"archive {sTag}: Inst n={len(daInst)} [{daInst[0]:.4f},{daInst[-1]:.4f}] Obl={daObl}")
        if sTag == "exp2a":
            print("    implied a (au) =", np.round(1 / np.sqrt(daInst), 4))
        daT = np.array([float(r[4]) for r in listRows])
        print(f"    Tglob range {daT.min():.1f}-{daT.max():.1f} K")


if __name__ == "__main__":
    fnAuditBenchmarks()
    print("=== Regenerated experiments")
    for sTag in ["exp1", "exp1a", "exp2", "exp2a", "exp3", "exp4"]:
        fnAuditSweep(sTag)
    print("=== Archive vs regenerated")
    for sTag in ["exp1", "exp2", "exp2a", "exp3", "exp4"]:
        fnCompareArchive(sTag)
    print("=== Grids")
    fnGridChecks()
