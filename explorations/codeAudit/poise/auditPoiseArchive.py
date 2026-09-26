"""Audit the archived VPLanet/POISE FILLET submission: grids, columns, OLR fingerprint, ice lines."""
import sys
import numpy as np

sArchive = "/workspace/fillet/Results/poise"
listExps = ["ben1", "ben2", "ben3", "exp1", "exp1a", "exp2", "exp2a",
            "exp3_warm", "exp3_cold", "exp4_warm", "exp4_cold"]
# Column order actually written by src/proc_poise_out.py (lines 262-270)
listWritten = ["Case", "Inst", "Obl", "XCO2", "Tglob",
               "NMaxLand", "NMinLand", "SMaxLand", "SMinLand",
               "NMaxSea", "NMinSea", "SMaxSea", "SMinSea", "Diff", "OLR"]
listFifteenV11 = ["Case", "Inst", "Obl", "XCO2", "Tglob",
                  "NMaxLand", "NMinLand", "NMaxSea", "NMinSea",
                  "SMaxLand", "SMinLand", "SMaxSea", "SMinSea", "Diff", "OLRglob"]
dPlanckB = 2.09


def fdaLoad(sPath):
    return np.atleast_2d(np.loadtxt(sPath, comments="#"))


def fsHeader(sPath):
    sLast = ""
    for sLine in open(sPath):
        if sLine.startswith("# Case"):
            sLast = sLine.strip()
    return sLast


def fnReportGlobal(sExp):
    sPath = f"{sArchive}/{sExp}/global_output.dat"
    daData = fdaLoad(sPath)
    iRows, iCols = daData.shape
    sHead = fsHeader(sPath)
    print(f"\n=== {sExp}: rows={iRows} cols={iCols} header_cols={len(sHead.split())-1}")
    print("  header:", sHead)
    daInst, daObl, daX = daData[:, 1], daData[:, 2], daData[:, 3]
    print(f"  Inst unique n={len(np.unique(np.round(daInst,6)))} "
          f"range=[{daInst.min():.6f},{daInst.max():.6f}]")
    print(f"  Obl unique={np.unique(np.round(daObl,3))}")
    daXu = np.unique(daX)
    print(f"  XCO2 unique n={len(daXu)} range=[{daXu.min():g},{daXu.max():g}]")
    if len(daXu) > 1:
        daStep = np.diff(np.log10(daXu))
        print(f"  XCO2 log10 step min/max={daStep.min():.4f}/{daStep.max():.4f}")
    daT = daData[:, 4]
    print(f"  Tglob range=[{daT.min():.3f},{daT.max():.3f}]  zero rows={np.sum(daT==0)}")
    daOLR = daData[:, -1]
    for dA in (203.3, 204.05):
        daRes = daOLR - (dA + dPlanckB * (daT - 273.15))
        print(f"  OLR - (A={dA} + B(T-273.15)): max|res|={np.abs(daRes).max():.4f}")
    print(f"  Diff unique={np.unique(daData[:, -2])}")
    if iCols == 15:
        fnCheckIceLines(daData)
    return daData


def fnCheckIceLines(daData):
    dictCol = {s: i for i, s in enumerate(listWritten)}
    for sSurf in ("Land", "Sea"):
        daNMax = daData[:, dictCol["NMax" + sSurf]]
        daNMin = daData[:, dictCol["NMin" + sSurf]]
        daSMax = daData[:, dictCol["SMax" + sSurf]]
        daSMin = daData[:, dictCol["SMin" + sSurf]]
        iAsym = np.sum(~np.isclose(daNMin, -daSMax) | ~np.isclose(daNMax, -daSMin))
        iBad = np.sum(daNMax < daNMin) + np.sum(daSMax < daSMin)
        print(f"  {sSurf}: N/S asymmetric rows={iAsym}, max<min rows={iBad}, "
              f"NMin values(sample)={np.unique(np.round(daNMin,2))[:8]}")
    iDiffer = np.sum(~np.isclose(daData[:, dictCol['NMinLand']], daData[:, dictCol['NMinSea']]))
    print(f"  rows where land and sea northern min edge differ: {iDiffer}")


def fnImpliedSemi(daData, sExp, dLumFactor):
    daA = np.sqrt(dLumFactor / daData[:, 1])
    daAu = np.unique(np.round(daA, 5))
    print(f"  {sExp}: implied a (L factor {dLumFactor}) n={len(daAu)} "
          f"range=[{daAu.min():.5f},{daAu.max():.5f}] steps={np.unique(np.round(np.diff(daAu),5))}")


def fnReportLat(sExp, daGlob):
    sPath = f"{sArchive}/{sExp}/case_0/lat_output.dat"
    daLat = fdaLoad(sPath)
    daX = np.sin(np.radians(daLat[:, 0]))
    iN = len(daLat)
    daXexp = -1 + (np.arange(iN) + 0.5) * 2.0 / iN
    print(f"  lat file: rows={iN} cols={daLat.shape[1]} lat range=[{daLat[0,0]:.3f},{daLat[-1,0]:.3f}] "
          f"equal-area max|dx|={np.abs(daX-daXexp).max():.2e}")
    print(f"  mean Tsurf={daLat[:,1].mean():.4f} vs Tglob={daGlob[0,4]:.4f}; "
          f"mean OLR={daLat[:,4].mean():.4f} vs OLRglob={daGlob[0,-1]:.4f}")
    print(f"  Asurf range=[{daLat[:,2].min():.4f},{daLat[:,2].max():.4f}] "
          f"ATOA range=[{daLat[:,3].min():.4f},{daLat[:,3].max():.4f}] "
          f"rows Asurf==ATOA: {np.sum(np.isclose(daLat[:,2],daLat[:,3]))}")
    daSym = daLat[::-1, 1] - daLat[:, 1]
    print(f"  N/S Tsurf asymmetry max={np.abs(daSym).max():.4f} K")


def fnCompareWarmCold(sWarm, sCold, dictData):
    daW, daC = dictData[sWarm], dictData[sCold]
    if daW.shape != daC.shape:
        print(f"  {sWarm}/{sCold} shapes differ")
        return
    daDiff = np.abs(daW[:, 4] - daC[:, 4])
    print(f"  {sWarm} vs {sCold}: same Inst grid={np.allclose(daW[:,1],daC[:,1])}, "
          f"same XCO2 grid={np.allclose(daW[:,3],daC[:,3])}, rows |dT|>0.01 K: {np.sum(daDiff>0.01)}")


def main():
    dictData = {s: fnReportGlobal(s) for s in listExps}
    print("\n--- lat files")
    for sExp in ("ben1", "ben2", "ben3"):
        print(sExp)
        fnReportLat(sExp, dictData[sExp])
    print("\n--- implied semi-major axes / luminosity factors")
    for sExp in ("exp1a", "exp2a"):
        for dFac in (1.004825, 0.999861):
            fnImpliedSemi(dictData[sExp], sExp, dFac)
    dInst3c = dictData["exp3_cold"][:, 1]
    dInst3w = dictData["exp3_warm"][:, 1]
    print(f"  exp3 warm/cold Inst ratio unique={np.unique(np.round(dInst3w/dInst3c,5))} "
          f"(1.02^2={1.02**2:.5f})")
    print(f"  exp4 Inst {dictData['exp4_warm'][0,1]:.6f} vs 1.004825/1.02^2="
          f"{1.004825/1.02**2:.6f}")
    print("\n--- warm vs cold")
    fnCompareWarmCold("exp3_warm", "exp3_cold", dictData)
    fnCompareWarmCold("exp4_warm", "exp4_cold", dictData)
    print("\n--- column naming: position -> written / v1.1-template")
    for i, (sW, sT) in enumerate(zip(listWritten, listFifteenV11)):
        sFlag = "" if sW == sT else "   <-- MISMATCH"
        print(f"  col {i:2d}: written={sW:9s} v1.1={sT:9s}{sFlag}")


if __name__ == "__main__":
    sys.exit(main())
