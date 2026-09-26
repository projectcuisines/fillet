"""Compare OPS archived FILLET submission against runs kept in the OPS tree; report duplicates and consistency checks."""
import hashlib, itertools, math, os, re, sys

sArchive = "/workspace/fillet/Results/ops"
sTree = "/workspace/TOTAL_EBM_constantpCO2_36belts_FILLET/out"

def flistReadRows(sPath):
    listRows = []
    for sLine in open(sPath):
        listTok = sLine.split()
        if len(listTok) >= 7 and re.fullmatch(r"-?\d+", listTok[0]):
            listRows.append([listTok[0]] + [float(s.replace("d", "e").replace("D", "e")) for s in listTok[1:]])
    return listRows

def fsMd5(sPath):
    return hashlib.md5(open(sPath, "rb").read()).hexdigest()[:12]

def fnCompareFiles(sA, sB):
    listA, listB = flistReadRows(sA), flistReadRows(sB)
    dictB = {(round(r[1], 5), round(r[2], 2), round(r[3], 3)): r for r in listB}
    iSame = iDiff = iMissing = 0
    fMaxDiff = 0.0
    for r in listA:
        tKey = (round(r[1], 5), round(r[2], 2), round(r[3], 3))
        if tKey not in dictB:
            iMissing += 1
            continue
        fD = max(abs(x - y) for x, y in zip(r[4:], dictB[tKey][4:]))
        fMaxDiff = max(fMaxDiff, fD)
        iSame += fD == 0.0
        iDiff += fD != 0.0
    print(f"  {os.path.basename(os.path.dirname(sA)) or sA} vs {os.path.relpath(sB, sTree)}: rowsA={len(listA)} rowsB={len(listB)} "
          f"matchedIdentical={iSame} matchedDiffer={iDiff} unmatched={iMissing} maxAbsDiff={fMaxDiff:.4g}")

def fnSectionArchiveVsTree():
    print("### md5 of archived global files")
    for sExp in sorted(os.listdir(sArchive)):
        print(" ", sExp, fsMd5(f"{sArchive}/{sExp}/global_output.dat"), len(flistReadRows(f"{sArchive}/{sExp}/global_output.dat")), "rows")
    listTree = [f"{sTree}/Experiments/{s}" for s in sorted(os.listdir(f"{sTree}/Experiments")) if s.startswith("global_output")]
    print("### archive experiment files vs every tree experiment file")
    for sExp in ["exp1", "exp2", "exp3_warm", "exp3_cold", "exp4_warm", "exp4_cold"]:
        for sB in listTree:
            fnCompareFiles(f"{sArchive}/{sExp}/global_output.dat", sB)
    print("### benchmark global files")
    for i in (1, 2, 3):
        print("  archive:", open(f"{sArchive}/ben{i}/global_output.dat").read().strip().splitlines()[-1])
        print("  tree   :", open(f"{sTree}/Benchmarks/{i}/global_output_bench{i}.dat").read().strip().splitlines()[-1])

def fnSectionDuplicates():
    print("### internal duplicates in archive")
    for sA, sB in [("exp4_warm", "exp4_cold"), ("exp3_warm", "exp3_cold"), ("exp1", "exp2")]:
        print(f"  {sA} vs {sB}: identical bytes = {fsMd5(f'{sArchive}/{sA}/global_output.dat') == fsMd5(f'{sArchive}/{sB}/global_output.dat')}")
        fnCompareFiles(f"{sArchive}/{sA}/global_output.dat", f"{sArchive}/{sB}/global_output.dat")
    listE1 = flistReadRows(f"{sArchive}/exp1/global_output.dat")
    listE2 = flistReadRows(f"{sArchive}/exp2/global_output.dat")
    dictE2 = {(round(r[1], 4), r[2]): r for r in listE2}
    listSame = [(r[1], r[2]) for r in listE1 if (round(r[1], 4), r[2]) in dictE2 and dictE2[(round(r[1], 4), r[2])][4:] == r[4:]]
    listOverlap = [(r[1], r[2]) for r in listE1 if (round(r[1], 4), r[2]) in dictE2]
    print(f"  exp1/exp2 overlapping (S,obl) cells={len(listOverlap)}, bit-identical rows={len(listSame)}")
    for sExp in ["exp1", "exp2", "exp3_warm"]:
        listR = flistReadRows(f"{sArchive}/{sExp}/global_output.dat")
        print(f"  {sExp}: S grid per obliquity:")
        for fObl, itR in itertools.groupby(listR, key=lambda r: r[2]):
            listR2 = list(itR)
            print(f"    obl={fObl}: n={len(listR2)} S=[{', '.join(f'{r[1]:g}' for r in listR2)}]")
            print(f"       Tglob=[{', '.join(f'{r[4]:.1f}' for r in listR2)}]")
            if sExp != "exp1": break

def flistReadLat(sPath):
    return [[float(x) for x in s.split()] for s in open(sPath) if re.match(r"\s*-?\d", s)]

def fnSectionLat():
    print("### lat outputs: archive vs tree, symmetry, global means")
    for i in (1, 2, 3):
        listA = flistReadLat(f"{sArchive}/ben{i}/case_0/lat_output.dat")
        listT = flistReadLat(f"{sTree}/Benchmarks/{i}/lat_output_bench{i}.dat")
        fMax = max(abs(a - b) for ra, rb in zip(listA, listT) for a, b in zip(ra, rb))
        listLatTrue = [-87.5 + 5 * k for k in range(36)]
        listArea = [abs(math.sin(math.radians(l + 2.5)) - math.sin(math.radians(l - 2.5))) / 2 for l in listLatTrue]
        fTmean = sum(a * r[1] for a, r in zip(listArea, listA))
        fOLR = sum(a * r[4] for a, r in zip(listArea, listA))
        fAsym = max(abs(listA[k][1] - listA[35 - k][1]) for k in range(18))
        fLatErr = max(abs(r[0] - l) for r, l in zip(listA, listLatTrue))
        print(f"  ben{i}: rows={len(listA)} maxAbsDiff(archive,tree)={fMax:.3g} areaMeanTsurf={fTmean:.3f} areaMeanOLR={fOLR:.3f} "
              f"maxNSasymTsurf={fAsym:.3f} K maxLatColumnError={fLatErr:.2f} deg")
        print("     Lat column:", [r[0] for r in listA[:6]], "... true centres", listLatTrue[:6])

def fnSectionBudget():
    print("### TOA budget from tree benchmark files (palb.dat col4 = global insolation-weighted albedo, col5 = annual-mean insolation)")
    for i in (1,):
        listP = [[float(x) for x in s.split()] for s in open(f"{sTree}/Benchmarks/{i}/palb.dat")]
        listO = [[float(x) for x in s.split()] for s in open(f"{sTree}/Benchmarks/{i}/olr.dat")]
        fS = sum(r[1] * r[4] for r in listP)
        fAlbPlanet = listP[0][3]
        fASRplanet = fS * (1 - fAlbPlanet)
        fASRlat = sum(r[1] * r[4] * (1 - r[0]) for r in listP)
        fOLR = sum(r[1] * r[0] for r in listO)
        print(f"  ben{i}: <S>={fS:.3f} planetAlb={fAlbPlanet:.5f} ASR(planet alb)={fASRplanet:.3f} OLR={fOLR:.3f} "
              f"imbalance={fASRplanet - fOLR:.3f} W/m2; ASR from archived ATOA x annual S = {fASRlat:.3f} (bias {fASRlat - fASRplanet:+.3f})")
        print(f"     area-weighted mean of archived ATOA = {sum(r[1]*r[0] for r in listP):.4f} vs insolation-weighted planetary albedo {fAlbPlanet:.4f}")

def fnSectionModelOut():
    print("### model*.out summaries")
    for i in (1, 2, 3):
        sTxt = open(f"{sTree}/Benchmarks/{i}/model{i}.out").read()
        dictV = {k: float(v) for k, v in re.findall(r"planet average (\w+(?: \w+)?) =\s*([-\d.E+]+)", sTxt)}
        fS, fA, fOLR = dictV["insolation"], dictV["albedo"], dictV["outgoing infrared"]
        print(f"  ben{i}: S={fS} alb={fA} OLR={fOLR} ASR-OLR={fS*(1-fA)-fOLR:+.3f} (albedo printed to 3 dp -> +/-{fS*0.0005:.2f})",
              re.search(r"coefficient \(D\) =\s*(\S+)", sTxt).group(1), re.search(r"after\s+(\d+) orbits", sTxt).group(1), "orbits")

def fnSectionPeriod():
    print("### orbital period implied by a0 = sqrt(1/S) (driver_fillet1_2.f90:650-651, 796-799)")
    fG, fM, fAU = 6.6732e-11, 1.9891e30, 1.49597892e11
    for fS in (0.8, 1.0, 1.25, 1.5):
        fA = math.sqrt(1 / fS) * fAU
        print(f"  S={fS}: a={fA/fAU:.4f} AU  P={2*math.pi*math.sqrt(fA**3/(fG*fM))/86400:.2f} d")

def fnSectionPsat():
    print("### saturation mixing ratio and latefac (driver.f90:1651, 1722, 2074, PSATH2O 2782-2847), igeog=6 => RH=1")
    for fT in (230.0, 250.0, 263.0, 273.0, 280.0, 287.9):
        if fT <= 273.16:
            fP = 6.103e-3 * math.exp(-677 / (1.9872 / 18) * (1 / fT - 1 / 273.15))
        else:
            listPvap = [6.116560e-3, 8.725050e-3, 1.228000e-2, 1.705380e-2, 2.338590e-2]
            fTC = fT - 273.15; iN = int(fTC / 5.0) + 1
            listTT = [0.01] + [5.0 * j for j in range(1, 10)]
            fFr = (fTC - listTT[iN - 1]) / (listTT[iN] - listTT[iN - 1])
            fP = fFr * listPvap[iN] + (1 - fFr) * listPvap[iN - 1]
        fP *= 1.01325
        fFH2O = fP / (1.0 + 2.8e-4 + fP)
        print(f"  T={fT}: pH2O={fP:.5f} bar FH2O={fFH2O:.5f} latefac={fFH2O/0.0155:.3f}")
    print("  CO2 table coordinate for 280 ppm: pco2r = 44/28 * f/(1-f) * Pdry =", 44 * 2.8e-4 / 1.00028 / (28 * (1 - 2.8e-4 / 1.00028)))
    for fPpm in (1, 10, 280, 1e5):
        fP = fPpm * 1e-6; fF = fP / (1 + fP)
        fR = 44 * fF / (28 * (1 - fF)) if fF <= 0.1 else fP
        print(f"  {fPpm:g} ppm -> pco2r={fR:.3e} bar ({'below' if fR < 1e-5 else 'inside'} table axis min 1e-5)")

if __name__ == "__main__":
    fnSectionArchiveVsTree(); fnSectionDuplicates(); fnSectionLat(); fnSectionBudget(); fnSectionModelOut(); fnSectionPeriod(); fnSectionPsat()

def fsRegime(r):
    if r[6] == 0.0 and r[7] == 0.0: return "snowball"
    if r[6] == 90.0 and r[7] == -90.0: return "icefree" if r[4] < 330 else "runaway"
    return "caps"

def fnSectionHysteresis():
    print("### exp1 (warm) vs exp2 (cold) regimes in overlapping cells")
    listE1 = flistReadRows(f"{sArchive}/exp1/global_output.dat")
    dictE2 = {(round(r[1], 3), r[2]): r for r in flistReadRows(f"{sArchive}/exp2/global_output.dat")}
    iDiffReg = 0
    for r in listE1:
        tKey = (round(r[1], 3), r[2])
        if tKey in dictE2:
            r2 = dictE2[tKey]
            bIdent = r2[4:] == r[4:]
            if fsRegime(r) != fsRegime(r2):
                iDiffReg += 1
                print(f"   S={r[1]} obl={r[2]}: warm {fsRegime(r)} {r[4]} vs cold {fsRegime(r2)} {r2[4]}")
    print("   cells with different regime:", iDiffReg)
    listIdent = [(r[1], r[2]) for r in listE1 if (round(r[1], 3), r[2]) in dictE2 and dictE2[(round(r[1], 3), r[2])][4:] == r[4:]]
    print("   bit-identical cells:", listIdent)
    print("### runaway / clamp states (Tglob > 330 K) per archived file")
    for sExp in ["exp1", "exp2", "exp3_warm", "exp4_warm"]:
        listR = flistReadRows(f"{sArchive}/{sExp}/global_output.dat")
        listHot = [r for r in listR if r[4] > 330]
        if listHot:
            print(f"   {sExp}: {len(listHot)}/{len(listR)} rows with Tglob>330 K; max Tglob={max(r[4] for r in listHot)}; OLR range {min(r[10] for r in listHot)}-{max(r[10] for r in listHot)}")

fnSectionHysteresis()
