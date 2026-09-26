"""Audit archived HEXTOR FILLET global outputs (format, grids, ice-line logic) and drift vs regenerated outputs.

Usage: python auditHextorArchive.py <archive_dir> <regenerated_fillet_dir>
"""
import os
import sys
import numpy as np

LIST_COLS = ["Case", "Inst", "Obl", "XCO2", "Tglob", "NMax", "NMin", "SMax", "SMin", "Diff", "OLRglob"]
DICT_REGEN = {"ben1": ("ben1", None), "ben2": ("ben2", None), "ben3": ("ben3", None),
              "exp1": ("exp1", None), "exp1a": ("exp1a", None), "exp2": ("exp2", None),
              "exp2a": ("exp2a", None), "exp3_cold": ("exp3", "cold"), "exp3_warm": ("exp3", "warm"),
              "exp4_cold": ("exp4", "cold"), "exp4_warm": ("exp4", "warm")}


def flistReadRows(sPath):
    listRows, listHeaderCols = [], None
    for sLine in open(sPath):
        s = sLine.strip()
        if s.startswith("# Case"):
            listHeaderCols = s[1:].split()
        if not s or s.startswith("#"):
            continue
        listRows.append(s.split())
    return listHeaderCols, listRows


def fdaNumeric(listRows):
    return np.array([[float(v) for v in r] for r in listRows])


def fsClassify(da):
    nmax, nmin, smax, smin = da
    if nmin == 90 and smax == -90:
        return "icefree"
    if nmax == 90 and nmin == 0 and smax == 0 and smin == -90:
        return "snowball"
    if nmax == 90 and smin == -90:
        return "caps"
    if nmin == 0 and smax == 0:
        return "belt"
    return "other"


def fnAuditOne(sName, sPath):
    listHdr, listRows = flistReadRows(sPath)
    iaNcol = sorted(set(len(r) for r in listRows))
    da = fdaNumeric(listRows)
    print("\n== %s: %d rows, column counts %s, header cols=%s" % (sName, len(da), iaNcol, len(listHdr or [])))
    print("   Inst %s..%s unique=%d ; Obl unique=%s ; XCO2 unique=%d (%.1f..%.1f)"
          % (da[:, 1].min(), da[:, 1].max(), len(np.unique(da[:, 1])), np.unique(da[:, 2]).tolist(),
             len(np.unique(da[:, 3])), da[:, 3].min(), da[:, 3].max()))
    print("   Tglob %.2f..%.2f ; Diff unique %s ; OLRglob %.2f..%.2f"
          % (da[:, 4].min(), da[:, 4].max(), np.unique(da[:, 9]).tolist(), da[:, 10].min(), da[:, 10].max()))
    daIce = da[:, 5:9]
    listCls = [fsClassify(r) for r in daIce]
    print("   states:", {s: listCls.count(s) for s in sorted(set(listCls))})
    iaAsym = np.where(np.abs(daIce[:, 1] + daIce[:, 2]) > 0.5)[0]
    iaAsym = [i for i in iaAsym if listCls[i] in ("caps", "other")]
    if len(iaAsym):
        print("   hemispherically asymmetric ice edges (NMin != -SMax) rows:", [int(da[i, 0]) for i in iaAsym][:20])
    iaOther = [int(da[i, 0]) for i, s in enumerate(listCls) if s == "other"]
    if iaOther:
        print("   unclassifiable rows:", iaOther[:20])
    iaCase = da[:, 0].astype(int)
    if not np.array_equal(iaCase, np.arange(iaCase[0], iaCase[0] + len(iaCase))):
        print("   NOTE: case numbers not contiguous")
    # identical physical results for distinct forcing (plateaus)
    iaDup = [i for i in range(1, len(da)) if da[i, 4] == da[i - 1, 4] and da[i, 10] == da[i - 1, 10]
             and (da[i, 1] != da[i - 1, 1] or da[i, 3] != da[i - 1, 3])]
    if iaDup:
        print("   consecutive cases with identical Tglob and OLRglob but different forcing:", [int(da[i, 0]) for i in iaDup])
    return da, listCls


def fnDrift(sName, daArc, listClsArc, sRegenDir):
    sExp, sMode = DICT_REGEN[sName]
    sPath = os.path.join(sRegenDir, sExp, "global_output_HEXTOR_%s.dat" % sExp)
    _, listRows = flistReadRows(sPath)
    da = fdaNumeric(listRows)
    if sMode is not None:
        iHalf = len(da) // 2
        da = da[:iHalf] if sMode == "cold" else da[iHalf:]
    if len(da) != len(daArc):
        print("   DRIFT: row count differs archive %d regenerated %d" % (len(daArc), len(da)))
        return
    bSameGrid = np.allclose(da[:, 1:4], daArc[:, 1:4])
    daDt = da[:, 4] - daArc[:, 4]
    listCls = [fsClassify(r) for r in da[:, 5:9]]
    iChanged = sum(a != b for a, b in zip(listCls, listClsArc))
    print("   DRIFT vs regenerated: same grid=%s  dTglob mean %+.2f median %+.2f min %+.2f max %+.2f ; state changes %d/%d"
          % (bSameGrid, daDt.mean(), np.median(daDt), daDt.min(), daDt.max(), iChanged, len(da)))
    print("      regenerated states:", {s: listCls.count(s) for s in sorted(set(listCls))})


def fnMain(sArc, sRegen):
    dictDa = {}
    for sName in DICT_REGEN:
        sPath = os.path.join(sArc, sName, "global_output.dat")
        da, listCls = fnAuditOne(sName, sPath)
        dictDa[sName] = da
        fnDrift(sName, da, listCls, sRegen)
    daProt = {"exp1": np.arange(0.8, 1.2501, 0.025), "exp2": np.arange(1.05, 1.5001, 0.025),
              "exp1a": 1 / np.arange(0.875, 1.1001, 0.0125) ** 2, "exp2a": 1 / np.arange(0.8, 0.9751, 0.0125) ** 2,
              "exp3_cold": np.arange(0.8, 1.5001, 0.0125)}
    print("\nProtocol Inst grids vs archive (rounded to 2 dp):")
    for sName, daP in daProt.items():
        daA = np.unique(dictDa[sName][:, 1]); daPr = np.unique(np.round(daP, 2))
        print("  %-9s protocol %.3f..%.3f (%d values)  archive %.2f..%.2f (%d values)  match=%s"
              % (sName, daP.min(), daP.max(), len(daP), daA.min(), daA.max(), len(daA),
                 len(daA) == len(daPr) and np.allclose(daA, daPr)))
    daA, daB = dictDa["exp1a"], dictDa["exp2a"]
    print("exp1a vs exp2a: identical forcing columns=%s ; identical Tglob rows=%d/%d"
          % (np.array_equal(daA[:, 1:4], daB[:, 1:4]), int(np.sum(daA[:, 4] == daB[:, 4])), len(daA)))
    daE = np.logspace(0, 5, 50)
    print("exp4 XCO2 grid matches 50 log points 1..1e5 ppm:",
          np.allclose(dictDa["exp4_cold"][:, 3], np.round(daE, 1), rtol=1e-3))


if __name__ == "__main__":
    fnMain(sys.argv[1], sys.argv[2])
