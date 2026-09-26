"""Measure N/S asymmetry of ice edges (global files) and Tsurf (regenerated lat files) in symmetric HEXTOR FILLET runs.

Usage: python measureHemisphericAsymmetry.py <archive_dir> <regenerated_fillet_dir>
"""
import glob
import os
import sys
import numpy as np

LIST_SYM = ["exp1", "exp1a", "exp2", "exp2a", "exp3_cold", "exp3_warm", "exp4_cold", "exp4_warm"]


def fdaGlobal(sPath):
    return np.array([[float(v) for v in l.split()] for l in open(sPath) if l.strip() and not l.startswith("#")])


def fnIceEdgeAsym(sLabel, da):
    daA = np.maximum(np.abs(da[:, 6] + da[:, 7]), np.abs(da[:, 5] + da[:, 8]))
    iaBad = np.where(daA > 0.3)[0]
    print("%-22s max |N+S| ice-edge asymmetry %.2f deg; cases >0.3 deg: %s"
          % (sLabel, daA.max(), [(int(da[i, 0]), round(float(daA[i]), 2)) for i in iaBad]))


def fnLatAsym(sRegen):
    listRes = []
    for sPath in glob.glob(os.path.join(sRegen, "exp*", "lat_output_HEXTOR_*.dat")):
        da = np.loadtxt(sPath, comments="#")
        if da.ndim != 2 or da.shape[0] != 18:
            listRes.append((np.inf, sPath)); continue
        listRes.append((np.max(np.abs(da[:, 1] - da[::-1, 1])), sPath))
    listRes.sort(reverse=True)
    daV = np.array([r[0] for r in listRes])
    print("\nregenerated lat files: %d ; max |T(phi)-T(-phi)| median %.3f K, 95th pct %.3f K, max %.3f K"
          % (len(daV), np.median(daV), np.percentile(daV, 95), daV.max()))
    print("   n files with asymmetry > 1 K: %d ; > 5 K: %d" % (np.sum(daV > 1), np.sum(daV > 5)))
    for dV, sP in listRes[:6]:
        print("   %.2f K  %s" % (dV, os.path.basename(sP)))


def fnMain(sArc, sRegen):
    for s in LIST_SYM:
        fnIceEdgeAsym("archive " + s, fdaGlobal(os.path.join(sArc, s, "global_output.dat")))
    for s in ("exp1", "exp1a", "exp2", "exp2a", "exp3", "exp4"):
        fnIceEdgeAsym("regenerated " + s, fdaGlobal(os.path.join(sRegen, s, "global_output_HEXTOR_%s.dat" % s)))
    fnLatAsym(sRegen)


if __name__ == "__main__":
    fnMain(sys.argv[1], sys.argv[2])
