"""Check Shields-Bitz grid construction, solar constant, insolation normalization, ice-line axis, and albedo ordering without running the model."""
import os
import sys

for sVar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(sVar, "1")
import numpy as np

sys.path.insert(0, "/workspace/shieldsEBM")
import EBM_one_file as oEbm


def fnCheckGrid():
    for iJmx in (18, 36, 60, 90, 100, 120, 150, 180, 240):
        fDelx = 2.0 / iJmx
        aX = np.arange(-1.0 + fDelx, 1.0, fDelx)
        aXfull = np.arange(-1 + fDelx / 2, 1, fDelx)
        print("jmx=%4d len(x)=%4d (expect %d) len(xfull)=%4d (expect %d)"
              % (iJmx, len(aX), iJmx - 1, len(aXfull), iJmx))


def fnCheckSolar():
    fT1 = 2808 / 2.0754
    print("hard-coded solar constant t1 = %.4f W/m2; scaleQ for 1361 = %.6f" % (fT1, 1361 / fT1))
    for fObl in (0.0, 23.5, 60.0, 90.0):
        dictS = oEbm.build_setup(dict(oEbm.DEFAULTS, obl=fObl))
        aQ = dictS["insol"]
        fGlob = np.mean(np.mean(aQ, axis=1))
        aQann = np.mean(aQ, axis=1)
        print("obl=%5.1f global annual-mean insolation %.4f (t1/4=%.4f); N/S asym max %.2e; nan=%d"
              % (fObl, fGlob, fT1 / 4, np.max(np.abs(aQann - aQann[::-1])), int(np.isnan(aQ).sum())))


def fnCheckIceLineAxis(iJmx=120):
    fDelx = 2.0 / iJmx
    aXfull = np.arange(-1 + fDelx / 2, 1, fDelx)
    aTrue = np.degrees(np.arcsin(aXfull))
    aLocator = np.linspace(-90, 90, iJmx)
    for fLat in (30, 45, 55, 60, 65, 70, 75, 80):
        fJ = np.interp(fLat, aTrue, np.arange(iJmx))
        print("jmx=%d true lat %4.1f -> locator reports %5.2f" % (iJmx, fLat, np.interp(fJ, np.arange(iJmx), aLocator)))


def fnCheckAlbedo():
    dictB = oEbm.get_broadband_albedo("G")
    aX = np.linspace(0, 1, 11)
    aTwarm = np.full(aX.shape, 10.0)
    aTcold = np.full(aX.shape, -10.0)
    aAlbL, aAlbW = oEbm.albedo_seasonal(aTwarm, aTwarm, aX, dictB["A_o"], dictB["A_l"], dictB["A_50"])
    aIceL, aIceW = oEbm.albedo_seasonal(aTcold, aTcold, aX, dictB["A_o"], dictB["A_l"], dictB["A_50"])
    print("lat   land  ocean  ice(land) ice(ocean)")
    for fX, a, b, c, d in zip(aX, aAlbL, aAlbW, aIceL, aIceW):
        print("%5.1f %.4f %.4f %.4f %.4f" % (np.degrees(np.arcsin(fX)), a, b, c, d))
    fXcross = np.sqrt((2 * (dictB["A_50"] - dictB["A_l"] - 0.05) / 0.08 + 1) / 3)
    print("snow-free land albedo exceeds ice albedo poleward of %.2f deg" % np.degrees(np.arcsin(fXcross)))


if __name__ == "__main__":
    fnCheckGrid()
    fnCheckSolar()
    fnCheckIceLineAxis()
    fnCheckAlbedo()
