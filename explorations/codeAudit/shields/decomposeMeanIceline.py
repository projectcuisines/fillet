"""Separate the latitude-axis error from the -2.013 C threshold error in Shields-Bitz mean_iceline for Benchmark 2, and check the multi-file solar routine for NaNs at high obliquity."""
import os
import sys

for sVar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(sVar, "1")
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, "/workspace/shieldsEBM")
import EBM_one_file as oOne  # noqa: E402
import seasonal_solar as oSolarMulti  # noqa: E402


def fdDailyMean(aField, aPhiNH, fThr, bLinspaceAxisUsed):
    aIce = np.empty(aField.shape[1])
    for d in range(aField.shape[1]):
        aX = aField[:, d] - 1e-6 * (np.arange(aField.shape[0]) + 1) - 1e-6 * (d + 1)
        v = interp1d(aX, aPhiNH, bounds_error=False, fill_value=np.nan)(fThr)
        if np.isnan(v):
            v = 0 if np.min(aX) < fThr else 90
        aIce[d] = v
    return float(np.nanmean(aIce))


def fdHEdgeDailyMean(aH, aPhiNH):
    """Daily equatorward-most NH sea-ice cell edge (interface latitude), averaged over the year."""
    aOut = []
    for d in range(aH.shape[1]):
        aJ = np.nonzero(aH[:, d] > 0.001)[0]
        aOut.append(90.0 if not len(aJ) else aPhiNH[aJ.min()])
    return float(np.mean(aOut))


def main():
    iJmx = 120
    dictR = oOne.seasonal_run(dict(oOne.DEFAULTS, land="fillet", obl=23.5, jmx=iJmx))
    aJ = np.arange(iJmx // 2, iJmx)
    aW = dictR["W_out"][aJ, -360:]
    aH = dictR["h_out"][aJ, -360:]
    aTrue = dictR["setup"]["phi"][aJ]
    aLin = np.linspace(-90, 90, iJmx)[aJ]
    print("package mean_iceline:                 %.2f" % oOne.mean_iceline(dictR))
    print("replica, linspace axis, -2.013 C:     %.2f" % fdDailyMean(aW, aLin, -2.013, True))
    print("true axis, -2.013 C:                  %.2f" % fdDailyMean(aW, aTrue, -2.013, False))
    print("true axis, -1.999 C (includes W=-2):  %.2f" % fdDailyMean(aW, aTrue, -1.999, False))
    aXedge = np.arange(-1 + 2.0 / iJmx, 1, 2.0 / iJmx)
    aPhiEdgeNH = np.degrees(np.arcsin(np.concatenate([aXedge[iJmx // 2 - 1:], [1.0]])))[:-1]
    print("h>0.001 daily edge (cell centre), mean:    %.2f" % fdHEdgeDailyMean(aH, aTrue))
    print("h>0.001 daily edge (equatorward interface): %.2f" % fdHEdgeDailyMean(aH, aPhiEdgeNH))
    fDelx = 2.0 / 60
    aXfull = np.arange(-1 + fDelx / 2, 1, fDelx)
    for fObl in (60.0, 80.0, 89.0, 90.0):
        with np.errstate(all="ignore"):
            aQ, _, _ = oSolarMulti.sun(aXfull, fObl, 0.0, 102.07, "G")
        print("multi-file sun obl=%.0f: NaN count %d" % (fObl, int(np.isnan(aQ).sum())))


if __name__ == "__main__":
    main()
