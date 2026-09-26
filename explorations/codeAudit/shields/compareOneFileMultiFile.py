"""Run the multi-file and one-file Shields-Bitz models on an identical configuration and compare their final-year fields; also reproduce the scaleQ=1 row of the shipped G_dwarf_ws.txt."""
import os
import sys

for sVar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(sVar, "1")
import numpy as np
from scipy.interpolate import interp1d

sRepo = "/workspace/shieldsEBM"
sys.path.insert(0, sRepo)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import seasonal_setup as oSetup  # noqa: E402
import seasonal as oSeas  # noqa: E402
import EBM_one_file as oOne  # noqa: E402


def fnReportUnpack():
    print("multi-file seasonal_setup globals after tuple unpack:")
    for sName in ("star", "ice_model", "land", "casename", "jmx"):
        print("  %-10s = %r" % (sName, getattr(oSetup, sName)))


def fdWarmstartIceLine(aW, aH, iJmx, bTrueAxis):
    """Replicates warmstart.py:89-119 (NH ocean, -2.013 C threshold, daily then nanmean)."""
    fDelx = 2.0 / iJmx
    aPhi = (np.degrees(np.arcsin(np.arange(-1 + fDelx / 2, 1, fDelx))) if bTrueAxis
            else np.linspace(-90, 90, iJmx))
    aJ = np.arange(iJmx // 2, iJmx)
    aWly = aW[np.ix_(aJ, np.arange(aW.shape[1] - 360, aW.shape[1]))].copy()
    aWly -= 1e-6 * (np.arange(aWly.shape[0])[:, None] + 1) + 1e-6 * (np.arange(360)[None, :] + 1)
    aIce = np.empty(360)
    for d in range(360):
        f = interp1d(aWly[:, d], aPhi[aJ], bounds_error=False, fill_value=np.nan)
        v = f(-2.013)
        if np.isnan(v):
            v = 0 if np.min(aWly[:, d]) < -2.013 else 90
        aIce[d] = v
    return float(np.nanmean(aIce))


def main():
    fnReportUnpack()
    dictMulti = oSeas.seasonal_run()
    dictOne = oOne.seasonal_run(dict(oOne.DEFAULTS, jmx=60, land="stepModern"))
    fl, fw = oSetup.fl, oSetup.fw
    print("land fraction identical:", np.allclose(fl, dictOne["setup"]["fl"]))
    for sKey in ("L_out", "W_out", "h_out"):
        a, b = dictMulti[sKey], dictOne[sKey]
        print("%s shapes %s %s max|diff| = %.3e" % (sKey, a.shape, b.shape, np.max(np.abs(a - b))))
    aTg = (dictMulti["L_out"].T @ fl + dictMulti["W_out"].T @ fw) / 60
    print("multi-file final-year mean Tg = %.6f C (G_dwarf_ws.txt row scaleQ=1: 10.904874)" % np.mean(aTg[-360:]))
    print("warmstart ice line, linspace axis = %.6f (file: 70.607277)"
          % fdWarmstartIceLine(dictMulti["W_out"], dictMulti["h_out"], 60, False))
    print("warmstart ice line, true sin-lat axis = %.6f"
          % fdWarmstartIceLine(dictMulti["W_out"], dictMulti["h_out"], 60, True))


if __name__ == "__main__":
    main()
