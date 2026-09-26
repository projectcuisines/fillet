"""Test whether EBM_one_file applies scaleQ once or twice, compare against the multi-file code and the shipped G_dwarf_ws.txt scaleQ=0.9 row, and check diffusion-operator conservation."""
import os
import sys

for sVar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(sVar, "1")
import numpy as np

sys.path.insert(0, "/workspace/shieldsEBM")
import EBM_one_file as oOne  # noqa: E402
from defaults import defaults  # noqa: E402
import seasonal as oSeas  # noqa: E402
import seasonal_setup as oSetup  # noqa: E402


def fnInsolScaling():
    fBase = np.mean(oOne.build_setup(dict(oOne.DEFAULTS, scaleQ=1.0))["insol"])
    for fQ in (0.8, 0.9, 1.005919, 1.2):
        dictR = oOne.seasonal_run(dict(oOne.DEFAULTS, scaleQ=fQ, runlength=1.0, jmx=36))
        fBase36 = np.mean(oOne.build_setup(dict(oOne.DEFAULTS, scaleQ=1.0, jmx=36))["insol"])
        print("one-file scaleQ=%.6f: insolation actually used / unscaled = %.6f (scaleQ^2 = %.6f)"
              % (fQ, np.mean(dictR["setup"]["insol"]) / fBase36, fQ ** 2))
    return fBase


def fdTg(aL, aW, fl, fw):
    return float(np.mean((aL[:, -360:].T @ fl + aW[:, -360:].T @ fw) / len(fl)))


def fnSweepRow(fQ=0.9):
    defaults["scaleQdef"] = fQ
    dictM = oSeas.seasonal_run()
    print("multi-file scaleQ=%.2f final-year Tg = %.6f C (G_dwarf_ws.txt: -3.676214)"
          % (fQ, fdTg(dictM["L_out"], dictM["W_out"], oSetup.fl, oSetup.fw)))
    for fQone in (fQ, np.sqrt(fQ)):
        dictO = oOne.seasonal_run(dict(oOne.DEFAULTS, jmx=60, land="stepModern", scaleQ=fQone))
        dictS = dictO["setup"]
        print("one-file scaleQ=%.6f final-year Tg = %.6f C"
              % (fQone, fdTg(dictO["L_out"], dictO["W_out"], dictS["fl"], dictS["fw"])))


def fnDiffusionConservation():
    for iJmx in (60, 120):
        for fH in (0.0, 1.0):
            dictS = oOne.build_setup(dict(oOne.DEFAULTS, jmx=iJmx, hadleyflag=fH))
            aOp = dictS["Diff_Op"]
            aT = np.random.default_rng(0).normal(size=iJmx)
            print("jmx=%d hadley=%d: len(x)=%d, |sum(Diff_Op@T)|=%.3e, max|colsum|=%.3e, symmetric=%s, last lam=%.3e"
                  % (iJmx, fH, len(dictS["x"]), abs(np.sum(aOp @ aT)), np.max(np.abs(aOp.sum(0))),
                     np.allclose(aOp, aOp.T), -dictS["c"][-1] if len(dictS["x"]) == iJmx else 0.0))


if __name__ == "__main__":
    fnInsolScaling()
    fnDiffusionConservation()
    fnSweepRow()
