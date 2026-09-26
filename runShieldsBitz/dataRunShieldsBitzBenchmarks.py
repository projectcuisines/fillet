"""Run the Shields-Bitz model on the three FILLET benchmarks and reduce each run to the protocol's global and latitudinal quantities.

Shields-Bitz is the one participating code whose FILLET deliverables are not
in the archive, and the only one distributed as source that this pipeline can
execute.  Rather than describe its schemes and stop, this step runs it under
the benchmark configurations the other four codes were run under, so that it
enters every comparison on the same footing as an archived submission.

The configuration is read off the protocol rather than invented.  Benchmark 1
is present-day Earth geography at $23.5\\degr$ obliquity; Benchmarks 2 and 3
replace it with the uniform 25% land, 75% ocean surface FILLET prescribes, at
$23.5\\degr$ and $60\\degr$.  The repository ships that surface already, as the
`fillet` case of its land-fraction table, so no geography is supplied here
that the authors did not write.

Everything else is left at the package defaults, including the 100-orbit
integration length.  Convergence is not assumed: the global mean of the final
orbit is reported beside that of the preceding orbit, so the residual drift
can be read rather than argued.

Two ice-line latitudes are reported per run.  One is the package's own
`mean_iceline`, the seasonal mean over the final orbit of the daily ocean
freezing latitude.  The other is what the protocol asks for -- the latitude
at which the annual-mean surface temperature crosses 263.15 K -- recomputed
here on the model's own sin(latitude) grid.  The two are not the same
quantity, and reporting both is what lets the difference be attributed.
"""

import argparse
import json
import os
import sys

# The integration solves a 2*jmx square system 360 times per orbit.  At
# jmx = 120 that matrix is small enough that a threaded BLAS spends more time
# synchronising than solving: measured on this machine, the default threading
# turns 62 s of CPU into over two hours of wall clock.  Pinning to one thread
# has to happen before numpy is imported to take effect.
for sVar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(sVar, "1")

import numpy as np

# Benchmark configurations, from Deitrick+2023 Table 2 as filed by the
# archived submissions: geography, obliquity, instellation.
BENCHMARKS = {
    "ben1": {"land": "modern", "obl": 23.5, "scaleQ": 1.0},
    "ben2": {"land": "fillet", "obl": 23.5, "scaleQ": 1.0},
    "ben3": {"land": "fillet", "obl": 60.0, "scaleQ": 1.0},
}

ICE_THRESHOLD_K = 263.15


def fdictRunOne(oEbm, sBench, fRunLength, iJmx):
    """One benchmark integration, returned as the raw package result."""
    dictCfg = oEbm.DEFAULTS.copy()
    dictCfg.update(BENCHMARKS[sBench])
    dictCfg["runlength"] = fRunLength
    dictCfg["jmx"] = iJmx
    dictCfg["casename"] = sBench
    return dictCfg, oEbm.seasonal_run(dictCfg)


def faAreaWeights(aXFull):
    """Equal-area weights: the grid is uniform in sin(latitude) by construction."""
    aW = np.ones(len(aXFull), dtype=float)
    return aW / aW.sum()


def fdPreviousOrbitMean(dictRes, dictSetup):
    """Global mean of the orbit before the last, as a convergence residual."""
    iSteps = int(dictSetup["nstepinyear"])
    aL, aW = dictRes["L_out"], dictRes["W_out"]
    if aL.shape[1] < 2 * iSteps:
        return None
    aLPrev = aL[:, -2 * iSteps:-iSteps]
    aWPrev = aW[:, -2 * iSteps:-iSteps]
    aT = (dictSetup["fl"] * np.mean(aLPrev, axis=1)
          + dictSetup["fw"] * np.mean(aWPrev, axis=1))
    return float(np.mean(aT) + 273.15)


def fdIceEdgeFromProfile(aLatDeg, aTK, fThresholdK):
    """Northern ice edge: the poleward-most crossing of the threshold, interpolated."""
    aLat, aT = np.asarray(aLatDeg), np.asarray(aTK)
    aMask = aLat >= 0.0
    aLatN, aTN = aLat[aMask], aT[aMask]
    if np.all(aTN > fThresholdK):
        return 90.0
    if np.all(aTN <= fThresholdK):
        return 0.0
    aCross = np.nonzero((aTN[:-1] - fThresholdK) * (aTN[1:] - fThresholdK) < 0)[0]
    if not len(aCross):
        return 90.0
    i = int(aCross[-1])
    fFrac = (fThresholdK - aTN[i]) / (aTN[i + 1] - aTN[i])
    return float(aLatN[i] + fFrac * (aLatN[i + 1] - aLatN[i]))


def fdictReduce(oEbm, sBench, dictCfg, dictRes):
    """The protocol quantities, plus the budget terms the benchmark figure needs."""
    dictSetup = dictRes["setup"]
    dictMean = oEbm.final_year_annual_means(dictRes)
    aLat = np.asarray(dictMean["lat"], dtype=float)
    aTK = np.asarray(dictMean["T_avg"], dtype=float) + 273.15
    aW = faAreaWeights(dictSetup["xfull"])

    aAlb = np.asarray(dictMean["A_avg"], dtype=float)
    aInsolAnn = np.mean(dictSetup["insol"], axis=1)
    aASR = (1.0 - aAlb) * aInsolAnn
    aOLR = dictSetup["A"] + dictSetup["B"] * np.asarray(dictMean["T_avg"])

    return {
        "sModel": "shields_bitz",
        "sBenchmark": sBench,
        "dictConfig": {k: v for k, v in dictCfg.items()},
        "aLat": aLat.tolist(),
        "aTsurf": aTK.tolist(),
        "aAsurf": aAlb.tolist(),
        "aATOA": aAlb.tolist(),
        "aOLR": aOLR.tolist(),
        "aASR": aASR.tolist(),
        "aInsolation": aInsolAnn.tolist(),
        "iNumLatitudes": int(len(aLat)),
        "fTglobK": float(dictMean["Tglob"]),
        "fTglobPreviousOrbitK": fdPreviousOrbitMean(dictRes, dictSetup),
        "fAlbedoGlobal": float(np.sum(aW * aAlb)),
        "fOLRGlobal": float(np.sum(aW * aOLR)),
        "fASRGlobal": float(np.sum(aW * aASR)),
        "fIceEdgeProtocolDeg": fdIceEdgeFromProfile(aLat, aTK, ICE_THRESHOLD_K),
        "fIceEdgePackageDeg": float(oEbm.mean_iceline(dictRes)),
    }


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--shields-repo", required=True)
    oParser.add_argument("--benchmarks", default="ben1,ben2,ben3")
    oParser.add_argument("--runlength", type=float, default=100.0)
    oParser.add_argument("--jmx", type=int, default=120)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    sys.path.insert(0, oArgs.shields_repo)
    import EBM_one_file as oEbm

    listOut = []
    for sBench in oArgs.benchmarks.split(","):
        dictCfg, dictRes = fdictRunOne(oEbm, sBench, oArgs.runlength, oArgs.jmx)
        listOut.append(fdictReduce(oEbm, sBench, dictCfg, dictRes))
        print("%s: Tglob = %.3f K, ice edge = %.2f deg"
              % (sBench, listOut[-1]["fTglobK"],
                 listOut[-1]["fIceEdgeProtocolDeg"]), flush=True)

    json.dump({"fRunLengthYears": oArgs.runlength, "iJmx": oArgs.jmx,
               "listBenchmarks": listOut}, open(oArgs.out, "w"))
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
