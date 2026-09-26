"""Re-run a HEXTOR FILLET experiment from the namelist template and sweep the repository ships, using a locally built executable.

HEXTOR drives its FILLET experiments from `fillet/<exp>/fillet_<exp>.sh`, a csh
script that loops over the protocol's instellation and obliquity grid,
substitutes each pair into the `XXX` and `QQQ` placeholders of
`input.nml.<exp>`, runs the model once per case, and appends the last line of
`out/fillet_global.out` to the experiment's global output file.

This script does the same thing, in Python, because csh is not installed here.
The loop bounds, the loop order, the substitution targets and the output
assembly are read from or transcribed from the shipped script rather than
chosen, so the sweep is the author's and only the interpreter differs.  The
model itself is unmodified; only the compiler differs from the one the
submission used, which is what the comparison at the end measures.

Nothing is written inside the HEXTOR repository: the working directory is
supplied by the caller and the shipped reference outputs are read only.
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import numpy as np

# The grids `fillet_exp1.sh` and `fillet_exp2.sh` set, as they set them.
SWEEPS = {
    "exp1": {"fInstMin": 0.80, "fInstMax": 1.25, "fInstStep": 0.025,
             "fOblMin": 0.0, "fOblMax": 90.0, "fOblStep": 10.0},
    "exp2": {"fInstMin": 1.05, "fInstMax": 1.50, "fInstStep": 0.025,
             "fOblMin": 0.0, "fOblMax": 90.0, "fOblStep": 10.0},
}


def faSeq(fMin, fMax, fStep):
    """`seq` semantics: inclusive of the endpoint within rounding."""
    iN = int(round((fMax - fMin) / fStep)) + 1
    return np.round(fMin + fStep * np.arange(iN), 6)


def fsRenderNamelist(sTemplate, fInst, fObl):
    """Substitute the two placeholders the shipped script substitutes."""
    sBody = open(sTemplate).read()
    sBody = sBody.replace("XXX", "%g" % fInst)
    return sBody.replace("QQQ", "%g" % fObl)


def flistRunSweep(sWorkDir, sExe, sTemplate, dictGrid, sLibPath, bVerbose):
    """One case per (instellation, obliquity) pair, in the shipped loop order."""
    aInst = faSeq(dictGrid["fInstMin"], dictGrid["fInstMax"], dictGrid["fInstStep"])
    aObl = faSeq(dictGrid["fOblMin"], dictGrid["fOblMax"], dictGrid["fOblStep"])
    dictEnv = dict(os.environ)
    if sLibPath:
        dictEnv["LD_LIBRARY_PATH"] = sLibPath + ":" + dictEnv.get("LD_LIBRARY_PATH", "")

    listRows, iCase = [], 0
    for fInst in aInst:
        for fObl in aObl:
            with open(os.path.join(sWorkDir, "input.nml"), "w") as oFile:
                oFile.write(fsRenderNamelist(sTemplate, fInst, fObl))
            oRun = subprocess.run([sExe], cwd=sWorkDir, env=dictEnv,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
            if oRun.returncode != 0:
                raise RuntimeError("case %d (S=%g, obl=%g) failed: %s"
                                   % (iCase, fInst, fObl,
                                      oRun.stderr.decode()[:400]))
            sLast = [s for s in open(os.path.join(sWorkDir, "out",
                                                  "fillet_global.out"))
                     if s.strip() and not s.lstrip().startswith("#")][-1]
            listTok = sLast.split()
            listTok[0] = str(iCase)
            listRows.append([float(t) for t in listTok])
            if bVerbose and iCase % 20 == 0:
                print("  case %3d  S=%.3f obl=%4.1f -> Tglob %s"
                      % (iCase, fInst, fObl, listTok[4]), flush=True)
            iCase += 1
    return np.array(listRows, dtype=float)


def fdictCompare(aNew, sReferencePath):
    """The new sweep against the output the repository ships for it."""
    listRef = [[float(t) for t in s.split()]
               for s in open(sReferencePath)
               if s.strip() and not s.lstrip().startswith("#")]
    aRef = np.array(listRef, dtype=float)
    if aRef.shape != aNew.shape:
        return {"bComparable": False, "aShapeReference": list(aRef.shape),
                "aShapeNew": list(aNew.shape)}
    aD = np.abs(aRef - aNew)
    iTglob, listIce = 4, [5, 6, 7, 8]
    listNames = ["Case", "Inst", "Obl", "XCO2", "Tglob", "IceLineNMax",
                 "IceLineNMin", "IceLineSMax", "IceLineSMin", "Diff", "OLRglob"]
    # A case whose ice edge moves is a case whose reported climate could be
    # classified differently, so those are counted rather than averaged.
    aIceMoved = np.any(aD[:, listIce] > 0.05, axis=1)
    aSnowRef = np.all(np.abs(aRef[:, listIce]) < 1e-9, axis=1)
    aSnowNew = np.all(np.abs(aNew[:, listIce]) < 1e-9, axis=1)
    return {"bComparable": True, "iCases": int(aRef.shape[0]),
            "fMaxAbsDifference": float(np.nanmax(aD)),
            "fMaxTglobDifference_K": float(np.nanmax(aD[:, iTglob])),
            "fRMSTglobDifference_K": float(np.sqrt(np.nanmean(aD[:, iTglob] ** 2))),
            "iCasesTglobDiffersAbovePointOne": int(np.count_nonzero(aD[:, iTglob] > 0.1)),
            "iCasesIceEdgeMoved": int(np.count_nonzero(aIceMoved)),
            "fMaxIceEdgeShift_deg": float(np.nanmax(aD[:, listIce])),
            "iCasesSnowballStateChanged": int(np.count_nonzero(aSnowRef != aSnowNew)),
            "dictMaxDifferenceByColumn": {listNames[i]: float(np.nanmax(aD[:, i]))
                                          for i in range(aRef.shape[1])},
            "bIdentical": bool(np.nanmax(aD) == 0.0)}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--experiment", required=True, choices=sorted(SWEEPS))
    oParser.add_argument("--hextor-repo", required=True)
    oParser.add_argument("--executable", required=True)
    oParser.add_argument("--work-dir", required=True)
    oParser.add_argument("--ld-library-path", default="")
    oParser.add_argument("--out", required=True)
    oParser.add_argument("--quiet", action="store_true")
    oArgs = oParser.parse_args()

    sExpDir = os.path.join(oArgs.hextor_repo, "fillet", oArgs.experiment)
    sTemplate = os.path.join(sExpDir, "input.nml.%s" % oArgs.experiment)
    sReference = os.path.join(sExpDir,
                              "global_output_HEXTOR_%s.dat" % oArgs.experiment)

    os.makedirs(os.path.join(oArgs.work_dir, "out"), exist_ok=True)
    for sName in ("data", "radiation"):
        sLink = os.path.join(oArgs.work_dir, sName)
        if not os.path.exists(sLink):
            os.symlink(os.path.join(oArgs.hextor_repo, "model", sName), sLink)
    shutil.copy(oArgs.executable, os.path.join(oArgs.work_dir, "driver"))

    fT0 = time.time()
    aNew = flistRunSweep(oArgs.work_dir,
                         os.path.join(oArgs.work_dir, "driver"),
                         sTemplate, SWEEPS[oArgs.experiment],
                         oArgs.ld_library_path, not oArgs.quiet)
    fElapsed = time.time() - fT0

    dictOut = {"sExperiment": oArgs.experiment,
               "iCases": int(aNew.shape[0]),
               "fWallSeconds": fElapsed,
               "dictGrid": SWEEPS[oArgs.experiment],
               "dictAgainstShippedOutput": fdictCompare(aNew, sReference),
               "aaRows": aNew.tolist()}
    json.dump(dictOut, open(oArgs.out, "w"))

    d = dictOut["dictAgainstShippedOutput"]
    print("\n%s: %d cases in %.1f s (%.2f s/case)"
          % (oArgs.experiment, dictOut["iCases"], fElapsed,
             fElapsed / max(1, dictOut["iCases"])))
    if d["bComparable"]:
        if d["bIdentical"]:
            print("against the shipped output: identical in every field")
        else:
            print("against the shipped output, this build vs the submitted one:")
            print("  Tglob     max %.3f K, RMS %.4f K, %d of %d cases above 0.1 K"
                  % (d["fMaxTglobDifference_K"], d["fRMSTglobDifference_K"],
                     d["iCasesTglobDiffersAbovePointOne"], d["iCases"]))
            print("  ice edge  max %.2f deg, moved in %d of %d cases"
                  % (d["fMaxIceEdgeShift_deg"], d["iCasesIceEdgeMoved"],
                     d["iCases"]))
            print("  snowball classification changed in %d cases"
                  % d["iCasesSnowballStateChanged"])
            print("  per column: %s"
                  % ", ".join("%s %.3g" % (k, v) for k, v
                              in d["dictMaxDifferenceByColumn"].items() if v > 0))
    else:
        print("against the shipped output: shapes differ, %s vs %s"
              % (d["aShapeReference"], d["aShapeNew"]))
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
