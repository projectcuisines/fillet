"""Time one Shields-Bitz seasonal integration so the cost of running the FILLET grids can be budgeted."""

import argparse
import json
import os
import sys
import time


def fdictTimeOneRun(sRepo, fRunLength, iJmx):
    """Run the model once at the given length and report wall time."""
    sys.path.insert(0, sRepo)
    import EBM_one_file as oEbm

    dictCfg = oEbm.DEFAULTS.copy()
    dictCfg["runlength"] = fRunLength
    dictCfg["jmx"] = iJmx
    fT0 = time.time()
    dictRes = oEbm.seasonal_run(dictCfg)
    fElapsed = time.time() - fT0
    return dictRes, fElapsed


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--shields-repo", required=True)
    oParser.add_argument("--runlength", type=float, default=5.0)
    oParser.add_argument("--jmx", type=int, default=120)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictRes, fElapsed = fdictTimeOneRun(oArgs.shields_repo, oArgs.runlength,
                                        oArgs.jmx)
    dictOut = {"fRunLengthYears": oArgs.runlength, "iJmx": oArgs.jmx,
               "fWallSeconds": fElapsed,
               "fSecondsPerModelYear": fElapsed / oArgs.runlength,
               "listResultKeys": sorted(dictRes.keys())}
    json.dump(dictOut, open(oArgs.out, "w"), indent=1)
    print(json.dumps(dictOut, indent=1)[:900])


if __name__ == "__main__":
    main()
