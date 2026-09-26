"""Recover the Shields-Bitz scheme parameters from its source, test the latitude axis its ice-line locator uses against the one its dynamics use, and read the instellation sweep the repository ships.

Shields-Bitz differs from the other four participants in kind rather than in
tuning.  Its longwave is the linear North (1981) form rather than a lookup
table, so it carries no CO2 dependence at all; it integrates the seasonal
cycle with separate land and ocean columns rather than solving for an annual
mean; and it carries a thermodynamic sea-ice layer with a freezing
temperature and a latent heat, so its ice line is a property of the ice, not
a threshold applied to a temperature field afterwards.  Each of those decides
which parts of the protocol it can answer, and this step establishes them
from the source.

The one measurement here that is not a transcription is the axis test.  The
model's boxes are uniform in sin(latitude), and `build_setup` places them at
arcsin of the box centres.  The ice-line locator `mean_iceline` builds its
own axis instead, `linspace(-90, 90, jmx)`, which is uniform in latitude.
The two disagree, most at the poles, so the reported ice-line latitude is
displaced from the temperature field it was read off.  The displacement is
computed here from the grid, and then applied to the actual benchmark
profiles so the size of the effect is stated where it matters rather than at
its worst.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

# The protocol controls, and the handle Shields-Bitz offers for each.
PROTOCOL_CONTROLS = [
    ("instellation", "scaleQ", True),
    ("obliquity", "obl", True),
    ("CO2 abundance", "-", False),
    ("diffusion coefficient", "Dmag", True),
    ("land fraction", "land", True),
    ("land albedo", "A_l", False),
    ("ocean albedo", "A_o", False),
    ("ice albedo", "Asnow / A_75 / A_50 / A_25 / A_bi", False),
]


def fdictSourceConstants(sRepo):
    """The defaults and compiled-in constants, read out of the source."""
    sSrc = open(os.path.join(sRepo, "EBM_one_file.py")).read()

    def fdFind(sPat, fDefault=None):
        oM = re.search(sPat, sSrc, re.M)
        return float(oM.group(1)) if oM else fDefault

    dictDefaults = {}
    oBlock = re.search(r"DEFAULTS = \{(.*?)\n\}", sSrc, re.S)
    for sKey, sVal in re.findall(r'"(\w+)":\s*([^,\n]+)', oBlock.group(1)):
        sVal = sVal.strip().strip('"')
        try:
            dictDefaults[sKey] = float(sVal)
        except ValueError:
            dictDefaults[sKey] = sVal

    return {
        "dictDefaults": dictDefaults,
        "fPlanckA": dictDefaults["A"],
        "fPlanckB": dictDefaults["B"],
        "fDiffusionMagnitude": dictDefaults["Dmag"],
        "iNumBoxes": int(dictDefaults["jmx"]),
        "fFreezingC": fdFind(r"Tfrz = ([-\d.]+)"),
        "fIceConductivity": fdFind(r"conduct = ([\d.]+)"),
        "fOceanHeatCapacity": dictDefaults["Cw"],
        "fLandHeatCapacity": dictDefaults["Cl"],
        "fLandOceanCoupling": dictDefaults["nu"],
        "iStepsPerYear": int(fdFind(r"nstepinyear = (\d+)")),
        "fFilletLandFraction": fdFind(r"key.startswith\('fillet'\):\s*\n\s*fl\[:\] = ([\d.]+)"),
        "sDiffusionProfile": (re.search(r"D = (Dmag \* \(1 \+ [^\n]+)", sSrc).group(1).strip()
                              if re.search(r"D = (Dmag \* \(1 \+ [^\n]+)", sSrc) else None),
        "fIceLineThresholdC": fdFind(r"def mean_iceline\(results, threshold=([-\d.]+)\)"),
        "sIceLineField": "W_out",
        "sIceLineAxis": (re.search(r"def mean_iceline.*?phi = (np\.linspace\([^\n]+)",
                                   sSrc, re.S).group(1).strip()
                         if re.search(r"def mean_iceline.*?phi = np\.linspace", sSrc, re.S)
                         else None),
        "sDynamicsAxis": (re.search(r"phi = (np\.arcsin\([^\n]+)", sSrc).group(1).strip()
                          if re.search(r"phi = np\.arcsin\(", sSrc) else None),
        "bCarriesCO2": bool(re.search(r"\bpco2|\bfco2|CO2", sSrc)),
        "bSeasonal": True,
        "bThermodynamicIce": bool(re.search(r"def icebalance\(", sSrc)),
    }


def fdictBroadbandAlbedo(sRepo, sStar):
    """The stellar-type broadband albedo set the model uses for its surfaces."""
    sys.path.insert(0, sRepo)
    import EBM_one_file as oEbm
    return {k: float(v) for k, v in oEbm.get_broadband_albedo(sStar).items()}


def faModelAxis(iJmx):
    """Box-centre latitudes as the dynamics place them: uniform in sin(latitude)."""
    fDelx = 2.0 / iJmx
    aXFull = np.arange(-1 + fDelx / 2, 1, fDelx)
    return np.rad2deg(np.arcsin(np.clip(aXFull, -1.0, 1.0)))


def faLocatorAxis(iJmx):
    """Box-centre latitudes as the ice-line locator assumes them: uniform in latitude."""
    return np.linspace(-90.0, 90.0, iJmx)


def fdictAxisDiscrepancy(iJmx):
    """How far the locator's axis sits from the axis the temperatures live on."""
    aModel, aLocator = faModelAxis(iJmx), faLocatorAxis(iJmx)
    aD = aLocator - aModel
    iWorst = int(np.argmax(np.abs(aD)))
    return {"iJmx": iJmx,
            "fMaxOffsetDeg": float(np.max(np.abs(aD))),
            "fRMSOffsetDeg": float(np.sqrt(np.mean(aD ** 2))),
            "fOffsetAtWorstBoxDeg": float(aD[iWorst]),
            "fModelLatitudeAtWorstBox": float(aModel[iWorst]),
            "fLocatorLatitudeAtWorstBox": float(aLocator[iWorst])}


def fdEdgeOnAxis(aLatDeg, aTK, fThresholdK):
    """Northern ice edge on a supplied axis, interpolated between box centres."""
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


def flistAxisEffectOnBenchmarks(listBench, fThresholdK=263.15):
    """The ice edge of each benchmark read off both axes, so the error is priced."""
    listOut = []
    for dictB in listBench:
        aT = np.asarray(dictB["aTsurf"], dtype=float)
        iJmx = len(aT)
        fModel = fdEdgeOnAxis(faModelAxis(iJmx), aT, fThresholdK)
        fLocator = fdEdgeOnAxis(faLocatorAxis(iJmx), aT, fThresholdK)
        listOut.append({"sBenchmark": dictB["sBenchmark"],
                        "fEdgeOnModelAxisDeg": fModel,
                        "fEdgeOnLocatorAxisDeg": fLocator,
                        "fDisplacementDeg": fLocator - fModel})
    return listOut


def fdictShippedSweep(sRepo):
    """The instellation sweep the repository ships, in protocol quantities."""
    sPath = os.path.join(sRepo, "G_dwarf_ws.txt")
    if not os.path.exists(sPath):
        return None
    aRaw = np.loadtxt(sPath)
    return {"sFile": "G_dwarf_ws.txt",
            "sBranch": "warm start",
            "iCases": int(aRaw.shape[0]),
            "aInstellation": aRaw[:, 0].tolist(),
            "aIceEdgeDeg": aRaw[:, 1].tolist(),
            "aTglobK": (aRaw[:, 2] + 273.15).tolist(),
            "sNote": ("written by `warmstart_sweep` at the package defaults, so "
                      "the 25% land FILLET surface; temperatures are stored in "
                      "Celsius and converted here")}


def flistReadinessFindings(dictConst):
    """Which protocol controls Shields-Bitz exposes, and which it cannot express."""
    listOut = []
    for sQuantity, sHandle, bSettable in PROTOCOL_CONTROLS:
        sNote = ("set in the run configuration" if bSettable
                 else "compiled into the broadband albedo table for the stellar type")
        if sQuantity == "CO2 abundance":
            sNote = ("not represented: the longwave is A + B T with A and B "
                     "constants, so no CO2 abundance can be imposed")
        listOut.append({"sQuantity": sQuantity, "sHandle": sHandle,
                        "bNamelistSettable": bSettable, "sNote": sNote})
    listOut.append({
        "sQuantity": "Experiment 4", "sHandle": "-", "bNamelistSettable": False,
        "sNote": "cannot be run: the experiment varies CO2, which the model has no term for"})
    listOut.append({
        "sQuantity": "ice-line definition", "sHandle": "mean_iceline",
        "bNamelistSettable": False,
        "sNote": ("the seasonal mean of the daily ocean freezing latitude, not the "
                  "263.15 K crossing of the annual-mean temperature the protocol asks for")})
    return listOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--shields-repo", required=True)
    oParser.add_argument("--benchmarks-json", required=True)
    oParser.add_argument("--star", default="G")
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictConst = fdictSourceConstants(oArgs.shields_repo)
    listBench = json.load(open(oArgs.benchmarks_json))["listBenchmarks"]
    dictOut = {
        "dictSourceConstants": dictConst,
        "dictBroadbandAlbedo": fdictBroadbandAlbedo(oArgs.shields_repo, oArgs.star),
        "dictAxisDiscrepancy": fdictAxisDiscrepancy(dictConst["iNumBoxes"]),
        "listAxisEffectOnBenchmarks": flistAxisEffectOnBenchmarks(listBench),
        "dictShippedSweep": fdictShippedSweep(oArgs.shields_repo),
        "listFindings": flistReadinessFindings(dictConst),
    }
    json.dump(dictOut, open(oArgs.out, "w"), indent=1)

    print("Shields-Bitz constants:")
    for sK, oV in sorted(dictConst.items()):
        if sK != "dictDefaults":
            print("  %-28s %s" % (sK, oV))
    d = dictOut["dictAxisDiscrepancy"]
    print("\nice-line axis against the dynamics axis (jmx = %d):" % d["iJmx"])
    print("  worst offset %.2f deg at model latitude %.2f (locator says %.2f)"
          % (d["fMaxOffsetDeg"], d["fModelLatitudeAtWorstBox"],
             d["fLocatorLatitudeAtWorstBox"]))
    print("  RMS offset over the grid %.2f deg" % d["fRMSOffsetDeg"])
    for r in dictOut["listAxisEffectOnBenchmarks"]:
        print("  %s: edge %.2f on the model axis, %.2f on the locator axis (%+.2f)"
              % (r["sBenchmark"], r["fEdgeOnModelAxisDeg"],
                 r["fEdgeOnLocatorAxisDeg"], r["fDisplacementDeg"]))
    print("\nprotocol quantities, and where Shields-Bitz keeps them:")
    for r in dictOut["listFindings"]:
        print("  %-24s %-8s %s" % (r["sQuantity"],
                                   "settable" if r["bNamelistSettable"] else "COMPILED",
                                   r["sNote"]))
    print("\nwrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
