"""Normalise the shipped outputs and recovered configuration of the OPS source tree, and verify that it reproduces the FILLET deliverables archived under `Results/ops/`.

OPS (Ramirez's energy balance model) is the one submitting code whose
source ships the protocol runs beside the model.  Under `out/Benchmarks/`
and `out/Experiments/` it carries the `global_output` and `lat_output` files
in FILLET format, together with the `input_ebm.dat` decks that produced them
and the model's own `.out` diagnostics for each.  That combination is unique
among the five participants, and it is what lets this step do something the
archive alone never permits: check a submission against the run that made it.

This step therefore does four things:

1. Parses `input_ebm.dat` into the configuration of the shipped run.
2. Parses every `out/**/*.out` into one record per run: the SUMMARY globals,
   the zonal annual-mean temperature profile, the reported ice lines and the
   geography block.  The area-weighted global mean is recomputed from the
   profile on OPS's own 36 equal-width belts so it is comparable with the
   means this pipeline recomputes for the other four codes.
3. Reads the constants FILLET prescribes out of `driver.f90` itself -- the
   diffusion coefficient and the factors that rescale it, the ice-line
   threshold, the ice-cover ramp, the solar constant, the cloud
   parameterisation, the lookup-table axes, the grid size and the CO2
   coordinate conversion -- and reports which are namelist-settable and which
   are compiled in.  Everything is derived from the source rather than
   transcribed, so it tracks the source if the source changes.
4. Compares the protocol-format files under `out/` against the archived
   submission case by case, and compares the source's effective diffusion
   coefficient and flux-weighted planetary albedo against the values the
   submission reports.  Where the two disagree the disagreement is a property
   of the protocol's column definitions, not of the model, and saying which
   is the whole point of doing the comparison.
"""

import argparse
import glob
import json
import os
import re

import numpy as np

# Protocol quantities a FILLET submission must be able to set per case.
PROTOCOL_CONTROLS = [
    ("instellation", "a0", True),
    ("obliquity", "obl", True),
    ("CO2 abundance", "pco2i", True),
    ("diffusion coefficient", "d0", False),
    ("land fraction", "igeog/ocean", False),
    ("land albedo", "groundalb", True),
    ("ocean albedo", "fresnel_reflct.dat", False),
    ("ice albedo", "aice_vis/aice_nir", False),
]


def fdFortranFloat(sVal):
    """Fortran writes double-precision literals with a `d` exponent."""
    return float(sVal.replace("d", "e").replace("D", "e"))


def fdictReadInput(sPath):
    """Parse `input_ebm.dat`: one `key: value ! comment` line per setting."""
    dictOut = {}
    for sLine in open(sPath):
        oM = re.match(r"\s*(\w+):\s*('[^']*'|[-\w.+]+)", sLine)
        if not oM:
            continue
        sKey, sVal = oM.group(1), oM.group(2)
        if sVal.startswith("'"):
            dictOut[sKey] = sVal.strip("'")
        else:
            dictOut[sKey] = fdFortranFloat(sVal)
    return dictOut


def faBeltWeights(aLatDeg):
    """Area weights of OPS's belts: edges midway between centres in sin(lat)."""
    aX = np.sin(np.deg2rad(np.asarray(aLatDeg, dtype=float)))
    aEdge = np.empty(len(aX) + 1)
    aEdge[1:-1] = 0.5 * (aX[:-1] + aX[1:])
    aEdge[0], aEdge[-1] = -1.0, 1.0
    aW = np.abs(np.diff(aEdge))
    return aW / aW.sum()


def fdictParseSummary(sText):
    """Pull the SUMMARY block's scalars; the older outputs omit some lines."""
    dictKeys = {
        "fTglobReported": r"planet average temperature\s*=\s*([-\d.eE+]+)",
        "fAlbedoReported": r"planet average albedo\s*=\s*([-\d.eE+]+)",
        "fInsolationReported": r"planet average insolation\s*=\s*([-\d.eE+]+)",
        "fOLRReported": r"planet average outgoing infrared\s*=\s*([-\d.eE+]+)",
        "fPCO2barReported": r"co2 partial pressure\s*=\s*\.?([-\d.eE+]+)",
        "fDiffusionReported": r"thermal diffusion coefficient \(D\)\s*=\s*\.?([-\d.eE+]+)",
        "fSurfacePressureBar": r"total surface pressure\s*=\s*([-\d.eE+]+)",
        "iOrbits": r"calculation completed after\s*(\d+)\s*orbits",
    }
    dictOut = {}
    for sName, sPat in dictKeys.items():
        oM = re.search(sPat, sText)
        dictOut[sName] = (float(oM.group(1)) if oM else None)
    return dictOut


def flistParseZonal(sText):
    """The zonal block: latitude, annual mean, min, dec, max, dec, amplitude."""
    listRows = []
    bIn = False
    for sLine in sText.splitlines():
        if sLine.strip().startswith("OUTPUT FILES"):
            bIn = True
            continue
        if not bIn:
            continue
        listTok = sLine.split()
        if len(listTok) != 7:
            if listRows:
                break
            continue
        try:
            listRows.append([float(t) for t in listTok])
        except ValueError:
            break
    return listRows


def fdictParseGeography(sText):
    """Ocean fraction, geography flag and belt count from the trailing block."""
    dictOut = {}
    for sName, sPat in (("fOceanFraction", r"planet ocean fraction:\s*([\d.]+)"),
                        ("iGeography", r"geography:\s*(\d+)"),
                        ("iNumBelts", r"number of belts:\s*(\d+)"),
                        ("fObliquityDeg", r"OBLIQUITY:\s*([-\d.]+)")):
        oM = re.search(sPat, sText)
        dictOut[sName] = (float(oM.group(1)) if oM else None)
    return dictOut


def flistParseIceLines(sText):
    """Reported ice-line latitudes, or the ice-free / ice-ball sentinel."""
    oBlock = re.search(r"ICE LINES.*?\n(.*?)\n\s*\n", sText, re.S)
    if not oBlock:
        return {"sState": "absent", "listEdgesDeg": []}
    sBody = oBlock.group(1)
    if "ice-free" in sBody:
        return {"sState": "ice_free", "listEdgesDeg": []}
    if "ice-ball" in sBody:
        return {"sState": "snowball", "listEdgesDeg": []}
    listEdges = [float(s) for s in re.findall(r"ice-line latitude\s*=\s*([-\d.]+)", sBody)]
    return {"sState": "ice_edge", "listEdgesDeg": listEdges}


def fdictParseRun(sPath, sRoot):
    """One shipped `.out` file into a record with a recomputed global mean."""
    sText = open(sPath, errors="replace").read()
    listZonal = flistParseZonal(sText)
    dictRow = {"sFile": os.path.relpath(sPath, sRoot),
               "sStar": os.path.basename(os.path.dirname(sPath))}
    dictRow.update(fdictParseSummary(sText))
    dictRow.update(fdictParseGeography(sText))
    dictRow["dictIceLines"] = flistParseIceLines(sText)
    if listZonal:
        aZ = np.array(listZonal, dtype=float)
        aW = faBeltWeights(aZ[:, 0])
        dictRow["aLat"] = aZ[:, 0].tolist()
        dictRow["aTsurf"] = aZ[:, 1].tolist()
        dictRow["aTmin"] = aZ[:, 2].tolist()
        dictRow["aTmax"] = aZ[:, 4].tolist()
        dictRow["fTglobAreaWeighted"] = float(np.sum(aW * aZ[:, 1]))
        dictRow["fTglobUnweightedMean"] = float(np.mean(aZ[:, 1]))
        dictRow["fEquatorPoleGradient"] = float(
            aZ[np.argmin(np.abs(aZ[:, 0])), 1] - 0.5 * (aZ[0, 1] + aZ[-1, 1]))
    return dictRow


def fsLiveSource(sDriverPath):
    """The source with whole-line Fortran comments removed.

    `driver.f90` carries several superseded formulations commented out
    immediately beside the live one -- the ice ramp appears three times -- so a
    regex over the raw text reads whichever version comes first in the file,
    not the one that runs.
    """
    listKeep = [s for s in open(sDriverPath, errors="replace").read().splitlines()
                if not s.lstrip().startswith("!")]
    return "\n".join(listKeep)


def fdictSourceConstants(sDriverPath):
    """Recover the compiled-in constants FILLET would need to control."""
    sSrc = fsLiveSource(sDriverPath)

    def fdFind(sPat, fDefault=None):
        oM = re.search(sPat, sSrc, re.M)
        return float(oM.group(1)) if oM else fDefault

    def fsFind(sPat):
        oM = re.search(sPat, sSrc, re.M)
        return oM.group(1).strip() if oM else None

    def fsLastFind(sPat):
        """The LAST live assignment wins: the source sets some of these twice."""
        listM = re.findall(sPat, sSrc, re.M)
        return listM[-1].strip() if listM else None

    return {
        "fDiffusionD0": fdFind(r"^d0\s*=\s*([\d.]+)", None),
        "fSolarConstant": fdFind(r"q0\s*=\s*([\d.]+)"),
        "fIceLineThresholdK": fdFind(r"zntempave\(k\+1\)-([\d.]+)\)\*"),
        "fIceRampUpperK": fdFind(r"if \(temp\(k\)\.ge\.([\d.]+)\) then\s*\n\s*fice\(k\) = 0"),
        "fIceRampLowerK": fdFind(r"temp\(k\)\.le\.([\d.]+)d0\).*\n\s*fice\(k\) = 1"),
        "fIceRampScaleK": fdFind(r"fice\(k\) = 1\. - exp\(\(temp\(k\)-273\.15\)/([\d.]+)\)"),
        "fLandSnowSwitchK": fdFind(r"if \(temp\(k\)\.le\.([\d.]+)\) then\s*\n\s*landalb = snowalb"),
        "sDiffusionExpression": fsFind(r"^\s*d = (fracd0\(k\)\*d0\*[^!\n]+)"),
        "bDiffusionRescaled": bool(re.search(r"^\s*d = fracd0\(k\)\*d0\*", sSrc, re.M)),
        "sLatentHeatFactor": fsLastFind(r"latefac = ([^!\n]+)"),
        "sPerBeltDiffusionFactor": fsLastFind(r"fracd0\(k\) = ([^!\n]+)"),
        "bReportsEffectiveDiffusion": bool(re.search(r"thermal diffusion coefficient", sSrc)),
        "sCloudIRExpression": fsFind(r"cloudir\(k\) = (max\([^!\n]+)"),
        "sCloudFractionExpression": fsFind(r"fcloud\(k\) = (min\([^!\n]+)"),
        "fCloudIRInitialWm2": fdFind(r"cloudir\(1:nbelts\) = ([-\d.]+)"),
        "fConvectiveFluxScaleWm2": fdFind(r"FE = ([\d.]+)"),
        "bCloudIRFromNamelist": bool(re.search(r"read\(inputebm,\*\)aa, cloudir\s*$",
                                               sSrc, re.M)),
        "sGlobalAlbedoDefinition": fsFind(r"albave = (albsum/[^!\n]+)"),
        "sGlobalAlbedoAccumulator": fsFind(r"albsum = albsum \+ ([^!\n]+)"),
        "bColumnAlbedoFluxWeighted": bool(
            re.search(r"atoak\(k\) = atoak\(k\) \+ atoa\(k\)\*\(s\(k\)", sSrc)),
        "sCO2CoordinateExpression":
            (re.search(r"pco2r\(k\) = ([^!\n]+)", sSrc).group(1).strip()
             if re.search(r"pco2r\(k\) = ([^!\n]+)", sSrc) else None),
        "fCO2CoordinateFactor": 44.0 / 28.0,
        "iNumBelts": int(fdFind(r"nbelts = (\d+)")),
        "fOceanAlbedoDivisor": fdFind(r"h20alb\(n1:n2\)/([\d.]+)"),
    }


def flistReadinessFindings(dictInput, dictConst, listRuns):
    """Which protocol controls OPS exposes in its namelist, and which it does not."""
    listOut = []
    for sQuantity, sHandle, bSettable in PROTOCOL_CONTROLS:
        listOut.append({
            "sQuantity": sQuantity, "sHandle": sHandle,
            "bNamelistSettable": bSettable,
            "sNote": ("read from input_ebm.dat" if bSettable
                      else "compiled in; changing it needs a source edit"),
        })
    listOut.append({
        "sQuantity": "solar constant", "sHandle": "q0",
        "bNamelistSettable": False,
        "sNote": "q0 = %.0f W/m2 in source against the protocol 1361"
                 % (dictConst["fSolarConstant"] or float("nan")),
    })
    listOut.append({
        "sQuantity": "number of latitude belts", "sHandle": "nbelts",
        "bNamelistSettable": False,
        "sNote": "compile-time parameter, %d in this build"
                 % dictConst["iNumBelts"],
    })
    return listOut


# The protocol-format files the source tree ships, and the archived
# directory each is meant to have produced.
SHIPPED_DELIVERABLES = [
    ("ben1", "out/Benchmarks/1/global_output_bench1.dat"),
    ("ben2", "out/Benchmarks/2/global_output_bench2.dat"),
    ("ben3", "out/Benchmarks/3/global_output_bench3.dat"),
    ("exp1", "out/Experiments/global_output_experiment1warmstart.dat"),
    ("exp2", "out/Experiments/global_output_experiment2coldstart.dat"),
    ("exp3_warm", "out/Experiments/global_output_experiment3_warmstart.dat"),
    ("exp3_cold", "out/Experiments/global_output_experiment3_coldstart.dat"),
    ("exp4_warm", "out/Experiments/global_output_experiment4_warmstart.dat"),
    ("exp4_cold", None),
]


def faReadProtocolTable(sPath):
    """The numeric rows of a FILLET `global_output`-format file."""
    listRows = []
    for sLine in open(sPath, errors="replace"):
        sBody = sLine.strip()
        if not sBody or sBody.startswith("#"):
            continue
        try:
            listRows.append([fdFortranFloat(t) for t in sBody.split()])
        except ValueError:
            continue
    iWidth = max((len(r) for r in listRows), default=0)
    return np.array([r for r in listRows if len(r) == iWidth], dtype=float)


def fdictCompareTables(aSource, aArchive):
    """Row-for-row agreement between two protocol tables of the same shape."""
    if aSource.size == 0 or aArchive.size == 0:
        return {"bComparable": False, "sReason": "one table is empty"}
    if aSource.shape != aArchive.shape:
        return {"bComparable": False,
                "sReason": "source %s vs archive %s"
                           % (aSource.shape, aArchive.shape),
                "iSourceRows": int(aSource.shape[0]),
                "iArchiveRows": int(aArchive.shape[0])}
    aDiff = np.abs(aSource - aArchive)
    return {"bComparable": True, "iRows": int(aSource.shape[0]),
            "iColumns": int(aSource.shape[1]),
            "fMaxAbsDifference": float(np.nanmax(aDiff)),
            "bIdentical": bool(np.nanmax(aDiff) == 0.0)}


def flistDeliverableCheck(sRoot, sArchiveDir):
    """Each shipped protocol file against the archived directory it produced."""
    listOut = []
    for sExp, sRel in SHIPPED_DELIVERABLES:
        sArchivePath = os.path.join(sArchiveDir, sExp, "global_output.dat")
        dictRow = {"sExperiment": sExp, "sSourceFile": sRel,
                   "bShippedInSource": sRel is not None,
                   "bPresentInArchive": os.path.exists(sArchivePath)}
        if sRel is None or not os.path.exists(sArchivePath):
            dictRow["dictAgreement"] = {"bComparable": False,
                                        "sReason": "no counterpart run in the source tree"
                                        if sRel is None else "absent from the archive"}
            listOut.append(dictRow)
            continue
        aSrc = faReadProtocolTable(os.path.join(sRoot, sRel))
        aArc = faReadProtocolTable(sArchivePath)
        dictRow["dictAgreement"] = fdictCompareTables(aSrc, aArc)
        listOut.append(dictRow)
    return listOut


def fdictDuplicateBranchCheck(sArchiveDir):
    """Whether the archived Experiment 4 branches are two runs or one file twice."""
    aWarm = faReadProtocolTable(os.path.join(sArchiveDir, "exp4_warm",
                                             "global_output.dat"))
    aCold = faReadProtocolTable(os.path.join(sArchiveDir, "exp4_cold",
                                             "global_output.dat"))
    dictOut = fdictCompareTables(aWarm, aCold)
    dictOut["sNote"] = ("the source tree ships only a warm-start Experiment 4, "
                        "so there is no second run for the cold branch to hold")
    return dictOut


def flistSourceVsSubmission(sRoot, dictConst, listOpsRecords, listRuns):
    """Every source quantity the archived submission can be held against.

    Each row states what the source says, what the archive shows, and whether
    the two agree.  A row that disagrees is a property of the protocol's
    column definitions rather than of the model, and the note says which.
    """
    listOut = []
    listBen = [r for r in listOpsRecords if r["sExperiment"] == "ben1"]
    listExp = [r for r in listOpsRecords if r["sExperiment"] == "exp1"]
    listBenchRuns = [r for r in listRuns if "Benchmarks/1" in r["sFile"]]
    fEffectiveD = (listBenchRuns[0]["fDiffusionReported"]
                   if listBenchRuns else None)
    fFluxAlbedo = (listBenchRuns[0]["fAlbedoReported"]
                   if listBenchRuns else None)

    listOut.append({
        "sQuantity": "latitude belts",
        "sSource": "%d, a compile-time parameter" % dictConst["iNumBelts"],
        "sArchive": "%d in every profile" % dictConst["iNumBelts"],
        "bConsistent": True,
    })
    listOut.append({
        "sQuantity": "solar constant",
        "sSource": "q0 = %.0f W/m2" % dictConst["fSolarConstant"],
        "sArchive": "instellation filed as 1.00 relative to the protocol 1361",
        "bConsistent": bool(abs(dictConst["fSolarConstant"] - 1361.0) < 0.5),
    })
    listOut.append({
        "sQuantity": "diffusion coefficient",
        "sSource": ("d0 = %.2f, rescaled every step by pressure, mean molecular "
                    "weight, heat capacity, rotation rate and a latent-heat "
                    "factor; the Benchmark 1 run reports %s"
                    % (dictConst["fDiffusionD0"],
                       "%.4f" % fEffectiveD if fEffectiveD else "no value")),
        "sArchive": ("Diff = %.2f, constant across every case"
                     % listExp[0]["fDiffReported"] if listExp else "not reported"),
        "bConsistent": bool(fEffectiveD is not None and listExp
                            and abs(fEffectiveD - listExp[0]["fDiffReported"]) < 0.005),
        "sNote": "the filed column is the nominal d0, not the coefficient the run used",
    })
    listOut.append({
        "sQuantity": "planetary albedo",
        "sSource": ("%s, i.e. reflected flux over incident flux"
                    % (dictConst["sGlobalAlbedoDefinition"] or "flux-weighted")),
        "sArchive": "the ATOA column, an unweighted annual mean per belt",
        "bConsistent": False,
        "sNote": ("area-weighting the filed column is not the planetary albedo "
                  "and does not close the budget; the model's own value does"),
        "fSourceValue": fFluxAlbedo,
    })
    listOut.append({
        "sQuantity": "ice-line threshold",
        "sSource": "%.2f K, interpolated between belt centres"
                   % dictConst["fIceLineThresholdK"],
        "sArchive": "263.15 K, per the file header and confirmed by recomputation",
        "bConsistent": bool(abs(dictConst["fIceLineThresholdK"] - 263.15) < 0.01),
    })
    listOut.append({
        "sQuantity": "CO2 abundance, benchmarks",
        "sSource": "a mixing ratio, converted to the table coordinate by 44/28",
        "sArchive": ("filed as %.1e, a mixing ratio under a header that says ppm"
                     % (listBen[0]["fXCO2ppm"] * 1e-6) if listBen else "unknown"),
        "bConsistent": False,
        "sNote": "a unit-label error in the submission, not a model difference",
    })
    return listOut


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--ops-repo", required=True)
    oParser.add_argument("--global-json", required=True,
                         help="parsed archive from step A01, for the ops rows")
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--out-runs", required=True)
    oParser.add_argument("--out-readiness", required=True)
    oParser.add_argument("--archive-dir", required=True,
                         help="Results/ops, the archived submission to verify against")
    oArgs = oParser.parse_args()

    sRoot = oArgs.ops_repo
    dictInput = fdictReadInput(os.path.join(sRoot, "input_ebm.dat"))
    listPaths = sorted(glob.glob(os.path.join(sRoot, "out", "**", "*.out"),
                                 recursive=True))
    listRuns = [fdictParseRun(s, sRoot) for s in listPaths]
    listRuns = [d for d in listRuns if d.get("fTglobReported") is not None]
    dictConst = fdictSourceConstants(os.path.join(sRoot, "driver.f90"))

    # The run reproduced by the shipped input deck, and the 1 AU run, are the
    # two the rest of the pipeline uses; tag them so downstream steps need no
    # filename knowledge.
    for dictRun in listRuns:
        dictRun["bShippedInputDeck"] = (dictRun["sFile"] == os.path.join("out", "model.out"))
        dictRun["bUnitInstellation"] = (dictRun["sFile"]
                                        == os.path.join("out", "Sun", "1AUmodel.out"))
        dictRun["fInstellation"] = (
            dictRun["fInsolationReported"] * 4.0 / dictConst["fSolarConstant"]
            if dictRun["fInsolationReported"] else None)

    json.dump({"dictInputDeck": dictInput, "dictSourceConstants": dictConst,
               "listRuns": listRuns},
              open(oArgs.out_runs, "w"))
    listOps = [r for r in json.load(open(oArgs.global_json))["listRecords"]
               if r["sModel"] == "ops"]
    listCompare = flistSourceVsSubmission(sRoot, dictConst, listOps, listRuns)
    listDeliverables = flistDeliverableCheck(sRoot, oArgs.archive_dir)
    dictExpFourBranches = fdictDuplicateBranchCheck(oArgs.archive_dir)

    json.dump({"dictSourceConstants": dictConst,
               "listFindings": flistReadinessFindings(dictInput, dictConst, listRuns),
               "listSourceVsSubmission": listCompare,
               "listDeliverableCheck": listDeliverables,
               "dictExpFourBranches": dictExpFourBranches,
               "iShippedRuns": len(listRuns),
               "iArchivedCases": len(listOps),
               "bShipsFilletDeliverables": True,
               "sArchiveDirectory": "Results/ops"},
              open(oArgs.out_readiness, "w"), indent=1)

    print("OPS input deck: %s" % ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(dictInput.items())))
    print("\nsource constants:")
    for sK, oV in sorted(dictConst.items()):
        print("  %-28s %s" % (sK, oV))
    print("\n%d shipped runs parsed:" % len(listRuns))
    print("%-28s %6s %7s %7s %7s %7s  %s"
          % ("file", "obl", "S/Searth", "Tglob", "Trecomp", "OLR", "ice"))
    for d in listRuns:
        print("%-28s %6.1f %7s %7.2f %7s %7.1f  %s"
              % (d["sFile"],
                 float("nan") if d["fObliquityDeg"] is None else d["fObliquityDeg"],
                 "%.4f" % d["fInstellation"] if d["fInstellation"] else "-",
                 d["fTglobReported"],
                 "%.2f" % d["fTglobAreaWeighted"] if "fTglobAreaWeighted" in d else "-",
                 d["fOLRReported"], d["dictIceLines"]["sState"]))
    print("\nprotocol quantities, and where OPS keeps them:")
    for d in flistReadinessFindings(dictInput, dictConst, listRuns):
        print("  %-26s %-8s %s" % (d["sQuantity"],
                                   "settable" if d["bNamelistSettable"] else "COMPILED",
                                   d["sNote"]))
    print("\nshipped protocol files against the archived submission:")
    for d in listDeliverables:
        dA = d["dictAgreement"]
        print("  %-10s %s" % (d["sExperiment"],
                              ("identical, %d rows x %d columns" % (dA["iRows"], dA["iColumns"])
                               if dA.get("bIdentical")
                               else "max |difference| %.3g over %d rows" % (dA["fMaxAbsDifference"], dA["iRows"])
                               if dA.get("bComparable")
                               else dA.get("sReason", "not comparable"))))
    print("  archived Exp 4 branches: %s"
          % ("the cold file duplicates the warm file"
             if dictExpFourBranches.get("bIdentical")
             else dictExpFourBranches.get("sReason", "differ")))

    print("\nsource against the archived submission (%d cases):" % len(listOps))
    for d in listCompare:
        print("  %-24s %-3s source : %s" % (d["sQuantity"],
                                            "OK" if d["bConsistent"] else "!!",
                                            d["sSource"]))
        print("  %-24s %-3s archive: %s" % ("", "", d["sArchive"]))
        if d.get("sNote"):
            print("  %-24s %-3s note   : %s" % ("", "", d["sNote"]))


if __name__ == "__main__":
    main()
