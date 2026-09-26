"""Normalise the FILLET Results/ archive for AVALON, HEXTOR, VPLanet-POISE and OPS into one tidy dataset, recording each model's actual column layout.

The four codes write `global_output.dat` files that are nominally the same
FILLET v1.1 product but in fact differ in column count, column ORDER and units.
This step resolves each file's layout explicitly (never by position alone) and
emits (a) a tidy per-case record set, (b) the latitudinal profiles, and (c) an
audit of the layout differences themselves, which is a result in its own right.
"""

import argparse
import json
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# Layout declarations.
#
# Each entry states, for one (model, file-kind) pair, the meaning of every
# numeric column as actually written on disk.  These were established by
# reading the writers:
#   AVALON  : avalon.jl  fillet_global()      -> FILLET v1.1 Table 5 order
#   HEXTOR  : driver.f   line ~1779 (format 768) -> v1.1 four-value form
#   POISE   : fillet/src/proc_poise_out.py    -> land-block then sea-block
#   OPS   : filed under Results/ops/; benchmarks in the v1.0 single-ice-line
#             form, experiments in the v1.1 four-value form
# ---------------------------------------------------------------------------

V11_ICE_KEYS = [
    "IceLineNMaxLand", "IceLineNMinLand", "IceLineNMaxSea", "IceLineNMinSea",
    "IceLineSMaxLand", "IceLineSMinLand", "IceLineSMaxSea", "IceLineSMinSea",
]

LAYOUTS = {
    # AVALON writes the protocol (v1.1 Table 5) order verbatim.
    "avalon": {
        "columns": ["Case", "Inst", "Obl", "XCO2", "Tglob"] + V11_ICE_KEYS
                   + ["Diff", "OLRglob"],
        "trailing_string": "Branch",      # present only in exp3/exp4
        "xco2_unit": "ppm",
        "ice_surfaces": "duplicated",     # land and sea columns carry the same value
    },
    # POISE emits all eight values but groups them land-block then sea-block,
    # i.e. NH-land, SH-land, NH-sea, SH-sea -- NOT the Table 5 order.
    "poise": {
        "columns": ["Case", "Inst", "Obl", "XCO2", "Tglob",
                    "IceLineNMaxLand", "IceLineNMinLand",
                    "IceLineSMaxLand", "IceLineSMinLand",
                    "IceLineNMaxSea", "IceLineNMinSea",
                    "IceLineSMaxSea", "IceLineSMinSea",
                    "Diff", "OLRglob"],
        "trailing_string": None,
        "xco2_unit": "mixing_ratio_in_exp4",   # see resolve_xco2()
        "ice_surfaces": "distinct",
    },
    # HEXTOR has a single surface temperature per belt, so it legitimately
    # reports one max/min pair per hemisphere (11 columns total).
    "hextor": {
        "columns": ["Case", "Inst", "Obl", "XCO2", "Tglob",
                    "IceLineNMax", "IceLineNMin", "IceLineSMax", "IceLineSMin",
                    "Diff", "OLRglob"],
        "trailing_string": None,
        "xco2_unit": "ppm",
        "ice_surfaces": "single",
    },
    # OPS is archived under the directory name `ops`.  Its experiment files
    # are the v1.1 four-value form in protocol order, like HEXTOR's; its
    # BENCHMARK files are the older v1.0 single-ice-line form and carry
    # neither Diff nor OLRglob (see OPS_V10_BENCHMARK_LAYOUT).
    "ops": {
        "columns": ["Case", "Inst", "Obl", "XCO2", "Tglob",
                    "IceLineNMax", "IceLineNMin", "IceLineSMax", "IceLineSMin",
                    "Diff", "OLRglob"],
        "trailing_string": None,
        "xco2_unit": "mixing_ratio_in_benchmarks",
        "ice_surfaces": "single",
    },
}

# POISE's benchmark files predate the v1.1 change and carry only 4 ice columns.
POISE_V10_LAYOUT = ["Case", "Inst", "Obl", "XCO2", "Tglob",
                    "IceLineNMax", "IceLineNMin", "IceLineSMax", "IceLineSMin",
                    "Diff", "OLRglob"]

# OPS's benchmark files are older still: one ice line per hemisphere, and
# no diffusion or OLR column at all.
OPS_V10_BENCHMARK_LAYOUT = ["Case", "Inst", "Obl", "XCO2", "Tglob",
                            "IceLineN", "IceLineS"]

LAT_COLUMNS = ["Lat", "Tsurf", "Asurf", "ATOA", "OLR"]

EXPERIMENTS = ["ben1", "ben2", "ben3", "exp1", "exp1a", "exp2", "exp2a",
               "exp3_cold", "exp3_warm", "exp4_cold", "exp4_warm",
               "exp3", "exp4"]


def flistDataRows(sPath):
    """Return (numeric rows, trailing string tokens, comment lines) of a .dat file."""
    listRows, listTags, listComments = [], [], []
    with open(sPath) as f:
        for sLine in f:
            sStrip = sLine.strip()
            if not sStrip:
                continue
            if sStrip.startswith("#"):
                listComments.append(sStrip)
                continue
            listTok = sStrip.split()
            listNum, sTag = [], None
            for sTok in listTok:
                try:
                    listNum.append(float(sTok))
                except ValueError:
                    sTag = sTok
            listRows.append(listNum)
            listTags.append(sTag)
    return listRows, listTags, listComments


def fsHeaderColumnLine(listComments):
    """Return the '# Case ...' declared-column comment, if the file has one."""
    for sLine in listComments:
        if re.match(r"^#\s*Case\s+Inst\s+Obl", sLine):
            return sLine.lstrip("#").strip()
    return None


def flistResolveLayout(sModel, sExperiment, iNumCols):
    """Choose the on-disk column names for one file, from the writer, not the header."""
    dictLayout = LAYOUTS[sModel]
    listCols = list(dictLayout["columns"])
    if sModel == "poise" and iNumCols == len(POISE_V10_LAYOUT):
        return list(POISE_V10_LAYOUT), "v1.0 (4 ice values)"
    if sModel == "ops" and iNumCols == len(OPS_V10_BENCHMARK_LAYOUT):
        return list(OPS_V10_BENCHMARK_LAYOUT), "v1.0 (2 ice values)"
    if len(listCols) != iNumCols:
        raise ValueError(
            "%s/%s: %d numeric columns on disk but layout declares %d"
            % (sModel, sExperiment, iNumCols, len(listCols)))
    return listCols, "v1.1 (%d ice values)" % sum(1 for c in listCols if c.startswith("IceLine"))


def fdictCanonicalIce(dictRow, sModel):
    """Map a model's ice-edge columns onto the canonical eight v1.1 fields.

    A model with one surface per band (HEXTOR, AVALON) has its single pair
    copied into both the land and the sea slots, and `bIceSurfacesDistinct`
    records that the duplication is ours, not the model's.
    """
    dictOut = {}
    if "IceLineN" in dictRow:
        # v1.0 single ice line per hemisphere.  The protocol's own examples fix
        # the mapping: an ice-free hemisphere puts the line at the pole, so
        # IceLineN = 90 means max = min = 90; otherwise the cap runs from the
        # pole to the reported edge.
        fN, fS = dictRow["IceLineN"], dictRow["IceLineS"]
        dictRow = dict(dictRow)
        dictRow["IceLineNMax"] = 90.0
        dictRow["IceLineNMin"] = fN
        dictRow["IceLineSMax"] = fS
        dictRow["IceLineSMin"] = -90.0
        if fN >= 90.0:
            dictRow["IceLineNMax"] = dictRow["IceLineNMin"] = 90.0
        if fS <= -90.0:
            dictRow["IceLineSMax"] = dictRow["IceLineSMin"] = -90.0
    if "IceLineNMaxLand" in dictRow:
        for sKey in V11_ICE_KEYS:
            dictOut[sKey] = dictRow[sKey]
        dictOut["bIceSurfacesDistinct"] = (LAYOUTS[sModel]["ice_surfaces"] == "distinct")
    else:
        for sSurf in ("Land", "Sea"):
            dictOut["IceLineNMax" + sSurf] = dictRow["IceLineNMax"]
            dictOut["IceLineNMin" + sSurf] = dictRow["IceLineNMin"]
            dictOut["IceLineSMax" + sSurf] = dictRow["IceLineSMax"]
            dictOut["IceLineSMin" + sSurf] = dictRow["IceLineSMin"]
        dictOut["bIceSurfacesDistinct"] = False
    return dictOut


def fdResolveXCO2(fRaw, sModel, sExperiment):
    """Return CO2 abundance in ppm.

    POISE writes `dpCO2` straight through for Experiment 4, which is the CO2
    partial pressure in bar (numerically the mixing ratio at p_total = 1 bar),
    not ppm as Table 5 requires.  The FILLET plotting script patches this with
    `XCO2 *= 1e6`; we do the same, but record it.
    """
    if sModel == "poise" and sExperiment.startswith("exp4") and fRaw < 1.0:
        return fRaw * 1.0e6, True
    # OPS writes the benchmark CO2 as a volume mixing ratio (2.8e-4) under a
    # header that says ppm, while its own experiment files use ppm.  One
    # submission, two units for one quantity.
    if sModel == "ops" and sExperiment.startswith("ben") and fRaw < 1.0:
        return fRaw * 1.0e6, True
    return fRaw, False


def fsClassifyState(fNMax, fNMin):
    """FILLET climate state from the northern-hemisphere ice-edge pair."""
    if fNMax == fNMin:
        return "ice_free"
    if fNMax >= 90.0 and fNMin > 0.0:
        return "ice_cap"
    if fNMax >= 90.0 and fNMin == 0.0:
        return "snowball"
    if fNMax < 90.0 and fNMin == 0.0:
        return "ice_belt"
    return "unclassified"


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--results-dir", required=True,
                         help="Path to the FILLET Results/ directory")
    oParser.add_argument("--models", nargs="+",
                         default=["avalon", "hextor", "poise", "ops"])
    oParser.add_argument("--out-global", required=True)
    oParser.add_argument("--out-lat", required=True)
    oParser.add_argument("--out-audit", required=True)
    oArgs = oParser.parse_args()

    listGlobal, listAudit = [], []
    dictLat = {}

    for sModel in oArgs.models:
        for sExp in EXPERIMENTS:
            sGlobPath = os.path.join(oArgs.results_dir, sModel, sExp,
                                     "global_output.dat")
            if not os.path.exists(sGlobPath):
                continue
            listRows, listTags, listComments = flistDataRows(sGlobPath)
            iNumCols = len(listRows[0])
            listCols, sFormatTag = flistResolveLayout(sModel, sExp, iNumCols)
            sDeclared = fsHeaderColumnLine(listComments)

            listAudit.append({
                "sModel": sModel,
                "sExperiment": sExp,
                "sPath": os.path.relpath(sGlobPath, oArgs.results_dir),
                "iNumericColumns": iNumCols,
                "iNumCases": len(listRows),
                "sResolvedFormat": sFormatTag,
                "saResolvedColumns": listCols,
                "sDeclaredHeader": sDeclared,
                "bHeaderMatchesData": (sDeclared is not None
                                       and len(sDeclared.split()) == iNumCols),
                "bHasTrailingBranchColumn": listTags[0] is not None,
                "sIceSurfaceTreatment": LAYOUTS[sModel]["ice_surfaces"],
                "bHasDiffusionColumn": "Diff" in listCols,
                "bHasOLRColumn": "OLRglob" in listCols,
            })

            for iRow, listVals in enumerate(listRows):
                dictRow = dict(zip(listCols, listVals))
                fXCO2, bPatched = fdResolveXCO2(dictRow["XCO2"], sModel, sExp)
                dictRec = {
                    "sModel": sModel,
                    "sExperiment": sExp,
                    "iCase": int(dictRow["Case"]),
                    "fInst": dictRow["Inst"],
                    "fObl": dictRow["Obl"],
                    "fXCO2ppm": fXCO2,
                    "bXCO2UnitPatched": bPatched,
                    "fTglob": dictRow["Tglob"],
                    "fDiffReported": dictRow.get("Diff"),
                    "fOLRglob": dictRow.get("OLRglob"),
                    "sBranch": listTags[iRow],
                }
                dictRec.update(fdictCanonicalIce(dictRow, sModel))
                dictRec["sStateNH"] = fsClassifyState(
                    dictRec["IceLineNMaxSea"], dictRec["IceLineNMinSea"])
                dictRec["sStateNHLand"] = fsClassifyState(
                    dictRec["IceLineNMaxLand"], dictRec["IceLineNMinLand"])
                listGlobal.append(dictRec)

            # Latitudinal profiles, where archived (benchmarks only).
            sLatPath = os.path.join(oArgs.results_dir, sModel, sExp,
                                    "case_0", "lat_output.dat")
            if os.path.exists(sLatPath):
                listLatRows, _, _ = flistDataRows(sLatPath)
                aLat = np.array([r[:5] for r in listLatRows], dtype=float)
                dictLat["%s/%s" % (sModel, sExp)] = {
                    "saColumns": LAT_COLUMNS,
                    "iNumLatitudes": aLat.shape[0],
                    "iNumColumnsOnDisk": len(listLatRows[0]),
                    "aaData": aLat.tolist(),
                }

    with open(oArgs.out_global, "w") as f:
        json.dump({"listRecords": listGlobal}, f, indent=1)
    with open(oArgs.out_lat, "w") as f:
        json.dump(dictLat, f, indent=1)
    with open(oArgs.out_audit, "w") as f:
        json.dump({"listFiles": listAudit}, f, indent=1)

    print("parsed %d global cases across %d files; %d latitudinal profiles"
          % (len(listGlobal), len(listAudit), len(dictLat)))


if __name__ == "__main__":
    main()
