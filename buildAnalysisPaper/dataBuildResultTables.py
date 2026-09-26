"""Emit the LaTeX tables of the code comparison directly from the analysis outputs, so that every number printed in the report is produced by the pipeline rather than transcribed."""

import argparse
import json
import os

S_EARTH = 1361.0



# --- table assembly --------------------------------------------------------
#
# Each table is emitted as a COMPLETE tabular environment, because LaTeX's
# \input cannot be used from inside a tabular (the file hooks it runs are not
# alignment-safe), so the report inputs whole tabulars inside its own
# \begin{table} wrappers.

HEAD_BENCH = (r"Case & Model & $N_{\rm lat}$ & $T_{\rm glob}$ & "
              r"$\alpha_{\rm surf}$ & $\alpha_{\rm TOA}$ & ASR & OLR & imbal. \\"
              "\n"
              r" & & & (K) & & & ($\Wm$) & ($\Wm$) & ($\Wm$) \\")

HEAD_RADIATION = (r"Model & Scheme & OLR & $A_{\rm eff}$ & $B$ & "
                  r"$-\mathrm{d\,OLR}/\mathrm{d}\log X$ & "
                  r"$1\to10^5$~ppm \\" "\n"
                  r" & & ($\Wm$) & ($\Wm$) & ($\WmK$) & ($\Wm$/dec) & ($\Wm$) \\")

HEAD_ATTRIBUTION = (r"Case & Pair & $S$ & $\alpha$ & $A$ & $B$ & "
                    r"$\Delta T_{\rm 0D}$ & $\Delta T_{\rm act}$ \\")

HEAD_COMPLIANCE = (r"Model & Experiment & Cases & Columns & Resolved format & "
                   r"Header matches \\")

HEAD_ICELINES = (r" & & \multicolumn{2}{c}{$-10\degr$C} & "
                 r"\multicolumn{2}{c}{$-2\degr$C} & "
                 r"\multicolumn{2}{c}{$0\degr$C} & \\" "\n"
                 r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
                 r"Model & $N_{\rm lat}$ & interp. & snap & interp. & snap & "
                 r"interp. & snap & range \\")

HEAD_STATES = (r"Exp. & Model & ice free & caps & belt & snowball & "
               r"$T_{\min}$ (K) & $T_{\max}$ (K) & land$\ne$sea \\")

HEAD_DRIFT = (r"Model & Experiment & Cases & mean $\Delta T$ & RMS $\Delta T$ & "
              r"$\max|\Delta T|$ & state changes \\" "\n"
              r" & & & (K) & (K) & (K) & \\")

HEAD_OPSREADY = (r"Quantity & Source tree & Archived submission \\")

HEAD_BRANCHORDER = (r"Model & Comparison & Matched & Identical & "
                    r"\multicolumn{2}{c}{inside protocol grid} & "
                    r"\multicolumn{2}{c}{outside it} \\" "\n"
                    r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
                    r" & & & & viol. & worst (K) & viol. & worst (K) \\")

HEAD_HYSTERESIS = (r" & \multicolumn{4}{c}{Experiment 3 (instellation)} & "
                   r"\multicolumn{2}{c}{Experiment 4 (CO$_2$, ppm)} & \\" "\n"
                   r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}" "\n"
                   r"Model & glaciation & deglaciation & width & ($\Wm$) & "
                   r"glaciation & deglaciation & same grid \\")


def fsWrap(sCols, sHead, sBody, sSize="small"):
    """Assemble one complete, self-contained tabular environment."""
    return "\n".join([r"\%s" % sSize,
                      r"\begin{tabular}{%s}" % sCols,
                      r"\toprule", sHead, r"\midrule", sBody,
                      r"\bottomrule", r"\end{tabular}"])


def fsEsc(s):
    return s.replace("_", r"\_")


LIST_MODEL_NAMES = (("avalon", "AVALON"), ("hextor", "HEXTOR"),
                    ("poise", "VPLanet-POISE"), ("ops", "OPS"),
                    ("shields_bitz", "Shields-Bitz"))


def fsTableBenchmarkBudget(dictBench):
    listRows = []
    for sB in ("ben1", "ben2", "ben3"):
        for sM, sName in LIST_MODEL_NAMES:
            d = dictBench.get((sM, sB))
            if d is None:
                continue
            listRows.append(
                r"%s & %s & %d & %.2f & %.4f & %.4f & %.1f & %.1f & %+.1f \\"
                % (sB.replace("ben", "Ben "), sName, d["iNumLatitudes"],
                   d["fTglobAreaWeighted"], d["fAsurfAreaWeighted"],
                   d["fATOAInsolationWeighted"], d["fAbsorbedShortwave"],
                   d["fOLRAreaWeighted"], d["fNetImbalance"]))
        if sB != "ben3":
            listRows.append(r"\addlinespace")
    return "\n".join(listRows)


def fsTableRadiation(dictRad):
    listRows = []
    listOrder = [("avalon", "AVALON", "linear, Myhre CO$_2$"),
                 ("hextor", "HEXTOR", "lookup table"),
                 ("ops", "OPS", "lookup table"),
                 ("shields_bitz", "Shields-Bitz", "linear, no CO$_2$ term"),
                 ("poise_fixed", "VPLanet-POISE", "fixed linearization, Ben, Exp 1--3"),
                 ("poise_wk97", "VPLanet-POISE", "WK97 polynomial, Exp 4"),
                 ("poise_hm16", "VPLanet-POISE", "HM16 polynomial"),
                 ("poise_sms09", "VPLanet-POISE", "SMS09 grey, code default")]
    for sKey, sName, sDesc in listOrder:
        d = dictRad["dictLocalCoefficients"][sKey]["at288K_280ppm"]
        e = dictRad["dictExp4Forcing"][sKey]
        # A CO2-independent scheme differences to a signed zero; print it as 0.
        fForcing = d["fCO2ForcingPerDecade"] + 0.0
        listRows.append(
            r"%s & %s & %.1f & %.2f & %.3f & %.2f & %.1f \\"
            % (sName, sDesc, d["fOLR"], d["fEffectiveA"], d["fPlanckB"],
               0.0 if abs(fForcing) < 5e-3 else fForcing,
               e["fTotalForcing_Wm2"]))
    return "\n".join(listRows)


def fsTableAttribution(listAttr):
    """Only the pairs whose zero-dimensional balance closes.

    The decomposition is exact only to the extent that Equation (1) reproduces
    each model's own temperature.  It does so for four of the five codes and
    not for OPS, whose archived annual-mean energy budget does not close
    because of the ATOA column definition, so OPS's pairs are omitted here
    and its residual is reported in the text instead of being decomposed into
    terms that do not add up.
    """
    dictShort = {"shields_bitz": "SHIELDS-BITZ"}
    listRows = []
    for d in listAttr:
        for p in [q for q in d["listPairs"]
                  if "ops" not in (q["sModelA"], q["sModelB"])]:
            s = p["dictShapley"]
            listRows.append(
                r"%s & %s $\rightarrow$ %s & %+.2f & %+.2f & %+.2f & %+.2f & %+.2f & %+.2f \\"
                % (d["sBenchmark"].replace("ben", "Ben "),
                   dictShort.get(p["sModelA"], p["sModelA"].upper()),
                   dictShort.get(p["sModelB"], p["sModelB"].upper()),
                   s["fS0"], s["fAlbedo"], s["fA"], s["fB"],
                   p["fDeltaTZeroD"], p["fDeltaTActual"]))
        listRows.append(r"\addlinespace")
    return "\n".join(listRows[:-1])


def fsTableCompliance(dictAudit, dictLayout):
    listRows = []
    for r in sorted(dictAudit["listGridCompliance"],
                    key=lambda x: (x["sExperiment"], x["sModel"])):
        sKey = (r["sModel"], r["sExperiment"])
        d = dictLayout.get(sKey, {})
        listRows.append(
            r"%s & %s & %d & %d & %s & %s \\"
            % (fsEsc(r["sModel"].upper()), fsEsc(r["sExperiment"]),
               r["iNumCases"], d.get("iNumericColumns", 0),
               d.get("sResolvedFormat", "--").replace("v1.", "v1."),
               "yes" if d.get("bHeaderMatchesData") else "\\textbf{no}"))
    return "\n".join(listRows)


LIST_ICELINE_DEFS = ["263.15/interpolate", "263.15/snapToNode",
                     "271.15/interpolate", "271.15/snapToNode",
                     "273.15/interpolate", "273.15/snapToNode"]


def fsTableIceLines(listBench, listOps):
    dictByModel = {d["sModel"]: d for d in listBench if d["sBenchmark"] == "ben1"}
    listRows = []
    for sM, sName in LIST_MODEL_NAMES:
        d = dictByModel.get(sM)
        if d is None:
            continue
        listRows.append(
            r"%s & %d & %s & %.1f \\"
            % (sName.replace("VPLanet-", ""), d["iNumLatitudes"],
               " & ".join("%.1f" % d["dictEdges"][k] for k in LIST_ICELINE_DEFS),
               d["fSpreadAcrossDefinitions_deg"]))
    return "\n".join(listRows)


def fsTableOpsReadiness(dictReady):
    listRows = []
    for d in dictReady["listSourceVsSubmission"]:
        listRows.append(
            r"%s & %s & %s \\"
            % (d["sQuantity"], fsEsc(d["sSource"]),
               fsEsc(d["sArchive"])
               + ("" if d["bConsistent"] else r" $\ast$")))
    return "\n".join(listRows)


def fsTableBranchOrder(dictAudit):
    listRows = []
    dictName = dict(LIST_MODEL_NAMES)
    for d in dictAudit["listBranchOrderTest"]:
        listRows.append(
            r"%s & %s & %d & %d & %d & %.2f & %d & %.2f \\"
            % (dictName.get(d["sModel"], d["sModel"]).replace("VPLanet-", ""),
               d["sComparison"].replace("Experiment", "Exp."),
               d["iMatchedCases"], d["iIdenticalCases"],
               d["iViolationsInsideProtocolGrid"],
               d["fWorstViolationInsideGrid_K"],
               d["iViolationsOutsideProtocolGrid"],
               d["fWorstViolationOutsideGrid_K"]))
    return "\n".join(listRows)


def fsTableStateCensus(dictStates):
    listRows = []
    for sExp in ("exp1", "exp2"):
        for sM, sName in LIST_MODEL_NAMES:
            if sM not in dictStates[sExp]:
                continue
            d = dictStates[sExp][sM]
            c = d["dictStateCounts"]
            listRows.append(
                r"%s & %s%s & %d & %d & %d & %d & %.1f & %.1f & %s \\"
                % (sExp.replace("exp", "Exp "),
                   sName.replace("VPLanet-", ""),
                   "" if d.get("bFilesAsLabelled", True) else r"$^\dagger$",
                   c["ice_free"],
                   c["ice_cap"], c["ice_belt"], c["snowball"],
                   d["fTglobMin"], d["fTglobMax"],
                   str(d["iLandSeaStateDisagreements"])
                   if d["bLandSeaDistinct"] else "n/a"))
        if sExp == "exp1":
            listRows.append(r"\addlinespace")
    return "\n".join(listRows)


def fsTableDrift(listDrift):
    listRows = []
    for d in listDrift:
        listRows.append(
            r"%s & %s & %d & %+.2f & %.2f & %+.1f & %d \\"
            % (d["sModel"].upper(), fsEsc(d["sExperiment"]),
               d["iMatchedCases"], d["fMeanDeltaTglob"], d["fRmsDeltaTglob"],
               d["fMaxAbsDeltaTglob"], d["iStateChanges"]))
    return "\n".join(listRows)


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    for sName in ("benchmark", "radiation", "attribution", "audit", "layout",
                  "harmonized", "states", "drift", "bifurcation", "albedo",
                  "ops-readiness", "co2coord", "shields-scheme"):
        oParser.add_argument("--%s-json" % sName, required=True)
    oParser.add_argument("--out-dir", required=True)
    oArgs = oParser.parse_args()

    os.makedirs(oArgs.out_dir, exist_ok=True)

    dictBench = {(d["sModel"], d["sBenchmark"]): d
                 for d in json.load(open(oArgs.benchmark_json))["listBenchmarks"]}
    dictRad = json.load(open(oArgs.radiation_json))
    listAttr = json.load(open(oArgs.attribution_json))["listBenchmarks"]
    dictAudit = json.load(open(oArgs.audit_json))
    dictLayout = {(r["sModel"], r["sExperiment"]): r
                  for r in json.load(open(oArgs.layout_json))["listFiles"]}
    dictHarm = json.load(open(oArgs.harmonized_json))
    dictStates = json.load(open(oArgs.states_json))
    listDrift = json.load(open(oArgs.drift_json))["listComparisons"]
    dictBif = json.load(open(oArgs.bifurcation_json))
    dictAlb = json.load(open(oArgs.albedo_json))
    dictReady = json.load(open(oArgs.ops_readiness_json))
    dictCoord = json.load(open(oArgs.co2coord_json))
    listOpsRuns = json.load(open(oArgs.benchmark_json))["listOpsReference"]
    dictOpsRun = [d for d in listOpsRuns if d["bShippedInputDeck"]][0]

    dictTables = {
        "tabBenchmarkBudget": fsWrap(
            "llrrrrrrr", HEAD_BENCH, fsTableBenchmarkBudget(dictBench)),
        "tabRadiation": fsWrap("llrrrrr", HEAD_RADIATION,
                               fsTableRadiation(dictRad),
                               sSize="scriptsize"),
        "tabAttribution": fsWrap("llrrrrrr", HEAD_ATTRIBUTION,
                                 fsTableAttribution(listAttr)),
        "tabCompliance": fsWrap("llrrlc", HEAD_COMPLIANCE,
                                fsTableCompliance(dictAudit, dictLayout),
                                sSize="scriptsize"),
        "tabIceLines": fsWrap("lrrrrrrrr", HEAD_ICELINES,
                              fsTableIceLines(dictHarm["listBenchmarkEdges"],
                                              dictHarm["listOpsRunEdges"])),
        "tabOpsReadiness": fsWrap("p{2.6cm}p{5.6cm}p{6.2cm}", HEAD_OPSREADY,
                                    fsTableOpsReadiness(dictReady),
                                    sSize="footnotesize"),
        "tabBranchOrder": fsWrap("llrrrrrr", HEAD_BRANCHORDER,
                                 fsTableBranchOrder(dictAudit),
                                 sSize="footnotesize"),
        "tabStateCensus": fsWrap("llrrrrrrr", HEAD_STATES,
                                 fsTableStateCensus(dictStates)),
        "tabDrift": fsWrap("llrrrrr", HEAD_DRIFT, fsTableDrift(listDrift)),
    }

    # Hysteresis table.  A model whose two branches are the same run has no
    # measurable loop, and printing a width for it would be printing the grid
    # spacing, so those cells are struck out instead.
    listRows = []
    for sM, sName in LIST_MODEL_NAMES:
        d3 = dictBif["dictExp3"].get(sM, {})
        d4 = dictBif["dictExp4"].get(sM, {})
        h3 = d3.get("dictHysteresis", {})
        g3 = d3.get("dictBranchGrid", {})
        h4 = d4.get("dictHysteresis", {})
        # A loop that appears at only a handful of grid points is the grid
        # spacing, not a measurement, so it is reported as degenerate rather
        # than as a number.  The threshold is stated in the caption.
        def fbDegenerate(dictM):
            d = dictM.get("dictBranchDistinctness") or {}
            if not d.get("bComparable"):
                return False
            return d["iIdenticalCases"] >= 0.9 * d["iCases"]
        b3, b4 = fbDegenerate(d3), fbDegenerate(d4)
        bNoBranchPair3 = "cold" not in d3
        def f(v, s="%.3f"):
            return "--" if v is None else s % v
        sSame = r"\textit{degenerate}"
        listRows.append(
            r"%s & %s & %s & %s & %s & %s & %s & %s \\"
            % (sName.replace("VPLanet-", ""),
               (r"\textit{warm branch only}" if bNoBranchPair3
                else sSame if b3 else f(h3.get("fGlaciationInst"))),
               "" if (b3 or bNoBranchPair3) else f(h3.get("fDeglaciationInst")),
               "" if (b3 or bNoBranchPair3) else f(h3.get("fBistableWidthInst")),
               "" if (b3 or bNoBranchPair3) else f(h3.get("fBistableWidthAbsorbedFlux_Wm2"), "%.0f"),
               (r"\textit{no CO$_2$ term}" if ("warm" not in d4 and "cold" not in d4)
                else sSame if b4 else f(h4.get("fGlaciationCO2ppm"), "%.3g")),
               ("" if (b4 or ("warm" not in d4 and "cold" not in d4))
                else f(h4.get("fDeglaciationCO2ppm"), "%.3g")),
               # A model with only one branch has no pair of grids to
               # compare, which is not the same as having two that disagree.
               ("--" if "cold" not in d3
                else "yes" if g3.get("bSameGrid") else r"\textbf{no}")))
    dictTables["tabHysteresis"] = fsWrap("lrrrrrrc", HEAD_HYSTERESIS,
                                         "\n".join(listRows))

    # Selected scalar facts, as LaTeX macros.
    dC = json.load(open(oArgs.bifurcation_json))
    listMacros = []
    dictClamp = dictRad["dictHextorTableAxes"]
    listMacros.append(r"\newcommand{\hextorTableCOtwoFloor}{%.0f}"
                      % (dictClamp["fFCO2Min"] * 1e6))
    listMacros.append(r"\newcommand{\hextorTableTmin}{%.0f}"
                      % min(dictClamp["aTemperatureK"]))
    listMacros.append(r"\newcommand{\hextorTableTmax}{%.0f}"
                      % max(dictClamp["aTemperatureK"]))
    dPoise = dC["dictPoiseExp4OLRIdentification"]
    listMacros.append(r"\newcommand{\poiseExpFourBestOLR}{%s}"
                      % dPoise["sBestCandidate"].upper())
    for sName, fVal, sFmt in (
            ("avalonBenOneAlbedo", dictBench[("avalon", "ben1")]["fATOAInsolationWeighted"], "%.3f"),
            ("hextorBenOneAlbedo", dictBench[("hextor", "ben1")]["fATOAInsolationWeighted"], "%.3f"),
            ("poiseBenOneAlbedo", dictBench[("poise", "ben1")]["fATOAInsolationWeighted"], "%.3f"),
            ("avalonBenOneOLR", dictBench[("avalon", "ben1")]["fOLRAreaWeighted"], "%.1f"),
            ("hextorBenOneOLR", dictBench[("hextor", "ben1")]["fOLRAreaWeighted"], "%.1f"),
            ("poiseBenOneOLR", dictBench[("poise", "ben1")]["fOLRAreaWeighted"], "%.1f")):
        listMacros.append(r"\newcommand{\%s}{%s}" % (sName, sFmt % fVal))
    # OPS, the fourth code: it has no submission, so every number here comes
    # from its source or from its own shipped run.
    dictTot = dictRad["dictOpsTableAxes"]
    listMacros.append(r"\newcommand{\opsTableCOtwoFloor}{%.1f}"
                      % dictTot["fTableMinCO2ppmEquivalent"])
    listMacros.append(r"\newcommand{\opsTableTmin}{%.0f}"
                      % min(dictTot["aTemperatureK"]))
    listMacros.append(r"\newcommand{\opsTableTmax}{%.0f}"
                      % max(dictTot["aTemperatureK"]))
    listMacros.append(r"\newcommand{\molarMassRatio}{%.4f}"
                      % dictCoord["fMolarMassRatio"])
    listMacros.append(r"\newcommand{\piOverTwo}{%.4f}" % dictCoord["fPiOverTwo"])
    listMacros.append(r"\newcommand{\coordRmsSeparation}{%.4f}"
                      % max(abs(d["fRmsDifference"])
                            for d in dictCoord["listHextorBenchmarks"]))
    dictW = dictAlb["dictTransitionWidthsK"]
    dictS = dictAlb["dictMeltingPointStep"]
    for sName, sKey in (("hextorSurfaceRamp", "hextor_surface"),
                        ("hextorPlanetaryRamp", "hextor_planetary"),
                        ("opsSurfaceRamp", "ops_surface"),
                        ("opsPlanetaryRamp", "ops_planetary")):
        listMacros.append(r"\newcommand{\%sWidth}{%.1f}" % (sName, dictW[sKey]))
    listMacros.append(r"\newcommand{\opsMeltStep}{%+.3f}"
                      % dictS["ops_surface"])
    listMacros.append(r"\newcommand{\opsGlaciatedAlbedo}{%.3f}"
                      % dictAlb["dictGlaciatedAlbedo"]["ops_surface"])
    # Only the two port residuals from the source tree's own run are quoted
    # in the report; that run's other summary numbers are not, so they are
    # not emitted.
    for sName, fVal, sFmt in (
            ("opsPortResidual",
             dictOpsRun["fOLRResidualWithMolarMassFactor"], "%+.2f"),
            ("opsPortResidualNoFactor",
             dictOpsRun["fOLRResidualAsPartialPressure"], "%+.2f")):
        listMacros.append(r"\newcommand{\%s}{%s}" % (sName, sFmt % fVal))
    listMacros.append(r"\newcommand{\opsExpFourForcing}{%.1f}"
                      % dictRad["dictExp4Forcing"]["ops"]["fTotalForcing_Wm2"])
    listMacros.append(r"\newcommand{\opsPlanckB}{%.3f}"
                      % dictRad["dictLocalCoefficients"]["ops"]["at288K_280ppm"]["fPlanckB"])
    listMacros.append(r"\newcommand{\opsOLRat}{%.1f}"
                      % dictRad["dictLocalCoefficients"]["ops"]["at288K_280ppm"]["fOLR"])

    # OPS as a submitter: the numbers that come from Results/ops.
    for sName, sBench, sKey, sFmt in (
            ("opsBenOneTglob", "ben1", "fTglobAreaWeighted", "%.2f"),
            ("opsBenOneAlbedo", "ben1", "fATOAInsolationWeighted", "%.3f"),
            ("opsBenOneOLR", "ben1", "fOLRAreaWeighted", "%.1f"),
            ("opsBenOneASR", "ben1", "fAbsorbedShortwave", "%.1f"),
            ("opsBenOneImbalance", "ben1", "fNetImbalance", "%+.1f"),
            ("opsBenThreeImbalance", "ben3", "fNetImbalance", "%+.1f"),
            ("opsPortRmsBenOne", "ben1", "fOLRPortRMSAboutFittedOffset", "%.2f"),
            ("opsPortRmsBenThree", "ben3", "fOLRPortRMSAboutFittedOffset", "%.2f")):
        listMacros.append(r"\newcommand{\%s}{%s}"
                          % (sName, sFmt % dictBench[("ops", sBench)][sKey]))
    listSbIce = [d for d in dictHarm["listBenchmarkEdges"]
                 if d["sModel"] == "shields_bitz" and d["sBenchmark"] == "ben1"]
    if listSbIce:
        listMacros.append(r"\newcommand{\sbIceEdgeSpread}{%.1f}"
                          % listSbIce[0]["fSpreadAcrossDefinitions_deg"])
        listMacros.append(r"\newcommand{\sbIceEdgeHarmonised}{%.1f}"
                          % listSbIce[0]["dictEdges"]["271.15/interpolate"])
        if listSbIce[0].get("fNativeReportedEdge_deg") is not None:
            listMacros.append(r"\newcommand{\sbIceEdgePackage}{%.1f}"
                              % listSbIce[0]["fNativeReportedEdge_deg"])

    dictOpsIce = [d for d in dictHarm["listBenchmarkEdges"]
                  if d["sModel"] == "ops" and d["sBenchmark"] == "ben1"][0]
    listMacros.append(r"\newcommand{\opsIceEdgeSpread}{%.1f}"
                      % dictOpsIce["fSpreadAcrossDefinitions_deg"])
    listMacros.append(r"\newcommand{\opsIceEdgeNative}{%.2f}"
                      % dictOpsIce["dictEdges"]["263.15/interpolate"])
    dictBranch = {(d["sModel"], d["sComparison"]): d
                  for d in dictAudit["listBranchOrderTest"]}
    dOps = dictBranch[("ops", "warm start vs cold start")]
    for sName, oVal, sFmt in (
            ("opsBranchViolations", dOps["iViolationsAsLabelled"], "%d"),
            ("opsBranchWorst", dOps["fWorstViolationAsLabelled_K"], "%.1f"),
            ("opsBranchViolationsSwapped", dOps["iViolationsIfSwapped"], "%d"),
            ("opsBranchWorstSwapped", dOps["fWorstViolationIfSwapped_K"], "%.1f"),
            ("opsBranchInside", dOps["iViolationsInsideProtocolGrid"], "%d"),
            ("opsBranchWorstInside", dOps["fWorstViolationInsideGrid_K"], "%.2f"),
            ("opsBranchOutside", dOps["iViolationsOutsideProtocolGrid"], "%d"),
            ("opsBranchWorstOutside", dOps["fWorstViolationOutsideGrid_K"], "%.1f"),
            ("opsBranchMatched", dOps["iMatchedCases"], "%d")):
        listMacros.append(r"\newcommand{\%s}{%s}" % (sName, sFmt % oVal))

    # OPS's effective diffusion, measured against what it files.
    dDiff = dictAudit.get("dictOpsDiffusionMeasured") or {}
    if dDiff:
        for sName, sKey, sFmt in (
                ("opsDiffFiled", "fDFiledInSubmission", "%.2f"),
                ("opsDiffUsed", "fDReportedByRun", "%.4f"),
                ("opsDiffDry", "fDDryRecomputation", "%.4f"),
                ("opsDiffMoistFactor", "fImpliedMoistAndLatentFactor", "%.3f")):
            listMacros.append(r"\newcommand{\%s}{%s}" % (sName, sFmt % dDiff[sKey]))
        listMacros.append(r"\newcommand{\opsDiffOffsetPercent}{%.1f}"
                          % (100.0 * dDiff["fFractionalOffsetVsFiled"]))

    # The source tree's protocol files against the archived submission.
    listDeliv = dictReady.get("listDeliverableCheck") or []
    iIdentical = sum(1 for d in listDeliv
                     if (d["dictAgreement"] or {}).get("bIdentical"))
    iComparable = sum(1 for d in listDeliv
                      if (d["dictAgreement"] or {}).get("bComparable"))
    listMacros.append(r"\newcommand{\opsDeliverablesIdentical}{%d}" % iIdentical)
    listMacros.append(r"\newcommand{\opsDeliverablesCompared}{%d}" % iComparable)

    # Shields-Bitz: every number below comes from a run made here.
    if ("shields_bitz", "ben1") in dictBench:
        for sName, sBench, sKey, sFmt in (
                ("sbBenOneTglob", "ben1", "fTglobAreaWeighted", "%.2f"),
                ("sbBenTwoTglob", "ben2", "fTglobAreaWeighted", "%.2f"),
                ("sbBenThreeTglob", "ben3", "fTglobAreaWeighted", "%.2f"),
                ("sbBenOneAlbedo", "ben1", "fATOAInsolationWeighted", "%.3f"),
                ("sbBenOneOLR", "ben1", "fOLRAreaWeighted", "%.1f"),
                ("sbBenOneASR", "ben1", "fAbsorbedShortwave", "%.1f"),
                ("sbBenOneImbalance", "ben1", "fNetImbalance", "%+.2f")):
            listMacros.append(r"\newcommand{\%s}{%s}"
                              % (sName, sFmt % dictBench[("shields_bitz", sBench)][sKey]))
        fPrev = dictBench[("shields_bitz", "ben1")].get("fTglobPreviousOrbitK")
        if fPrev is not None:
            listMacros.append(
                r"\newcommand{\sbBenOneDrift}{%.0e}"
                % abs(dictBench[("shields_bitz", "ben1")]["fTglobAreaWeighted"] - fPrev))
    dictSB = json.load(open(oArgs.shields_scheme_json))
    dAx = dictSB["dictAxisDiscrepancy"]
    listMacros.append(r"\newcommand{\sbAxisWorst}{%.1f}" % dAx["fMaxOffsetDeg"])
    listMacros.append(r"\newcommand{\sbAxisRms}{%.1f}" % dAx["fRMSOffsetDeg"])
    listMacros.append(r"\newcommand{\sbBoxes}{%d}" % dAx["iJmx"])
    listMacros.append(r"\newcommand{\sbFilletLand}{%.2f}"
                      % dictSB["dictSourceConstants"]["fFilletLandFraction"])
    listAxB = dictSB["listAxisEffectOnBenchmarks"]
    if listAxB:
        fWorst = max(abs(r["fDisplacementDeg"]) for r in listAxB)
        listMacros.append(r"\newcommand{\sbAxisBenchmarkShift}{%.1f}" % fWorst)
    dSBAlb = dictSB["dictBroadbandAlbedo"]
    for sName, sKey in (("sbAlbedoOcean", "A_o"), ("sbAlbedoLand", "A_l"),
                        ("sbAlbedoIce", "A_50")):
        listMacros.append(r"\newcommand{\%s}{%.3f}" % (sName, dSBAlb[sKey]))
    listMacros.append(r"\newcommand{\sbIceFreeAlbedo}{%.3f}"
                      % dictAlb["dictIceFreeAlbedo"]["shields_bitz"])
    listMacros.append(r"\newcommand{\sbGlaciatedAlbedo}{%.3f}"
                      % dictAlb["dictGlaciatedAlbedo"]["shields_bitz"])
    dSweep = dictSB.get("dictShippedSweep") or {}
    if dSweep:
        listMacros.append(r"\newcommand{\sbSweepCases}{%d}" % dSweep["iCases"])
        listMacros.append(r"\newcommand{\sbSweepMin}{%.2f}"
                          % min(dSweep["aInstellation"]))
        listMacros.append(r"\newcommand{\sbSweepMax}{%.2f}"
                          % max(dSweep["aInstellation"]))
    dictH12 = dictBif["dictExp12Hysteresis"]
    for sM, sName in (("hextor", "hextorExpOneTwoWidth"),
                      ("ops", "opsExpOneTwoWidth"),
                      ("avalon", "avalonExpOneTwoWidth"),
                      ("poise", "poiseExpOneTwoWidth")):
        listMacros.append(r"\newcommand{\%s}{%.4f}"
                          % (sName, dictH12[sM]["fMeanBistableWidthInst"]))
    dictAttrOps = {d["sBenchmark"]: d for d in listAttr}
    for sBench, sName in (("ben1", "opsZeroDResidualBenOne"),
                          ("ben2", "opsZeroDResidualBenTwo"),
                          ("ben3", "opsZeroDResidualBenThree")):
        listMacros.append(
            r"\newcommand{\%s}{%+.1f}"
            % (sName, dictAttrOps[sBench]["dictModels"]["ops"]["fZeroDResidual"]))
    for sBench, sName in (("ben1", "sbZeroDResidualBenOne"),
                          ("ben2", "sbZeroDResidualBenTwo"),
                          ("ben3", "sbZeroDResidualBenThree")):
        dM = dictAttrOps[sBench]["dictModels"].get("shields_bitz")
        if dM:
            listMacros.append(r"\newcommand{\%s}{%+.1f}"
                              % (sName, dM["fZeroDResidual"]))

    dictTables["macros"] = "\n".join(listMacros)

    for sName, sBody in dictTables.items():
        sPath = os.path.join(oArgs.out_dir, "%s.tex" % sName)
        with open(sPath, "w") as f:
            f.write(sBody + "\n")
        print("wrote %s (%d lines)" % (sPath, sBody.count("\n") + 1))


if __name__ == "__main__":
    main()
