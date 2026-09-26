"""Attribute the inter-model differences in Benchmark 1 global mean temperature to the individual parameterisation choices, using an exact Shapley decomposition of the zero-dimensional equilibrium.

Every one of the five codes satisfies, in the global mean at equilibrium,

    S (1 - alpha) / 4  =  A + B (T - 273.15),

where alpha is the insolation-weighted planetary albedo diagnosed from the
model's own archived profile and (A, B) are the intercept and Planck slope of
the model's own longwave scheme, evaluated at the benchmark's temperature and
CO2 abundance.  The temperature predicted by this relation is compared with
the model's actual Tglob; the residual measures what the zero-dimensional
view omits (meridional structure interacting with a nonlinear OLR).

Differences between any two models are then attributed to S, alpha, A and B
by the Shapley value, i.e. by averaging each factor's marginal contribution
over all 4! orderings in which the four factors could be exchanged.  Unlike a
one-at-a-time perturbation, this is exact: the four contributions sum to the
total difference with no interaction residual.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "compareRadiationSchemes"))
from dataCompareRadiationSchemes import (HextorTable,  # noqa: E402
                                         OpsOlrTable)

S_EARTH = 1361.0

# The cloud offset assumed when the benchmark step computed the OPS port
# residual; the fitted offset is relative to it, so it is added back here.
# OPS diagnoses the offset per belt at run time, so this is its initialisation.
BENCH_CLOUDIR_OPS = -4.0
FACTORS = ["fS0", "fAlbedo", "fA", "fB"]


def fdPredictT(dictP):
    """Equilibrium global temperature of the zero-dimensional balance, in K."""
    fASR = dictP["fS0"] * (1.0 - dictP["fAlbedo"]) / 4.0
    return 273.15 + (fASR - dictP["fA"]) / dictP["fB"]


def fdictShapley(dictA, dictB):
    """Exact Shapley attribution of fdPredictT(dictB) - fdPredictT(dictA)."""
    dictPhi = {s: 0.0 for s in FACTORS}
    iN = 0
    for tOrder in itertools.permutations(FACTORS):
        dictCur = dict(dictA)
        fPrev = fdPredictT(dictCur)
        for sFac in tOrder:
            dictCur[sFac] = dictB[sFac]
            fNow = fdPredictT(dictCur)
            dictPhi[sFac] += fNow - fPrev
            fPrev = fNow
        iN += 1
    return {s: v / iN for s, v in dictPhi.items()}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--benchmark-json", required=True)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listB = json.load(open(oArgs.benchmark_json))["listBenchmarks"]
    oTab = HextorTable(oArgs.hextor_table)
    oOps = OpsOlrTable(oArgs.ops_olr_table)
    dictB = {(d["sModel"], d["sBenchmark"]): d for d in listB}

    # Longwave coefficients, each evaluated in the model's own scheme at the
    # benchmark's own global mean temperature and CO2 abundance.
    def fdictHextorAB(fT, fFCO2, fCloudIR):
        fO1 = oTab.fdOLR(fT + 1.0, fFCO2) - fCloudIR
        fO0 = oTab.fdOLR(fT - 1.0, fFCO2) - fCloudIR
        fB = (fO1 - fO0) / 2.0
        fA = (oTab.fdOLR(fT, fFCO2) - fCloudIR) - fB * (fT - 273.15)
        return fA, fB

    def fdictOpsAB(fT, fFCO2, fCloudIR):
        """OPS's table linearised at the benchmark's own global mean."""
        fO1 = oOps.faOLR(np.array([fT + 1.0]), fFCO2)[0] - fCloudIR
        fO0 = oOps.faOLR(np.array([fT - 1.0]), fFCO2)[0] - fCloudIR
        fB = (fO1 - fO0) / 2.0
        fA = (oOps.faOLR(np.array([fT]), fFCO2)[0] - fCloudIR) - fB * (fT - 273.15)
        return fA, fB

    listState = []
    for sBench in ("ben1", "ben2", "ben3"):
        dictModels = {}

        d = dictB[("avalon", sBench)]
        dictModels["avalon"] = {
            "fS0": S_EARTH * 1.0,
            "fAlbedo": d["fATOAInsolationWeighted"],
            "fA": 210.0 if sBench == "ben1" else 210.0,
            "fB": 2.0,
            "fTglobActual": d["fTglobAreaWeighted"],
            "sLongwave": "linear A=210, B=2.0 (Myhre CO2 forcing)",
        }

        d = dictB[("poise", sBench)]
        fInstP = 1.004825 if sBench == "ben1" else 0.999899
        dictModels["poise"] = {
            "fS0": S_EARTH * fInstP,
            "fAlbedo": d["fATOAInsolationWeighted"],
            "fA": 203.30 if sBench == "ben1" else 204.05,
            "fB": 2.09,
            "fTglobActual": d["fTglobAreaWeighted"],
            "sLongwave": "fixed linearisation dPlanckA/dPlanckB (bCalcAB = 0)",
        }

        d = dictB[("hextor", sBench)]
        # The archived HEXTOR run predates the pi/2 CO2-coordinate correction,
        # so its effective CO2 abundance was (pi/2) x 280 ppm.
        fFCO2Eff = 2.8e-4 * np.pi / 2.0
        fA, fB = fdictHextorAB(d["fTglobAreaWeighted"], fFCO2Eff, -6.3)
        dictModels["hextor"] = {
            "fS0": S_EARTH * 1.0,
            "fAlbedo": d["fATOAInsolationWeighted"],
            "fA": fA, "fB": fB,
            "fTglobActual": d["fTglobAreaWeighted"],
            "sLongwave": ("lookup table linearised at Tglob, effective CO2 = "
                          "(pi/2) x 280 ppm, cloudir = -6.3 W/m2"),
        }

        # OPS, archived as `ops`.  Its `cloudir` is recorded nowhere, so the
        # constant that makes the ported table reproduce its own archived OLR
        # profile is used; that is the only choice under which the 0-D balance
        # and the archived profile describe the same run.
        d = dictB[("ops", sBench)]
        fCloudIROps = BENCH_CLOUDIR_OPS - d["fOLRPortFittedOffset"]
        fA, fB = fdictOpsAB(d["fTglobAreaWeighted"], 2.8e-4, fCloudIROps)
        dictModels["ops"] = {
            "fS0": S_EARTH * 1.0,
            "fAlbedo": d["fATOAInsolationWeighted"],
            "fA": fA, "fB": fB,
            "fTglobActual": d["fTglobAreaWeighted"],
            "sLongwave": ("lookup table linearised at Tglob, cloudir recovered "
                          "from the archived profile = %.2f W/m2" % fCloudIROps),
        }

        # Shields-Bitz.  Its longwave IS A + B T with these two constants, so
        # unlike the other four there is nothing to linearise and no offset to
        # recover: the zero-dimensional balance uses the model's own scheme
        # exactly as written.
        if ("shields_bitz", sBench) in dictB:
            d = dictB[("shields_bitz", sBench)]
            dictModels["shields_bitz"] = {
                "fS0": S_EARTH * 1.0,
                "fAlbedo": d["fATOAInsolationWeighted"],
                "fA": 203.3, "fB": 2.09,
                "fTglobActual": d["fTglobAreaWeighted"],
                "sLongwave": "A + B T exactly, A = 203.3, B = 2.09, no CO2 term",
            }

        for sM, dictP in dictModels.items():
            dictP["fTglobZeroD"] = fdPredictT(dictP)
            dictP["fZeroDResidual"] = dictP["fTglobActual"] - dictP["fTglobZeroD"]

        listPairs = []
        for sA, sBm in itertools.combinations(sorted(dictModels), 2):
            dictPhi = fdictShapley(dictModels[sA], dictModels[sBm])
            fTot = (dictModels[sBm]["fTglobZeroD"] - dictModels[sA]["fTglobZeroD"])
            listPairs.append({
                "sModelA": sA, "sModelB": sBm,
                "fDeltaTZeroD": fTot,
                "fDeltaTActual": (dictModels[sBm]["fTglobActual"]
                                  - dictModels[sA]["fTglobActual"]),
                "dictShapley": dictPhi,
                "fShapleySum": float(sum(dictPhi.values())),
                "fClosureError": float(sum(dictPhi.values()) - fTot),
            })

        listState.append({"sBenchmark": sBench,
                          "dictModels": dictModels,
                          "listPairs": listPairs})

    with open(oArgs.out, "w") as f:
        json.dump({"listBenchmarks": listState}, f, indent=1)

    for d in listState:
        print("\n=== %s ===" % d["sBenchmark"].upper())
        print("%-7s %9s %9s %9s %9s %10s %10s %9s"
              % ("model", "S0", "albedo", "A", "B", "T_0D", "T_actual", "resid"))
        for sM, p in d["dictModels"].items():
            print("%-7s %9.1f %9.4f %9.2f %9.3f %10.2f %10.2f %+9.2f"
                  % (sM, p["fS0"], p["fAlbedo"], p["fA"], p["fB"],
                     p["fTglobZeroD"], p["fTglobActual"], p["fZeroDResidual"]))
        print("  Shapley attribution of the zero-dimensional difference (K):")
        print("  %-18s %9s %9s %9s %9s %9s %9s"
              % ("pair", "dT(0D)", "S0", "albedo", "A", "B", "dT(act)"))
        for p in d["listPairs"]:
            s = p["dictShapley"]
            print("  %-18s %+9.2f %+9.2f %+9.2f %+9.2f %+9.2f %+9.2f"
                  % ("%s->%s" % (p["sModelA"], p["sModelB"]), p["fDeltaTZeroD"],
                     s["fS0"], s["fAlbedo"], s["fA"], s["fB"], p["fDeltaTActual"]))


if __name__ == "__main__":
    main()
