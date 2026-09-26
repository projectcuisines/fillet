"""Reimplement the outgoing-longwave-radiation parameterisation of AVALON, HEXTOR, VPLanet-POISE, OPS and Shields-Bitz from their own source and tabulate them on a common (T, pCO2) grid.

Each function below is a line-by-line transcription of the model's own code, so
that the comparison is between the parameterisations themselves and not between
whatever settings happened to be used for a given experiment:

  AVALON  avalon.jl  `A_eff` + `imex_step!`
          OLR = (A - F_CO2 ln(CO2/CO2_ref)) + B T[C],  A=210, B=2.0, F_CO2=5.35

  POISE   src/poise.c
          `fdOLRwk97`  Williams & Kasting (1997) quartic in (T, ln pCO2)
          `fdOLRhm16`  Haqq-Misra et al. (2016) quartic in (log10 T, log10 pCO2)
          `fdOLRsms09` Spiegel et al. (2009) grey, no CO2 dependence
          plus the fixed linearisation dPlanckA + dPlanckB T used for the
          FILLET benchmarks and Experiments 1-3 (bCalcAB = 0)

  HEXTOR  model/radiation/radiation.f90 `getOLR` on the shipped 1 bar Sun
          lookup table, including the log10 CO2 axis, the linear T axis and
          the fitted power-law extrapolation outside [T_min, T_max].

  OPS   driver.f90 `INTERPOLR` plus the bilinear evaluation in the main
          loop, on the shipped `data/OLRparaminterp.325` table: linear in T
          over 25 nodes from 150 to 390 K, linear in ln(pCO2) over 16 nodes
          from 1e-5 to 35 bar, and -- unlike HEXTOR -- LINEARLY EXTRAPOLATING
          outside both axes rather than clamping, because the bracketing pair
          is taken from the end of the grid and the weights are evaluated at
          the true coordinate.  The table is queried not at the CO2 mixing
          ratio but at `pco2r = 44 P_dry f / (28 (1 - f))`, a factor
          44/28 = 1.5714 above the partial pressure (see
          `explorations/testCO2CoordinateMolarMassHypothesis.py`).

  SHIELDS-BITZ  EBM_one_file.py, the `A` and `B` of its DEFAULTS block
          OLR = A + B T[C], A=203.3, B=2.09, with NO CO2 term of any kind.
          This is not a linearisation of a table, as POISE's fixed A and B
          are; it is the model's entire longwave scheme, so the CO2 axis of
          every surface below is flat for it by construction and Experiment 4
          cannot be posed to it at all.

Outputs a JSON of OLR(T, pCO2) surfaces plus the derived quantities that
control EBM behaviour: the local Planck slope B = dOLR/dT and the CO2 forcing
dOLR/dlog10(pCO2).
"""

import argparse
import json

import numpy as np

SIGMA = 5.670367e-8

# ---------------------------------------------------------------------------
# AVALON
# ---------------------------------------------------------------------------

def faOLRAvalon(aTK, fPCO2bar, fA=210.0, fB=2.0, fFCO2=5.35, fCO2Ref=280.0e-6):
    """AVALON linear OLR with logarithmic CO2 forcing (Myhre et al. 1998)."""
    fAeff = fA - fFCO2 * np.log(fPCO2bar / fCO2Ref)
    return fAeff + fB * (aTK - 273.15)


# ---------------------------------------------------------------------------
# VPLanet-POISE
# ---------------------------------------------------------------------------

def faOLRPoiseFixed(aTK, fA=204.05, fB=2.09):
    """The fixed linearisation used in the FILLET benchmarks (bCalcAB = 0)."""
    return fA + fB * (aTK - 273.15)


def faOLRShieldsBitz(aTK, fA=203.3, fB=2.09):
    """The whole Shields-Bitz longwave: North (1981) linear, no CO2 dependence."""
    return fA + fB * (aTK - 273.15)


def faOLRPoiseWK97(aTK, fPCO2bar):
    """fdOLRwk97 (poise.c) -- Williams & Kasting (1997)."""
    phi = np.log(fPCO2bar / 3.3e-4)
    T = np.asarray(aTK, dtype=float)
    Int = (9.468980 - 7.714727e-5 * phi - 2.794778 * T - 3.244753e-3 * phi * T
           - 3.547406e-4 * phi ** 2 + 2.212108e-2 * T ** 2
           + 2.229142e-3 * phi ** 2 * T + 3.088497e-5 * phi * T ** 2
           - 2.789815e-5 * (phi * T) ** 2 - 3.442973e-3 * phi ** 3
           - 3.361939e-5 * T ** 3 + 9.173169e-3 * phi ** 3 * T
           - 7.775195e-5 * phi ** 3 * T ** 2 - 1.679112e-7 * phi * T ** 3
           + 6.590999e-8 * phi ** 2 * T ** 3 + 1.528125e-7 * phi ** 3 * T ** 3
           - 3.367567e-2 * phi ** 4 - 1.631909e-4 * phi ** 4 * T
           + 3.663871e-6 * phi ** 4 * T ** 2 - 9.255646e-9 * phi ** 4 * T ** 3)
    Int = np.minimum(Int, 300.0)                 # saturation clamp in the code
    return np.where(T < 190, SIGMA * T ** 4, Int)


def faOLRPoiseHM16(aTK, fPCO2bar):
    """fdOLRhm16 (poise.c) -- Haqq-Misra et al. (2016)."""
    phi = np.log10(fPCO2bar)
    T = np.asarray(aTK, dtype=float)
    tmpk = np.log10(np.maximum(T, 1e-3))
    f = (9.12805643869791438760 * tmpk ** 4
         + 4.58408794768168803557 * tmpk ** 3 * phi
         - 8.47261075643147449910e+01 * tmpk ** 3
         + 4.35517381112690282752e-01 * (tmpk * phi) ** 2
         - 2.86355036260417961103e+01 * tmpk ** 2 * phi
         + 2.96626642498045896446e+02 * tmpk ** 2
         - 6.01082900358299240806e-02 * tmpk * phi ** 3
         - 2.60414691486954641420 * tmpk * phi ** 2
         + 5.69812976563675661623e+01 * tmpk * phi
         - 4.62596100127381816947e+02 * tmpk
         + 2.18159373001564722491e-03 * phi ** 4
         + 1.61456772400726950023e-01 * phi ** 3
         + 3.75623788187470086797 * phi ** 2
         - 3.53347289223180354156e+01 * phi
         + 2.75011005409836684521e+02)
    return np.where(T > 150, 10.0 ** f / 1000.0, SIGMA * T ** 4)


def faOLRPoiseSMS09(aTK):
    """fdOLRsms09 (poise.c) -- Spiegel et al. (2009) grey; CO2-independent."""
    T = np.asarray(aTK, dtype=float)
    tau = 0.79 * (T / 273.15) ** 3
    return SIGMA * T ** 4 / (1.0 + 0.75 * tau)


# ---------------------------------------------------------------------------
# HEXTOR
# ---------------------------------------------------------------------------

class HextorTable(object):
    """Faithful Python port of radiation.f90's v1 loader + getOLR."""

    def __init__(self, sPath):
        import h5py
        with h5py.File(sPath, "r") as f:
            aOLR = np.array(f["/olr"])            # (1748, 4)
        iNCO2, iNT = 92, 19
        self.aFCO2 = aOLR[:iNCO2, 1]
        self.aTemp = aOLR[::iNCO2, 2]
        # OLR stored negated, in mW/m^2; driver.f divides by 1000.
        self.aTable = np.zeros((iNT, iNCO2))
        for j in range(iNT):
            self.aTable[j, :] = -aOLR[j * iNCO2:(j + 1) * iNCO2, 3]
        self.aTable /= 1000.0
        self._fit_extrap()

    def _fit_extrap(self):
        """Power-law exponents fitted from the table's own boundary gradient."""
        aT, aTab = self.aTemp, self.aTable
        self.aNEffUpper = np.zeros(aTab.shape[1])
        self.aNEffLower = np.zeros(aTab.shape[1])
        for ic in range(aTab.shape[1]):
            dTop = (aTab[-1, ic] - aTab[-2, ic]) / (aT[-1] - aT[-2])
            dBot = (aTab[1, ic] - aTab[0, ic]) / (aT[1] - aT[0])
            self.aNEffUpper[ic] = max(0.0, dTop * aT[-1] / aTab[-1, ic])
            self.aNEffLower[ic] = dBot * aT[0] / aTab[0, ic]

    def fdOLR(self, fTK, fFCO2):
        """getOLR(): bilinear in (log10 fco2, T) with power-law extrapolation."""
        aT, aC = self.aTemp, self.aFCO2
        fTEval = min(max(fTK, aT[0]), aT[-1])
        ic = int(np.clip(np.searchsorted(aC, fFCO2) - 1, 0, len(aC) - 2))
        it = int(np.clip(np.searchsorted(aT, fTEval) - 1, 0, len(aT) - 2))
        if fFCO2 <= aC[0]:
            ic, fc = 0, 0.0
        elif fFCO2 >= aC[-1]:
            ic, fc = len(aC) - 2, 1.0
        else:
            fc = (np.log10(fFCO2) - np.log10(aC[ic])) / (np.log10(aC[ic + 1]) - np.log10(aC[ic]))
        ft = (fTEval - aT[it]) / (aT[it + 1] - aT[it])
        fVal = ((1 - fc) * (1 - ft) * self.aTable[it, ic]
                + fc * (1 - ft) * self.aTable[it, ic + 1]
                + (1 - fc) * ft * self.aTable[it + 1, ic]
                + fc * ft * self.aTable[it + 1, ic + 1])
        if fTK < aT[0] or fTK > aT[-1]:
            aN = self.aNEffUpper if fTK > aT[-1] else self.aNEffLower
            fN = (1 - fc) * aN[ic] + fc * aN[ic + 1]
            fVal *= (fTK / fTEval) ** fN
        return fVal

    def faOLR(self, aTK, fFCO2):
        return np.array([self.fdOLR(float(t), fFCO2) for t in np.atleast_1d(aTK)])


# ---------------------------------------------------------------------------
# OPS
# ---------------------------------------------------------------------------

# CO2/N2 molar-mass ratio; driver.f90 converts the mixing ratio to the table's
# CO2 coordinate with pco2r = 44 P_dry f / (28 (1 - f)).
CO2_COORD_FACTOR = 44.0 / 28.0


class OpsOlrTable(object):
    """Port of driver.f90's OLR table read + INTERPOLR + bilinear evaluation."""

    aTemp = np.array([150.0 + 10.0 * i for i in range(25)])
    aPress = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 3.0, 4.0, 5.0,
                       10.0, 15.0, 20.0, 25.0, 30.0, 35.0])

    def __init__(self, sPath, iColumn=4):
        aRaw = np.loadtxt(sPath, skiprows=1, usecols=(iColumn,))
        # driver.f90 reads pressure in the outer loop, temperature in the inner.
        self.aTable = aRaw[:self.aTemp.size * self.aPress.size].reshape(
            self.aPress.size, self.aTemp.size).T

    @staticmethod
    def _brk(aLev, fVal):
        """INTERPOLR's bracket: the end pair when outside, so it extrapolates."""
        iR = int(np.searchsorted(aLev, fVal, side="left"))
        if iR <= 0:
            return 0, 1
        if iR >= len(aLev) - 1:
            return len(aLev) - 2, len(aLev) - 1
        return iR - 1, iR

    def fdOLR(self, fTK, fPCO2barTable):
        """Bilinear in (T, ln pCO2) on the table's own coordinate."""
        iTL, iTR = self._brk(self.aTemp, fTK)
        iPL, iPR = self._brk(self.aPress, fPCO2barTable)
        fPL, fPR = np.log(self.aPress[iPL]), np.log(self.aPress[iPR])
        fP = np.log(max(fPCO2barTable, 1e-300))
        fDen = (self.aTemp[iTR] - self.aTemp[iTL]) * (fPR - fPL)
        fNA = (self.aTemp[iTR] - fTK) * (fP - fPL) / fDen
        fNB = (fTK - self.aTemp[iTL]) * (fP - fPL) / fDen
        fNC = (self.aTemp[iTR] - fTK) * (fPR - fP) / fDen
        fND = (fTK - self.aTemp[iTL]) * (fPR - fP) / fDen
        return float(fNA * self.aTable[iTL, iPR] + fNB * self.aTable[iTR, iPR]
                     + fNC * self.aTable[iTL, iPL] + fND * self.aTable[iTR, iPL])

    def faOLR(self, aTK, fFCO2, fPdryBar=1.0, fFactor=CO2_COORD_FACTOR):
        """OLR against a CO2 MIXING RATIO, converted as driver.f90 converts it."""
        fCoord = fFactor * fPdryBar * fFCO2 / (1.0 - fFCO2)
        return np.array([self.fdOLR(float(t), fCoord) for t in np.atleast_1d(aTK)])


# ---------------------------------------------------------------------------

def fdictSlopeAndForcing(fnOLR, fT0=288.0, fPCO2=280e-6, fDT=1.0, fDlog=0.05):
    """Local Planck slope B = dOLR/dT and CO2 forcing -dOLR/dlog10(pCO2)."""
    fB = (fnOLR(np.array([fT0 + fDT]), fPCO2)[0]
          - fnOLR(np.array([fT0 - fDT]), fPCO2)[0]) / (2 * fDT)
    fHi = fnOLR(np.array([fT0]), fPCO2 * 10 ** fDlog)[0]
    fLo = fnOLR(np.array([fT0]), fPCO2 * 10 ** (-fDlog))[0]
    fF = -(fHi - fLo) / (2 * fDlog)
    return {"fOLR": float(fnOLR(np.array([fT0]), fPCO2)[0]),
            "fPlanckB": float(fB),
            "fCO2ForcingPerDecade": float(fF),
            "fEffectiveA": float(fnOLR(np.array([fT0]), fPCO2)[0] - fB * (fT0 - 273.15))}


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-olr-table", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oTab = HextorTable(oArgs.hextor_table)
    oOps = OpsOlrTable(oArgs.ops_olr_table)

    aT = np.linspace(200.0, 330.0, 131)
    aCO2ppm = np.array([1.0, 10.0, 100.0, 280.0, 1000.0, 1.0e4, 1.0e5])

    dictSurfaces = {}
    for fPPM in aCO2ppm:
        fBar = fPPM * 1e-6
        dictSurfaces["%g" % fPPM] = {
            "avalon": faOLRAvalon(aT, fBar).tolist(),
            "poise_fixed": faOLRPoiseFixed(aT).tolist(),
            "poise_wk97": faOLRPoiseWK97(aT, fBar).tolist(),
            "poise_hm16": faOLRPoiseHM16(aT, fBar).tolist(),
            "poise_sms09": faOLRPoiseSMS09(aT).tolist(),
            "hextor": oTab.faOLR(aT, fBar).tolist(),
            "ops": oOps.faOLR(aT, fBar).tolist(),
            "shields_bitz": faOLRShieldsBitz(aT).tolist(),
        }

    dictLocal = {}
    listSchemes = [
        ("avalon", lambda t, p: faOLRAvalon(t, p)),
        ("poise_fixed", lambda t, p: faOLRPoiseFixed(t)),
        ("poise_wk97", faOLRPoiseWK97),
        ("poise_hm16", faOLRPoiseHM16),
        ("poise_sms09", lambda t, p: faOLRPoiseSMS09(t)),
        ("hextor", lambda t, p: oTab.faOLR(t, p)),
        ("ops", lambda t, p: oOps.faOLR(t, p)),
        ("shields_bitz", lambda t, p: faOLRShieldsBitz(t)),
    ]
    for sName, fn in listSchemes:
        dictLocal[sName] = {
            "at288K_280ppm": fdictSlopeAndForcing(fn, 288.0, 280e-6),
            "at273K_280ppm": fdictSlopeAndForcing(fn, 273.15, 280e-6),
            "at250K_280ppm": fdictSlopeAndForcing(fn, 250.0, 280e-6),
            "at288K_1ppm": fdictSlopeAndForcing(fn, 288.0, 1e-6),
            "at288K_1e5ppm": fdictSlopeAndForcing(fn, 288.0, 1e-1),
        }

    # Total CO2 forcing across the v1.1 Experiment 4 range, at fixed T = 288 K.
    dictExp4Forcing = {}
    for sName, fn in listSchemes:
        fLo = float(fn(np.array([288.0]), 1e-6)[0])
        fHi = float(fn(np.array([288.0]), 1e-1)[0])
        dictExp4Forcing[sName] = {
            "fOLR_1ppm": fLo, "fOLR_1e5ppm": fHi,
            "fTotalForcing_Wm2": fLo - fHi,
        }

    dictOut = {
        "aTemperatureK": aT.tolist(),
        "aCO2ppm": aCO2ppm.tolist(),
        "dictSurfaces": dictSurfaces,
        "dictLocalCoefficients": dictLocal,
        "dictExp4Forcing": dictExp4Forcing,
        "dictOpsTableAxes": {
            "aTemperatureK": oOps.aTemp.tolist(),
            "aCO2CoordinateBar": oOps.aPress.tolist(),
            "fCO2CoordinateFactor": CO2_COORD_FACTOR,
            "fTableMinCO2ppmEquivalent":
                float(oOps.aPress[0] / CO2_COORD_FACTOR * 1e6),
            "bExtrapolatesOutsideAxes": True,
        },
        "dictHextorTableAxes": {
            "aTemperatureK": oTab.aTemp.tolist(),
            "fFCO2Min": float(oTab.aFCO2.min()),
            "fFCO2Max": float(oTab.aFCO2.max()),
            "iNumCO2": int(len(oTab.aFCO2)),
            "fNEffUpperMean": float(oTab.aNEffUpper.mean()),
            "fNEffLowerMean": float(oTab.aNEffLower.mean()),
        },
    }
    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f)

    print("HEXTOR table: T in [%.1f, %.1f] K over %d nodes; fCO2 in [%.3g, %.3g]"
          % (oTab.aTemp.min(), oTab.aTemp.max(), len(oTab.aTemp),
             oTab.aFCO2.min(), oTab.aFCO2.max()))
    print("OPS table:  T in [%.1f, %.1f] K over %d nodes; CO2 coordinate in "
          "[%.3g, %.3g] bar (= %.1f ppm upward at 1 bar, after the 44/28 factor)"
          % (oOps.aTemp.min(), oOps.aTemp.max(), len(oOps.aTemp),
             oOps.aPress.min(), oOps.aPress.max(),
             oOps.aPress[0] / CO2_COORD_FACTOR * 1e6))
    print("\n%-13s %8s %8s %8s %10s" % ("scheme", "OLR288", "B", "A_eff", "F/decade"))
    for sName, _ in listSchemes:
        d = dictLocal[sName]["at288K_280ppm"]
        print("%-13s %8.2f %8.3f %8.2f %10.2f"
              % (sName, d["fOLR"], d["fPlanckB"], d["fEffectiveA"],
                 d["fCO2ForcingPerDecade"]))
    print("\nExperiment 4 total forcing, 1 ppm -> 1e5 ppm, at 288 K:")
    for sName, _ in listSchemes:
        print("  %-13s %7.1f W/m2" % (sName, dictExp4Forcing[sName]["fTotalForcing_Wm2"]))


if __name__ == "__main__":
    main()
