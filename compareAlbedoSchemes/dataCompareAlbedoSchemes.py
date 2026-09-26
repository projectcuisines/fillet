"""Reimplement the surface- and planetary-albedo parameterisation of each model from its own source and tabulate the four schemes against temperature and zenith angle.

The protocol prescribes surface albedos (0.30 land / 0.20 ocean / 0.60 ice)
but leaves the top-of-atmosphere albedo to "the nominal configuration of your
model".  The five codes interpret that instruction in structurally different
ways:

  AVALON  There is no atmosphere in the shortwave.  The prescribed surface
          albedo IS the planetary albedo, applied as a hard step at `T_ice`
          (avalon.jl `albedo`): alpha = 0.60 below the threshold, otherwise
          f_land*0.30 + (1-f_land)*0.20 = 0.225 at the protocol land fraction.
          No zenith-angle dependence.

  POISE   With bCalcAB = 0 (the FILLET benchmark configuration) the albedo is
          the prescribed surface value plus a zenith-angle term
          (poise.c fvAlbedoSeasonal):
              alpha = alpha_surf + 0.08 (3 sin^2 z - 1) / 2,
          overridden by dIceAlbedo when the column is frozen (T <= -2 C on
          both land and ocean) or carries ice-sheet mass.  Land and ocean are
          separate columns with separate temperatures.

  HEXTOR  A surface albedo with FRACTIONAL ice cover over a 10 K window
          (driver.f):  f_ice = 1 - exp((T - 273.15)/10) for 263.15 < T < 273.15,
          then a full radiative-transfer lookup of the planetary albedo as a
          function of (p, CO2, T, zenith, surface albedo).

  OPS   The same Kasting-lineage construction as HEXTOR, but with every
          piece widened (driver.f90).  Fractional sea ice over a THIRTY-FOUR
          kelvin window, f_ice = 1 - exp((T - 273.15)/12.5) down to a hard
          floor at 239 K; a snow/ice albedo that is itself temperature
          dependent and spectrally weighted by the host star,
          alpha_snow = 0.5 f_ir + 0.7 f_vis at depth, relaxing to
          0.22 f_ir + 0.5 f_vis at the melting point; a Fresnel ocean albedo
          read from `data/fresnel_reflct.dat` as a function of incidence
          angle; a cloud albedo alpha_cloud = -0.078 + 0.65 z[rad]; and a
          planetary-albedo lookup that is quadrilinear in
          (T, pCO2, zenith, surface albedo) on a 19-node zenith axis.  Clouds
          enter by BLENDING above 273.15 K and by FLOORING below it, which
          puts a step in the albedo at the melting point independent of the
          ice.

  SHIELDS-BITZ  Like AVALON there is no shortwave atmosphere, so the surface
          albedo IS the planetary albedo (`rprimew = A - (1 - alb_w) S`).  The
          form is North's zenith term written in sin(latitude) rather than
          zenith angle (`albedo_seasonal`):
              alpha_ocean = A_o + 0.08 (3 x^2 - 1)/2 - 0.05,
              alpha_land  = A_l + 0.08 (3 x^2 - 1)/2 + 0.05,
          with a hard step to A_50 on either surface at T <= -2 C.  The three
          constants are NOT the protocol's prescribed values: they are
          broadband albedos weighted by the host star's spectrum, and for a G
          dwarf they are A_o = 0.319, A_l = 0.415 and A_50 = 0.514 against the
          protocol's 0.20, 0.30 and 0.60.  The scheme has no ramp, so its ice
          feedback acts over zero width, as AVALON's and POISE's do.

The consequence measured here is the effective planetary albedo each scheme
assigns to the same surface state, and the width in temperature over which
the ice-albedo feedback acts -- the quantity that sets how sharp each model's
bifurcations are.
"""

import argparse
import json
import os

import numpy as np

LAND_FRAC = 0.25
A_LAND, A_OCEAN, A_ICE = 0.30, 0.20, 0.60

# OPS's own defaults, from input_ebm.dat and driver.f90.
OPS_FCLOUD, OPS_LANDSNOWFRAC = 0.55, 0.5
OPS_CLOUD_ALPHA, OPS_CLOUD_BETA = -0.078, 0.65
OPS_FVIS_SUN, OPS_FIR_SUN = 0.533, 0.467
CO2_COORD_FACTOR = 44.0 / 28.0


def faAlbedoAvalon(aTK, fTIceC=0.0):
    """avalon.jl `albedo`: hard step, planetary albedo == surface albedo."""
    fIceFree = LAND_FRAC * A_LAND + (1 - LAND_FRAC) * A_OCEAN
    return np.where(np.asarray(aTK) - 273.15 < fTIceC, A_ICE, fIceFree)


def faAlbedoPoise(aTK, fZenithDeg, fSurfAlbedo, fFrzC=-2.0):
    """poise.c fvAlbedoSeasonal with bCalcAB = 0."""
    fZ = np.deg2rad(fZenithDeg)
    fBase = fSurfAlbedo + 0.08 * (3.0 * np.sin(fZ) ** 2 - 1.0) / 2.0
    return np.where(np.asarray(aTK) - 273.15 <= fFrzC, A_ICE, fBase)


# Shields-Bitz broadband albedos for a G dwarf, from `get_broadband_albedo`.
SB_A_OCEAN, SB_A_LAND, SB_A_ICE = 0.31948, 0.41484, 0.51363


def faAlbedoShieldsBitz(aTK, fSinLat, fLandFrac=LAND_FRAC, fFrzC=-2.0):
    """`albedo_seasonal`: the North zenith term in sin(latitude), stepping at -2 C."""
    fGeom = 0.08 * (3.0 * fSinLat ** 2 - 1.0) / 2.0
    aOcean = np.where(aTK - 273.15 <= fFrzC, SB_A_ICE, SB_A_OCEAN + fGeom - 0.05)
    aLand = np.where(aTK - 273.15 <= fFrzC, SB_A_ICE, SB_A_LAND + fGeom + 0.05)
    return fLandFrac * aLand + (1.0 - fLandFrac) * aOcean


def faIceFractionHextor(aTK):
    """driver.f fractional ice cover over the 263.15-273.15 K window."""
    aT = np.asarray(aTK, dtype=float)
    return np.where(aT >= 273.15, 0.0,
                    np.where(aT < 263.15, 1.0, 1.0 - np.exp((aT - 273.15) / 10.0)))


def faSurfaceAlbedoHextor(aTK, fOceanFrac=0.75, fSnowAlb=0.6, fGroundAlb=0.3,
                          fOceanAlb=0.2):
    """driver.f surface albedo with fractional sea ice and snow-covered land."""
    aT = np.asarray(aTK, dtype=float)
    aFIce = faIceFractionHextor(aT)
    aLandAlb = np.where(aT <= 273.15, fSnowAlb, fGroundAlb)
    return ((1 - fOceanFrac) * aLandAlb
            + fOceanFrac * ((1 - aFIce) * fOceanAlb + aFIce * fSnowAlb))


class HextorPalb(object):
    """Python port of radiation.f90 read_table_v1 + getPALB (pentalinear)."""

    def __init__(self, sPath):
        import h5py
        with h5py.File(sPath, "r") as f:
            aOLR = np.array(f["/olr"])
            aAlb = np.array(f["/palb"])
        iNCO2, iNT = 92, 19
        self.aFCO2 = aOLR[:iNCO2, 1]
        self.aTemp = aOLR[::iNCO2, 2]
        self.aZen = np.array([0.0, 30.0, 60.0, 90.0])
        self.aSab = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.aTable = np.zeros((len(self.aSab), len(self.aZen), iNT, iNCO2))
        for aRow in aAlb:
            ic = int(np.argmin(np.abs(self.aFCO2 - aRow[1])))
            it = int(np.argmin(np.abs(self.aTemp - aRow[2])))
            iz = int(np.argmin(np.abs(self.aZen - aRow[3])))
            ia = int(np.argmin(np.abs(self.aSab - aRow[4])))
            self.aTable[ia, iz, it, ic] = aRow[5]

    @staticmethod
    def _brk(aLev, fVal, bLog=False):
        if fVal <= aLev[0]:
            return 0, 0, 0.0
        if fVal >= aLev[-1]:
            return len(aLev) - 1, len(aLev) - 1, 0.0
        i = int(np.searchsorted(aLev, fVal) - 1)
        if bLog:
            fF = ((np.log10(fVal) - np.log10(aLev[i]))
                  / (np.log10(aLev[i + 1]) - np.log10(aLev[i])))
        else:
            fF = (fVal - aLev[i]) / (aLev[i + 1] - aLev[i])
        return i, i + 1, fF

    def fdPalb(self, fTK, fFCO2, fZenDeg, fSurfAlb):
        ic, ic2, fc = self._brk(self.aFCO2, fFCO2, True)
        it, it2, ft = self._brk(self.aTemp, min(max(fTK, self.aTemp[0]),
                                                self.aTemp[-1]))
        iz, iz2, fz = self._brk(self.aZen, fZenDeg)
        ia, ia2, fa = self._brk(self.aSab, fSurfAlb)
        fVal = 0.0
        for kc, wc in ((ic, 1 - fc), (ic2, fc)):
            for kt, wt in ((it, 1 - ft), (it2, ft)):
                for kz, wz in ((iz, 1 - fz), (iz2, fz)):
                    for ka, wa in ((ia, 1 - fa), (ia2, fa)):
                        w = wc * wt * wz * wa
                        if w:
                            fVal += w * self.aTable[ka, kz, kt, kc]
        return fVal


# ---------------------------------------------------------------------------
# OPS
# ---------------------------------------------------------------------------

def faIceFractionOps(aTK):
    """driver.f90 fractional ice cover: a 34 K ramp, not HEXTOR's 10 K one."""
    aT = np.asarray(aTK, dtype=float)
    return np.where(aT >= 273.15, 0.0,
                    np.where(aT <= 239.0, 1.0,
                             1.0 - np.exp((aT - 273.15) / 12.5)))


def faSnowAlbedoOps(aTK, fFVis=OPS_FVIS_SUN, fFIr=OPS_FIR_SUN):
    """Temperature-dependent, spectrally weighted snow/ice albedo."""
    aT = np.asarray(aTK, dtype=float)
    aVis = np.where(aT <= 263.15, 0.7,
                    np.where(aT < 273.15, 0.7 - 0.020 * (aT - 263.15), 0.5))
    aNir = np.where(aT <= 263.15, 0.5,
                    np.where(aT < 273.15, 0.5 - 0.028 * (aT - 263.15), 0.22))
    return aNir * fFIr + aVis * fFVis


def faOceanAlbedoTable(sPath):
    """`fresnel_reflct.dat` as driver.f90 reads it: percent, divided by 90."""
    listRows = []
    for sLine in open(sPath):
        listTok = sLine.split()
        if len(listTok) == 11 and listTok[0].isdigit():
            listRows.extend(float(t) for t in listTok[1:])
    aAlb = np.array(listRows) / 90.0
    return np.append(aAlb, 1.00)          # driver.f90 sets h20alb(90) = 1.00


def fdCloudAlbedoOps(fZenDeg):
    """acloud = alpha + beta z, z in radians, floored at 0.1."""
    fVal = OPS_CLOUD_ALPHA + OPS_CLOUD_BETA * np.deg2rad(fZenDeg)
    return max(fVal, 0.1) if fVal <= 0 else fVal


def faSurfaceAlbedoOps(aTK, aOceanAlb, fZenDeg, fOceanFrac=1 - LAND_FRAC,
                         fGroundAlb=A_LAND, fFCloud=OPS_FCLOUD):
    """driver.f90's three-branch surface albedo, clouds blended then floored."""
    aT = np.asarray(aTK, dtype=float)
    aFIce = faIceFractionOps(aT)
    aSnow = faSnowAlbedoOps(aT)
    aLandAlb = np.where(aT <= 263.15,
                        aSnow * OPS_LANDSNOWFRAC
                        + fGroundAlb * (1 - OPS_LANDSNOWFRAC), fGroundAlb)
    aBare = ((1 - fOceanFrac) * aLandAlb
             + fOceanFrac * ((1 - aFIce) * aOceanAlb + aFIce * aSnow))
    fCloud = fFCloud * fdCloudAlbedoOps(fZenDeg)
    return np.where(aT >= 273.15, (1 - fFCloud) * aBare + fCloud,
                    np.maximum(aBare, fCloud))


class OpsPalb(object):
    """Port of driver.f90's PALB table read + INTERPPALB (quadrilinear)."""

    aTemp = np.array([150.0 + 10.0 * i for i in range(25)])
    aPress = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 3.0, 4.0, 5.0,
                       10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    aZen = np.array([5.0 * i for i in range(19)])
    aSab = np.array([0.05 * i for i in range(21)])

    def __init__(self, sPath):
        aRaw = np.loadtxt(sPath, skiprows=1, usecols=(2,))
        # File order is zenith, then surface albedo, then pressure, then T.
        self.aTable = aRaw.reshape(len(self.aZen), len(self.aSab),
                                   len(self.aPress), len(self.aTemp))

    @staticmethod
    def _brk(aLev, fVal):
        """INTERPPALB's bracket: the end pair when outside, so it extrapolates."""
        iR = int(np.searchsorted(aLev, fVal, side="left"))
        if iR <= 0:
            return 0, 1
        if iR >= len(aLev) - 1:
            return len(aLev) - 2, len(aLev) - 1
        return iR - 1, iR

    def fdPalb(self, fTK, fFCO2, fZenDeg, fSurfAlb, fPdryBar=1.0):
        fCoord = CO2_COORD_FACTOR * fPdryBar * fFCO2 / (1.0 - fFCO2)
        iTL, iTR = self._brk(self.aTemp, fTK)
        iPL, iPR = self._brk(self.aPress, fCoord)
        iZL, iZR = self._brk(self.aZen, fZenDeg)
        iAL, iAR = self._brk(self.aSab, fSurfAlb)
        fT = (fTK - self.aTemp[iTL]) / (self.aTemp[iTR] - self.aTemp[iTL])
        fPLo, fPHi = np.log(self.aPress[iPL]), np.log(self.aPress[iPR])
        fP = (np.log(max(fCoord, 1e-300)) - fPLo) / (fPHi - fPLo)
        fZ = (fZenDeg - self.aZen[iZL]) / (self.aZen[iZR] - self.aZen[iZL])
        fA = (fSurfAlb - self.aSab[iAL]) / (self.aSab[iAR] - self.aSab[iAL])
        fVal = 0.0
        for iT, wT in ((iTL, 1 - fT), (iTR, fT)):
            for iP, wP in ((iPL, 1 - fP), (iPR, fP)):
                for iZ, wZ in ((iZL, 1 - fZ), (iZR, fZ)):
                    for iA, wA in ((iAL, 1 - fA), (iAR, fA)):
                        fVal += wT * wP * wZ * wA * self.aTable[iZ, iA, iP, iT]
        return float(fVal)


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--hextor-table", required=True)
    oParser.add_argument("--ops-palb-table", required=True)
    oParser.add_argument("--ops-fresnel-table", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    oPalb = HextorPalb(oArgs.hextor_table)
    oOpsPalb = OpsPalb(oArgs.ops_palb_table)
    aOceanTable = faOceanAlbedoTable(oArgs.ops_fresnel_table)
    aT = np.linspace(230.0, 310.0, 321)

    aAvalon0 = faAlbedoAvalon(aT, 0.0)
    aAvalonM10 = faAlbedoAvalon(aT, -10.0)
    aPoise = faAlbedoPoise(aT, 45.0, LAND_FRAC * A_LAND + (1 - LAND_FRAC) * A_OCEAN)
    aHexSurf = faSurfaceAlbedoHextor(aT)
    aHexPalb = np.array([oPalb.fdPalb(float(t), 2.8e-4, 45.0, float(s))
                         for t, s in zip(aT, aHexSurf)])

    # OPS at the same 45 deg zenith angle and the same protocol land fraction.
    fOceanAlb45 = float(aOceanTable[45])
    aOpsSurf = faSurfaceAlbedoOps(aT, fOceanAlb45, 45.0)
    aOpsSurfNoCloud = faSurfaceAlbedoOps(aT, fOceanAlb45, 45.0, fFCloud=0.0)
    aOpsPalb = np.array([oOpsPalb.fdPalb(float(t), 2.8e-4, 45.0, float(s))
                         for t, s in zip(aT, aOpsSurf)])

    # Shields-Bitz at the latitude whose sin(lat) matches the 45 deg the other
    # schemes are evaluated at, so the geometric term is compared like for like.
    aShields = faAlbedoShieldsBitz(aT, np.sin(np.deg2rad(45.0)))

    aBelowMelt = aT <= 273.15

    def fdTransitionWidth(aAlb):
        """Width in T over which the ice feedback spans 10-90% of its range.

        Measured BELOW the melting point only.  Three of the four schemes are
        flat in T above 273.15 K, so the restriction changes nothing for them;
        OPS is not, because its cloud term switches from a floor to a blend
        at 273.15 K and its planetary albedo keeps falling with temperature
        through the water-vapour axis of its lookup table.  Measured over the
        whole range, that non-monotonicity would put every warm temperature
        back inside the 10-90% band and report a spurious width.
        """
        aSub, aTSub = aAlb[aBelowMelt], aT[aBelowMelt]
        fLo, fHi = aSub.min(), aSub.max()
        if fHi - fLo < 1e-6:
            return 0.0
        aN = (aSub - fLo) / (fHi - fLo)
        aIn = aTSub[(aN > 0.1) & (aN < 0.9)]
        return float(aIn.max() - aIn.min()) if aIn.size else 0.0

    def fdMeltingPointStep(aAlb):
        """Albedo discontinuity across 273.15 K (positive = jumps up on melting)."""
        iHi = int(np.searchsorted(aT, 273.15))
        return float(aAlb[iHi] - aAlb[iHi - 1])

    dictOut = {
        "aTemperatureK": aT.tolist(),
        "aAlbedoAvalon_Tice0C": aAvalon0.tolist(),
        "aAlbedoAvalon_TiceM10C": aAvalonM10.tolist(),
        "aAlbedoPoise_zenith45": aPoise.tolist(),
        "aSurfaceAlbedoHextor": aHexSurf.tolist(),
        "aPlanetaryAlbedoHextor": aHexPalb.tolist(),
        "aSurfaceAlbedoOps": aOpsSurf.tolist(),
        "aSurfaceAlbedoOpsNoCloud": aOpsSurfNoCloud.tolist(),
        "aPlanetaryAlbedoOps": aOpsPalb.tolist(),
        "aAlbedoShieldsBitz": aShields.tolist(),
        "dictTransitionWidthsK": {
            "avalon": fdTransitionWidth(aAvalon0),
            "poise": fdTransitionWidth(aPoise),
            "hextor_surface": fdTransitionWidth(aHexSurf),
            "hextor_planetary": fdTransitionWidth(aHexPalb),
            "ops_surface": fdTransitionWidth(aOpsSurf),
            "ops_planetary": fdTransitionWidth(aOpsPalb),
            "shields_bitz": fdTransitionWidth(aShields),
        },
        "dictIceFreeAlbedo": {
            "avalon": float(aAvalon0[-1]),
            "poise_zenith45": float(aPoise[-1]),
            "hextor_surface": float(aHexSurf[-1]),
            "hextor_planetary": float(aHexPalb[-1]),
            "ops_surface": float(aOpsSurf[-1]),
            "ops_planetary": float(aOpsPalb[-1]),
            "shields_bitz": float(aShields[-1]),
        },
        "dictGlaciatedAlbedo": {
            "avalon": float(aAvalon0[0]),
            "poise_zenith45": float(aPoise[0]),
            "hextor_surface": float(aHexSurf[0]),
            "hextor_planetary": float(aHexPalb[0]),
            "ops_surface": float(aOpsSurf[0]),
            "ops_planetary": float(aOpsPalb[0]),
            "shields_bitz": float(aShields[0]),
        },
        "dictMeltingPointStep": {
            "avalon": fdMeltingPointStep(aAvalon0),
            "poise": fdMeltingPointStep(aPoise),
            "hextor_surface": fdMeltingPointStep(aHexSurf),
            "hextor_planetary": fdMeltingPointStep(aHexPalb),
            "ops_surface": fdMeltingPointStep(aOpsSurf),
            "ops_planetary": fdMeltingPointStep(aOpsPalb),
            "shields_bitz": fdMeltingPointStep(aShields),
        },
        "dictOpsCloudTreatment": {
            "fCloudFraction": OPS_FCLOUD,
            "fCloudAlbedoAt45deg": fdCloudAlbedoOps(45.0),
            "fOceanAlbedoAt45deg": fOceanAlb45,
            "fOceanAlbedoDivisorInSource": 90.0,
            "fOceanAlbedoAt45degIfDividedBy100": float(aOceanTable[45] * 0.9),
            "fStepAtMeltingPoint": fdMeltingPointStep(aOpsSurf),
        },
    }

    # POISE zenith-angle dependence, and HEXTOR's zenith dependence at fixed
    # surface albedo, are the two mechanisms that break the surface/TOA identity.
    aZen = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    dictOut["dictZenithDependence"] = {
        "aZenithDeg": aZen.tolist(),
        "aPoiseIceFree": [float(faAlbedoPoise(np.array([300.0]), float(z), 0.225)[0])
                          for z in aZen],
        "aHextorPlanetaryIceFree": [float(oPalb.fdPalb(300.0, 2.8e-4, float(z), 0.225))
                                    for z in aZen],
        "aHextorPlanetaryGlaciated": [float(oPalb.fdPalb(240.0, 2.8e-4, float(z), 0.60))
                                      for z in aZen],
        "aAvalonAnyState": [float(faAlbedoAvalon(np.array([300.0]))[0])] * len(aZen),
        "aOpsPlanetaryIceFree":
            [float(oOpsPalb.fdPalb(300.0, 2.8e-4, float(z), 0.225))
             for z in aZen],
        "aOpsPlanetaryGlaciated":
            [float(oOpsPalb.fdPalb(240.0, 2.8e-4, float(z), 0.60))
             for z in aZen],
        "aOpsOceanFresnel": [float(aOceanTable[int(z)]) for z in aZen],
    }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f)

    print("Albedo of the ice-free and fully glaciated states:")
    print("%-20s %14s %14s" % ("scheme", "ice-free", "glaciated"))
    for sK in dictOut["dictIceFreeAlbedo"]:
        print("%-20s %14.4f %14.4f"
              % (sK, dictOut["dictIceFreeAlbedo"][sK],
                 dictOut["dictGlaciatedAlbedo"][sK]))
    print("\nWidth of the ice-albedo transition below the melting point")
    print("(10-90%% of range), and the albedo step across 273.15 K:")
    for sK, fV in dictOut["dictTransitionWidthsK"].items():
        print("  %-20s %6.2f K   step %+7.4f"
              % (sK, fV, dictOut["dictMeltingPointStep"][sK]))
    print("\nZenith-angle dependence of the planetary albedo (ice-free):")
    print("  %-26s %s" % ("zenith (deg)",
                          " ".join("%7.1f" % z for z in aZen)))
    for sK in ("aAvalonAnyState", "aPoiseIceFree", "aHextorPlanetaryIceFree",
               "aOpsPlanetaryIceFree", "aOpsOceanFresnel"):
        print("  %-26s %s" % (sK[1:],
                              " ".join("%7.4f" % v
                                       for v in dictOut["dictZenithDependence"][sK])))


if __name__ == "__main__":
    main()
