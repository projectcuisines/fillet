"""Reproduce HEXTOR's runtime diffusion rescaling (driver.f diffadj) and the first-step pole ghost-cell transient.

Formulas transcribed from /workspace/hextor/model/driver.f (HEAD) and commit 852fd03.
"""
import numpy as np

MP = 1.67e-24
AVEMOL0 = 28.89 * MP
HCP0, HCPCO2, HCPN2 = 0.2401, 0.2105, 0.2484
NBELTS = 18


def fdDiffFactor(dFco2, bArchived):
    """d/d0 with rot = rot0; archived: pn2=1, pg0=1+pco2; HEAD: pg0=1, pn2=1-pco2."""
    if bArchived:
        dPco2 = dFco2; dPn2 = 1.0; dPg0 = 1.0 + dPco2
    else:
        dPg0 = 1.0; dPco2 = dFco2; dPn2 = dPg0 - dPco2
    dAvemol = MP * (28.0 * dPn2 + 44.0 * dPco2) / dPg0
    dHcp = (HCPN2 * dPn2 + dPco2 * HCPCO2) / dPg0
    return dPg0 * (AVEMOL0 / dAvemol) ** 2 * (dHcp / HCP0)


def fnDiffusion():
    print("Effective D used when namelist d0 is reported as Diff (diffadj=.true., rot=rot0):")
    for dF in (2.8e-4, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1):
        print("  XCO2=%8.1f ppm  HEAD d/d0=%.4f (D=%.4f for d0=0.5)   archived d/d0=%.4f (D=%.4f)"
              % (dF * 1e6, fdDiffFactor(dF, False), 0.5 * fdDiffFactor(dF, False),
                 fdDiffFactor(dF, True), 0.5 * fdDiffFactor(dF, True)))
    print("  Ben1 d0=0.38 -> D = %.4f (HEAD), %.4f (archived)" % (0.38 * fdDiffFactor(2.8e-4, False),
                                                                  0.38 * fdDiffFactor(2.8e-4, True)))


def ftGrid():
    daLat = np.zeros(NBELTS + 2)
    daLat[0], daLat[-1] = -np.pi / 2, np.pi / 2
    daLat[1:-1] = -np.pi / 2 + (1 + 2 * np.arange(NBELTS)) * np.pi / (2 * NBELTS)
    daX = np.sin(daLat)
    daDx = np.abs(np.diff(daX))
    return daX, daDx


def fdaT2prime(daT, daX, daDx):
    daTp = (daT[1:] - daT[:-1]) / daDx
    da2 = np.zeros(NBELTS + 2)
    for k in range(1, NBELTS + 1):
        dxk, dxm = daDx[k], daDx[k - 1]
        dAve = (daTp[k] * dxk + daTp[k - 1] * dxm) / (dxk + dxm)
        da2[k] = (((dxk / 2) ** 2 - (dxm / 2) ** 2) * (1 - daX[k] ** 2) * dAve
                  + (dxm / 2) ** 2 * (1 - (daX[k] + dxk / 2) ** 2) * daTp[k]
                  - (dxk / 2) ** 2 * (1 - (daX[k] - dxm / 2) ** 2) * daTp[k - 1]) \
            / ((dxk / 2) * (dxm / 2) * (dxk / 2 + dxm / 2))
    return da2


def fnPoleTransient():
    daX, daDx = ftGrid()
    print("\nGrid: dx(0) pole->first centre = %.5f, dx(1) = %.5f" % (daDx[0], daDx[1]))
    dD = 0.5 * fdDiffFactor(2.8e-4, False); dDt = 4.32e4
    for dTinit, dC in ((233.0, 1.0e7), (288.0, 0.75 * 4e8 + 0.25 * 1e7)):
        daT = np.full(NBELTS + 2, dTinit); daT[0] = daT[-1] = 273.0  # DATA temp/20*273./ ghost cells
        da2 = fdaT2prime(daT, daX, daDx)
        dFlux = dD * da2[1]
        print("tempinit=%.0f K: ghost=273 K -> D*T'' at polar belt = %.0f W/m2 -> dT in first step = %+.1f K (C=%.2e)"
              % (dTinit, dFlux, dFlux * dDt / dC, dC))


def fnIceDiscontinuity():
    dT = 263.15
    dFiceAbove = 1 - np.exp((dT + 1e-6 - 273.15) / 10.)
    print("\nfice just above icetemp=263.15 K: %.3f ; below: 1.000" % dFiceAbove)
    for dF in (dFiceAbove, 1.0):
        dAs = 0.25 * 0.6 + 0.75 * ((1 - dF) * 0.2 + dF * 0.6)
        dC = (1 - dF) * 0.75 * 4e8 + dF * 0.75 * 1e7 + 0.25 * 1e7
        print("  fice=%.3f -> surface albedo %.3f, heat capacity %.3e J/m2/K" % (dF, dAs, dC))
    print("  land at T<=273.15 K: albedo jumps 0.30 -> 0.60 (landsnowfrac=1)")


if __name__ == "__main__":
    fnDiffusion(); fnPoleTransient(); fnIceDiscontinuity()
