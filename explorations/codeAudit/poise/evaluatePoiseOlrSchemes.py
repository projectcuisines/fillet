"""Evaluate POISE's coded OLR fits (poise.c fdOLRwk97/fdOLRhm16/fdOLRsms09) at FILLET-relevant T and pCO2."""
import numpy as np

dSigma = 5.670367e-8


def fdOlrWk97(dT, dPco2):
    """poise.c:6236-6267 verbatim (T in K, pCO2 in bar)."""
    p = np.log(dPco2 / 3.3e-4)
    T = dT
    I = (9.468980 - 7.714727e-5 * p - 2.794778 * T - 3.244753e-3 * p * T
         - 3.547406e-4 * p**2 + 2.212108e-2 * T**2 + 2.229142e-3 * p**2 * T
         + 3.088497e-5 * p * T**2 - 2.789815e-5 * (p * T * p * T)
         - 3.442973e-3 * p**3 - 3.361939e-5 * T**3 + 9.173169e-3 * p**3 * T
         - 7.775195e-5 * p**3 * T**2 - 1.679112e-7 * p * T**3
         + 6.590999e-8 * p**2 * T**3 + 1.528125e-7 * p**3 * T**3
         - 3.367567e-2 * p**4 - 1.631909e-4 * p**4 * T
         + 3.663871e-6 * p**4 * T**2 - 9.255646e-9 * p**4 * T**3)
    if I >= 300:
        I = 300.0
    if T < 190:
        I = dSigma * T**4
    return I


def fdOlrWk97Raw(dT, dPco2):
    """WK97 polynomial without the 300 cap or 190 K switch."""
    p = np.log(dPco2 / 3.3e-4)
    T = dT
    return (9.468980 - 7.714727e-5 * p - 2.794778 * T - 3.244753e-3 * p * T
            - 3.547406e-4 * p**2 + 2.212108e-2 * T**2 + 2.229142e-3 * p**2 * T
            + 3.088497e-5 * p * T**2 - 2.789815e-5 * (p * T * p * T)
            - 3.442973e-3 * p**3 - 3.361939e-5 * T**3 + 9.173169e-3 * p**3 * T
            - 7.775195e-5 * p**3 * T**2 - 1.679112e-7 * p * T**3
            + 6.590999e-8 * p**2 * T**3 + 1.528125e-7 * p**3 * T**3
            - 3.367567e-2 * p**4 - 1.631909e-4 * p**4 * T
            + 3.663871e-6 * p**4 * T**2 - 9.255646e-9 * p**4 * T**3)


def fdOlrHm16(dT, dPco2):
    """poise.c:6112-6143 verbatim."""
    phi = np.log10(dPco2)
    if dT <= 150:
        return dSigma * dT**4
    t = np.log10(dT)
    f = (9.12805643869791438760 * t**4 + 4.58408794768168803557 * t**3 * phi
         - 8.47261075643147449910e+01 * t**3 + 4.35517381112690282752e-01 * t * phi * t * phi
         - 2.86355036260417961103e+01 * t**2 * phi + 2.96626642498045896446e+02 * t**2
         - 6.01082900358299240806e-02 * t * phi**3 - 2.60414691486954641420 * t * phi**2
         + 5.69812976563675661623e+01 * t * phi - 4.62596100127381816947e+02 * t
         + 2.18159373001564722491e-03 * phi**4 + 1.61456772400726950023e-01 * phi**3
         + 3.75623788187470086797 * phi**2 - 3.53347289223180354156e+01 * phi
         + 2.75011005409836684521e+02)
    return 10**f / 1000.0


def fdOlrSms09(dT, dPco2):
    """poise.c:6189-6201 (no CO2 argument used)."""
    tau = 0.79 * (dT / 273.15)**3
    return dSigma * dT**4 / (1 + 0.75 * tau)


def fdOlrLinear(dT, dPco2, dA=204.05, dB=2.09):
    return dA + dB * (dT - 273.15)


dictSchemes = {"wk97": fdOlrWk97, "hm16": fdOlrHm16, "sms09": fdOlrSms09, "linear": fdOlrLinear}


def fnTable():
    print("OLR (W/m2) by scheme at selected (T, pCO2 bar):")
    for dT, dP in [(288.0, 2.8e-4), (218.73, 1e-6), (218.73, 2.8e-4), (250.0, 2.8e-4),
                   (312.0, 2.8e-4), (330.0, 2.8e-4), (344.0, 0.1)]:
        sRow = "  ".join(f"{s}={f(dT, dP):7.2f}" for s, f in dictSchemes.items())
        print(f"  T={dT:6.2f} K pCO2={dP:8.1e}: {sRow}")


def fnCo2Sensitivity():
    print("\nd OLR / d log10(pCO2) at 288 K, around 280 ppm (W/m2 per decade):")
    for s, f in dictSchemes.items():
        dS = f(288.0, 2.8e-3) - f(288.0, 2.8e-4)
        print(f"  {s}: {dS:+.3f}")


def fnWk97Edges():
    print("\nWK97 as coded: discontinuity at 190 K (pCO2=2.8e-4, 1e-6):")
    for dP in (2.8e-4, 1e-6):
        print(f"  pCO2={dP:g}: poly(190.0001)={fdOlrWk97Raw(190.0001, dP):.2f}  sigmaT^4(189.9999)={dSigma*189.9999**4:.2f}")
    for dP in (2.8e-4, 1e-2, 0.1):
        daT = np.arange(250.0, 400.0, 0.05)
        daI = np.array([fdOlrWk97Raw(t, dP) for t in daT])
        iCap = np.argmax(daI >= 300.0)
        print(f"  pCO2={dP:g}: raw WK97 reaches 300 W/m2 at T={daT[iCap]:.2f} K"
              if daI.max() >= 300 else f"  pCO2={dP:g}: raw max {daI.max():.1f}")
    print("  WK97 fit range per WK97 paper: 1e-5 <= pCO2 <= 10 bar, 190 <= T <= 380 K;"
          f" archive Exp4 XCO2 below 1e-5 bar: {np.sum(10**np.arange(-6, -0.95, 0.1) < 1e-5)} of 51")


def main():
    fnTable()
    fnCo2Sensitivity()
    fnWk97Edges()


if __name__ == "__main__":
    main()
