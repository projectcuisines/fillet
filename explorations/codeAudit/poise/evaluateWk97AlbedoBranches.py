"""Evaluate POISE AlbedoTOA280 vs AlbedoTOA370 (poise.c:6422-6459) below 190 K, and AlbedoTaylor ocean albedo."""
import numpy as np


def fdAlb280(dTc, dPco2, dZen, dAs):
    mu, T = np.cos(dZen), dTc + 273.15
    return (-6.891e-1 + 1.046 * dAs + 7.8054e-3 * T - 2.8373e-3 * dPco2 - 2.8899e-1 * mu
            - 3.7412e-2 * dAs * dPco2 - 6.3499e-3 * mu * dPco2 + 2.0122e-1 * dAs * mu
            - 1.8508e-3 * dAs * T + 1.3649e-4 * mu * T + 9.8581e-5 * dPco2 * T
            + 7.3239e-2 * dAs**2 - 1.6555e-5 * T**2 + 6.5817e-4 * dPco2**2 + 8.1218e-2 * mu**2)


def fdAlb370(dTc, dPco2, dZen, dAs):
    mu, T = np.cos(dZen), dTc + 273.15
    return (1.1082 + 1.5172 * dAs - 5.7993e-3 * T + 1.9705e-2 * dPco2 - 1.867e-1 * mu
            - 3.1355e-2 * dAs * dPco2 - 1.0214e-2 * mu * dPco2 + 2.0986e-1 * dAs * mu
            - 3.7098e-3 * dAs * T - 1.1335e-4 * mu * T + 5.3714e-5 * dPco2 * T
            + 7.5887e-2 * dAs**2 + 9.269e-6 * T**2 - 4.1327e-4 * dPco2**2 + 6.3298e-2 * mu**2)


def fdTaylor(dZen):
    mu = np.cos(dZen)
    return 0.037 / (1.1 * mu**1.4 + 0.15) if mu > 0 else 0.037 / 0.15


print("Ice surface (0.6), pCO2=1e-6 bar: TOA albedo from the branch actually used (370) vs intended fallback")
for dTc in (-85.0, -90.0, -100.0):
    for dZdeg in (40.0, 70.0, 85.0):
        dZ = np.radians(dZdeg)
        print(f"  T={dTc+273.15:6.1f} K zen={dZdeg:4.0f}: used(370)={fdAlb370(dTc,1e-6,dZ,0.6):.3f} "
              f"cold-fit(280)={fdAlb280(dTc,1e-6,dZ,0.6):.3f} intended ice albedo=0.600")
print("\nAlbedoTaylor ocean surface albedo used instead of dAlbedoWater when bCalcAB=1:")
for dZdeg in (0.0, 30.0, 60.0, 80.0):
    print(f"  noon zenith {dZdeg:4.0f} deg: {fdTaylor(np.radians(dZdeg)):.3f}  (protocol ocean albedo 0.2)")
