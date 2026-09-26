"""Reproduce AVALON's insolation sampling and quantify 12-sample seasonal quadrature and old-grid global-mean errors."""
import numpy as np

fS0 = 1361.0


def faDailyMean(daPhi, fDelta):
    """Daily-mean insolation (Berger 1978), as in avalon.jl insolation_instant."""
    daT = -np.tan(daPhi) * np.tan(fDelta)
    daH0 = np.where(daT <= -1, np.pi, np.where(daT >= 1, 0.0, np.arccos(np.clip(daT, -1, 1))))
    return fS0 / np.pi * (daH0 * np.sin(daPhi) * np.sin(fDelta) + np.cos(daPhi) * np.cos(fDelta) * np.sin(daH0))


def faAnnualMean(daPhi, fObl, iSamples, bMidpoint):
    """Annual mean with iSamples solar longitudes (midpoint or left-endpoint sampling)."""
    fEps = np.radians(fObl)
    daQ = np.zeros_like(daPhi)
    for k in range(iSamples):
        fLam = (k + 0.5 if bMidpoint else k) * 2 * np.pi / iSamples
        daQ += faDailyMean(daPhi, np.arcsin(np.sin(fEps) * np.sin(fLam)))
    return daQ / iSamples


def fnMain():
    """Print quadrature error of the 12-step seasonal forcing and old-grid global-mean bias."""
    n = 90
    daX = np.linspace(-1 + 1 / n, 1 - 1 / n, n)
    daPhi = np.arcsin(daX)
    for fObl in [0.0, 23.5, 60.0, 90.0]:
        daRef = faAnnualMean(daPhi, fObl, 3600, True)
        daQ12 = faAnnualMean(daPhi, fObl, 12, False)
        daErr = daQ12 - daRef
        i = int(np.argmax(np.abs(daErr)))
        print(f"obl={fObl:5.1f}: 12-sample annual-mean Q error max={daErr[i]:+.2f} W/m2 at lat {np.degrees(daPhi[i]):.1f} "
              f"(ref {daRef[i]:.1f}); polemost node err {daErr[-1]:+.2f}; equator err {daErr[n//2]:+.2f}; "
              f"global mean Q12 {daQ12.mean():.3f} vs S0/4 {fS0/4:.3f}")
    daXold = np.linspace(-1, 1, n)
    daPhiOld = np.arcsin(daXold)
    for fObl in [0.0, 23.5, 60.0, 90.0]:
        daRef = faAnnualMean(daPhiOld, fObl, 3600, True)
        print(f"old node-on-pole grid obl={fObl:5.1f}: sum/n global-mean Q = {daRef.mean():.2f} W/m2 (exact {fS0/4:.2f})")
    for fObl in [23.5, 60.0]:
        fEps = np.radians(fObl)
        listG = []
        for k in range(12):
            fLam = k * 2 * np.pi / 12
            listG.append(faDailyMean(daPhi, np.arcsin(np.sin(fEps) * np.sin(fLam))).mean())
        print(f"obl={fObl}: per-sample global mean Q on current grid ranges {min(listG):.3f}-{max(listG):.3f}")


if __name__ == "__main__":
    fnMain()
