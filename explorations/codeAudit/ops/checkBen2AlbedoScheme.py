"""Bound OPS clear-sky surface albedo of the Ben2 polar belt under protocol (bench) and native albedo schemes (driver.f90:1118-1229)."""
import math
listFresnel = [2.1]*10 + [2.1]*10 + [2.1]*5 + [2.2]*5 + [2.2,2.3,2.3,2.3,2.3,2.4,2.4,2.4,2.4,2.5] + \
    [2.5,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.4] + [3.5,3.6,3.8,4.0,4.2,4.4,4.7,5.0,5.4,5.8] + \
    [6.2,6.6,7.0,7.5,8.2,9.6,10.4,11.4,11.4,12.4] + [13.6,14.8,16.2,17.8,19.6,21.5,23.8,26.0,28.8,31.4] + \
    [35.0,38.6,42.8,47.6,52.9,58.6,65.0,72.0,80.6,89.6]
def ffIce(fT):
    return 0.0 if fT >= 273.15 else (1.0 if fT <= 239 else 1 - math.exp((fT - 273.15) / 12.5))
def ffSnow(fT):
    fVis = 0.7 if fT <= 263.15 else (0.7 - 0.02 * (fT - 263.15) if fT < 273.15 else 0.5)
    fNir = 0.5 if fT <= 263.15 else (0.5 - 0.028 * (fT - 263.15) if fT < 273.15 else 0.22)
    return fNir * 0.467 + fVis * 0.533
def ffSurf(fT, bBench, fLandSnow, iZen):
    fIce = ffIce(fT)
    fSnow = 0.6 if bBench else ffSnow(fT)
    fGround = 0.3 if bBench else 0.22
    fH2o = 1.0 if iZen >= 90 else listFresnel[iZen] / 90.
    fOcean = 0.2 if bBench else (fH2o if fT >= 273.15 else (fSnow if fT <= 239 else fH2o * (1 - fIce) + fSnow * fIce))
    fLand = fSnow * fLandSnow + fGround * (1 - fLandSnow) if fT <= 263.15 else fGround
    fIceAlb = fSnow   # icealb(k) set only when T<=263.15; assume it was set
    return 0.25 * fLand + 0.75 * ((1 - fIce) * fOcean + fIce * fIceAlb)
for bBench in (True, False):
    listV = [ffSurf(fT / 10, bBench, fLs, iZ) for fT in range(2548, 2648) for fLs in (0.5, 1.0) for iZ in (60, 75, 85, 90)]
    print(f"bench={bBench}: clear-sky surface albedo range over T=254.8-264.8 K, landsnowfrac 0.5/1, zenith 60-90: {min(listV):.3f}-{max(listV):.3f}")
print("Archived Ben2 Asurf at the pole belts: 0.5415 (-88), 0.5413 (88); water-cloud term is 0 for T<263 K (driver.f90:1771)")
