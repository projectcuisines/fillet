"""Find the temperature above which POISE's WK97 OLR fit has dI/dT <= 0 (uncapped) for several pCO2, plus Exp4 hot-row lat temps."""
import io
import subprocess
import numpy as np
from evaluatePoiseOlrSchemes import fdOlrWk97Raw

for dP in (2.8e-4, 1e-3, 1e-2, 3e-2, 1e-1):
    daT = np.arange(250.0, 420.0, 0.05)
    daI = np.array([fdOlrWk97Raw(t, dP) for t in daT])
    daSlope = np.gradient(daI, daT)
    iNeg = np.argmax(daSlope <= 0) if np.any(daSlope <= 0) else None
    sNeg = f"{daT[iNeg]:.1f} K (I={daI[iNeg]:.1f})" if iNeg is not None else "none below 420 K"
    print(f"pCO2={dP:g} bar: dI/dT<=0 first at {sNeg}")

for iCase in (40, 45, 50):
    sText = subprocess.run(["git", "-C", "/workspace/fillet", "show",
                            f"bedeec3:output/poise/exp4_warm/case_{iCase}/lat_output.dat"],
                           capture_output=True, text=True, check=True).stdout
    daLat = np.loadtxt(io.StringIO(sText), comments="#")
    print(f"exp4_warm case {iCase}: max Tsurf={daLat[:,1].max():.2f} K  max OLR={daLat[:,4].max():.2f} W/m2  "
          f"cells with OLR>=299.9: {np.sum(daLat[:,4] >= 299.9)}")
