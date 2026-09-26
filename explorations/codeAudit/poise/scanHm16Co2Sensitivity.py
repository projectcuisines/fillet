"""Scan the sign of dOLR/dlog10(pCO2) for POISE's coded HM16 and WK97 fits over T and pCO2."""
import numpy as np
from evaluatePoiseOlrSchemes import fdOlrHm16, fdOlrWk97Raw

daT = np.array([200.0, 230.0, 260.0, 288.0, 310.0, 340.0])
daLogP = np.arange(-5.0, 0.01, 1.0)

for sName, fdOlr in (("hm16", fdOlrHm16), ("wk97", fdOlrWk97Raw)):
    print(f"{sName}: OLR (W/m2); rows T, cols log10 pCO2(bar) = {daLogP}")
    for dT in daT:
        print(f"  T={dT:5.0f}: " + " ".join(f"{fdOlr(dT, 10**p):7.1f}" for p in daLogP))
