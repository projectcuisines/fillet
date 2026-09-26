"""Rebuild OPS driver.f90 diffusion stencil (lines 455-546, 990-1008) and test conservation and explicit-Euler stability."""
import numpy as np, re, sys
iBelts = 36
def ftupleGrid():
    fPi = np.float32(np.pi)
    daX = np.zeros(iBelts + 2)
    daX[0], daX[-1] = -1.0, 1.0
    daLat = np.array([-np.pi / 2 + (1 + 2 * k) * np.pi / (2 * iBelts) for k in range(iBelts)])
    daLat[0] = np.radians(-87.5)
    daX[1:-1] = np.sin(daLat)
    daDel = np.abs(np.diff(daX))                       # delx(0..nbelts)
    daArea = np.abs(np.sin(daLat + np.pi / 72) - np.sin(daLat - np.pi / 72)) / 2
    return daX, daDel, daArea, daLat

def fdaT2prime(daT, daX, daDel):
    daTf = np.concatenate([[daT[0]], daT, [daT[-1]]])  # temp(0)=temp(1), temp(n+1)=temp(n)
    daTp = np.diff(daTf) / daDel                       # tprime(0..nbelts)
    daOut = np.zeros(iBelts)
    for k in range(1, iBelts + 1):
        fDk, fDm = daDel[k], daDel[k - 1]
        fX = daX[k]
        fTpa = (daTp[k] * fDk + daTp[k - 1] * fDm) / (fDk + fDm)
        daOut[k - 1] = ((((fDk / 2) ** 2) - ((fDm / 2) ** 2)) * (1 - fX ** 2) * fTpa
            + ((fDm / 2) ** 2) * (1 - (fX + fDk / 2) ** 2) * daTp[k]
            - ((fDk / 2) ** 2) * (1 - (fX - fDm / 2) ** 2) * daTp[k - 1]) / ((fDk / 2) * (fDm / 2) * (fDk / 2 + fDm / 2))
    return daOut

def fnMain(sLatFile):
    daX, daDel, daArea, daLat = ftupleGrid()
    daMat = np.column_stack([fdaT2prime(np.eye(iBelts)[j], daX, daDel) for j in range(iBelts)])
    print("area sum =", daArea.sum())
    daColumn = daArea @ daMat
    print("max |sum_k area_k L_kj| (should be 0 for conservative diffusion):", np.abs(daColumn).max())
    daT = np.array([[float(x) for x in s.split()][1] for s in open(sLatFile) if re.match(r"\s*-?\d", s)])
    for fD in (0.5, 3.9):
        print(f"D={fD}: global-mean diffusive heating for Ben2 annual-mean T profile = {fD * daArea @ fdaT2prime(daT, daX, daDel):+.4f} W/m2;"
              f" pole-belt D*t2prime={fD*fdaT2prime(daT, daX, daDel)[0]:+.2f}, equator={fD*fdaT2prime(daT, daX, daDel)[17]:+.2f} W/m2")
    daEig = np.linalg.eigvals(daMat)
    print("eigenvalue range of stencil:", daEig.real.min(), daEig.real.max(), "max |imag|", np.abs(daEig.imag).max())
    for fD, fC in ((0.5, 1e7), (3.9, 1e7), (0.5, 0.75 * 4e8 + 0.25e7)):
        print(f"  explicit Euler dt=8640 s, D={fD}, C={fC:.3g}: max |lambda|*D*dt/C = {abs(daEig.real.min())*fD*8640/fC:.4f} (stable if < 2)")

fnMain(sys.argv[1])
