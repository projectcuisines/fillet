"""Classify whether archived POISE Asurf values are threshold-reconstructed (discrete) or a time-averaged model output."""
import io
import subprocess
import numpy as np

sRepo = "/workspace/fillet"


def fdaLat(sExp, iCase):
    sText = subprocess.run(["git", "-C", sRepo, "show", f"bedeec3:output/poise/{sExp}/case_{iCase}/lat_output.dat"],
                           capture_output=True, text=True, check=True).stdout
    return np.loadtxt(io.StringIO(sText), comments="#")


def fsClassify(daA, dFland):
    daAllowed = np.array([dFland * l + (1 - dFland) * o for l in (0.3, 0.6) for o in (0.2, 0.6)])
    bDiscrete = np.all(np.min(np.abs(daA[:, None] - daAllowed[None, :]), axis=1) < 1e-6)
    return "threshold-reconstructed" if bDiscrete else "continuous (model output)"


for sExp, listCases in (("exp1", [0, 75, 150]), ("exp1a", [80]), ("exp2", [100]), ("exp2a", [60]),
                        ("exp3_warm", [5, 10]), ("exp3_cold", [5, 25]), ("exp4_warm", [10, 24]), ("exp4_cold", [44])):
    for iCase in listCases:
        daLat = fdaLat(sExp, iCase)
        daA = daLat[:, 2]
        print(f"{sExp} case {iCase}: Asurf n_unique={len(np.unique(np.round(daA,6)))} "
              f"mean={daA.mean():.4f} source={fsClassify(daA, 0.25)}; "
              f"OLR - (204.05+2.09(T-273.15)) max|res|={np.abs(daLat[:,4]-(204.05+2.09*(daLat[:,1]-273.15))).max():.3f}")
