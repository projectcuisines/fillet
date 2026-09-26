"""Test whether archived POISE benchmark ice lines are land-fraction-weighted blends of land/sea edges."""
import numpy as np

sArchive = "/workspace/fillet/Results/poise"
iNumLats = 150
daLatN = np.degrees(np.arcsin(-1 + (np.arange(iNumLats) + 0.5) * 2.0 / iNumLats))
daLatN = daLatN[daLatN >= 0]
daCandidates = np.concatenate([[0.0, 90.0], daLatN])

for sExp, dLandFrac in (("ben1", 0.34), ("ben2", 0.25), ("ben3", 0.25)):
    daGlob = np.atleast_2d(np.loadtxt(f"{sArchive}/{sExp}/global_output.dat", comments="#"))
    dTarget = daGlob[0, 6]  # IceLineNMin
    listHits = []
    for dLand in daCandidates:
        for dSea in daCandidates:
            if abs(dLandFrac * dLand + (1 - dLandFrac) * dSea - dTarget) < 5e-6:
                listHits.append((dLand, dSea))
    bOnGrid = np.any(np.isclose(daCandidates, dTarget, atol=5e-6))
    print(f"{sExp}: IceLineNMin={dTarget:.6f} on-grid={bOnGrid} "
          f"blend solutions (land, sea) with f_land={dLandFrac}: {[(round(a,4), round(b,4)) for a,b in listHits]}")
