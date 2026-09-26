"""Read POISE Exp4/Exp3 lat outputs from FILLET git history (commit bedeec3) and report cold-cell and grid facts."""
import io
import subprocess
import numpy as np

sRepo = "/workspace/fillet"
sCommit = "bedeec3"


def fdaLatFromGit(sExp, iCase):
    sText = subprocess.run(["git", "-C", sRepo, "show", f"{sCommit}:output/poise/{sExp}/case_{iCase}/lat_output.dat"],
                           capture_output=True, text=True, check=True).stdout
    return np.loadtxt(io.StringIO(sText), comments="#")


def fnScan(sExp, iNumCases):
    for iCase in range(iNumCases):
        daLat = fdaLatFromGit(sExp, iCase)
        iCold = np.sum(daLat[:, 1] < 190.0)
        if iCase in (0, 5, 10) or iCold > 0:
            print(f"  {sExp} case {iCase}: nlat={len(daLat)} minTsurf={daLat[:,1].min():.2f} K "
                  f"cells<190K={iCold} ATOA range=[{daLat[:,3].min():.3f},{daLat[:,3].max():.3f}] "
                  f"Asurf range=[{daLat[:,2].min():.3f},{daLat[:,2].max():.3f}]")


for sExp, iN in (("exp4_cold", 51), ("exp4_warm", 12)):
    fnScan(sExp, iN)
daLat = fdaLatFromGit("exp3_warm", 0)
print(f"exp3_warm case 0: nlat={len(daLat)} lat[0]={daLat[0,0]:.4f} has equator cell={np.any(np.isclose(daLat[:,0],0))}")
daLat = fdaLatFromGit("exp4_warm", 24)
print(f"exp4_warm case 24 (~2.5e-4 bar): Tsurf eq={daLat[len(daLat)//2,1]:.2f} K pole={daLat[0,1]:.2f} K, "
      f"mean ATOA={daLat[:,3].mean():.3f} mean Asurf={daLat[:,2].mean():.3f}, OLR max={daLat[:,4].max():.2f}")
