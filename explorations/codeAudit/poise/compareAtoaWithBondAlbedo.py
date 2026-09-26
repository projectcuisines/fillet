"""Compare area-mean archived ATOA with the Bond albedo implied by OLRglob = (S/4)(1-A) for POISE benchmarks."""
import numpy as np

sArchive = "/workspace/fillet/Results/poise"
dS0 = 1361.0

for sExp in ("ben1", "ben2", "ben3"):
    daGlob = np.atleast_2d(np.loadtxt(f"{sArchive}/{sExp}/global_output.dat", comments="#"))
    daLat = np.loadtxt(f"{sArchive}/{sExp}/case_0/lat_output.dat", comments="#")
    dInst, dOlr = daGlob[0, 1], daGlob[0, -1]
    dBond = 1.0 - dOlr / (dS0 * dInst / 4.0)
    print(f"{sExp}: area-mean ATOA={daLat[:,3].mean():.4f}  area-mean Asurf={daLat[:,2].mean():.4f}  "
          f"Bond albedo implied by OLRglob={dBond:.4f}  (difference ATOA-Bond={daLat[:,3].mean()-dBond:+.4f})")
