"""List archived POISE rows whose land/sea ice edges are not N/S mirror images, with neighbors."""
import numpy as np

sArchive = "/workspace/fillet/Results/poise"
listExps = ["exp1", "exp1a", "exp2", "exp2a", "exp4_warm", "exp4_cold"]
# written order (proc_poise_out.py): Land N max/min, Land S max/min, Sea N max/min, Sea S max/min


def fnReport(sExp):
    daData = np.loadtxt(f"{sArchive}/{sExp}/global_output.dat", comments="#")
    for iOff, sSurf in ((5, "Land"), (9, "Sea")):
        daNMax, daNMin, daSMax, daSMin = (daData[:, iOff + k] for k in range(4))
        baBad = ~np.isclose(daNMin, -daSMax) | ~np.isclose(daNMax, -daSMin)
        for iRow in np.where(baBad)[0]:
            print(f"{sExp} {sSurf} asymmetric case {int(daData[iRow,0])}:")
            for jRow in range(max(0, iRow - 1), min(len(daData), iRow + 2)):
                print("   ", " ".join(f"{v:.3f}" for v in daData[jRow, :13]))


for sExp in listExps:
    fnReport(sExp)
