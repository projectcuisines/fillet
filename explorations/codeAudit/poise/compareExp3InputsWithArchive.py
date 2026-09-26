"""Compare final TGlobal/FluxOutGlobal of Protocol1.0/Exp3_POISE case runs with the archived exp3 global_output.dat."""
import numpy as np

sProto = "/workspace/fillet/Protocol1.0/Exp3_POISE"
sArchive = "/workspace/fillet/Results/poise"


def fdaFinalRows(sStart, sFolder, iNum):
    listRows = []
    for i in range(iNum):
        daF = np.loadtxt(f"{sProto}/{sStart}/{sFolder}/instell_{i:02d}/tilted.earth.forward")
        listRows.append(daF[-1])
    return np.array(listRows)


def fnCompare(sStart, sFolder, sExp):
    daArc = np.loadtxt(f"{sArchive}/{sExp}/global_output.dat", comments="#")
    daFin = fdaFinalRows(sStart, sFolder, len(daArc))
    daDT = (daFin[:, 2] + 273.15) - daArc[:, 4]
    daDO = daFin[:, 4] - daArc[:, -1]
    print(f"{sExp}: n={len(daArc)}  max|dTglob|={np.abs(daDT).max():.3e} K  "
          f"max|dOLRglob|={np.abs(daDO).max():.3e} W/m2  "
          f"final time(yr) unique={np.unique(daFin[:,0])}")
    iWorst = np.argmax(np.abs(daDT))
    print(f"   worst case {iWorst}: proto T={daFin[iWorst,2]+273.15:.6f} archive T={daArc[iWorst,4]:.6f}")


fnCompare("WarmStart", "Exp3_WarmStart", "exp3_warm")
fnCompare("ColdStart", "Exp3_ColdStart", "exp3_cold")


def fnTableWarm():
    """Print per-case warm-start Tglob from Protocol1.0 runs vs archive."""
    daArc = np.loadtxt(f"{sArchive}/exp3_warm/global_output.dat", comments="#")
    daFin = fdaFinalRows("WarmStart", "Exp3_WarmStart", len(daArc))
    print("case Inst  T_proto  T_archive  dT")
    for i in range(len(daArc)):
        dTp = daFin[i, 2] + 273.15
        print(f"{i:3d} {daArc[i,1]:.4f} {dTp:8.3f} {daArc[i,4]:8.3f} {dTp-daArc[i,4]:+8.3f}")


fnTableWarm()
