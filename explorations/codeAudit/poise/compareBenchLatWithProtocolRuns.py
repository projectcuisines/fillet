"""Compare archived POISE benchmark lat_output.dat with final block of Protocol1.0 Ben*_POISE .Climate files."""
import numpy as np

sProto = "/workspace/fillet/Protocol1.0"
sArchive = "/workspace/fillet/Results/poise"
iNumLats = 150


def fnCompare(sBen, sArc):
    daC = np.loadtxt(f"{sProto}/{sBen}/solarsys.Earth.Climate")
    daFinal = daC[-iNumLats:]
    daLat = np.loadtxt(f"{sArchive}/{sArc}/case_0/lat_output.dat", comments="#")
    dT = np.abs(daFinal[:, 2] + 273.15 - daLat[:, 1]).max()
    dA = np.abs(daFinal[:, 3] - daLat[:, 3]).max()
    print(f"{sBen} vs {sArc}: final time={daFinal[0,0]}  max|dTsurf|={dT:.3e} K  max|dATOA|={dA:.3e}")
    daLandT, daWaterT = daFinal[:, 10], daFinal[:, 11]
    print(f"   annual-mean T_land range [{daLandT.min():.2f},{daLandT.max():.2f}] C; "
          f"T_water range [{daWaterT.min():.2f},{daWaterT.max():.2f}] C; "
          f"max land-water difference {np.abs(daLandT-daWaterT).max():.2f} K")


fnCompare("Ben2_POISE", "ben2")
fnCompare("Ben1_POISE", "ben1")
