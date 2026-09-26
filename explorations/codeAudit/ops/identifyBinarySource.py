"""Search OPS object/binary files for float32 array constants that distinguish driver.f90 from the driver_fillet*.f90 variants."""
import struct, sys
dictSig = {
    "driver.f90 warm S array (0.95,0.975,1.0,...)": [0.95, 0.975, 1.0, 1.025, 1.05],
    "fillet1_2 warm S array (0.8,0.825,...)": [0.8, 0.825, 0.85, 0.875, 0.9],
    "fillet1_2 cold S array (0.9,0.925,0.95,...)": [0.9, 0.925, 0.95, 0.975, 1.0],
    "driver.f90 cold S array (0.95..1.151..)": [1.1, 1.125, 1.151, 1.175],
    "fillet3 S array (0.8,0.8125,...)": [0.8, 0.8125, 0.825, 0.8375],
    "fillet4 pCO2 array (1e-6,1.2589e-6,...)": [1.0e-06, 1.25892541e-06, 1.58489319e-06],
    "obliquity array (0,10,...,90)": [0.0, 10.0, 20.0, 30.0],
}
for sPath in sys.argv[1:]:
    baData = open(sPath, "rb").read()
    print("==", sPath)
    for sName, listV in dictSig.items():
        for sFmt in ("<%df", "<%dd"):
            baPat = struct.pack(sFmt % len(listV), *listV)
            iPos = baData.find(baPat)
            if iPos >= 0:
                print(f"   FOUND {sName} as {'float32' if 'f' in sFmt else 'float64'} at offset {iPos}")
