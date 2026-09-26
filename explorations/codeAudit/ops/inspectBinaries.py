"""Extract identifying printable strings (compiler, file names, headers) from OPS binaries."""
import re, sys, hashlib
listPatterns = [rb"Experiment [0-9][^\x00']{0,20}", rb"filletout\.out", rb"PLANETARY ALBEDO\(EQUATOR\)",
    rb"AVERAGE SURFACE ALBEDO", rb"GCC: \([^)]*\)[^\x00]{0,40}", rb"Intel\(R\)[^\x00]{0,80}",
    rb"data/OLR[^\x00 ]{0,30}", rb"data/PALB[^\x00 ]{0,30}", rb"ifort[^\x00]{0,40}", rb"driver\.f90",
    rb"lat_output[^\x00]{0,20}", rb"model\.out", rb"total_ebm[^\x00]{0,10}", rb"Version [0-9][^\x00]{0,40}"]
for sPath in sys.argv[1:]:
    baData = open(sPath, "rb").read()
    print("==", sPath, len(baData), hashlib.md5(baData).hexdigest()[:10])
    for bPat in listPatterns:
        setHits = sorted(set(re.findall(bPat, baData)))
        for bHit in setHits[:6]:
            print("   ", bHit.decode("latin-1").strip())
