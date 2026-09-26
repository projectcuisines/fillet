"""Compare a newly supplied set of FILLET output files against the copies already in the Results archive, field by field.

A participant re-sending results raises one question before any science is done
with them: are these the files the archive already holds, or different ones?
The answer has to distinguish three cases, and a plain `diff` conflates them:

1. Byte-identical -- nothing to do.
2. Identical numbers, different formatting (whitespace, exponent style,
   trailing newline, header wording).  The archive is already correct and the
   resend is cosmetic.
3. Different numbers.  Then the archive is stale, and the size and location of
   the difference decides whether anything downstream changes.

So the comparison is made twice: once on the raw bytes, and once on the
parsed numeric table, with the maximum absolute and relative difference per
column reported for case 3.  Header lines are compared separately from data,
because a changed header is a metadata correction and a changed datum is not.
"""

import argparse
import hashlib
import json
import os

import numpy as np


def flistNumericRows(sPath):
    """The numeric rows of a FILLET output file, ignoring comments and blanks."""
    listRows = []
    for sLine in open(sPath, errors="replace"):
        sBody = sLine.strip()
        if not sBody or sBody.startswith("#"):
            continue
        try:
            listRows.append([float(t.replace("d", "e").replace("D", "e"))
                             for t in sBody.split()])
        except ValueError:
            continue
    iWidth = max((len(r) for r in listRows), default=0)
    return np.array([r for r in listRows if len(r) == iWidth], dtype=float)


def flistHeaderLines(sPath):
    """The comment lines, which carry the protocol metadata."""
    return [s.rstrip("\n") for s in open(sPath, errors="replace")
            if s.strip().startswith("#")]


def fsDigest(sPath):
    return hashlib.sha256(open(sPath, "rb").read()).hexdigest()[:16]


def fdictComparePair(sNew, sArchived):
    """One new file against its archived counterpart."""
    dictRow = {"sNewFile": sNew, "sArchivedFile": sArchived,
               "bArchivedExists": os.path.exists(sArchived)}
    if not dictRow["bArchivedExists"]:
        dictRow["sVerdict"] = "absent from the archive"
        return dictRow

    dictRow["bBytesIdentical"] = (fsDigest(sNew) == fsDigest(sArchived))
    aNew, aOld = flistNumericRows(sNew), flistNumericRows(sArchived)
    listHNew, listHOld = flistHeaderLines(sNew), flistHeaderLines(sArchived)
    dictRow["bHeadersIdentical"] = (listHNew == listHOld)
    dictRow["iHeaderLinesDiffering"] = int(sum(
        1 for a, b in zip(listHNew, listHOld) if a != b)
        + abs(len(listHNew) - len(listHOld)))
    dictRow["aShapeNew"] = list(aNew.shape)
    dictRow["aShapeArchived"] = list(aOld.shape)

    if aNew.shape != aOld.shape:
        dictRow["bNumbersIdentical"] = False
        dictRow["sVerdict"] = ("different shape: %s against %s"
                               % (aNew.shape, aOld.shape))
        return dictRow

    aDiff = np.abs(aNew - aOld)
    with np.errstate(divide="ignore", invalid="ignore"):
        aRel = np.where(np.abs(aOld) > 0, aDiff / np.abs(aOld), 0.0)
    dictRow["bNumbersIdentical"] = bool(np.nanmax(aDiff) == 0.0)
    dictRow["fMaxAbsDifference"] = float(np.nanmax(aDiff))
    dictRow["fMaxRelDifference"] = float(np.nanmax(aRel))
    dictRow["alistMaxAbsByColumn"] = [float(v) for v in np.nanmax(aDiff, axis=0)]
    dictRow["iRowsDiffering"] = int(np.count_nonzero(np.any(aDiff > 0, axis=1)))

    if dictRow["bBytesIdentical"]:
        dictRow["sVerdict"] = "byte-identical"
    elif dictRow["bNumbersIdentical"] and dictRow["bHeadersIdentical"]:
        dictRow["sVerdict"] = "same numbers and headers, formatting differs"
    elif dictRow["bNumbersIdentical"]:
        dictRow["sVerdict"] = "same numbers, header text differs"
    else:
        dictRow["sVerdict"] = ("numbers differ, max %.6g absolute"
                               % dictRow["fMaxAbsDifference"])
    return dictRow


def flistPairs(sNewRoot, sArchiveRoot):
    """Match each supplied file to its archived counterpart.

    The two trees name cases differently -- `case0` against `case_0` -- so the
    match is made on the benchmark directory and the case index rather than on
    the path as written.
    """
    listOut = []
    for sDir, _, listFiles in os.walk(sNewRoot):
        for sName in sorted(listFiles):
            if not sName.endswith(".dat"):
                continue
            sRel = os.path.relpath(os.path.join(sDir, sName), sNewRoot)
            listParts = sRel.split(os.sep)
            listNorm = [p.replace("case", "case_") if p.startswith("case")
                        and not p.startswith("case_") else p for p in listParts]
            listOut.append((os.path.join(sNewRoot, sRel),
                            os.path.join(sArchiveRoot, *listNorm)))
    return sorted(listOut)


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--new-dir", required=True)
    oParser.add_argument("--archive-dir", required=True)
    oParser.add_argument("--label", default="submission")
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    listPairs = flistPairs(oArgs.new_dir, oArgs.archive_dir)
    listRows = [fdictComparePair(a, b) for a, b in listPairs]

    iSame = sum(1 for d in listRows if d.get("bNumbersIdentical"))
    dictOut = {"sLabel": oArgs.label, "iFiles": len(listRows),
               "iNumericallyIdentical": iSame,
               "bArchiveUpToDate": bool(iSame == len(listRows) and listRows),
               "listComparisons": listRows}
    json.dump(dictOut, open(oArgs.out, "w"), indent=1)

    print("%s: %d files supplied, %d numerically identical to the archive"
          % (oArgs.label, len(listRows), iSame))
    for d in listRows:
        print("  %-34s %s"
              % (os.path.relpath(d["sNewFile"], oArgs.new_dir), d["sVerdict"]))
        if not d.get("bHeadersIdentical", True):
            print("  %-34s   (%d header line(s) differ)"
                  % ("", d["iHeaderLinesDiffering"]))
    print("\nverdict: %s" % ("the archive already holds these results"
                             if dictOut["bArchiveUpToDate"]
                             else "the archive differs from what was supplied"))
    print("wrote %s" % oArgs.out)


if __name__ == "__main__":
    main()
