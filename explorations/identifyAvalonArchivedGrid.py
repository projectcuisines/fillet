"""Identify which latitude grid produced the archived AVALON FILLET profiles and test it against the grid the current avalon.jl source generates."""

import argparse
import json

import numpy as np


def faCandidateGrids(iN):
    """Candidate grids an EBM of this family might use, as sin(lat) node arrays."""
    dictC = {}
    # Current avalon.jl make_grid(): equal-area cell centres, no node at a pole.
    dictC["cellCentredEqualArea_n%d" % iN] = np.linspace(-1 + 1.0 / iN, 1 - 1.0 / iN, iN)
    # Equal-area cell EDGES of an (n-1)-cell grid: nodes at both poles.
    dictC["cellEdgesEqualArea_n%d" % iN] = np.linspace(-1.0, 1.0, iN)
    # Equal-width in latitude, nodes at the poles.
    dictC["equalWidthLatitude_n%d" % iN] = np.sin(np.deg2rad(np.linspace(-90.0, 90.0, iN)))
    # Gauss-Legendre nodes in sin(lat) (a common spectral choice).
    aGL, _ = np.polynomial.legendre.leggauss(iN)
    dictC["gaussLegendre_n%d" % iN] = aGL
    # Equal-area cell centres with the two outermost nodes snapped to the poles.
    aSnap = np.linspace(-1 + 1.0 / iN, 1 - 1.0 / iN, iN).copy()
    aSnap[0], aSnap[-1] = -1.0, 1.0
    dictC["cellCentredPolesSnapped_n%d" % iN] = aSnap
    return dictC


def main():
    oParser = argparse.ArgumentParser(description=__doc__)
    oParser.add_argument("--lat-json", required=True)
    oParser.add_argument("--out", required=True)
    oArgs = oParser.parse_args()

    dictLat = json.load(open(oArgs.lat_json))
    aDat = np.array(dictLat["avalon/ben1"]["aaData"], dtype=float)
    aLat = aDat[:, 0]
    aX = np.sin(np.deg2rad(aLat))
    iN = len(aX)

    dictScore = {}
    for sName, aCand in faCandidateGrids(iN).items():
        # Latitudes are archived to 0.01 deg, so compare in degrees.
        aCandLat = np.rad2deg(np.arcsin(np.clip(aCand, -1, 1)))
        dictScore[sName] = float(np.max(np.abs(aCandLat - aLat)))

    sBest = min(dictScore, key=dictScore.get)

    # How near-uniform is the archived grid in sin(lat)?
    aDX = np.diff(aX)
    dictOut = {
        "iNumNodes": iN,
        "aArchivedLatFirst5": aLat[:5].tolist(),
        "aArchivedLatLast5": aLat[-5:].tolist(),
        "fDxMin": float(aDX.min()), "fDxMax": float(aDX.max()),
        "fDxMean": float(aDX.mean()),
        "fDxRelativeSpread": float(np.ptp(aDX) / aDX.mean()),
        "dictMaxAbsLatitudeMismatch_deg": dictScore,
        "sBestMatchingGrid": sBest,
        "fBestMismatch_deg": dictScore[sBest],
        "bCurrentSourceGridMatches":
            bool(dictScore["cellCentredEqualArea_n%d" % iN] < 0.01),
        "fCurrentSourceGridMismatch_deg":
            dictScore["cellCentredEqualArea_n%d" % iN],
    }

    with open(oArgs.out, "w") as f:
        json.dump(dictOut, f, indent=1)

    print("archived AVALON grid: %d nodes, lat[0..4] = %s"
          % (iN, np.round(aLat[:5], 2).tolist()))
    print("sin(lat) spacing: min %.5f max %.5f mean %.5f (relative spread %.3g)"
          % (aDX.min(), aDX.max(), aDX.mean(), np.ptp(aDX) / aDX.mean()))
    for sName in sorted(dictScore, key=dictScore.get):
        print("  %-34s max |dlat| = %8.3f deg" % (sName, dictScore[sName]))
    print("\ncurrent avalon.jl make_grid() matches archive: %s"
          % dictOut["bCurrentSourceGridMatches"])


if __name__ == "__main__":
    main()
