"""Re-run the previous analysis's Shields-Bitz benchmark configurations and diagnose ice edges, albedo averaging, and convergence with the model's own ice criteria; optionally run protocol-constant variants."""
import argparse
import json
import os
import sys

for sVar in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(sVar, "1")
import numpy as np

sys.path.insert(0, "/workspace/shieldsEBM")
import EBM_one_file as oEbm  # noqa: E402

SEC_PER_YR = 3.15576e7
PREVIOUS = {"ben1": {"land": "modern", "obl": 23.5, "scaleQ": 1.0},
            "ben2": {"land": "fillet", "obl": 23.5, "scaleQ": 1.0},
            "ben3": {"land": "fillet", "obl": 60.0, "scaleQ": 1.0}}
# scaleQ is square-rooted because EBM_one_file.py applies it twice (lines 406 and 546).
PROTOCOL = {"scaleQ": np.sqrt(1361.0 / (2808 / 2.0754)), "hadleyflag": 0.0, "Dmag": 0.5,
            "Cw": 4e8 / SEC_PER_YR, "Cl": 1e7 / SEC_PER_YR, "land": "fillet"}


def fnPatchProtocolAlbedo():
    """In-memory only: constant 0.3/0.2/0.6 albedos, no zenith term (source untouched)."""
    def albedo_protocol(L, W, x, A_o, A_l, A_50):
        aL = np.where(L <= -2, 0.6, 0.3)
        aW = np.where(W <= -2, 0.6, 0.2)
        return aL, aW
    oEbm.albedo_seasonal = albedo_protocol


def fdEdges(aLat, aMask):
    """Poleward/equatorward ice edges per hemisphere from a per-cell ice mask (cell centres)."""
    dictOut = {}
    for sHem, aSel in (("N", aLat > 0), ("S", aLat < 0)):
        aL, aM = aLat[aSel], aMask[aSel]
        if not aM.any():
            dictOut[sHem] = None
            continue
        aIce = np.abs(aL[aM])
        dictOut[sHem] = (float(np.min(aIce)), float(np.max(aIce)), int(aM.sum()))
    return dictOut


def fdCrossing(aLat, aT, fThr):
    aN = aLat > 0
    aL, aTT = aLat[aN], aT[aN]
    aC = np.nonzero((aTT[:-1] - fThr) * (aTT[1:] - fThr) < 0)[0]
    if not len(aC):
        return 90.0 if np.all(aTT > fThr) else 0.0
    i = int(aC[-1])
    return float(aL[i] + (fThr - aTT[i]) / (aTT[i + 1] - aTT[i]) * (aL[i + 1] - aL[i]))


def fdictDiagnose(dictRes):
    dictS = dictRes["setup"]
    aLat, fl, fw = dictS["phi"], dictS["fl"], dictS["fw"]
    aH = dictRes["h_out"][:, -360:]
    aW = dictRes["W_out"][:, -360:]
    aL = dictRes["L_out"][:, -360:]
    aSeaIce = aH > 0.001
    aLandSnow = aL <= -2.0
    dictMean = oEbm.final_year_annual_means(dictRes)
    aT = dictMean["T_avg"]
    iSteps = 360
    aTprev = fl * dictRes["L_out"][:, -2 * iSteps:-iSteps].mean(1) + fw * dictRes["W_out"][:, -2 * iSteps:-iSteps].mean(1)
    aTyr10 = fl * dictRes["L_out"][:, -11 * iSteps:-10 * iSteps].mean(1) + fw * dictRes["W_out"][:, -11 * iSteps:-10 * iSteps].mean(1)
    aQ = dictS["insol"]
    aDay = dictRes["thedays"]
    aQfy = aQ[:, aDay - 1]
    aAlbL, aAlbW = dictRes["alb_l_ann"], dictRes["alb_w_ann"]
    aAlbCell = fl[:, None] * aAlbL + fw[:, None] * aAlbW
    aRefl = np.sum(aAlbCell * aQfy, axis=1)
    aInc = np.sum(aQfy, axis=1)
    aAlbWeighted = np.where(aInc > 0, aRefl / np.maximum(aInc, 1e-12), np.nan)
    iIceDays = int(aSeaIce.sum())
    iIceAtMinus2 = int((aSeaIce & (aW > -2.013)).sum())
    return {
        "Tglob": float(dictMean["Tglob"]),
        "TglobPrevOrbit": float(np.mean(aTprev) + 273.15),
        "TglobTenOrbitsEarlier": float(np.mean(aTyr10) + 273.15),
        "seaIceEver": fdEdges(aLat, aSeaIce.any(1)),
        "seaIceHalfYear": fdEdges(aLat, aSeaIce.mean(1) >= 0.5),
        "seaIcePerennial": fdEdges(aLat, aSeaIce.all(1)),
        "landSnowEver": fdEdges(aLat, aLandSnow.any(1)),
        "landSnowHalfYear": fdEdges(aLat, aLandSnow.mean(1) >= 0.5),
        "landSnowPerennial": fdEdges(aLat, aLandSnow.all(1)),
        "crossTavg263": fdCrossing(aLat, aT + 273.15, 263.15),
        "crossTavgMinus2C": fdCrossing(aLat, aT, -2.0),
        "crossToceanMinus2C": fdCrossing(aLat, dictMean["T_ocean"], -2.0),
        "crossTlandMinus2C": fdCrossing(aLat, dictMean["T_land"], -2.0),
        "meanIcelinePackage": float(oEbm.mean_iceline(dictRes)),
        "fracIceCellDaysWithWabove_2p013": (iIceAtMinus2 / iIceDays) if iIceDays else None,
        "albedoGlobalUnweighted": float(np.mean(dictMean["A_avg"])),
        "albedoGlobalInsolWeighted": float(np.sum(aRefl) / np.sum(aInc)),
        "albedoPolarBoxUnweighted": float(dictMean["A_avg"][-1]),
        "albedoPolarBoxInsolWeighted": float(aAlbWeighted[-1]),
        "maxAbsAlbedoDiffPerLat": float(np.nanmax(np.abs(aAlbWeighted - dictMean["A_avg"]))),
        "OLRglob": float(np.mean(dictS["A"] + dictS["B"] * aT)),
        "ASRglob": float(np.mean(np.sum(fl[:, None] * (1 - aAlbL) * aQfy + fw[:, None] * (1 - aAlbW) * aQfy, axis=1) / 360)),
        "Sused": float(4 * np.mean(aQ)),
    }


def main():
    oP = argparse.ArgumentParser(description=__doc__)
    oP.add_argument("--mode", choices=["previous", "protocolCfg", "protocolAlbedo"], default="previous")
    oP.add_argument("--benchmarks", default="ben1,ben2,ben3")
    oP.add_argument("--runlength", type=float, default=100.0)
    oP.add_argument("--jmx", type=int, default=120)
    oP.add_argument("--out", required=True)
    oA = oP.parse_args()
    if oA.mode == "protocolAlbedo":
        fnPatchProtocolAlbedo()
    dictAll = {}
    for sB in oA.benchmarks.split(","):
        dictCfg = dict(oEbm.DEFAULTS, **PREVIOUS[sB])
        if oA.mode != "previous":
            dictCfg.update(PROTOCOL)
            if sB == "ben1":
                dictCfg["land"] = "modern"
        dictCfg.update(runlength=oA.runlength, jmx=oA.jmx, casename=sB)
        dictAll[sB] = {"cfg": dictCfg, "diag": fdictDiagnose(oEbm.seasonal_run(dictCfg))}
        print(sB, json.dumps(dictAll[sB]["diag"], indent=1), flush=True)
    json.dump(dictAll, open(oA.out, "w"), indent=1)


if __name__ == "__main__":
    main()
