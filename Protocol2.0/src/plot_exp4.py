import numpy as np
import matplotlib.pyplot as plt
import pathlib
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import pdb

model_dirs = ['poise','ops','estm_noclouds','hextor']
mod_labels = ['VPLanet','OPS','ESTM (no cloud)','HEXTOR']
outfile='FILLET_Protocol2_Experiment4.pdf'

labels = ['Cold start','Warm start']
colors = ['b','r']
colors_o = ['c','orange']

glob_output = ['exp4_cold/global_output.dat', 'exp4_warm/global_output.dat']



fig, axes = plt.subplots(ncols=1,nrows=len(model_dirs),figsize=(5,3*len(model_dirs)))

for imod in np.arange(len(model_dirs)):
    for istart in np.arange(len(glob_output)):
        globfile = pathlib.Path(model_dirs[imod]) / glob_output[istart]
        if not globfile.exists():
            print(str(globfile) + ' is missing')
        else:
            if 'poise' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLandNMax, IceLandNMin, IceLandSMax, \
                    IceLandSMin, IceOceanNMax, IceOceanNMin, IceOceanSMax, \
                    IceOceanSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)
                XCO2 *= 1e6
                IceLineNMax = IceLandNMax
                IceLineNMin = IceLandNMin
                IceLineSMax = IceLandSMax
                IceLineSMin = IceLandSMin
            if 'ops' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)
            if 'estm' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLineNMin, IceLineSMax, fice, fclo, atoa, dtep, Diff = np.loadtxt(str(globfile),comments='#',unpack=True)
                IceLineNMax = np.zeros_like(case) + 90.
                IceLineSMin = np.zeros_like(case) + -90.0
            if 'hextor' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)

            ax = axes[imod]
            ax.semilogx(XCO2/1e6,IceLineNMin,color=colors[istart],linestyle='-',marker='s',label=labels[istart],ms=2)
            if 'poise' in model_dirs[imod]:
                ax.plot(XCO2/1e6,IceOceanNMin,color=colors_o[istart],linestyle='-',marker='s',zorder=-1,lw=3,ms=3)
            ax.set(ylabel='Ice line latitude ($^{\circ}$)',xlabel='CO$_2$ partial pressure (bar)',title='%s'%mod_labels[imod])

    if imod == 0:
        ax.legend(loc='best')

plt.tight_layout()
plt.savefig(outfile)
plt.close()
