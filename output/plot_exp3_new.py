import numpy as np
import matplotlib.pyplot as plt
import pathlib
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import pdb

model_dirs = ['poise','ops_new','estm_noclouds','hextor']
#colors = ['b', 'r', 'c','orange']
mod_labels = ['VPLanet/POISE','OPS','ESTM (no cloud)','HEXTOR']
#lat_output = 'ben1/case_0/lat_output.dat'
glob_output = ['exp3_cold/global_output.dat', 'exp3_warm/global_output.dat']
labels = ['Cold start','Warm start']
colors = ['b','r']
colors_o = ['c','orange']

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
                IceLineNMax = IceLandNMax    #temporary fix
                IceLineNMin = IceLandNMin    #Idea: maybe use hatching on plot to indicate mixed land/ocean states
                IceLineSMax = IceLandSMax
                IceLineSMin = IceLandSMin
            if 'ops' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)
            if 'estm' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLineNMin, IceLineSMax, fice, fclo, atoa, dtep, Diff = np.loadtxt(str(globfile),comments='#',unpack=True)
                # Case Inst Obl XCO2 Tglob IceLineN IceLineS fice fclo ATOA dTep diff
                IceLineNMax = np.zeros_like(case) + 90.   #temporary fix for old template format
                IceLineSMin = np.zeros_like(case) + -90.0
            if 'hextor' in model_dirs[imod]:
                case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)

            ax = axes[imod]
            ax.plot(inst,IceLineNMin,color=colors[istart],linestyle='-',marker='s',label=labels[istart],ms=2)
            if 'poise' in model_dirs[imod]:
                ax.plot(inst,IceOceanNMin,color=colors_o[istart],linestyle='-',marker='s',zorder=-1,lw=3,ms=3)
            ax.set(ylabel='Ice line latitude ($^{\circ}$)',xlabel='Solar constant (S$_{\oplus}$)',title='%s'%mod_labels[imod])

    if imod == 0:
        ax.legend(loc='best')

plt.tight_layout()
plt.savefig('Exp3_hyst.pdf')
plt.close()
