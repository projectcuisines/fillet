import numpy as np
import matplotlib.pyplot as plt
import pathlib
from astropy import units as u
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import pdb

model_dirs = ['poise','ops','estm_noclouds','hextor']
labels = ['VPLanet','OPS','ESTM (no cloud)','HEXTOR']
outfile='FILLET_Protocol2_Experiment2.pdf'

glob_output = 'exp2/global_output.dat'

fig, axes = plt.subplots(ncols=2,nrows=len(model_dirs),figsize=(7.5,3*len(model_dirs)))

for imod in np.arange(len(model_dirs)):
    globfile = pathlib.Path(model_dirs[imod]) / glob_output
    if not globfile.exists():
        print(str(globfile) + ' is missing')
    else:
        if 'poise' in model_dirs[imod]:
            case, inst, obl, XCO2, Tglob, IceLandNMax, IceLandNMin, IceLandSMax, \
                IceLandSMin, IceOceanNMax, IceOceanNMin, IceOceanSMax, \
                IceOceanSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)
            IceLineNMax = IceLandNMax 
            IceLineNMin = IceLandNMin 
            IceLineSMax = IceLandSMax
            IceLineSMin = IceLandSMin
        if 'ops' in model_dirs[imod]:
            case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)
        if 'estm' in model_dirs[imod]:
            case, inst, obl, XCO2, Tglob, IceLineNMin, IceLineSMax, fice, fclo, atoa, dtep, diff = np.loadtxt(str(globfile),comments='#',unpack=True)
            IceLineNMax = np.zeros_like(case) + 90.
            IceLineSMin = np.zeros_like(case) + -90.0
        if 'hextor' in model_dirs[imod]:
            case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin, Diff, OLR = np.loadtxt(str(globfile),comments='#',unpack=True)


    inst_plot = np.unique(inst)
    obl_plot = np.unique(obl)

    Tglob_plot = np.reshape(Tglob,(len(inst_plot),len(obl_plot)))
    if 'ops' in model_dirs[imod]:
        Tglob_plot = np.reshape(Tglob,(len(obl_plot),len(inst_plot))).T

    ax = axes[imod][0]
    c = ax.pcolormesh(obl_plot,inst_plot,Tglob_plot,cmap='plasma')
    ax.set(title='%s: Temperature (K)'%labels[imod],xlabel='Obliquity ($^{\circ}$)',ylabel='Solar constant (S$_{\oplus}$)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(c,cax = cax)

    Cap = np.zeros_like(Tglob)
    Belt = np.zeros_like(Tglob)
    Free = np.zeros_like(Tglob)
    Sball = np.zeros_like(Tglob)
    State = np.zeros_like(Tglob)
    for isim in case:
        isim = int(isim)
        if IceLineNMax[isim] == IceLineNMin[isim]:
            State[isim] = 0
        elif IceLineNMax[isim]  == 90 and IceLineNMin[isim] > 0:
            State[isim] = 1
        elif IceLineNMax[isim]  == 90 and IceLineNMin[isim] == 0:
            State[isim] = 3
        elif IceLineNMax[isim]  < 90 and IceLineNMin[isim] == 0:
            State[isim] = 2
        else:
            raise Exception("Logic is broken here at sim %d"%isim)

    State_plot = np.reshape(State,(len(inst_plot),len(obl_plot)))
    if 'ops' in model_dirs[imod]:
        State_plot = np.reshape(State,(len(obl_plot),len(inst_plot))).T

    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap',['blue','purple','cyan','white'],4)


    ax = axes[imod][1]
    c = ax.pcolormesh(obl_plot,inst_plot,State_plot,cmap=cmap)
    ax.set(title='%s: Climate State'%labels[imod],xlabel='Obliquity ($^{\circ}$)',ylabel='Solar constant (S$_{\oplus}$)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    clb = plt.colorbar(c,cax = cax)
    clb.set_ticks([0.75*0.5,0.5*(1.5+0.75),0.5*(1.5+2.25),0.5*(2.25+3)])
    clb.ax.set_yticklabels(['Ice free','Ice caps','Ice belt','Snowball'])

plt.tight_layout()
plt.savefig(outfile)
plt.close()
