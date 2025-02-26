import numpy as np
import matplotlib.pyplot as plt
import pathlib
from astropy import units as u
import pdb

model_dirs = ['ops_new','poise','hextor','estm']
colors = ['b', 'r', 'c','orange']
labels = ['OPS','POISE', 'HEXTOR','ESTM']
lat_output = 'ben1/case_0/lat_output.dat'
glob_output = 'ben1/global_output.dat'

fig, axes = plt.subplots(ncols=2,nrows=2,figsize=(7.5,6))
ylims = np.array([[245,310],
         [0.05,0.675],
         [160,310],
         [0.1,0.7]])

for imod in np.arange(len(model_dirs)):

    latfile = pathlib.Path(model_dirs[imod]) / lat_output
    if not latfile.exists():
        print(str(latfile) + ' is missing')
    else:
        if model_dirs[imod] == 'estm':
          lat, Tsurf, Asurf, ATOA, OLR, fice, fclo, diff = np.loadtxt(str(latfile),comments='#',unpack=True)
        else:
          lat, Tsurf, Asurf, ATOA, OLR = np.loadtxt(str(latfile),comments='#',unpack=True)

    #Stuff below was used to add vertical lines indicating ice lines
    # globfile = pathlib.Path(model_dirs[imod]) / glob_output
    # if not globfile.exists():
    #     print(str(globfile) + ' is missing')
    # else:
    #     if 'poise' in model_dirs[imod] or 'hextor' in model_dirs[imod]:
    #         case, inst, obl, XCO2, Tglob, IceLineNMax, IceLineNMin, IceLineSMax, IceLineSMin = \
    #                    np.loadtxt(str(globfile),comments='#',unpack=True)
    #     else:
    #         case, inst, obl, XCO2, Tglob, IceLineN, IceLineS = \
    #                    np.loadtxt(str(globfile),comments='#',unpack=True)
    #     if IceLineS > 0:
    #         IceLineS *= -1.0

    axes[0][0].plot(lat,Tsurf,c=colors[imod],label=labels[imod],lw=2,zorder=1000)
    #   This stuff was for plotting vertical lines and global mean T
#    axes[0][0].hlines(Tglob,-90,90,colors=colors[imod],linestyles=':')
#    axes[0][0].vlines([IceLineS,IceLineN],ylims[0][0],ylims[0][1],colors=colors[imod],linestyles='--')

    axes[0][1].plot(lat,Asurf,c=colors[imod],lw=2,zorder=1000)
#    axes[0][1].vlines([IceLineS,IceLineN],ylims[1][0],ylims[1][1],colors=colors[imod],linestyles='--')

    axes[1][0].plot(lat,OLR,c=colors[imod],lw=2,zorder=1000)
#    axes[1][0].vlines([IceLineS,IceLineN],ylims[2][0],ylims[2][1],colors=colors[imod],linestyles='--')

    axes[1][1].plot(lat,ATOA,c=colors[imod],lw=2,zorder=1000)
#    axes[1][1].vlines([IceLineS,IceLineN],ylims[3][0],ylims[3][1],colors=colors[imod],linestyles='--')

axes[0][0].set(ylabel='Surface Temperature (K)',ylim=ylims[0])
axes[0][0].legend(loc='best')
axes[0][0].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

axes[0][1].set(ylabel='Surface Albedo',ylim=ylims[1])
axes[0][1].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

axes[1][0].set(xlabel='Latitude (deg)',ylabel='OLR (W m$^{-2}$)',ylim=ylims[2])
axes[1][0].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

axes[1][1].set(xlabel='Latitude (deg)',ylabel='Total Albedo',ylim=ylims[3])
axes[1][1].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

plt.tight_layout()
plt.savefig('ben1_comp_lat.pdf')
plt.close()
