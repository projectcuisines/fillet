import numpy as np
import matplotlib.pyplot as plt
import pathlib
from astropy import units as u
import pdb

model_dirs = ['ops','poise', 'hextor','estm_noclouds', 'estm']
labels = ['OPS','VPLanet', 'HEXTOR','ESTMnc', 'ESTM']
colors = ['b', 'r', 'c','orange', 'magenta']
outfile='FILLET_Protocol2_Benchmark3.pdf'

lat_output = 'ben3/case_0/lat_output.dat'
glob_output = 'ben3/global_output.dat'

fig, axes = plt.subplots(ncols=2,nrows=2,figsize=(7.5,6))
ylims = np.array([[270,313],
         [0.2,0.7],
         [230,280],
         [0.2,0.51]])

for imod in np.arange(len(model_dirs)):

    latfile = pathlib.Path(model_dirs[imod]) / lat_output
    if not latfile.exists():
        print(str(latfile) + ' is missing')
    else:
        if 'estm' in model_dirs[imod]:
          lat, Tsurf, Asurf, ATOA, OLR, fice, fclo, diff = np.loadtxt(str(latfile),comments='#',unpack=True)
        else:
          lat, Tsurf, Asurf, ATOA, OLR = np.loadtxt(str(latfile),comments='#',unpack=True)

    axes[0][0].plot(lat,Tsurf,c=colors[imod],label=labels[imod],lw=2,zorder=1000)
    axes[0][1].plot(lat,Asurf,c=colors[imod],lw=2,zorder=1000)
    axes[1][0].plot(lat,OLR,c=colors[imod],lw=2,zorder=1000)
    axes[1][1].plot(lat,ATOA,c=colors[imod],lw=2,zorder=1000)

axes[0][0].set(ylabel='Surface Temperature (K)',ylim=ylims[0])
axes[0][0].xaxis.set_ticks([-90,-60,-30,0,30,60,90])
axes[0][0].legend(loc='best')

axes[0][1].set(ylabel='Surface Albedo',ylim=ylims[1])
axes[0][1].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

axes[1][0].set(xlabel='Latitude (deg)',ylabel='OLR (W m$^{-2}$)',ylim=ylims[2])
axes[1][0].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

axes[1][1].set(xlabel='Latitude (deg)',ylabel='Total Albedo',ylim=ylims[3])
axes[1][1].xaxis.set_ticks([-90,-60,-30,0,30,60,90])

plt.tight_layout()
plt.savefig(outfile)
plt.close()
