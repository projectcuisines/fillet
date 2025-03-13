import numpy as np
import matplotlib.pyplot as plt
import vplot as vpl
import vplanet
import pathlib
from astropy import units as u

sims = ['Ben1_POISE','Ben2_POISE']
lines = ['-','--']
lolab = ['Average (Ben 1)','Average (Ben 2)']
lalab = ['Land (Ben 1)','Land (Ben 2)']
oclab = ['Ocean (Ben 1)','Ocean (Ben 2)']
fig, (ax1, ax2) = plt.subplots(ncols=2,nrows=1,figsize=(7.5,3))

for i in np.arange(len(sims)):
    path = pathlib.Path(sims[i])
    out = vplanet.run(path / "vpl.in")

    lat = np.unique(out.Earth.Latitude)
    nlats = len(lat)
    ntimes = len(out.Earth.Time)

    T = np.reshape(out.Earth.TempLat,(ntimes,nlats)).to(u.K,equivalencies=u.temperature())
    Tland = np.reshape(out.Earth.TempLandLat,(ntimes,nlats)).to(u.K,equivalencies=u.temperature())
    Toc = np.reshape(out.Earth.TempWaterLat,(ntimes,nlats)).to(u.K,equivalencies=u.temperature())

    A = np.reshape(out.Earth.AlbedoLat,(ntimes,nlats))
    Aland = np.reshape(out.Earth.AlbedoLandLat,(ntimes,nlats))
    Aoc = np.reshape(out.Earth.AlbedoWaterLat,(ntimes,nlats))

    ax1.plot(lat,Tland[1],'-',color=vpl.colors.orange,label=lalab[i],linestyle=lines[i])
    ax1.plot(lat,Toc[1],'-',color=vpl.colors.pale_blue,label=oclab[i],linestyle=lines[i])
    ax1.plot(lat,T[1],'k-',label=lolab[i],linestyle=lines[i])

    ax2.plot(lat,Aland[1],'-',color=vpl.colors.orange,label=lalab[i],linestyle=lines[i])
    ax2.plot(lat,Aoc[1],'-',color=vpl.colors.pale_blue,label=oclab[i],linestyle=lines[i])
    ax2.plot(lat,A[1],'k-',label=lolab[i],linestyle=lines[i])

#ax1.vlines([-58,58],255,320,linestyles=':',colors='0.5')
#ax1.hlines([271.15],-90,90,linestyles=':',colors='0.5')
#ax2.vlines([-58,58],0.17,0.65,linestyles=':',colors='0.5')

ax1.set(ylabel='Temperature [K]')
ax1.set_xticks([-90,-60,-30,0,30,60,90])
ax1.legend(loc='lower center',fontsize=7)
ax2.set(ylabel='TOA Albedo')
ax2.set_xticks([-90,-60,-30,0,30,60,90])

plt.tight_layout()
plt.savefig('fig1.pdf')
plt.close()
