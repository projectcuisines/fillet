import numpy as np
import matplotlib.pyplot as plt
import vplanet
import vplot as vpl
import pathlib

parents = ['Exp3_POISE/GDwarf/WarmStart/Exp3_WarmStart','Exp3_POISE/GDwarf/ColdStart/Exp3_ColdStart']
lacol = [vpl.colors.red,vpl.colors.pale_blue]
lalab = ['Land, warm start','Land, cold start']
occol = [vpl.colors.orange,vpl.colors.dark_blue]
oclab = ['Ocean, warm start','Ocean, cold start']
L = np.linspace(3.06202617e26,5.74129904e26,29)

for isim in np.arange(len(parents)):
    parent = pathlib.Path(parents[isim])
    trial_name = 'instell_'
    trial_len = 29  #should parse files for this

    land_line = np.zeros(trial_len)
    ocean_line = np.zeros(trial_len)

    #should replace the following with a parse of input files
    a = 1.0000001123 * 149597870700
    S = L / (4*np.pi*a**2)
    S0 = 1361.0

    for i in np.arange(trial_len):
        path = parent / (trial_name + '%02d'%i)
        out = vplanet.run(path / 'vpl.in')

        lat = np.unique(out.earth.Latitude)
        nlats = len(lat)
        ntimes = len(out.earth.Time)

        T = np.reshape(out.earth.TempLat,(ntimes,nlats))
        Tland = np.reshape(out.earth.TempLandLat,(ntimes,nlats))
        Toc = np.reshape(out.earth.TempWaterLat,(ntimes,nlats))

        TlandN = Tland[1][lat>=0].value
        TocN = Toc[1][lat>=0].value
        latN = lat[lat>=0].value

        if (TlandN > 0).all():
            icelandN = 90.0
        elif (TlandN <= 0).all():
            icelandN = 0.0
        else:
            icelandN = np.min(latN[TlandN<=0])
        if (TocN > -2).all():
            iceocN = 90.0
        elif (TocN <= -2).all():
            iceocN = 0.0
        else:
            iceocN = np.min(latN[TocN<=-2])

        land_line[i] = icelandN
        ocean_line[i] = iceocN

    plt.plot(S/S0, land_line, '-', color=lacol[isim], marker='^',ms=10,label=lalab[isim])
    plt.plot(S/S0, ocean_line, '-', color=occol[isim], marker='s',label=oclab[isim])

plt.legend(loc='center')
plt.xlabel('$S/S_{\oplus}$')
plt.ylabel('Ice line latitude (deg)')
plt.savefig('fig5.pdf')
plt.close()
