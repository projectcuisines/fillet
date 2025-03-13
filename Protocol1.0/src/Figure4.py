import os
import sys
import pandas
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#-------------------------------------------------------------------------------
# INPUT: path to data directory
#path1="./Experiment1_Tuned/"
#path2="./Experiment2_Tuned/"

# list of insolations
#insol_values1=np.arange(0.8,1.225,0.025)
#insol_values2=np.arange(1.0,1.425,0.025)

# list of obliquities
#obliq_values=np.arange(0.0,100.0,10.0)

# INPUT: path and file names
csvfile1="Exp1_ESTM/Experiment1_Tuned.csv"
csvfile2="Exp2_ESTM/Experiment2_Tuned.csv"

#-------------------------------------------------------------------------------
# FUNCTIONS

# this function identify the climate state of the planet (ice-free, snowball...)
def flagfunc(path,insol_values,obliq_values):
    # path to sub-directories
    pathglobal=path+"Global/"
    pathzonal=path+"Zonal/"
    # creates a list of insolations (dtype: str)
    insols=[]
    for i in insol_values:
        ins="%.3f" % i
        insols.append(ins)
    # creates a list of obliquities (dtype: str)
    obliqs=[]
    for o in obliq_values:
        obliqs.append(str(o))
    # fills the ice fraction (ices) and ice state (flags) arrays
    ices=np.zeros((len(insols),len(obliqs)))
    flags=np.zeros((len(insols),len(obliqs)))
    insolrange=range(0,len(insols))
    obliqrange=range(0,len(obliqs))
    for i in insolrange:
        for o in obliqrange:
            name_global="GlobalData_i"+insols[i]+"_ob"+obliqs[o]+".txt"
            ice=float(np.loadtxt(pathglobal+name_global,skiprows=1,usecols=(6)))
            ices[i,o]=ice
            if ice < 0.01:           #hothouse
                flags[i,o]=3
            elif ice > 0.99:         #snowball
                flags[i,o]=0
            else:                    #icehouse: checks whether it is ice caps or ice belt
                name_zonal="ZonalData_i"+insols[i]+"_ob"+obliqs[o]+".txt"
                runzonal=np.loadtxt(pathzonal+name_zonal,usecols=(0,6))
                polesidx=np.searchsorted(runzonal[:,0],-88.0)
                equatidx=np.searchsorted(runzonal[:,0],0.0)
                poles=runzonal[polesidx,1]         #south pole ice
                equat=runzonal[equatidx,1]         #equatorial ice
                if equat > poles:   #icebelt
                    flags[i,o]=2
                elif equat < poles: #icepoles
                    flags[i,o]=1
                else:                              #shoud not exist, but I check nonetheless
                    flags[i,o]=4
    return insols,obliqs,ices,flags

# consolidates various vectors in a flag array that can be read by pandas and seaborn
def tradfunc_integer(insols,obliqs,flags):
    insolrange=range(0,len(insols))
    obliqrange=range(0,len(obliqs))
    sbnflags=np.zeros((len(insols)*len(obliqs),3))
    for i in insolrange:
        for o in obliqrange:
            j=i*len(obliqrange)+o
            sbnflags[j,0]=float(insols[i])
            sbnflags[j,1]=float(obliqs[o])
            sbnflags[j,2]=int(flags[i,o])
    return sbnflags

# consolidates various vectors in a float array that can be read by pandas and seaborn
def tradfunc_float(insols,obliqs,flags):
    insolrange=range(0,len(insols))
    obliqrange=range(0,len(obliqs))
    sbnices=np.zeros((len(insols)*len(obliqs),3))
    for i in insolrange:
        for o in obliqrange:
            j=i*len(obliqrange)+o
            sbnices[j,0]=float(insols[i])
            sbnices[j,1]=float(obliqs[o])
            sbnices[j,2]=float(ices[i,o])
    return sbnices

# creates the flag lists
#insols1,obliqs1,ices1,flags1=flagfunc(path1,insol_values1,obliq_values)
#insols2,obliqs2,ices2,flags2=flagfunc(path2,insol_values2,obliq_values)

# rewrites the flag list in pandas/seaborn arrays
#sbnflags1=tradfunc_integer(insols1,obliqs1,flags1)
#sbnflags2=tradfunc_integer(insols2,obliqs2,flags2)

# creates pivot tables from flag arrays that can be read by seaborn
#pandaflags1=pandas.DataFrame(sbnflags1,columns=["insolation","obliquity","status"])
#pandaflags2=pandas.DataFrame(sbnflags2,columns=["insolation","obliquity","status"])

#pandaflags1=pandaflags1.pivot("insolation","obliquity","status")
#pandaflags2=pandaflags2.pivot("insolation","obliquity","status")

# stores the pivot tables as csv files
#pandaflags1.to_csv("Experiment1_Tuned.csv",index=False)
#pandaflags2.to_csv("Experiment2_Tuned.csv",index=False)

# reads the flag arrays from csv files
pandaflags1=pandas.read_csv(csvfile1,usecols=["insolation","obliquity","status"])
pandaflags2=pandas.read_csv(csvfile2,usecols=["insolation","obliquity","status"])

pandaflags1=pandaflags1.pivot("insolation","obliquity","status")
pandaflags2=pandaflags2.pivot("insolation","obliquity","status")

#-------------------------------------------------------------------------------
# PLOTS

font = {'family' : 'serif', 'size'   : 16}
plt.rc('font', **font)

# plots the two figs side by side and with the same width
fig, ax = plt.subplots(ncols=3,gridspec_kw=dict(width_ratios=[1,1,0.05]),figsize=(10,4.5))
# defines the 4 colors related to the 4 possible states
mycolors=((0.8,0.8,0.8,0.6),(0.6,0.1,0.8,0.6),(0.5,0.5,0.9,0.6),(0.0,0.0,0.9,0.6))
# defines the colorbar to be used
cmap=LinearSegmentedColormap.from_list("Bestemmie",mycolors,len(mycolors))
# yticks
yticks1=[0.8,0.9,1.0,1.1,1.2]
yticks2=[1.0,1.1,1.2,1.3,1.4]
yticks1=["0.8","","","","0.9","","","","1.0","","","","1.1","","","","1.2"]
yticks2=["1.0","","","","1.1","","","","1.2","","","","1.3","","","","1.4"]
# xticks
xticks=[0,10,20,30,40,50,60,70,80,90]
# sets font
sbn.set(font_scale=1.25, font='serif')
# creates the heatmap (finally...)
sbn.heatmap(pandaflags1,cmap=cmap,cbar=None,ax=ax[0],xticklabels=xticks,yticklabels=yticks1)
sbn.heatmap(pandaflags2,cmap=cmap,cbar=None,ax=ax[1],xticklabels=xticks,yticklabels=yticks2)
# places the colorbar
#colorbar = ax[1].collections[0].colorbar
colorbar=fig.colorbar(ax[1].collections[0],cax=ax[2])
# places the ticks on the colorbar
colorbar.set_ticks([0.375,1.125,1.875,2.625])
# labels for the ticks on the colorbar
colorbar.set_ticklabels(["SnBl","Caps","Belt","Free"])
# labels for the x and y axes of the plots
ax[0].set(ylabel=r'Instellation [S$_0$]')
ax[1].set(ylabel='')
ax[0].set(xlabel=r'Obliquity [deg]')
ax[1].set(xlabel=r'Obliquity [deg]')
#ax[0].set_yticks(yticks1)
ax[0].set_yticklabels(yticks1,rotation=0)
ax[0].set_xticklabels(xticks,rotation=90)
#ax[1].set_yticks(yticks2)
ax[1].set_yticklabels(yticks2,rotation=0)
ax[1].set_xticklabels(xticks,rotation=90)
# inverts axes since seaborn starts with inverted axes
ax[0].invert_yaxis()
ax[1].invert_yaxis()
# plt.show()
plt.tight_layout()
plt.savefig('fig4.pdf')
plt.close()
