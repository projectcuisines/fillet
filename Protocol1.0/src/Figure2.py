import numpy as np
import matplotlib.pylab as plt

font = {'family' : 'serif', 'size'   : 17}
plt.rc('font', **font)

# INPUT file & path
zonalquantEBM1='./Ben1_ESTM/Bench1_Tuned/ZonalData.txt'
zonalquantEBM2='./Ben2_ESTM/Bench2_Tuned/ZonalData.txt'
zonalquantEBM3='./Ben3_ESTM/Bench3_Tuned/ZonalData.txt'
zonalquantEBM4='./Ben1_ESTM/Bench1_Paper/ZonalData.txt'
zonalquantEBM5='./Ben2_ESTM/Bench2_Paper/ZonalData.txt'
zonalquantEBM6='./Ben3_ESTM/Bench3_Paper/ZonalData.txt'


# loads yearly-averaged zonal data at last orbit
bench1=np.loadtxt(zonalquantEBM1)
bench2=np.loadtxt(zonalquantEBM2)
bench3=np.loadtxt(zonalquantEBM3)
bench4=np.loadtxt(zonalquantEBM4)
bench5=np.loadtxt(zonalquantEBM5)
bench6=np.loadtxt(zonalquantEBM6)

#############   PLOT ZONAL TEMPERATURE

plt.plot(bench1[:,0],bench1[:,1],color='black',linewidth=3,label='Ben1')
plt.plot(bench2[:,0],bench2[:,1],color='red',linewidth=3,label='Ben2')
plt.plot(bench3[:,0],bench3[:,1],color='goldenrod',linewidth=3,label='Ben3')
plt.plot(bench4[:,0],bench4[:,1],color='black',linewidth=3,linestyle='dashed',label='Ben1 un-tuned')
plt.plot(bench5[:,0],bench5[:,1],color='red',linewidth=3,linestyle='dashed',label='Ben2 un-tuned')
plt.plot(bench6[:,0],bench6[:,1],color='goldenrod',linewidth=3,linestyle='dashed',label='Ben3 un-tuned')

plt.xlabel('Latitude [deg]')
plt.ylabel('Temperature [K]')
#plt.grid(visible=True)
plt.xlim(-90,90)
plt.xticks([-90.,-60.,-30.,0.,30.,60.,90.])
plt.ylim(190,310)
plt.legend(loc='lower center',prop={'size': 12}, ncol=2)
plt.subplots_adjust(left=0.15,right=0.97,top=0.97,bottom=0.13)
#plt.show()
plt.savefig('fig2a.pdf')
plt.close()

#############   PLOT ZONAL ALBEDO

plt.plot(bench1[:,0],bench1[:,5],color='black',linewidth=3,label='Ben1')
plt.plot(bench2[:,0],bench2[:,5],color='red',linewidth=3,label='Ben2')
plt.plot(bench3[:,0],bench3[:,5],color='goldenrod',linewidth=3,label='Ben3')
plt.plot(bench4[:,0],bench4[:,5],color='black',linewidth=3,linestyle='dashed')
plt.plot(bench5[:,0],bench5[:,5],color='red',linewidth=3,linestyle='dashed')
plt.plot(bench6[:,0],bench6[:,5],color='goldenrod',linewidth=3,linestyle='dashed')
plt.xlabel('Latitude [deg]')
plt.ylabel('TOA Albedo')
#plt.grid(visible=True)
plt.xlim(-90,90)
plt.xticks([-90.,-60.,-30.,0.,30.,60.,90.])
#plt.ylim(-0.03,1.03)
plt.ylim(0.2,0.8)
#plt.legend(loc='best')
plt.subplots_adjust(left=0.15,right=0.97,top=0.97,bottom=0.13)
#plt.show()
plt.savefig('fig2b.pdf')
plt.close()

#############   PLOT MEAN ANNUAL ICE COVERAGE

# plt.plot(bench1[:,0],bench1[:,6],color='black',linewidth=3,label='Ben1')
# plt.plot(bench2[:,0],bench2[:,6],color='red',linewidth=3,label='Ben2')
# plt.plot(bench3[:,0],bench3[:,6],color='goldenrod',linewidth=3,label='Ben3')
# plt.plot(bench4[:,0],bench4[:,6],color='black',linewidth=3,linestyle='dashed')
# plt.plot(bench5[:,0],bench5[:,6],color='red',linewidth=3,linestyle='dashed')
# plt.plot(bench6[:,0],bench6[:,6],color='goldenrod',linewidth=3,linestyle='dashed')
# #plt.grid(visible=True)
# plt.xlim([-90,90])
# plt.xticks([-90,-60,-30,0,30,60,90])
# plt.xlabel('Latitude [deg]')
# plt.ylim([-0.03,1.03])
# plt.ylabel("Ice Fraction")
# #plt.legend(loc='best')
# plt.subplots_adjust(left=0.15,right=0.97,top=0.97,bottom=0.13)
# plt.show()
