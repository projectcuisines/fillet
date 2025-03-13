import numpy as np
import matplotlib.pyplot as plt
import argparse
import polynomials as poly
#simple tool for calculating linear OLR coefficients and total albedo
#total albedo = surface + atmosphere (clouds, scattering)

# parse user input
parser = argparse.ArgumentParser()
parser.add_argument('-m','--method',nargs=1,default=['WK'],help='Method of calculation: options are "WK" (Williams & Kasting 1997), "HM" (Haqq-Misra et al. 2016), "KT" (Kadoya & Tajika 2019)')
parser.add_argument('-t','--temperature',nargs=1,default=[288.0],help='Surface temperature (K)')
parser.add_argument('-x','--XCO2',nargs=1,default=[280.0],help='CO2 mixing ratio (ppm)')
parser.add_argument('-a','--albedo',nargs=1,default=[0.3],help='Surface albedo')
parser.add_argument('-p','--plot',action='store_true',help='Make plot of linear OLR as function of T and compare to original')
args = parser.parse_args()

#check calculation method
if args.method[0] == 'WK':
    print('\nUsing Williams & Kasting 1997...')
    Tlimits = [190,380]
elif args.method[0] == 'HM':
    print('\nUsing Haqq-Misra et al 2016...')
    Tlimits = [150,350]
elif args.method[0] == 'KT':
    print('\nUsing Kadoya & Tajika 2019...')
    Tlimits = [150,350]
else:
    raise ValueError('Invalid calculation method')

### Main part of script

T = float(args.temperature[0])
pCO2 = float(args.XCO2[0])*1e-6  #assumes 1 bar atmosphere!
asurf = float(args.albedo[0])
print('  Input: T = %#.3f K, pCO2 = %#.3e bar, asurf = %#.3f'%(T,pCO2,asurf))

if T < Tlimits[0] or T > Tlimits[1]:
    print('\nWarning: temperature is outside limits of (%#.1f, %#.1f) K.'%(Tlimits[0],Tlimits[1]))
    print('         Resulting coefficients may be inaccurate.')

if args.method[0] == 'WK':
    olr = poly.olr_wk97(pCO2,T)
    B = poly.olr_wk97_dT(pCO2,T)
    atot = poly.alb_wk97(pCO2,np.array([T]),1,asurf)

elif args.method[0] == 'HM':
    olr = poly.olr_hm16(pCO2,T)
    B = poly.olr_hm16_dT(pCO2,T)
    atot = poly.alb_hm16(pCO2,np.array([T]),1,asurf)

elif args.method[0] == 'KT':
    olr = poly.olr_kt19(pCO2,T)
    B = poly.olr_kt19_dT(pCO2,T)
    atot = poly.alb_kt19(0.0,asurf)

A_K = olr - B*T
A_C = olr - B*(T-273.15)

print('\nOLR = %#.3f W m^-2'%olr)
print('\nLinear OLR coefficients:')
print('    A = %#.3f W m^-2 (if using T in K)'%A_K)
print('    A = %#.3f W m^-2 (if using T in deg C)'%A_C)
print('    B = %#.3f W m^-2 K^-1'%B)
print('\nAlbedo atm zenith angle = 0 deg:')
print('    atotal = %#.3f (with asurf = %#.3f)'%(atot,asurf))
print('    atotal is the albedo of surface + atmosphere')
if args.method[0] == 'KT':
   print('      *atmosphere albedo is not a function of pCO2 or T in KT model')
print('\n')


if args.plot:
    #temperature grid
    temps = np.linspace(150,400.,100)
    temps_lim = np.linspace(Tlimits[0],Tlimits[1],100)

    if args.method[0] == 'WK':
        olr_full = poly.olr_wk97(pCO2,temps)
        olr_limit = poly.olr_wk97(pCO2,temps_lim)
    elif args.method[0] == 'HM':
        olr_full = poly.olr_hm16(pCO2,temps)
        olr_limit = poly.olr_hm16(pCO2,temps_lim)
    elif args.method[0] == 'KT':
        olr_full = poly.olr_kt19(pCO2,temps)
        olr_limit = poly.olr_kt19(pCO2,temps_lim)

    plt.figure(figsize=(7.5,4))
    plt.subplot(1,2,1)
    plt.plot(temps,olr_full,'k--')
    plt.plot(temps_lim,olr_limit,'k-')
    plt.plot(np.array(Tlimits),[olr_limit[0],olr_limit[-1]],'k|')

    plt.plot(temps,A_K+B*temps,'-',color='orangered')
    plt.plot(T,A_K+B*T,'*',color='orangered')
    plt.title('$p_{CO2}$ = %#.3e bar'%(pCO2))

    plt.ylabel(r'OLR (W/m$^2$)')
    plt.xlabel('Surface Temperature (K)')

    #pressure grid
    pressures = np.logspace(np.log10(50),np.log10(5050),100)*1e-6
    if args.method[0] == 'WK':
        olr_full = poly.olr_wk97(pressures,T)
    elif args.method[0] == 'HM':
        olr_full = poly.olr_hm16(pressures,T)
    elif args.method[0] == 'KT':
        olr_full = poly.olr_kt19(pressures,T)

    plt.subplot(1,2,2)
    plt.semilogx(pressures,olr_full,'k-')
    plt.plot(pCO2,A_K+B*T,'*',color='orangered')

    plt.title('$T$ = %#.3f K'%(T))

    plt.ylabel(r'OLR (W/m$^2$)')
    plt.xlabel('$p_{CO2}$ (bar)')

    plt.tight_layout()
    plt.show()
