import numpy as np
import matplotlib.pyplot as plt
import argparse

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

###### Williams & Kasting 1997 functions ******************************************

def olr_wk97(pCO2, T):
  phi = np.log(pCO2/3.3e-4)
  # pdb.set_trace()
  Int = 9.468980 - 7.714727e-5*phi - 2.794778*T - 3.244753e-3*phi*T - 3.547406e-4*phi**2. + \
      2.212108e-2*T**2 + 2.229142e-3*phi**2*T + 3.088497e-5*phi*T**2 - 2.789815e-5*(phi*T)**2 \
      - 3.442973e-3*phi**3 - 3.361939e-5*T**3 + 9.173169e-3*phi**3*T - 7.775195e-5*phi**3*T**2\
      - 1.679112e-7*phi*T**3 + 6.590999e-8*phi**2*T**3 + 1.528125e-7*phi**3*T**3 \
      - 3.367567e-2*phi**4 - 1.631909e-4*phi**4*T + 3.663871e-6*phi**4*T**2 \
      - 9.255646e-9*phi**4*T**3
  return Int

def olr_wk97_dT(pCO2, T):
  phi = np.log(pCO2/3.3e-4)
  # pdb.set_trace()
  Int = - 2.794778 - 3.244753e-3*phi + \
      2.212108e-2*T*2 + 2.229142e-3*phi**2 + 3.088497e-5*phi*T*2 - 2.789815e-5*phi**2*T*2 \
      - 3.361939e-5*T**2*3 + 9.173169e-3*phi**3 - 7.775195e-5*phi**3*T*2\
      - 1.679112e-7*phi*T**2*3 + 6.590999e-8*phi**2*T**2*3 + 1.528125e-7*phi**3*T**2*3 \
      - 1.631909e-4*phi**4 + 3.663871e-6*phi**4*T*2 \
      - 9.255646e-9*phi**4*T**2*3
  return Int

def alb_wk97(p, T, mu, asurf):
    A = np.zeros_like(T)
    A[T<280] = -6.8910e-1 + 1.0460*asurf + 7.8054e-3*T[T<280] - 2.8373e-3*p - 2.8899e-1 * mu - 3.7412e-2*asurf*p \
        - 6.3499e-3*mu*p + 2.0122e-1*asurf*mu -1.8508e-3*asurf*T[T<280] + 1.3649e-4*mu*T[T<280] + 9.8581e-5*p*T[T<280]\
        + 7.3239e-2*asurf**2 - 1.6555e-5*T[T<280]**2 + 6.5817e-4*p**2 + 8.1218e-2*mu**2
    A[T>=280] = 1.1082 + 1.5172*asurf - 5.7993e-3*T[T>=280] + 1.9705e-2*p - 1.8670e-1*mu - \
                3.1355e-2*asurf*p - 1.0214e-2*mu*p + 2.0986e-1*asurf*mu - 3.7098e-3*asurf*T[T>=280]\
                - 1.1335e-4*mu*T[T>=280] + 5.3714e-5*p*T[T>=280] + 7.5887e-2*asurf**2 + 9.2690e-6*T[T>=280]**2 \
                - 4.1327e-4*p**2 + 6.3298e-2*mu**2
    return A

###### Haqq-Misra+ 2019 functions *************************************************

def olr_hm16(pCO2, T):
  phi = np.log10(pCO2)
  tmpk = np.log10(T)
  term1=9.12805643869791438760e+00*tmpk**4 + 4.58408794768168803557e+00*tmpk**3*phi-\
        8.47261075643147449910e+01*tmpk**3 + 4.35517381112690282752e-01*tmpk**2*phi**2-\
        2.86355036260417961103e+01*tmpk**2*phi + 2.96626642498045896446e+02*tmpk**2-\
        6.01082900358299240806e-02*tmpk*phi**3 - 2.60414691486954641420e+00*tmpk*phi**2+\
        5.69812976563675661623e+01*tmpk*phi

  term2=-4.62596100127381816947e+02*tmpk + 2.18159373001564722491e-03*phi**4+\
        1.61456772400726950023e-01*phi**3 + 3.75623788187470086797e+00*phi**2-\
        3.53347289223180354156e+01*phi + 2.75011005409836684521e+02
  ir = 10.0**(term1+term2)/1000.0
  return ir

def olr_hm16_dT(pCO2, T):
  phi = np.log10(pCO2)
  tmpk = np.log10(T)
  f    = 4 * 9.12805643869791438760 * (tmpk * tmpk * tmpk) +\
         3 * 4.58408794768168803557 * (tmpk * tmpk) * phi -\
         3 * 8.47261075643147449910e+01 * (tmpk * tmpk) +\
         2 * 4.35517381112690282752e-01 * tmpk * (phi * phi) -\
         2 * 2.86355036260417961103e+01 * tmpk * phi +\
         2 * 2.96626642498045896446e+02 * tmpk -\
         6.01082900358299240806e-02 * (phi * phi * phi) -\
         2.60414691486954641420 * (phi * phi) + 5.69812976563675661623e+01 * phi -\
         4.62596100127381816947e+02
  dIr = olr_hm16(pCO2,T) * f / T
  return dIr

def alb_hm16(pCO2, T, mu, albsurf):
  tmpk = np.log10(T)
  phi = np.log10(pCO2)
  A = np.zeros_like(T)
  A[T<=250] = -3.64301272050786051349e-01 * mu * mu * mu -\
             6.66571453035937344644e-01 * mu * mu * albsurf +\
             1.38761634791769922215e-01 * mu * mu * tmpk[T<=250] -\
             1.40826323888164368220e-02 * mu * mu * phi +\
             9.41440608298288128530e-01 * mu * mu +\
             7.10961643487220129600e-02 * mu * albsurf * albsurf -\
             2.19180456421237290776e-01 * mu * albsurf * tmpk[T<=250] -\
             1.82873271476295846949e-02 * mu * albsurf * phi +\
             1.48505536251773073708e+00 * mu * albsurf -\
             9.01309617860975631487e-01 * mu * tmpk[T<=250] * tmpk[T<=250] +\
             1.92113767482554841093e-02 * mu * tmpk[T<=250] * phi +\
             4.11334031794617160926e+00 * mu * tmpk[T<=250] +\
             6.80906172782627400891e-04 * mu * phi * phi -\
             1.66632232847024261413e-02 * mu * phi -\
             6.01321219414692986760e+00 * mu +\
             5.20833333338503734478e-02 * albsurf * albsurf * albsurf +\
             1.09511892935421337181e-01 * albsurf * albsurf * tmpk[T<=250] +\
             1.86369741605604787027e-02 * albsurf * albsurf * phi -\
             2.54092206932019781807e-01 * albsurf * albsurf -\
             4.00290429315177131997e+00 * albsurf * tmpk[T<=250] * tmpk[T<=250] -\
             4.60694421170402754195e-02 * albsurf * tmpk[T<=250] * phi +\
             1.79103047870275950970e+01 * albsurf * tmpk[T<=250] -\
             1.59834667195196747369e-02 * albsurf * phi * phi -\
             1.29954198131196525801e-02 * albsurf * phi -\
             1.97041106668471570629e+01 * albsurf -\
             9.28987827590191805882e+00 * tmpk[T<=250] * tmpk[T<=250] * tmpk[T<=250] +\
             2.33079221557892068972e-01 * tmpk[T<=250] * tmpk[T<=250] * phi +\
             6.58750181054108310263e+01 * tmpk[T<=250] * tmpk[T<=250] +\
             7.46763857253681870296e-03 * tmpk[T<=250] * phi * phi -\
             1.00561681124449076030e+00 * tmpk[T<=250] * phi -\
             1.55355955538023465579e+02 * tmpk[T<=250] +\
             7.11268878229609079374e-04 * phi * phi * phi -\
             3.36136500021004319336e-03 * phi * phi +\
             1.13977221457453326003e+00 * phi + 1.22439629486842392225e+02
  A[T>250] = -4.41391619954555503025e-01 * mu * mu * mu -\
             2.60017516002879089942e-01 * mu * mu * albsurf +\
             1.08110772295329837789e+00 * mu * mu * tmpk[T>250] -\
             3.93863285843020910493e-02 * mu * mu * phi -\
             1.46383456258096611435e+00 * mu * mu +\
             9.91383778608142668398e-02 * mu * albsurf * albsurf -\
             1.45914724229303338632e+00 * mu * albsurf * tmpk[T>250] -\
             2.72769392852398387395e-02 * mu * albsurf * phi +\
             3.99933641081463919775e+00 * mu * albsurf +\
             1.07231336256525633388e+00 * mu * tmpk[T>250] * tmpk[T>250] -\
             1.04302520934751417891e-02 * mu * tmpk[T>250] * phi -\
             6.10296439299006454604e+00 * mu * tmpk[T>250] +\
             2.69255203910960137434e-03 * mu * phi * phi +\
             9.50143253373007257157e-02 * mu * phi +\
             7.37864215757422226005e+00 * mu +\
             1.28580729156335171748e-01 * albsurf * albsurf * albsurf -\
             3.07800300913486257759e-01 * albsurf * albsurf * tmpk[T>250] +\
             2.27715594632176554502e-02 * albsurf * albsurf * phi +\
             6.11699085276039222769e-01 * albsurf * albsurf -\
             2.33213409642421742873e+00 * albsurf * tmpk[T>250] * tmpk[T>250] +\
             2.56011431303802661219e-01 * albsurf * tmpk[T>250] * phi +\
             1.05912148222549546972e+01 * albsurf * tmpk[T>250] -\
             1.85772688884413561539e-02 * albsurf * phi * phi -\
             7.55796861024326749323e-01 * albsurf * phi -\
             1.16485004141808623501e+01 * albsurf +\
             2.74062491988752192640e+01 * tmpk[T>250] * tmpk[T>250] * tmpk[T>250] +\
             5.46044240911252587445e-01 * tmpk[T>250] * tmpk[T>250] * phi -\
             2.05761674358916081928e+02 * tmpk[T>250] * tmpk[T>250] +\
             5.57943359123403426203e-02 * tmpk[T>250] * phi * phi -\
             2.49880329758542751861e+00 * tmpk[T>250] * phi +\
             5.14448995054491206247e+02 * tmpk[T>250] +\
             2.43702089287719950508e-03 * phi * phi * phi -\
             1.09384840764980617589e-01 * phi * phi +\
             2.92643187434628071486e+00 * phi - 4.27802454850920923946e+02
  return A


###### Kadoya & Tajika 2019 functions *********************************************

def olr_kt19(pCO2, T):
    tau = 0.01*(T-250)
    p = 0.2*np.log10(pCO2)
    I0 = -3.1
    B = np.array([[87.8373,-311.289,-504.408,-422.929,-134.611],
                  [54.9102,-677.741,-1440.63,-1467.04,-543.371],
                  [24.7875,31.3614,-363.617,-747.352,-395.401],
                  [75.8917,816.426,1565.03,1453.73,476.475],
                  [43.0076,339.957,996.723,1361.41,612.967],
                  [-31.4994,-261.362,-395.106,-261.6,-36.6589],
                  [-28.8846,-174.942,-378.436,-445.878,-178.948]])
    pvec = np.array([1,p,p**2,p**3,p**4])
    Tvec = np.array([1,tau,tau**2,tau**3,tau**4,tau**5,tau**6])
    ir = I0 = np.dot(Tvec,np.dot(B,pvec))
    return ir

def olr_kt19_dT(pCO2, T):
    tau = 0.01*(T-250)
    p = 0.2*np.log10(pCO2)
    I0 = -3.1
    B = np.array([[87.8373,-311.289,-504.408,-422.929,-134.611],
                  [54.9102,-677.741,-1440.63,-1467.04,-543.371],
                  [24.7875,31.3614,-363.617,-747.352,-395.401],
                  [75.8917,816.426,1565.03,1453.73,476.475],
                  [43.0076,339.957,996.723,1361.41,612.967],
                  [-31.4994,-261.362,-395.106,-261.6,-36.6589],
                  [-28.8846,-174.942,-378.436,-445.878,-178.948]])
    pvec = np.array([1,p,p**2,p**3,p**4])
    Tvec = np.array([0.0,1,2*tau,3*tau**2,4*tau**3,5*tau**4,6*tau**5])
    ir = I0 = 0.01*np.dot(Tvec,np.dot(B,pvec))
    return ir

def alb_kt19(sza, asurf):
    #adds cloud albedo to surface albedo
    #simple function of zenith angle in radians
    return asurf - 0.078 + 0.65*sza*np.pi/180.0


### Main part of script

T = float(args.temperature[0])
pCO2 = float(args.XCO2[0])*1e-6  #assumes 1 bar atmosphere!
asurf = float(args.albedo[0])
print('  Input: T = %#.3f K, pCO2 = %#.3e bar, asurf = %#.3f'%(T,pCO2,asurf))

if T < Tlimits[0] or T > Tlimits[1]:
    print('\nWarning: temperature is outside limits of (%#.1f, %#.1f) K.'%(Tlimits[0],Tlimits[1]))
    print('         Resulting coefficients may be inaccurate.')

if args.method[0] == 'WK':
    olr = olr_wk97(pCO2,T)
    B = olr_wk97_dT(pCO2,T)
    atot = alb_wk97(pCO2,np.array([T]),1,asurf)

elif args.method[0] == 'HM':
    olr = olr_hm16(pCO2,T)
    B = olr_hm16_dT(pCO2,T)
    atot = alb_hm16(pCO2,np.array([T]),1,asurf)

elif args.method[0] == 'KT':
    olr = olr_kt19(pCO2,T)
    B = olr_kt19_dT(pCO2,T)
    atot = alb_kt19(0.0,asurf)

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
        olr_full = olr_wk97(pCO2,temps)
        olr_limit = olr_wk97(pCO2,temps_lim)
    elif args.method[0] == 'HM':
        olr_full = olr_hm16(pCO2,temps)
        olr_limit = olr_hm16(pCO2,temps_lim)
    elif args.method[0] == 'KT':
        olr_full = olr_kt19(pCO2,temps)
        olr_limit = olr_kt19(pCO2,temps_lim)

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
        olr_full = olr_wk97(pressures,T)
    elif args.method[0] == 'HM':
        olr_full = olr_hm16(pressures,T)
    elif args.method[0] == 'KT':
        olr_full = olr_kt19(pressures,T)

    plt.subplot(1,2,2)
    plt.semilogx(pressures,olr_full,'k-')
    plt.plot(pCO2,A_K+B*T,'*',color='orangered')

    plt.title('$T$ = %#.3f K'%(T))

    plt.ylabel(r'OLR (W/m$^2$)')
    plt.xlabel('$p_{CO2}$ (bar)')

    plt.tight_layout()
    plt.show()
