import numpy as np
import matplotlib.pyplot as plt
import vplot as vpl
import vplanet
import pathlib
from astropy import units as u
import pdb
import os

sims = ['protocol_figures/Ben1_POISE_long','protocol_figures/Ben2_POISE_vnew_wAlbSurf','protocol_figures/Ben3_POISE_vnew_wAlbSurf',
         '../fillet_sims/Exp3_long','../fillet_sims/Exp3_long','../fillet_sims/Exp1_long/Exp1_sims/','../fillet_sims/Exp2_long/Exp2_sims/',
         '../fillet_sims/Exp4_long','../fillet_sims/Exp4_long','../fillet_sims/Exp1a_long/Exp1a_sims/','../fillet_sims/Exp2a_long/Exp2a_sims/']
out_paths = ['output/poise/ben1/','output/poise/ben2/','output/poise/ben3/','output/poise/exp3_cold/',
                'output/poise/exp3_warm/','output/poise/exp1','output/poise/exp2',
                'output/poise/exp4_cold','output/poise/exp4_warm','output/poise/exp1a','output/poise/exp2a',]
dim_space = [0,0,0,1,1,2,2,1,1,2,2]
do_these = [0,0,0,0,0,0,0,1,1,0,0]  #set to zero to skip set of sims
sim_number = [1,1,1,29,29,190,190,51,51,190,150]
iter_prefix = ['','','','/ColdStart/Exp3_ColdStart/instell_','/WarmStart/Exp3_WarmStart/instell_','exp1_S','exp2_S',
               '/ColdStart/Exp4_ColdStart/pCO2_p','/WarmStart/Exp4_WarmStart/pCO2_p','exp1a_semi','exp2a_semi',]

lat_template = 'output_templates/lat_output.dat'
glob_template = 'output_templates/global_output.dat'

FREE = 0
CAP = 1
BELT = 2
SBALL = 3

class read_dotin_file:
    def __init__(self,filename):
        f = open(filename)
        lines = f.readlines()
        for iline in lines:
            if iline[0] == '#' or len(iline) < 2:
                pass
            else:
                split_line = iline.split()
                setattr(self,split_line[0],split_line[1:])

for i in np.arange(len(sims)):
    if do_these[i]:
        case_glob = np.arange(sim_number[i])
        inst_glob = np.zeros_like(case_glob,dtype=float)
        obl_glob = np.zeros_like(case_glob,dtype=float)
        xco2_glob = np.zeros_like(case_glob,dtype=float)
        tglob_glob = np.zeros_like(case_glob,dtype=float)
        # icen_glob_max = np.zeros_like(case_glob,dtype=float)
        # icen_glob_min = np.zeros_like(case_glob,dtype=float)
        # ices_glob_max = np.zeros_like(case_glob,dtype=float)
        # ices_glob_min = np.zeros_like(case_glob,dtype=float)
        icen_land_max = np.zeros_like(case_glob,dtype=float)
        icen_land_min = np.zeros_like(case_glob,dtype=float)
        ices_land_max = np.zeros_like(case_glob,dtype=float)
        ices_land_min = np.zeros_like(case_glob,dtype=float)
        icen_ocean_max = np.zeros_like(case_glob,dtype=float)
        icen_ocean_min = np.zeros_like(case_glob,dtype=float)
        ices_ocean_max = np.zeros_like(case_glob,dtype=float)
        ices_ocean_min = np.zeros_like(case_glob,dtype=float)
        diff = np.zeros_like(case_glob,dtype=float)
        olr_glob = np.zeros_like(case_glob,dtype=float)

        print('Working on %s'%out_paths[i])
        for isim in case_glob:
            #read in vpl files
            path = pathlib.Path(sims[i])
            if sim_number[i] > 1:
                if dim_space[i] == 1:
                    path = pathlib.Path(sims[i] + iter_prefix[i] + '%02d'%isim)
                elif dim_space[i] == 2:
                    files_list = [filename for filename in os.listdir(str(path)) if filename.startswith(iter_prefix[i])]
                    files_list.sort()
                    path = pathlib.Path(sims[i] + files_list[isim])
                    # pdb.set_trace()
            out = vplanet.run(path / "vpl.in")

            if hasattr(out,'Earth'):
                body = out.Earth
                logbody = out.log.initial.Earth
            elif hasattr(out,'earth'):
                body = out.earth
                logbody = out.log.initial.earth

            if not np.isnan(body.TGlobal[-1]):
                inputf = read_dotin_file(str(path / 'earth.in'))
                nlat = int(inputf.iLatCellNum[0])
                ice_albedo = float(inputf.dIceAlbedo[0])
                land_albedo = float(inputf.dAlbedoLand[0])
                ocean_albedo = float(inputf.dAlbedoWater[0])
                if hasattr(inputf,'dLandFrac'):
                    land_frac = float(inputf.dLandFrac[0])
                else:
                    land_frac = 0.34 #default value in poise
                diff[isim] = float(inputf.dDiffusion[0])

                ntime = len(body.Time)
                lats = body.Latitude[nlat*(ntime-1):].value
                temp = (body.TempLat[nlat*(ntime-1):].to(u.K,equivalencies=u.temperature())).value
                temp_land = (body.TempLandLat[nlat*(ntime-1):].to(u.K,equivalencies=u.temperature())).value
                temp_ocean = (body.TempWaterLat[nlat*(ntime-1):].to(u.K,equivalencies=u.temperature())).value
                alblat = (body.AlbedoLat[nlat*(ntime-1):]).value
                alblat_land = (body.AlbedoLandLat[nlat*(ntime-1):]).value
                alblat_ocean = (body.AlbedoWaterLat[nlat*(ntime-1):]).value
                A = body.PlanckAAvg[nlat*(ntime-1):].value
                B = body.PlanckBAvg[nlat*(ntime-1):].value
                OLR = A + B*(temp-273.15)

                if hasattr(body,'AlbSurfLandLat'):
                    albsurf_land = (body.AlbSurfLandLat[nlat*(ntime-1):]).value
                    albsurf_ocean = (body.AlbSurfWaterLat[nlat*(ntime-1):]).value
                else:
                    #need to reconstruct surface albedo!
                    albsurf_land = np.zeros_like(temp) + land_albedo
                    albsurf_ocean = np.zeros_like(temp) + ocean_albedo
                    albsurf_land[temp_land<=273.15] = ice_albedo
                    albsurf_ocean[temp_ocean<=271.15] = ice_albedo

                albsurf = land_frac*albsurf_land + (1-land_frac)*albsurf_ocean

                #set up newly formatted outputs
                out_path = pathlib.Path(out_paths[i])
                if not out_path.exists():
                    out_path.mkdir(parents=True,exist_ok=True)

                #latitude varying file first
                lat_file = open(lat_template)
                lat_text = lat_file.readlines()
                new_lat_path = out_path / ('case_%d'%isim)
                if not new_lat_path.exists():
                    new_lat_path.mkdir(parents=True,exist_ok=True)
                new_lat_file = open(str(new_lat_path / 'lat_output.dat'),'w')
                for iline in lat_text:
                    new_lat_file.write(iline)

                for ilat in np.arange(nlat):
                    new_lat_file.write("%f %f %f %f %f\n"%(lats[ilat],temp[ilat],
                                                        albsurf[ilat],alblat[ilat],OLR[ilat]))
                new_lat_file.close()

                #update global arrays
                inst_glob[isim] = out.log.initial.sun.Luminosity.value/(4*np.pi*logbody.SemiMajorAxis.value**2)/1361.0
                obl_glob[isim] = logbody.Obliquity.value*180/np.pi
                tglob_glob[isim] = (body.TGlobal[ntime-1].to(u.K,equivalencies=u.temperature())).value
                olr_glob[isim] = body.FluxOutGlobal[ntime-1].value

                if hasattr(inputf,'dpCO2'):
                    xco2_glob[isim] = float(inputf.dpCO2[0])
                else:
                    xco2_glob[isim] = 280.0

                # pdb.set_trace()
                TlandN = temp_land[lats>=0]
                TocN = temp_ocean[lats>=0]
                latN = lats[lats>=0]

                if (TlandN > 273.15).all():
                    icelandN_max = 90.0  #no ice when max = min
                    icelandN_min = 90.0
                    landN = FREE
                elif (TlandN <= 273.15).all():
                    icelandN_max = 90.0
                    icelandN_min = 0.0
                    landN = SBALL
                else:
                    if TlandN[latN==np.max(latN)] <= 273.15: #cap
                        icelandN_max = 90.0 #ice extends to north pole
                        icelandN_min = np.min(latN[TlandN<=273.15])
                        landN = CAP
                    elif TlandN[latN==np.min(latN)] <= 273.15: #belt
                        icelandN_max = np.max(latN[TlandN<=273.15])
                        icelandN_min = 0.0 #ice extends to equator
                        landN = BELT
                if (TocN > 271.15).all():
                    iceocN_max = 90.0  #no ice when max = min
                    iceocN_min = 90.0
                    ocN = FREE
                elif (TocN <= 271.15).all():
                    iceocN_max = 90.0
                    iceocN_min = 0.0
                    ocN = SBALL
                else:
                    if TocN[latN==np.max(latN)] <= 271.15: #cap
                        iceocN_max = 90.0
                        iceocN_min = np.min(latN[TocN<=271.15])
                        ocN = CAP
                    if TocN[latN==np.min(latN)] <= 271.15: #belt
                        iceocN_max = np.max(latN[TocN<=271.15])
                        iceocN_min = 0.0
                        ocN = BELT

                # if landN == BELT and ocN == FREE:
                #     iceocN_max = 0.0  #no ice when max = min
                #     iceocN_min = 0.0
                # if ocN == BELT and landN == FREE:
                #     icelandN_max = 0.0  #no ice when max = min
                #     icelandN_min = 0.0
                # if landN == SBALL

                # icen_glob_max[isim] = land_frac*icelandN_max + (1-land_frac)*iceocN_max
                # icen_glob_min[isim] = land_frac*icelandN_min + (1-land_frac)*iceocN_min
                icen_land_max[isim] = icelandN_max
                icen_land_min[isim] = icelandN_min
                icen_ocean_max[isim] = iceocN_max
                icen_ocean_min[isim] = iceocN_min

                TlandS = temp_land[lats<=0]
                TocS = temp_ocean[lats<=0]
                latS = lats[lats<=0]

                if (TlandS > 273.15).all():
                    icelandS_max = -90.0
                    icelandS_min = -90.0
                    landS = FREE
                elif (TlandS <= 273.15).all():
                    icelandS_max = 0.0
                    icelandS_min = -90.0
                    landS = SBALL
                else:
                    if TlandS[latS==np.min(latS)] <= 273.15: #cap
                        icelandS_max = np.max(latS[TlandS<=273.15])
                        icelandS_min = -90.0
                        landS = CAP
                    elif TlandS[latS==np.max(latS)] <= 273.15: #belt
                        icelandS_max = 0.0
                        icelandS_min = np.min(latS[TlandS<=273.15])
                        landS = BELT
                if (TocS > 271.15).all():
                    iceocS_max = -90.0
                    iceocS_min = -90.0
                    ocS = FREE
                elif (TocS <= 271.15).all():
                    iceocS_max = 0.0
                    iceocS_min = -90.0
                    ocS = SBALL
                else:
                    if TocS[latS==np.min(latS)] <= 271.15: #cap
                        iceocS_max = np.max(latS[TocS<=271.15])
                        iceocS_min = -90.0
                        ocS = CAP
                    elif TocS[latS==np.max(latS)] <= 271.15: #belt
                        iceocS_max = 0.0
                        iceocS_min = np.min(latS[TocS<=271.15])
                        ocS = BELT

                # if landS == BELT and ocS == FREE:
                #     iceocS_max = 0.0  #no ice when max = min
                #     iceocS_min = 0.0
                # if ocS == BELT and landS == FREE:
                #     icelandS_max = 0.0  #no ice when max = min
                #     icelandS_min = 0.0

                # ices_glob_max[isim] = land_frac*icelandS_max + (1-land_frac)*iceocS_max
                # ices_glob_min[isim] = land_frac*icelandS_min + (1-land_frac)*iceocS_min

                ices_land_max[isim] = icelandS_max
                ices_land_min[isim] = icelandS_min
                ices_ocean_max[isim] = iceocS_max
                ices_ocean_min[isim] = iceocS_min

            #now write global file
            glob_file = open(glob_template)
            glob_text = glob_file.readlines()
            new_glob_file = open(str(out_path / 'global_output.dat'),'w')
            for iline in glob_text:
                new_glob_file.write(iline)

            for isim in case_glob:
                new_glob_file.write("%d %f %f %g %f %f %f %f %f %f %f %f %f %f %f\n"%(isim,inst_glob[isim],obl_glob[isim],
                                                                    xco2_glob[isim],tglob_glob[isim],
                                                                    # icen_glob_max[isim],icen_glob_min[isim],
                                                                    # ices_glob_max[isim],ices_glob_min[isim],
                                                                    icen_land_max[isim],icen_land_min[isim],
                                                                    ices_land_max[isim],ices_land_min[isim],
                                                                    icen_ocean_max[isim],icen_ocean_min[isim],
                                                                    ices_ocean_max[isim],ices_ocean_min[isim],
                                                                    diff[isim],olr_glob[isim]))
            new_glob_file.close()
