# Creates input for a DALES simulation with open boundary conditions from HARMONIE output
import funcDALES as DALES
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime

def main():
    # -------------------------- Input -------------------------- #
    # General
    iexpnr = 1                 # Experiment number 0-999
    pathRead  = '../harmonie/' # Path to harmonie data
    pathWrite = '../simulations/512x384/coarse/input/' # Path where to write input files
    it0    = 4*24              # Start HARMONIE time index for boundary input
    ite    = 5*24              # End HARMONIE time index for boundary input
    # LES grid
    itot = 512          # Number of grid points in x-direction
    jtot = 384          # Number of grid points in y-direction
    ktot = 150          # Number of grid points in z-direction
    xsize = itot/4*2500 # Size in x-direction (multiple of grid spacing harmonie 2500m)
    ysize = jtot/4*2500 # Size in y-direction (multiple of grid spacing harmonie 2500m)
    stretched = True    # Stretched grid True/False
    dz = 20.            # Stretched grid = False
    dz0 = 20.           # Stretched grid = True, resolution first height level
    alpha = 0.012       # Stretched grid = True, growth factor height levels
    lat0 = 13.3         # Latitude of domain centre
    lon0 = -57.8        # Longitude of domain centre
    # Files to create
    lprof = False       # Input profiles required True/False
    lboundary = True    # Boundary input required True/False
    linithetero = False # Heterogeneous input fields required True/False
    lsynturb  = True    # Synthetic turbulence input required True/False
    lrad = False        # Input for rrtmg input required True/False (not supported yet)
    # Synthetic turbulence input
    dxTurb = xsize      # Patch size for mass correction and synturb
    dyTurb = ysize      # Patch size for mass correction and synturb
    T0     = 293        # Temperature scale
    # Boundary input
    e12    = 0.1        # Turbulent kinetic energy (set to a constant)
    # Simulation constants (modglobal)
    p0   = 1e5          # Reference pressure
    Rd   = 287.04       # Gas constant for dry air
    Rv   = 461.5        # Gas constant for water vapor
    cp   = 1004.        # Specific heat at constant pressure (dry air)
    Lv   = 2.53e6       # Latent heat for vaporisation
    grav = 9.81         # Gravitational constant
    kappa= 0.4          # Von Karman constant

    # ------------------ Construct DALES grid ------------------- #
    grid = DALES.Grid(xsize,ysize,itot,jtot,ktot,dz,dz0,alpha,stretched)

    # ------------------- Open Harmonie data -------------------- #
    nch_u    = Dataset(f"{pathRead}ua_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_v    = Dataset(f"{pathRead}va_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_w    = Dataset(f"{pathRead}wa_Slev_fp_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_T    = Dataset(f"{pathRead}ta_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_qv   = Dataset(f"{pathRead}hus_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_ql   = Dataset(f"{pathRead}clw_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_ps   = Dataset(f"{pathRead}ps_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_Ts   = Dataset(f"{pathRead}tas_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_qts  = Dataset(f"{pathRead}huss_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_cb   = Dataset(f"{pathRead}cb_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nch_wTs  = Dataset(f"{pathRead}hfss_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    nch_uwsc = Dataset(f"{pathRead}uflx_conv_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    nch_uwst = Dataset(f"{pathRead}uflx_turb_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    nch_vwsc = Dataset(f"{pathRead}vflx_conv_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    nch_vwst = Dataset(f"{pathRead}vflx_turb_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    time   = nch_u.variables['time'][:]; time = (time-time[0])*3600*24; dt=time[1]-time[0]

    # ------------------- crop Harmonie data -------------------- #
    Nt     = ite-it0+1
    itotH  = int(xsize/2500)
    jtotH  = int(ysize/2500)
    x = np.arange(itotH+1)*2500
    y = np.arange(jtotH+1)*2500
    # Subset for 3d variables EUREC4Acircle_BES
    lats   = nch_u.variables['lat'][:,:]
    lons   = nch_u.variables['lon'][:,:]
    i0s    = np.argwhere((lats-lat0)**2+(lons-lon0)**2 == np.min((lats-lat0)**2+(lons-lon0)**2))[0]
    iy0s   = i0s[0]-int(jtotH/2); iyes=iy0s+jtotH
    ix0s   = i0s[1]-int(itotH/2); ixes=ix0s+itotH
    lats   = lats[iy0s:iyes+1,ix0s:ixes+1]
    lons   = lons[iy0s:iyes+1,ix0s:ixes+1]
    # Subset for surface variables BES_BES
    latl   = nch_Ts.variables['lat'][:,:]
    lonl   = nch_Ts.variables['lon'][:,:]
    i0l    = np.argwhere((latl-lat0)**2+(lonl-lon0)**2 == np.min((latl-lat0)**2+(lonl-lon0)**2))[0]
    iy0l   = i0l[0]-int(jtotH/2); iyel=iy0l+jtotH
    ix0l   = i0l[1]-int(itotH/2); ixel=ix0l+itotH
    latl   = latl[iy0l:iyel+1,ix0l:ixel+1]
    lonl   = lonl[iy0l:iyel+1,ix0l:ixel+1]

    # ------------------- Get DALES baseprof -------------------- #
    # Obtain dales reference profile
    Ts  = nch_Ts.variables['tas'][it0,iy0l:iyel+1,ix0l:ixel+1]
    ps  = nch_ps.variables['ps'][it0,iy0l:iyel+1,ix0l:ixel+1]
    psDALES    = np.mean(ps) # Mean surface pressure
    exnrs      = (psDALES/p0)**(Rd/cp)
    thlsDALES  = np.mean(Ts)/exnrs # Mean surface temp
    rhobf = DALES.calcBaseprof(grid.z,thlsDALES,psDALES) # Density profile DALES
    
    # -------- Create initial heterogeneous input fields -------- #
    if linithetero:
        start_time = datetime.now()
        print("Start creation of initfields.inp."+str(iexpnr).zfill(3)+".nc")
        # Create netcdf file
        heteroInp = DALES.HeteroInp(grid, iexpnr=iexpnr, path=pathWrite)
        # Load fields first time step
        u  = nch_u.variables['ua'][it0,::-1,iy0s:iyes+1,ix0s:ixes+1]
        v  = nch_v.variables['va'][it0,::-1,iy0s:iyes+1,ix0s:ixes+1]
        w  = nch_w.variables['wa'][it0,::-1,iy0s:iyes+1,ix0s:ixes+1]
        T  = nch_T.variables['ta'][it0,::-1,iy0s:iyes+1,ix0s:ixes+1]
        qv = nch_qv.variables['hus'][it0,::-1,iy0s:iyes+1,ix0s:ixes+1]
        ql = nch_ql.variables['clw'][it0,::-1,iy0s:iyes+1,ix0s:ixes+1]
        qt = qv+ql
        # Surface variables
        us  = np.zeros(np.shape(u)[1:3])
        vs  = np.zeros(np.shape(v)[1:3])
        ws  = np.zeros(np.shape(w)[1:3])
        qls = np.zeros(np.shape(w)[1:3])
        Ts  = nch_Ts.variables['tas'][it0,iy0l:iyel+1,ix0l:ixel+1]
        qts = nch_qts.variables['huss'][it0,iy0l:iyel+1,ix0l:ixel+1]
        ps  = nch_ps.variables['ps'][it0,iy0l:iyel+1,ix0l:ixel+1]
        # Concatenate surface fields to 3d fields
        u   = np.concatenate((us[None,:,:],u),axis=0)
        v   = np.concatenate((vs[None,:,:],v),axis=0)
        w   = np.concatenate((ws[None,:,:],w),axis=0)
        T   = np.concatenate((Ts[None,:,:],T),axis=0)
        qt  = np.concatenate((qts[None,:,:],qt),axis=0)
        ql  = np.concatenate((qls[None,:,:],ql),axis=0)
        # Calculate height levels using hydrostatic equilibrium
        p   = calcPressure(ps)
        p   = np.concatenate((ps[None,:,:],p),axis=0)
        rho = p/(Rd*T*(1+(Rv/Rd-1)*qt-Rv/Rd*ql))
        rhoh= (rho[:-1,:,:]+rho[1:,:,:])/2
        z   = np.zeros(np.shape(T))
        for k in np.arange(1,np.shape(u)[0]):
            z[k,:,:] = z[k-1,:,:]-(p[k,:,:]-p[k-1,:,:])/(rhoh[k-1,:,:]*grav)
        # Interpolate, convert and write harmonie field to DALES input
        Tint  = interp3d(x,y,z,T,grid.x,grid.y,grid.z)
        qtint = interp3d(x,y,z,qt,grid.x,grid.y,grid.z)
        qlint = interp3d(x,y,z,ql,grid.x,grid.y,grid.z)
        p     = rhobf*Rd*np.mean(Tint,axis=(1,2))*(1+(Rv/Rd-1)*np.mean(qtint,axis=(1,2))-Rv/Rd*np.mean(qlint,axis=(1,2))) # Ideal gas law
        exnr  = (p/p0)**(Rd/cp)
        # Write exnr function to exnr.inp.iexpnr
        file = open(f"{pathWrite}exnr.inp.{iexpnr:03}",'w')
        file.write('Exner function used for calculation DALES input\n')
        file.write(f"ps = {psDALES} thls = {thlsDALES} exnrs = {exnrs}\n")
        file.write('zf exnr\n')
        for k in range(ktot):
            file.write(f"{grid.z[k]} {exnr[k]}\n")
        file.close()
        heteroInp.thl[:,:,:] = Tint/exnr[:,None,None]-Lv*qlint/(cp*exnr[:,None,None])
        heteroInp.qt[:,:,:]  = qtint
        heteroInp.u[:,:,:]   = interp3d(x,y,z,u,grid.xh,grid.y,grid.z)
        heteroInp.v[:,:,:]   = interp3d(x,y,z,v,grid.x,grid.yh,grid.z)
        heteroInp.w[:,:,:]   = interp3d(x,y,z,w,grid.x,grid.y,grid.zh)
        heteroInp.e12[:,:,:] = e12
        # Close netcdf file
        heteroInp.exit()
        del heteroInp
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")
    else: # Read exnr.inp.iexpnr
        data = np.loadtxt(f"{pathWrite}exnr.inp.{iexpnr:03}",skiprows=3,delimiter=' ')
        exnr = data[:,1]

    # ------------------ Create boundary input ------------------ #
    if lboundary:
        start_time = datetime.now()
        print("Start creation of openboundaries.inp."+str(iexpnr).zfill(3)+".nc")
        # Create netcdf file
        boundaryInp = DALES.OpenBoundariesInp(grid, iexpnr=iexpnr, path=pathWrite)
        # Write fields per time step
        for it in range(Nt):
            # Load fields for current time step
            u  = nch_u.variables['ua'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
            v  = nch_v.variables['va'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
            w  = nch_w.variables['wa'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
            T  = nch_T.variables['ta'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
            qv = nch_qv.variables['hus'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
            ql = nch_ql.variables['clw'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
            qt = qv+ql
            # Surface variables
            us  = np.zeros(np.shape(u)[1:3])
            vs  = np.zeros(np.shape(v)[1:3])
            ws  = np.zeros(np.shape(w)[1:3])
            qls = np.zeros(np.shape(w)[1:3])
            Ts  = nch_Ts.variables['tas'][it0+it,iy0l:iyel+1,ix0l:ixel+1]
            qts = nch_qts.variables['huss'][it0+it,iy0l:iyel+1,ix0l:ixel+1]
            ps  = nch_ps.variables['ps'][it0+it,iy0l:iyel+1,ix0l:ixel+1]
            # Concatenate surface fields to 3d fields
            u   = np.concatenate((us[None,:,:],u),axis=0)
            v   = np.concatenate((vs[None,:,:],v),axis=0)
            w   = np.concatenate((ws[None,:,:],w),axis=0)
            T   = np.concatenate((Ts[None,:,:],T),axis=0)
            qt  = np.concatenate((qts[None,:,:],qt),axis=0)
            ql  = np.concatenate((qls[None,:,:],ql),axis=0)
            # Calculate height levels using hydrostatic equilibrium
            p   = calcPressure(ps)
            p   = np.concatenate((ps[None,:,:],p),axis=0)
            rho = p/(Rd*T*(1+(Rv/Rd-1)*qt-Rv/Rd*ql))
            rhoh= (rho[:-1,:,:]+rho[1:,:,:])/2
            z   = np.zeros(np.shape(T))
            for k in np.arange(1,np.shape(u)[0]):
                z[k,:,:] = z[k-1,:,:]-(p[k,:,:]-p[k-1,:,:])/(rhoh[k-1,:,:]*grav)
            # Interpolate fields and write fields
            boundaryInp.time[it] = time[it0+it]-time[it0]
            # Interpolate west boundary
            Tint  = interpLateral(y,z[:,:,0],T[:,:,0],grid.y,grid.z)
            qtint = interpLateral(y,z[:,:,0],qt[:,:,0],grid.y,grid.z)
            qlint = interpLateral(y,z[:,:,0],ql[:,:,0],grid.y,grid.z)
            boundaryInp.thlwest[it,:,:]  = Tint/exnr[:,None]-Lv*qlint/(cp*exnr[:,None])
            boundaryInp.qtwest[it,:,:]   = qtint
            boundaryInp.uwest[it,:,:]    = interpLateral(y,z[:,:,0],u[:,:,0],grid.y,grid.z)
            boundaryInp.vwest[it,:,:]    = interpLateral(y,z[:,:,0],v[:,:,0],grid.yh,grid.z)
            boundaryInp.wwest[it,:,:]    = interpLateral(y,z[:,:,0],w[:,:,0],grid.y,grid.zh)
            boundaryInp.e12west[it,:,:]  = e12
            # Interpolate east boundary
            Tint  = interpLateral(y,z[:,:,-1],T[:,:,-1],grid.y,grid.z)
            qtint = interpLateral(y,z[:,:,-1],qt[:,:,-1],grid.y,grid.z)
            qlint = interpLateral(y,z[:,:,-1],ql[:,:,-1],grid.y,grid.z)
            boundaryInp.thleast[it,:,:]  = Tint/exnr[:,None]-Lv*qlint/(cp*exnr[:,None])
            boundaryInp.qteast[it,:,:]   = qtint
            boundaryInp.ueast[it,:,:]    = interpLateral(y,z[:,:,-1],u[:,:,-1],grid.y,grid.z)
            boundaryInp.veast[it,:,:]    = interpLateral(y,z[:,:,-1],v[:,:,-1],grid.yh,grid.z)
            boundaryInp.weast[it,:,:]    = interpLateral(y,z[:,:,-1],w[:,:,-1],grid.y,grid.zh)
            boundaryInp.e12east[it,:,:]  = e12
            # Interpolate south boundary
            Tint  = interpLateral(x,z[:,0,:],T[:,0,:],grid.x,grid.z)
            qtint = interpLateral(x,z[:,0,:],qt[:,0,:],grid.x,grid.z)
            qlint = interpLateral(x,z[:,0,:],ql[:,0,:],grid.x,grid.z)
            boundaryInp.thlsouth[it,:,:] = Tint/exnr[:,None]-Lv*qlint/(cp*exnr[:,None])
            boundaryInp.qtsouth[it,:,:]  = qtint
            boundaryInp.usouth[it,:,:]   = interpLateral(x,z[:,0,:],u[:,0,:],grid.xh,grid.z)
            boundaryInp.vsouth[it,:,:]   = interpLateral(x,z[:,0,:],v[:,0,:],grid.x,grid.z)
            boundaryInp.wsouth[it,:,:]   = interpLateral(x,z[:,0,:],w[:,0,:],grid.x,grid.zh)
            boundaryInp.e12south[it,:,:] = e12
            # Interpolate north boundary
            Tint  = interpLateral(x,z[:,-1,:],T[:,-1,:],grid.x,grid.z)
            qtint = interpLateral(x,z[:,-1,:],qt[:,-1,:],grid.x,grid.z)
            qlint = interpLateral(x,z[:,-1,:],ql[:,-1,:],grid.x,grid.z)
            boundaryInp.thlnorth[it,:,:] = Tint/exnr[:,None]-Lv*qlint/(cp*exnr[:,None])
            boundaryInp.qtnorth[it,:,:]  = qtint
            boundaryInp.unorth[it,:,:]   = interpLateral(x,z[:,-1,:],u[:,-1,:],grid.xh,grid.z)
            boundaryInp.vnorth[it,:,:]   = interpLateral(x,z[:,-1,:],v[:,-1,:],grid.x,grid.z)
            boundaryInp.wnorth[it,:,:]   = interpLateral(x,z[:,-1,:],w[:,-1,:],grid.x,grid.zh)
            boundaryInp.e12north[it,:,:] = e12
            # Interpolate top boundary
            Tint  = interpTop(x,y,z,T,grid.x,grid.y,grid.zh[-1])
            qtint = interpTop(x,y,z,qt,grid.x,grid.y,grid.zh[-1])
            qlint = interpTop(x,y,z,ql,grid.x,grid.y,grid.zh[-1])
            exnrt  = exnr[-1]+(grid.zh[-1]-grid.z[-1])*(exnr[-1]-exnr[-2])/(grid.z[-1]-grid.z[-2])
            boundaryInp.thltop[it,:,:]   = Tint/exnrt-Lv*qlint/(cp*exnrt)
            boundaryInp.qttop[it,:,:]    = qtint
            boundaryInp.utop[it,:,:]     = interpTop(x,y,z,u,grid.xh,grid.y,grid.zh[-1])
            boundaryInp.vtop[it,:,:]     = interpTop(x,y,z,v,grid.x,grid.yh,grid.zh[-1])
            boundaryInp.wtop[it,:,:]     = interpTop(x,y,z,w,grid.x,grid.y,grid.zh[-1])
            boundaryInp.e12top[it,:,:]   = e12
        # Close netcdf file
        boundaryInp.exit()
        del boundaryInp
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")

    # ------------------ Create synturb input ------------------ #
    if lsynturb:
        start_time = datetime.now()
        print("Start creation of synturb for "+str(iexpnr).zfill(3))
        NxPatch = xsize/dxTurb
        NyPatch = ysize/dyTurb
        xpatch = (np.arange(NxPatch)+.5)*dxTurb
        ypatch = (np.arange(NyPatch)+.5)*dyTurb
        zpatch = grid.z
        # Create netcdf file
        synTurbInp = DALES.SynTurbInp(xpatch,ypatch,zpatch,iexpnr=iexpnr,path=pathWrite)
        # Write fields per time step
        timeint = np.cumsum(time)
        for it in range(Nt):
            if(it0+it+1<len(nch_uwsc.variables['time'][:])):
                # Load fields for current time step
                T  = nch_T.variables['ta'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
                qv = nch_qv.variables['hus'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
                ql = nch_ql.variables['clw'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1]
                qt = qv+ql
                # Surface variables
                qls = np.zeros(np.shape(T)[1:3])
                Ts  = nch_Ts.variables['tas'][it0+it,iy0l:iyel+1,ix0l:ixel+1]
                qts = nch_qts.variables['huss'][it0+it,iy0l:iyel+1,ix0l:ixel+1]
                ps  = nch_ps.variables['ps'][it0+it,iy0l:iyel+1,ix0l:ixel+1]
                # Concatenate surface fields to 3d fields
                T   = np.concatenate((Ts[None,:,:],T),axis=0)
                qt  = np.concatenate((qts[None,:,:],qt),axis=0)
                ql  = np.concatenate((qls[None,:,:],ql),axis=0)
                # Calculate height levels using hydrostatic equilibrium
                p   = calcPressure(ps)
                p   = np.concatenate((ps[None,:,:],p),axis=0)
                rho = p/(Rd*T*(1+(Rv/Rd-1)*qt-Rv/Rd*ql))
                rhoh= (rho[:-1,:,:]+rho[1:,:,:])/2
                z   = np.zeros(np.shape(T))
                for k in np.arange(1,np.shape(T)[0]):
                    z[k,:,:] = z[k-1,:,:]-(p[k,:,:]-p[k-1,:,:])/(rhoh[k-1,:,:]*grav)
                z = np.mean(z,axis=(1,2))
                # Preallocate and set to 0
                u2   = np.zeros(grid.ktot)
                v2   = np.zeros(grid.ktot)
                w2   = np.zeros(grid.ktot)
                uv   = np.zeros(grid.ktot)
                thl2 = np.zeros(grid.ktot)
                qt2  = np.zeros(grid.ktot)
                wthl = np.zeros(grid.ktot)
                wqt  = np.zeros(grid.ktot)
                # Calculate required parameters
                rhobs = rhobf[0]-grid.z[0]*(rhobf[1]-rhobf[0])/(grid.z[1]-grid.z[0])
                zi    = np.max([np.mean(nch_cb.variables['cb'][it0+it,iy0l:iyel+1,ix0l:ixel+1]),500])
                wthls = np.mean(nch_wTs.variables['hfss'][it0+it+1,iy0l:iyel+1,ix0l:ixel+1]-nch_wTs.variables['hfss'][it0+it,iy0l:iyel+1,ix0l:ixel+1])/(dt*exnrs*rhobs*cp)
                uw    = (np.mean(nch_uwsc.variables['uflx_conv'][it0+it+1,::-1,iy0s:iyes+1,ix0s:ixes+1]-nch_uwsc.variables['uflx_conv'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1],axis=(1,2))+
                         np.mean(nch_uwst.variables['uflx_turb'][it0+it+1,::-1,iy0s:iyes+1,ix0s:ixes+1]-nch_uwst.variables['uflx_turb'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1],axis=(1,2)))/(dt*rhobs)
                vw    = (np.mean(nch_vwsc.variables['vflx_conv'][it0+it+1,::-1,iy0s:iyes+1,ix0s:ixes+1]-nch_vwsc.variables['vflx_conv'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1],axis=(1,2))+
                         np.mean(nch_vwst.variables['vflx_turb'][it0+it+1,::-1,iy0s:iyes+1,ix0s:ixes+1]-nch_vwst.variables['vflx_turb'][it0+it,::-1,iy0s:iyes+1,ix0s:ixes+1],axis=(1,2)))/(dt*rhobs)
                uws   = 2*uw[0]-uw[1]
                vws   = 2*vw[0]-vw[1]
                uw    = np.concatenate((uws[None],uw),axis=0)
                vw    = np.concatenate((vws[None],vw),axis=0)
                uw    = np.interp(grid.z,z,uw)
                vw    = np.interp(grid.z,z,vw)
                ustar = np.sqrt(np.sqrt(uws**2+vws**2))
                wstar = (grav/T0*wthls*zi)**(1/3)
                Tstar = wthls/wstar
                Thstar= wthls/ustar
                L     = np.min([-ustar**2*T0/(kappa*grav*Thstar),-0.1])
                iz    = np.argwhere(grid.z<=zi)[-1][-1]+1
                # Get profiles
                u2[:iz]   = ( np.sqrt(np.abs(uws))*0.8*(-zi/L)**(1/3) )**2*np.minimum(5*grid.z[:iz]/zi,1)
                v2[:iz]   = ( np.sqrt(np.abs(vws))*0.8*(-zi/L)**(1/3) )**2*np.minimum(5*grid.z[:iz]/zi,1)
                w2[:iz]   = ( wstar*(grid.z[:iz]/zi)**(1/3) )**2
                # uv   = leave at zero
                thl2[:iz] = ( Tstar*(grid.z[:iz]/zi)**(-1/3) )**2
                thl2[:4] = thl2[:4]*grid.z[:4]/grid.z[3]
                # qt2  = Leave at zero
                wthl[:iz] = wthls*(1-1.2*(grid.z[:iz]/zi))
                # wqt[:iz]  = Leave at zero
                synTurbInp.u2west[it,:,:] = u2[:,None]
                synTurbInp.v2west[it,:,:] = v2[:,None]
                synTurbInp.w2west[it,:,:] = w2[:,None]
                synTurbInp.uvwest[it,:,:] = uv[:,None]
                synTurbInp.uwwest[it,:,:] = uw[:,None]
                synTurbInp.vwwest[it,:,:] = vw[:,None]
                synTurbInp.thl2west[it,:,:] = thl2[:,None]
                synTurbInp.qt2west[it,:,:] = qt2[:,None]
                synTurbInp.wthlwest[it,:,:] = wthl[:,None]
                synTurbInp.wqtwest[it,:,:] = wqt[:,None]
                synTurbInp.u2east[it,:,:] = u2[:,None]
                synTurbInp.v2east[it,:,:] = v2[:,None]
                synTurbInp.w2east[it,:,:] = w2[:,None]
                synTurbInp.uveast[it,:,:] = uv[:,None]
                synTurbInp.uweast[it,:,:] = uw[:,None]
                synTurbInp.vweast[it,:,:] = vw[:,None]
                synTurbInp.thl2east[it,:,:] = thl2[:,None]
                synTurbInp.qt2east[it,:,:] = qt2[:,None]
                synTurbInp.wthleast[it,:,:] = wthl[:,None]
                synTurbInp.wqteast[it,:,:] = wqt[:,None]
                synTurbInp.u2south[it,:,:] = u2[:,None]
                synTurbInp.v2south[it,:,:] = v2[:,None]
                synTurbInp.w2south[it,:,:] = w2[:,None]
                synTurbInp.uvsouth[it,:,:] = uv[:,None]
                synTurbInp.uwsouth[it,:,:] = uw[:,None]
                synTurbInp.vwsouth[it,:,:] = vw[:,None]
                synTurbInp.thl2south[it,:,:] = thl2[:,None]
                synTurbInp.qt2south[it,:,:] = qt2[:,None]
                synTurbInp.wthlsouth[it,:,:] = wthl[:,None]
                synTurbInp.wqtsouth[it,:,:] = wqt[:,None]
                synTurbInp.u2north[it,:,:] = u2[:,None]
                synTurbInp.v2north[it,:,:] = v2[:,None]
                synTurbInp.w2north[it,:,:] = w2[:,None]
                synTurbInp.uvnorth[it,:,:] = uv[:,None]
                synTurbInp.uwnorth[it,:,:] = uw[:,None]
                synTurbInp.vwnorth[it,:,:] = vw[:,None]
                synTurbInp.thl2north[it,:,:] = thl2[:,None]
                synTurbInp.qt2north[it,:,:] = qt2[:,None]
                synTurbInp.wthlnorth[it,:,:] = wthl[:,None]
                synTurbInp.wqtnorth[it,:,:] = wqt[:,None]
                synTurbInp.u2top[it,:,:] = 0.
                synTurbInp.v2top[it,:,:] = 0.
                synTurbInp.w2top[it,:,:] = 0.
                synTurbInp.uvtop[it,:,:] = 0.
                synTurbInp.uwtop[it,:,:] = 0.
                synTurbInp.vwtop[it,:,:] = 0.
                synTurbInp.thl2top[it,:,:] = 0.
                synTurbInp.qt2top[it,:,:] = 0.
                synTurbInp.wthltop[it,:,:] = 0.
                synTurbInp.wqttop[it,:,:] = 0.
            else:
                synTurbInp.u2west[it,:,:] = synTurbInp.u2west[it-1,:,:]
                synTurbInp.v2west[it,:,:] = synTurbInp.v2west[it-1,:,:]
                synTurbInp.w2west[it,:,:] = synTurbInp.w2west[it-1,:,:]
                synTurbInp.uvwest[it,:,:] = synTurbInp.uvwest[it-1,:,:]
                synTurbInp.uwwest[it,:,:] = synTurbInp.uwwest[it-1,:,:]
                synTurbInp.vwwest[it,:,:] = synTurbInp.vwwest[it-1,:,:]
                synTurbInp.thl2west[it,:,:] = synTurbInp.thl2west[it-1,:,:]
                synTurbInp.qt2west[it,:,:] = synTurbInp.qt2west[it-1,:,:]
                synTurbInp.wthlwest[it,:,:] = synTurbInp.wthlwest[it-1,:,:]
                synTurbInp.wqtwest[it,:,:] = synTurbInp.wqtwest[it-1,:,:]
                synTurbInp.u2east[it,:,:] = synTurbInp.u2east[it-1,:,:]
                synTurbInp.v2east[it,:,:] = synTurbInp.v2east[it-1,:,:]
                synTurbInp.w2east[it,:,:] = synTurbInp.w2east[it-1,:,:]
                synTurbInp.uveast[it,:,:] = synTurbInp.uveast[it-1,:,:]
                synTurbInp.uweast[it,:,:] = synTurbInp.uweast[it-1,:,:]
                synTurbInp.vweast[it,:,:] = synTurbInp.vweast[it-1,:,:]
                synTurbInp.thl2east[it,:,:] = synTurbInp.thl2east[it-1,:,:]
                synTurbInp.qt2east[it,:,:] = synTurbInp.qt2east[it-1,:,:]
                synTurbInp.wthleast[it,:,:] = synTurbInp.wthleast[it-1,:,:]
                synTurbInp.wqteast[it,:,:] = synTurbInp.wqteast[it-1,:,:]
                synTurbInp.u2south[it,:,:] = synTurbInp.u2south[it-1,:,:]
                synTurbInp.v2south[it,:,:] = synTurbInp.v2south[it-1,:,:]
                synTurbInp.w2south[it,:,:] = synTurbInp.w2south[it-1,:,:]
                synTurbInp.uvsouth[it,:,:] = synTurbInp.uvsouth[it-1,:,:]
                synTurbInp.uwsouth[it,:,:] = synTurbInp.uwsouth[it-1,:,:]
                synTurbInp.vwsouth[it,:,:] = synTurbInp.vwsouth[it-1,:,:]
                synTurbInp.thl2south[it,:,:] = synTurbInp.thl2south[it-1,:,:]
                synTurbInp.qt2south[it,:,:] = synTurbInp.qt2south[it-1,:,:]
                synTurbInp.wthlsouth[it,:,:] = synTurbInp.wthlsouth[it-1,:,:]
                synTurbInp.wqtsouth[it,:,:] = synTurbInp.wqtsouth[it-1,:,:]
                synTurbInp.u2north[it,:,:] = synTurbInp.u2north[it-1,:,:]
                synTurbInp.v2north[it,:,:] = synTurbInp.v2north[it-1,:,:]
                synTurbInp.w2north[it,:,:] = synTurbInp.w2north[it-1,:,:]
                synTurbInp.uvnorth[it,:,:] = synTurbInp.uvnorth[it-1,:,:]
                synTurbInp.uwnorth[it,:,:] = synTurbInp.uwnorth[it-1,:,:]
                synTurbInp.vwnorth[it,:,:] = synTurbInp.vwnorth[it-1,:,:]
                synTurbInp.thl2north[it,:,:] = synTurbInp.thl2north[it-1,:,:]
                synTurbInp.qt2north[it,:,:] = synTurbInp.qt2north[it-1,:,:]
                synTurbInp.wthlnorth[it,:,:] = synTurbInp.wthlnorth[it-1,:,:]
                synTurbInp.wqtnorth[it,:,:] = synTurbInp.wqtnorth[it-1,:,:]
                synTurbInp.u2top[it,:,:] = 0.
                synTurbInp.v2top[it,:,:] = 0.
                synTurbInp.w2top[it,:,:] = 0.
                synTurbInp.uvtop[it,:,:] = 0.
                synTurbInp.uwtop[it,:,:] = 0.
                synTurbInp.vwtop[it,:,:] = 0.
                synTurbInp.thl2top[it,:,:] = 0.
                synTurbInp.qt2top[it,:,:] = 0.
                synTurbInp.wthltop[it,:,:] = 0.
                synTurbInp.wqttop[it,:,:] = 0.
        # Close netcdf file
        synTurbInp.exit()
        del synTurbInp
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")

    # -------------------- Create profiles ---------------------- #
    # Read initfields and average for prof.inp
    if lprof:
        nc = Dataset(pathWrite+'initfields.inp.'+str(iexpnr).zfill(3)+'.nc','r')
        uprof   = np.mean(nc.variables['u0'][:,:,:],axis=(1,2))
        vprof   = np.mean(nc.variables['v0'][:,:,:],axis=(1,2))
        thlprof = np.mean(nc.variables['thl0'][:,:,:],axis=(1,2))
        qtprof  = np.mean(nc.variables['qt0'][:,:,:],axis=(1,2))
        e12prof = np.mean(nc.variables['e120'][:,:,:],axis=(1,2))
        DALES.writeProfInp(grid.z,thlprof,qtprof,uprof,vprof,e12prof,iexpnr=iexpnr,path=pathWrite,description='Initial profiles')
        nc.close()
        ug = np.zeros(ktot)
        vg = np.zeros(ktot)
        wfls = np.zeros(ktot)
        dqtdxls = np.zeros(ktot)
        dqtdyls = np.zeros(ktot)
        dqtdtls = np.zeros(ktot)
        dthlrad = np.zeros(ktot)
        DALES.writeLscaleInp(grid.z,ug,vg,wfls,dqtdxls,dqtdyls,dqtdtls,dthlrad,iexpnr=iexpnr,path=pathWrite,description='Large scale forcing input')

    # ------------------------ Finalize ------------------------- #
    # Close Harmonie netcdf files
    nch_u.close()
    nch_v.close()
    nch_w.close()
    nch_T.close()
    nch_qv.close()
    nch_ql.close()
    nch_ps.close()
    nch_Ts.close()
    nch_qts.close()
    nch_cb.close()
    nch_wTs.close()
    nch_uwsc.close()
    nch_uwst.close()
    nch_vwsc.close()
    nch_vwst.close()

def calcPressure(ps):
    # Calculate pressure with model coefficients given in H43_65lev.txt
    coeff = np.loadtxt('H43_65lev.txt')
    ph = coeff[:,1,None,None]+(ps[None,:,:]*coeff[:,2,None,None])
    p = 0.5*(ph[1:,:,:]+ph[:-1,:,:])
    return p[::-1,:,:]

def interp3d(xLS,yLS,zLS,val,x,y,z):
    # 3D interpolation function
    fieldLES = np.zeros((z.size,y.size,x.size))
    for k in range(z.size):
        fieldLES[k,:,:] = interpTop(xLS,yLS,zLS,val,x,y,z[k])
    return fieldLES

def interpTop(xLS,yLS,zLS,val,x,y,ztop):
    fieldLES = np.zeros((y.size,x.size))
    valtemp = np.zeros((yLS.size,xLS.size))
    # Vertical interpolation to DALES height level
    for i in range(xLS.size):
        for j in range(yLS.size):
            kb = np.where(zLS[:,j,i] - ztop <= 0)[0][-1]
            kt = kb+1
            fkb = (zLS[kt,j,i]-ztop)/(zLS[kt,j,i]-zLS[kb,j,i])
            fkt = 1-fkb
            valtemp[j,i] = fkb*val[kb,j,i]+fkt*val[kt,j,i]
    # Horizontal interpolation
    for i in range(x.size):
        il   = np.where(xLS - x[i] <= 0)[0][-1]
        if abs(x[i]-xLS[-1])<10**-5 :
            ir = il
            fil = 1
            fir = 0
        else:
            ir   = il+1
            fil  = (xLS[ir]-x[i])/(xLS[ir]-xLS[il])
            fir  = 1-fil
        for j in range(y.size):
            jl   = np.where(yLS - y[j] <= 0)[0][-1]
            if abs(y[j]-yLS[-1])<10**-5 :
                jr = jl
                fjl = 1
                fjr = 0
            else:
                jr   = jl+1
                fjl  = (yLS[jr]-y[j])/(yLS[jr]-yLS[jl])
                fjr  = 1-fjl
            fieldLES[j,i] = fil*(fjl*valtemp[jl,il]+fjr*valtemp[jr,il])+fir*(fjl*valtemp[jl,ir]+fjr*valtemp[jr,ir])
    return fieldLES

def interpLateral(xLS,zLS,val,x,z):
    fieldLES = np.zeros((z.size,x.size))
    for i in range(x.size):
        for k in range(z.size):
            # Get horizontal factors
            il   = np.where(xLS - x[i] <= 0)[0][-1]
            if abs(x[i]-xLS[-1])<10**-5 :
                ir = il
                fil = 1
                fir = 0
            else:
                ir = il+1
                fil  = (xLS[ir]-x[i])/(xLS[ir]-xLS[il])
                fir  = 1-fil
            # Get vertical factors
            kbl  = np.where(zLS[:,il]-z[k] <= 0)[0][-1]
            ktl  = kbl+1
            fkbl = (zLS[ktl,il]-z[k])/(zLS[ktl,il]-zLS[kbl,il])
            fktl = 1-fkbl

            kbr  = np.where(zLS[:,ir]-z[k] <= 0)[0][-1]
            ktr  = kbr+1
            fkbr = (zLS[ktr,ir]-z[k])/(zLS[ktr,ir]-zLS[kbr,ir])
            fktr = 1-fkbr
            fieldLES[k,i] = fil*(fkbl*val[kbl,il]+fktl*val[ktl,il])+fir*(fkbr*val[kbr,ir]+fktr*val[ktr,ir])
    return fieldLES

if __name__ == '__main__':
    main()
