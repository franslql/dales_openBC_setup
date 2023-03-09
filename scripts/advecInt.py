# Advective interpolation scheme to artificially increase the temporal resolution of the HARMONIE output
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def main():
    # -------------------------- Input -------------------------- #
    pathData = '../harmonie/' # Path of HARMONIE data
    pathWrite= '../harmonie/advecInt/' # Path to write interpolated data
    dt       = 30   # Advection time step (s)
    twrite   = 120  # Write interval (s)
    it0      = 4*24 # Start time index of HARMONIE data
    Nt       = 25   # Number of time steps in original HARMONIE data

    # ------------------- Open Harmonie data -------------------- #
    ncu   = Dataset(f"{pathData}ua_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    x     = ncu.variables['x'][:]; dx = x[1]-x[0]; Nx = len(x)
    y     = ncu.variables['y'][:]; dy = y[1]-y[0]; Ny = len(y)
    Nz    = ncu.dimensions['lev'].size
    time  = ncu.variables['time'][:]; time0 = time[0]; time = (time-time0)*3600*24; dtharm = time[1]-time[0]
    latS  = ncu.variables['lat'][:,:]
    lonS  = ncu.variables['lon'][:,:]
    ncv   = Dataset(f"{pathData}va_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncw   = Dataset(f"{pathData}wa_Slev_fp_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncT   = Dataset(f"{pathData}ta_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncqv  = Dataset(f"{pathData}hus_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncql  = Dataset(f"{pathData}clw_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncps  = Dataset(f"{pathData}ps_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc", 'r')
    xL    = ncps.variables['x'][:]; NxL = len(xL)
    yL    = ncps.variables['y'][:]; NyL = len(yL)
    latL  = ncps.variables['lat'][:,:]
    lonL  = ncps.variables['lon'][:,:]
    i0L   = np.unravel_index(np.argmin((latL-latS[0,0])**2+(lonL-lonS[0,0])**2),np.shape(latL))
    i1L   = i0L+np.array([Ny,Nx])
    ncTs  = Dataset(f"{pathData}tas_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncqvs = Dataset(f"{pathData}huss_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    nccb  = Dataset(f"{pathData}cb_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc",'r')
    ncwTs = Dataset(f"{pathData}hfss_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    ncuwsc= Dataset(f"{pathData}uflx_conv_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    ncuwst= Dataset(f"{pathData}uflx_turb_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    ncvwsc= Dataset(f"{pathData}vflx_conv_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')
    ncvwst= Dataset(f"{pathData}vflx_turb_Slev_his_EUREC4Acircle_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002110000.nc",'r')

    # -------- Create files for interpolated variables ---------- #
    ncoutu=createFile(ncu,'ua',pathWrite,lflux=False)
    ncoutv=createFile(ncv,'va',pathWrite,lflux=False)
    ncoutw=createFile(ncw,'wa',pathWrite,lflux=False)
    ncoutT=createFile(ncT,'ta',pathWrite,lflux=False)
    ncoutqv=createFile(ncqv,'hus',pathWrite,lflux=False)
    ncoutql=createFile(ncql,'clw',pathWrite,lflux=False)
    ncoutps=createFile(ncps,'ps',pathWrite,lflux=False)
    ncoutTs=createFile(ncTs,'tas',pathWrite,lflux=False)
    ncoutqvs=createFile(ncqvs,'huss',pathWrite,lflux=False)
    ncoutcb=createFile(nccb,'cb',pathWrite,lflux=False)
    ncoutwTs=createFile(ncwTs,'hfss',pathWrite,lflux=True)
    ncoutuwsc=createFile(ncuwsc,'uflx_conv',pathWrite,lflux=True)
    ncoutuwst=createFile(ncuwst,'uflx_turb',pathWrite,lflux=True)
    ncoutvwsc=createFile(ncvwsc,'vflx_conv',pathWrite,lflux=True)
    ncoutvwst=createFile(ncvwst,'vflx_turb',pathWrite,lflux=True)

    # ---------------- Advective interpolation ------------------ #
    # Loop over HARMONIE Data points
    for it in range(Nt-1):
        # Preallocate fields
        tint   = np.zeros((int(dtharm/twrite)+1))
        uint   = np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        vint   = np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        wint   = np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        Tint   = np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        qlint  = np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        qvint  = np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        psint  = np.zeros((int(dtharm/twrite)+1,NyL,NxL))
        Tsint  = np.zeros((int(dtharm/twrite)+1,NyL,NxL))
        qvsint = np.zeros((int(dtharm/twrite)+1,NyL,NxL))
        cbint  = np.zeros((int(dtharm/twrite)+1,NyL,NxL))
        wTsint = np.zeros((int(dtharm/twrite)+1,NyL,NxL))
        uwscint= np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        uwstint= np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        vwscint= np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        vwstint= np.zeros((int(dtharm/twrite)+1,Nz,Ny,Nx))
        # Read harmonie fields
        u    = ncu.variables['ua'][it0+it:it0+it+2,:,:,:]
        v    = ncv.variables['va'][it0+it:it0+it+2,:,:,:]
        w    = ncw.variables['wa'][it0+it:it0+it+2,:,:,:]
        T    = ncT.variables['ta'][it0+it:it0+it+2,:,:,:]
        qv   = ncqv.variables['hus'][it0+it:it0+it+2,:,:,:]
        ql   = ncql.variables['clw'][it0+it:it0+it+2,:,:,:]
        ps   = ncps.variables['ps'][it0+it:it0+it+2,:,:]
        Ts   = ncTs.variables['tas'][it0+it:it0+it+2,:,:]
        qvs  = ncqvs.variables['huss'][it0+it:it0+it+2,:,:]
        cb   = nccb.variables['cb'][it0+it:it0+it+2,:,:]
        wTs  = ncwTs.variables['hfss'][it0+it:it0+it+2,:,:]
        uwsc = ncuwsc.variables['uflx_conv'][it0+it:it0+it+2,:,:,:]
        uwst = ncuwst.variables['uflx_turb'][it0+it:it0+it+2,:,:,:]
        vwsc = ncvwsc.variables['vflx_conv'][it0+it:it0+it+2,:,:,:]
        vwst = ncvwst.variables['vflx_turb'][it0+it:it0+it+2,:,:,:]
        # Forward integration
        # Preallocate temporary fields
        T0  = np.zeros((Nz+2,Ny+2,Nx+2))
        ql0 = np.zeros((Nz+2,Ny+2,Nx+2))
        qv0 = np.zeros((Nz+2,Ny+2,Nx+2))
        dp  = np.zeros((Nz+1,Ny+2,Nx+2))
        # Set initial conditions
        t = time[it0+it]
        tnext = t+twrite
        idx = 0
        idxWrite = np.arange(int(dtharm/twrite)+1)+it*int(dtharm/twrite)
        T0[1:-1,1:-1,1:-1]  = T[0,:,:,:]
        ql0[1:-1,1:-1,1:-1] = ql[0,:,:,:]
        qv0[1:-1,1:-1,1:-1] = qv[0,:,:,:]
        # Set interpolation boundaries to input fields
        uint[0,:,:,:]     = u[0,:,:,:]
        uint[-1,:,:,:]    = u[1,:,:,:]
        vint[0,:,:,:]     = v[0,:,:,:]
        vint[-1,:,:,:]    = v[1,:,:,:]
        wint[0,:,:,:]     = w[0,:,:,:]
        wint[-1,:,:,:]    = w[1,:,:,:]
        Tint[0,:,:,:]     = T[0,:,:,:]
        Tint[-1,:,:,:]    = T[1,:,:,:]
        qlint[0,:,:,:]    = ql[0,:,:,:]
        qlint[-1,:,:,:]   = ql[1,:,:,:]
        qvint[0,:,:,:]    = qv[0,:,:,:]
        qvint[-1,:,:,:]   = qv[1,:,:,:]
        psint[0,:,:]      = ps[0,:,:]
        psint[-1,:,:]     = ps[1,:,:]
        Tsint[0,:,:]      = Ts[0,:,:]
        Tsint[-1,:,:]     = Ts[1,:,:]
        qvsint[0,:,:]     = qvs[0,:,:]
        qvsint[-1,:,:]    = qvs[1,:,:]
        cbint[0,:,:]      = cb[0,:,:]
        cbint[-1,:,:]     = cb[1,:,:]
        wTsint[0,:,:]     = wTs[0,:,:]
        wTsint[-1,:,:]    = wTs[1,:,:]
        uwscint[0,:,:,:]  = uwsc[0,:,:,:]
        uwscint[-1,:,:,:] = uwsc[1,:,:,:]
        uwstint[0,:,:,:]  = uwst[0,:,:,:]
        uwstint[-1,:,:,:] = uwst[1,:,:,:]
        vwscint[0,:,:,:]  = vwsc[0,:,:,:]
        vwscint[-1,:,:,:] = vwsc[1,:,:,:]
        vwstint[0,:,:,:]  = vwst[0,:,:,:]
        vwstint[-1,:,:,:] = vwst[1,:,:,:]
        tint[0]           = time[it0+it]
        tint[-1]          = time[it0+it+1]

        while t<time[it0+it+1]-twrite: # Start forward advection integration
            # Get surface values at correct grid and time
            am,ap = calcLinearCoeff(t,time[it0+it],time[it0+it+1])
            psS  = am*ps[0,i0L[0]:i1L[0],i0L[1]:i1L[1]]+ap*ps[1,i0L[0]:i1L[0],i0L[1]:i1L[1]]
            TsS  = am*Ts[0,i0L[0]:i1L[0],i0L[1]:i1L[1]]+ap*Ts[1,i0L[0]:i1L[0],i0L[1]:i1L[1]]
            qvsS = am*qvs[0,i0L[0]:i1L[0],i0L[1]:i1L[1]]+ap*qvs[1,i0L[0]:i1L[0],i0L[1]:i1L[1]]
            # Get pressure levels
            p  = calcPressure(psS)
            dp[1:-1,1:-1,1:-1] = p[1:,:,:]-p[:-1,:,:]
            # Set boundaries (homogeneous neumann)
            T0  = setBoundaries(T0)
            ql0 = setBoundaries(ql0)
            qv0 = setBoundaries(qv0)
            dp  = setBoundaries(dp)
            # Forward in time upwind discretisation scheme
            up  = np.maximum(0.,am*u[0,:,:,:]+ap*u[1,:,:,:])
            um  = np.minimum(0.,am*u[0,:,:,:]+ap*u[1,:,:,:])
            vp  = np.maximum(0.,am*v[0,:,:,:]+ap*v[1,:,:,:])
            vm  = np.minimum(0.,am*v[0,:,:,:]+ap*v[1,:,:,:])
            wp  = np.maximum(0.,am*w[0,:,:,:]+ap*w[1,:,:,:])
            wm  = np.minimum(0.,am*w[0,:,:,:]+ap*w[1,:,:,:])
            T0  = doAdvection(T0,um,up,vm,vp,wm,wp,dx,dy,dp,dt)
            ql0 = doAdvection(ql0,um,up,vm,vp,wm,wp,dx,dy,dp,dt)
            qv0 = doAdvection(qv0,um,up,vm,vp,wm,wp,dx,dy,dp,dt)
            # Update time
            t = t + dt
            if(t==tnext): # Write fields to variables
                idx = idx+1
                tint[idx] = t
                am,ap = calcLinearCoeff(t,time[it0+it],time[it0+it+1])
                # Advective interpolation for non-velocity fields
                Tint[idx,:,:,:]  = Tint[idx,:,:,:]  + am*T0[1:-1,1:-1,1:-1]
                qlint[idx,:,:,:] = qlint[idx,:,:,:] + am*ql0[1:-1,1:-1,1:-1]
                qvint[idx,:,:,:] = qvint[idx,:,:,:] + am*qv0[1:-1,1:-1,1:-1]
                # Linear interpolation for velocity fields and surface fields
                uint[idx,:,:,:]    = am*u[0,:,:,:]    + ap*u[1,:,:,:]
                vint[idx,:,:,:]    = am*v[0,:,:,:]    + ap*v[1,:,:,:]
                wint[idx,:,:,:]    = am*w[0,:,:,:]    + ap*w[1,:,:,:]
                psint[idx,:,:]     = am*ps[0,:,:]     + ap*ps[1,:,:]
                Tsint[idx,:,:]     = am*Ts[0,:,:]     + ap*Ts[1,:,:]
                qvsint[idx,:,:]    = am*qvs[0,:,:]    + ap*qvs[1,:,:]
                cbint[idx,:,:]     = am*cb[0,:,:]     + ap*cb[1,:,:]
                wTsint[idx,:,:]    = am*wTs[0,:,:]    + ap*wTs[1,:,:]
                uwscint[idx,:,:,:] = am*uwsc[0,:,:,:] + ap*uwsc[1,:,:,:]
                uwstint[idx,:,:,:] = am*uwst[0,:,:,:] + ap*uwst[1,:,:,:]
                vwscint[idx,:,:,:] = am*vwsc[0,:,:,:] + ap*vwsc[1,:,:,:]
                vwstint[idx,:,:,:] = am*vwst[0,:,:,:] + ap*vwst[1,:,:,:]
                tnext = tnext+twrite

        # Backward integration
        # Preallocate fields
        T0 = np.zeros((Nz+2,Ny+2,Nx+2))
        ql0 = np.zeros((Nz+2,Ny+2,Nx+2))
        qv0 = np.zeros((Nz+2,Ny+2,Nx+2))
        # Set initial conditions
        t = time[it0+it+1]
        tnext = t-twrite
        idx = int(dtharm/twrite)
        T0[1:-1,1:-1,1:-1]  = T[1,:,:,:]
        ql0[1:-1,1:-1,1:-1] = ql[1,:,:,:]
        qv0[1:-1,1:-1,1:-1] = qv[1,:,:,:]
        while t>time[it0+it]+twrite: # Start backward advection integration
            # Get surface values at correct grid and time
            am,ap = calcLinearCoeff(t,time[it0+it],time[it0+it+1])
            psS  = am*ps[0,i0L[0]:i1L[0],i0L[1]:i1L[1]]+ap*ps[1,i0L[0]:i1L[0],i0L[1]:i1L[1]]
            TsS  = am*Ts[0,i0L[0]:i1L[0],i0L[1]:i1L[1]]+ap*Ts[1,i0L[0]:i1L[0],i0L[1]:i1L[1]]
            qvsS = am*qvs[0,i0L[0]:i1L[0],i0L[1]:i1L[1]]+ap*qvs[1,i0L[0]:i1L[0],i0L[1]:i1L[1]]
            # Get pressure levels
            p  = calcPressure(psS)
            dp[1:-1,1:-1,1:-1] = p[1:,:,:]-p[:-1,:,:]
            # Set boundaries (homogeneous neumann)
            T0  = setBoundaries(T0)
            ql0 = setBoundaries(ql0)
            qv0 = setBoundaries(qv0)
            # Backward in time upwind discretisation scheme
            am,ap = calcLinearCoeff(t,time[it0+it],time[it0+it+1])
            up  = np.maximum(0.,-am*u[0,:,:,:]-ap*u[1,:,:,:])
            um  = np.minimum(0.,-am*u[0,:,:,:]-ap*u[1,:,:,:])
            vp  = np.maximum(0.,-am*v[0,:,:,:]-ap*v[1,:,:,:])
            vm  = np.minimum(0.,-am*v[0,:,:,:]-ap*v[1,:,:,:])
            wp  = np.maximum(0.,-am*w[0,:,:,:]-ap*w[1,:,:,:])
            wm  = np.minimum(0.,-am*w[0,:,:,:]-ap*w[1,:,:,:])
            T0  = doAdvection(T0,um,up,vm,vp,wm,wp,dx,dy,dp,dt)
            ql0 = doAdvection(ql0,um,up,vm,vp,wm,wp,dx,dy,dp,dt)
            qv0 = doAdvection(qv0,um,up,vm,vp,wm,wp,dx,dy,dp,dt)
            # Update time
            t = t - dt
            if(t==tnext): # Write fields to variables
                idx = idx-1
                am,ap = calcLinearCoeff(t,time[it0+it],time[it0+it+1])
                # Advective interpolation for non-velocity fields
                Tint[idx,:,:,:]  = Tint[idx,:,:,:]+ap*T0[1:-1,1:-1,1:-1]
                qlint[idx,:,:,:] = qlint[idx,:,:,:]+ap*ql0[1:-1,1:-1,1:-1]
                qvint[idx,:,:,:] = qvint[idx,:,:,:]+ap*qv0[1:-1,1:-1,1:-1]
                tnext = tnext-twrite
        # Save interpolated fields
        writeData(ncoutT,'ta',Tint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutql,'clw',qlint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutqv,'hus',qvint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutu,'ua',uint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutv,'va',vint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutw,'wa',wint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutps,'ps',psint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutTs,'tas',Tsint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutqvs,'huss',qvsint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutcb,'cb',cbint,tint/3600/24+time0,idxWrite,lflux=False)
        writeData(ncoutwTs,'hfss',wTsint,tint/3600/24+time0,idxWrite,lflux=True)
        writeData(ncoutuwsc,'uflx_conv',uwscint,tint/3600/24+time0,idxWrite,lflux=True)
        writeData(ncoutuwst,'uflx_turb',uwstint,tint/3600/24+time0,idxWrite,lflux=True)
        writeData(ncoutvwsc,'vflx_conv',vwscint,tint/3600/24+time0,idxWrite,lflux=True)
        writeData(ncoutvwst,'vflx_turb',vwstint,tint/3600/24+time0,idxWrite,lflux=True)
    ncoutT.close()
    ncoutql.close()
    ncoutqv.close()
    ncoutu.close()
    ncoutv.close()
    ncoutw.close()
    ncoutps.close()
    ncoutTs.close()
    ncoutqvs.close()
    ncoutcb.close()
    ncoutwTs.close()
    ncoutuwsc.close()
    ncoutuwst.close()
    ncoutvwsc.close()
    ncoutvwst.close()
    ncu.close()
    ncv.close()
    ncw.close()
    ncT.close()
    ncqv.close()
    ncql.close()
    ncps.close()
    nTs.close()
    nqvs.close()
    nccb.close()
    ncwTs.close()
    ncuwsc.close()
    ncuwst.close()
    ncvwsc.close()
    ncvwst.close()

def calcLinearCoeff(t,tm,tp):
    # Calculate coefficients for linear interpolation
    am = (tp-t)/(tp-tm)
    ap = (t-tm)/(tp-tm)
    return am,ap
def setBoundaries(var):
    # Set homogeneous neumann boundary conditions
    var[:,0,:]  = var[:,1,:]
    var[:,-1,:] = var[:,-2,:]
    var[:,:,0]  = var[:,:,1]
    var[:,:,-1] = var[:,:,-2]
    var[0,:,:]  = var[1,:,:]
    var[-1,:,:] = var[-2,:,:]
    return var
def doAdvection(var,um,up,vm,vp,wm,wp,dx,dy,dp,dt):
    # Do advection
    var[1:-1,1:-1,1:-1] = (1-up*dt/dx+um*dt/dx-vp*dt/dy+vm*dt/dy- \
                             wp*dt/dp[:-1,1:-1,1:-1]+wm*dt/dp[1:,1:-1,1:-1])*var[1:-1,1:-1,1:-1] + \
                          (  up*dt/dx)*var[1:-1,1:-1,0:-2] + \
                          ( -um*dt/dx)*var[1:-1,1:-1,2:] + \
                          (  vp*dt/dy)*var[1:-1,0:-2,1:-1] + \
                          ( -vm*dt/dy)*var[1:-1,2:,1:-1] + \
                          (  wp*dt/dp[:-1,1:-1,1:-1])*var[0:-2,1:-1,1:-1] + \
                          ( -wm*dt/dp[ 1:,1:-1,1:-1])*var[2:,1:-1,1:-1]
    return var
def calcPressure(ps):
    # Calculate pressure with model coefficients given in H43_65lev.txt
    coeff = np.loadtxt('H43_65lev.txt')
    ph = coeff[:,1,None,None]+(ps[None,:,:]*coeff[:,2,None,None])
    p = 0.5*(ph[1:,:,:]+ph[:-1,:,:])
    return p[::-1,:,:]

def createFile(ncin,varName,pathWrite,lflux=False):
    # Create files for interpolated fields
    excludeVar = [varName, 'time']
    if(lflux):
        excludeVar = [varName, 'time', 'time_bnds']
        lowerbnd   = ncin.variables['time_bnds'][0,0]
    filename = f"{varName}_int.nc"
    ncout = Dataset(f"{pathWrite}{filename}",'w')
    ncout.setncatts(ncin.__dict__) # copy global attributes all at once via dictionary
    for name, dimension in ncin.dimensions.items(): # copy dimensions
        ncout.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
    for name, variable in ncin.variables.items(): # copy all file data except for the excluded
        temp = ncout.createVariable(name, variable.datatype, variable.dimensions)
        if(name not in excludeVar): ncout[name][:] = ncin[name][:]
        ncout[name].setncatts(ncin[name].__dict__) # copy variable attributes all at once via dictionary
    if(lflux):
        ncout['time_bnds'][:,0] = lowerbnd
    return ncout

def writeData(ncout,varName,varint,tint,idx,lflux=False):
    # Write interpolated fields
    ncout['time'][idx] = tint[:]
    ncout[varName][idx] = varint[:]
    if(lflux):
        ncout['time_bnds'][idx,1] = tint[:]

if(__name__ == '__main__'):
    main()
