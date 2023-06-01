# Create input for a DALES simulation with open boundary conditions from coarser DALES slab output
import numpy as np
import scipy as sc
from scipy import interpolate as scInt
from netCDF4 import Dataset
from datetime import datetime
import funcDALES as DALES
import os

def main():
    # -------------------------- Input -------------------------- #
    # Coarse simulation
    iexpnr_coarse = 0 # Experiment number
    path_coarse  = "../simulations/512x384/coarse/"# Path coarse simulation data
    nsv = 2 # Number of scalar fields
    # Fine simulation
    iexpnr_fine   = 1 # Experiment number
    path_fine = "../simulations/512x384/fine/"# Path for input fine simulation
    x0 = 512*2500/4/4 # x-coordinate lower left corner fine simulation in coordinate system of coarse simulation
    y0 = 384*2500/4/4 # y-coordinate lower left corner fine simulation in coordinate system of coarse simulation
    itot = 512           # Number of x-grid points
    jtot = 384           # Number of y-grid points
    ktot = 150           # Number of z-grid points
    xsize = itot*2500/16 # Size in x-direction (multiple of coarse DALES resolution)
    ysize = jtot*2500/16 # Size in y-direction (multiple of coarse DALES resolution)
    stretched = True     # Stretched grid True/False
    dz = 20.             # Stretched grid = False
    dz0 = 20.            # Stretched grid = True, resolution first height level
    alpha = 0.012        # Stretched grid = True, growth factor height levels
    # Files to create
    lboundary = True    # Boundary input required True/False
    linithetero = True  # Heterogeneous input fields required True/False
    lsynturb  = False   # Synthetic turbulence input required True/False (not supported yet)

    # ------------------ Construct DALES grid ------------------- #
    grid = DALES.Grid(xsize,ysize,itot,jtot,ktot,dz,dz0,alpha,stretched)

    # ------------------ Create boundary input ------------------ #
    if(lboundary):
        start_time = datetime.now()
        print("Start creation of openboundaries.inp."+str(iexpnr_fine).zfill(3)+".nc")
        # Create netcdf file
        boundaryInp = DALES.OpenBoundariesInp(grid, iexpnr=iexpnr_fine, path=f"{path_fine}/input/",nsv=nsv)
        # Get initial lateral boundary input from initfields coarse simulation
        nc = Dataset(f"{path_coarse}input/initfields.inp.{iexpnr_coarse:03}.nc","r")
        xm_coarse  = nc.variables['xm'][:]
        xt_coarse  = nc.variables['xt'][:]
        ym_coarse  = nc.variables['ym'][:]
        yt_coarse  = nc.variables['yt'][:]
        zt_coarse  = nc.variables['zt'][:]
        zm_coarse  = nc.variables['zm'][:]
        # Get corner indices fine domain in coarse domain
        ix0 = np.argwhere(xm_coarse-x0==0)[0][0]
        iy0 = np.argwhere(ym_coarse-y0==0)[0][0]
        ixe = np.argwhere(xm_coarse-(x0+grid.xh[-1])==0)[0][0]
        iye = np.argwhere(ym_coarse-(y0+grid.yh[-1])==0)[0][0]
        # Interpolate and write initial boundary fields
        boundaryInp.time[0]=0.
        # West boundary
        f = scInt.interp2d(yt_coarse,zt_coarse,0.5*(nc.variables['thl0'][:,:,ix0]+nc.variables['thl0'][:,:,ix0-1]),bounds_error=False,kind='linear')
        boundaryInp.thlwest[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zt_coarse,0.5*(nc.variables['qt0'][:,:,ix0]+nc.variables['qt0'][:,:,ix0-1]),bounds_error=False,kind='linear')
        boundaryInp.qtwest[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zt_coarse,0.5*(nc.variables['e120'][:,:,ix0]+nc.variables['e120'][:,:,ix0-1]),bounds_error=False,kind='linear')
        boundaryInp.e12west[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zt_coarse,nc.variables['u0'][:,:,ix0],bounds_error=False,kind='linear')
        boundaryInp.uwest[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(ym_coarse,zt_coarse,0.5*(nc.variables['v0'][:,:,ix0]+nc.variables['v0'][:,:,ix0-1]),bounds_error=False,kind='linear')
        boundaryInp.vwest[0,:,:]  = f(grid.yh+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zm_coarse,0.5*(nc.variables['w0'][:,:,ix0]+nc.variables['w0'][:,:,ix0-1]),bounds_error=False,kind='linear')
        boundaryInp.wwest[0,:,:]  = f(grid.y+y0,grid.zh)
        boundaryInp.svwest[:,0,:,:] = 0.
        # East boundary
        f = scInt.interp2d(yt_coarse,zt_coarse,0.5*(nc.variables['thl0'][:,:,ixe]+nc.variables['thl0'][:,:,ixe-1]),bounds_error=False,kind='linear')
        boundaryInp.thleast[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zt_coarse,0.5*(nc.variables['qt0'][:,:,ixe]+nc.variables['qt0'][:,:,ixe-1]),bounds_error=False,kind='linear')
        boundaryInp.qteast[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zt_coarse,0.5*(nc.variables['e120'][:,:,ixe]+nc.variables['e120'][:,:,ixe-1]),bounds_error=False,kind='linear')
        boundaryInp.e12east[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zt_coarse,nc.variables['u0'][:,:,ixe],bounds_error=False,kind='linear')
        boundaryInp.ueast[0,:,:]  = f(grid.y+y0,grid.z)
        f = scInt.interp2d(ym_coarse,zt_coarse,0.5*(nc.variables['v0'][:,:,ixe]+nc.variables['v0'][:,:,ixe-1]),bounds_error=False,kind='linear')
        boundaryInp.veast[0,:,:]  = f(grid.yh+y0,grid.z)
        f = scInt.interp2d(yt_coarse,zm_coarse,0.5*(nc.variables['w0'][:,:,ixe]+nc.variables['w0'][:,:,ixe-1]),bounds_error=False,kind='linear')
        boundaryInp.weast[0,:,:]  = f(grid.y+y0,grid.zh)
        boundaryInp.sveast[:,0,:,:] = 0.
        # South boundary
        f = scInt.interp2d(xt_coarse,zt_coarse,0.5*(nc.variables['thl0'][:,iy0,:]+nc.variables['thl0'][:,iy0-1,:]),bounds_error=False,kind='linear')
        boundaryInp.thlsouth[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zt_coarse,0.5*(nc.variables['qt0'][:,iy0,:]+nc.variables['qt0'][:,iy0-1,:]),bounds_error=False,kind='linear')
        boundaryInp.qtsouth[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zt_coarse,0.5*(nc.variables['e120'][:,iy0,:]+nc.variables['e120'][:,iy0-1,:]),bounds_error=False,kind='linear')
        boundaryInp.e12south[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xm_coarse,zt_coarse,0.5*(nc.variables['u0'][:,iy0,:]+nc.variables['u0'][:,iy0-1,:]),bounds_error=False,kind='linear')
        boundaryInp.usouth[0,:,:]  = f(grid.xh+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zt_coarse,nc.variables['v0'][:,iy0,:],bounds_error=False,kind='linear')
        boundaryInp.vsouth[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zm_coarse,0.5*(nc.variables['w0'][:,iy0,:]+nc.variables['w0'][:,iy0-1,:]),bounds_error=False,kind='linear')
        boundaryInp.wsouth[0,:,:]  = f(grid.x+x0,grid.zh)
        boundaryInp.svsouth[:,0,:,:] = 0.
        # North boundary
        f = scInt.interp2d(xt_coarse,zt_coarse,0.5*(nc.variables['thl0'][:,iye,:]+nc.variables['thl0'][:,iye-1,:]),bounds_error=False,kind='linear')
        boundaryInp.thlnorth[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zt_coarse,0.5*(nc.variables['qt0'][:,iye,:]+nc.variables['qt0'][:,iye-1,:]),bounds_error=False,kind='linear')
        boundaryInp.qtnorth[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zt_coarse,0.5*(nc.variables['e120'][:,iye,:]+nc.variables['e120'][:,iye-1,:]),bounds_error=False,kind='linear')
        boundaryInp.e12north[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xm_coarse,zt_coarse,0.5*(nc.variables['u0'][:,iye,:]+nc.variables['u0'][:,iye-1,:]),bounds_error=False,kind='linear')
        boundaryInp.unorth[0,:,:]  = f(grid.xh+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zt_coarse,nc.variables['v0'][:,iye,:],bounds_error=False,kind='linear')
        boundaryInp.vnorth[0,:,:]  = f(grid.x+x0,grid.z)
        f = scInt.interp2d(xt_coarse,zm_coarse,0.5*(nc.variables['w0'][:,iye,:]+nc.variables['w0'][:,iye-1,:]),bounds_error=False,kind='linear')
        boundaryInp.wnorth[0,:,:]  = f(grid.x+x0,grid.zh)
        boundaryInp.svnorth[:,0,:,:] = 0.
        nc.close()
        # Top boundary input from openboundaries.inp coarse simulation
        nc = Dataset(f"{path_coarse}input/openboundaries.inp.{iexpnr_coarse:03}.nc","r")
        f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['thltop'][0,:,:],bounds_error=False,kind='linear')
        boundaryInp.thltop[0,:,:]  = f(grid.x+x0,grid.y+y0)
        f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['qttop'][0,:,:],bounds_error=False,kind='linear')
        boundaryInp.qttop[0,:,:]   = f(grid.x+x0,grid.y+y0)
        f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['e12top'][0,:,:],bounds_error=False,kind='linear')
        boundaryInp.e12top[0,:,:]  = f(grid.x+x0,grid.y+y0)
        f = scInt.interp2d(xm_coarse,yt_coarse,nc.variables['utop'][0,:,:],bounds_error=False,kind='linear')
        boundaryInp.utop[0,:,:]    = f(grid.xh+x0,grid.y+y0)
        f = scInt.interp2d(xt_coarse,ym_coarse,nc.variables['vtop'][0,:,:],bounds_error=False,kind='linear')
        boundaryInp.vtop[0,:,:]    = f(grid.x+x0,grid.yh+y0)
        f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['wtop'][0,:,:],bounds_error=False,kind='linear')
        boundaryInp.wtop[0,:,:]    = f(grid.x+x0,grid.y+y0)
        boundaryInp.svtop[:,0,:,:] = 0.
        nc.close()
        # Write rest boundary input from slab output coarse simulation
        # West boundary
        nc_thl   = Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/thlyz.{ix0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_qt   = Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/qtyz.{ix0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_e12   = Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/e120yz.{ix0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_u   = Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/uyz.{ix0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_v   = Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/vyz.{ix0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_w   = Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/wyz.{ix0+2:04}.{iexpnr_coarse:03}.nc","r")
        if(nsv>0): nc_sv = [Dataset(f"{path_coarse}output/crossyz/{ix0+2:04}/sv{isv+1:03}.{ix0+2:04}.{iexpnr_coarse:03}.nc","r") for isv in range(nsv)]
        time = nc_thl.variables['time'][:]; time = time-time[0]+(time[1]-time[0]); nt = len(time)
        for it in range(nt): # Interpolate and write per time step
            boundaryInp.time[it+1] = time[it]
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_thl.variables['thlyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.thlwest[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_qt.variables['qtyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.qtwest[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_e12.variables['e120yz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.e12west[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_u.variables['uyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.uwest[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(ym_coarse[:-1],zt_coarse,nc_v.variables['vyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.vwest[it+1,:,:] = f(grid.yh+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zm_coarse[:-1],nc_w.variables['wyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.wwest[it+1,:,:] = f(grid.y+y0,grid.zh)
            if(nsv>0):
                for isv in range(nsv):
                    f = scInt.interp2d(yt_coarse,zt_coarse,nc_sv[isv].variables[f"sv{isv+1:03}"][it,:,:],bounds_error=False,kind='linear')
                    boundaryInp.svwest[isv,it+1,:,:] = f(grid.y+y0,grid.z)
        nc_thl.close()
        nc_qt.close()
        nc_e12.close()
        nc_u.close()
        nc_v.close()
        nc_w.close()
        if(nsv>0): [nc_sv[isv].close() for isv in range(nsv)]
        print('finished west')
        # East boundary
        nc_thl   = Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/thlyz.{ixe+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_qt   = Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/qtyz.{ixe+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_e12   = Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/e120yz.{ixe+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_u   = Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/uyz.{ixe+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_v   = Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/vyz.{ixe+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_w   = Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/wyz.{ixe+2:04}.{iexpnr_coarse:03}.nc","r")
        if(nsv>0): nc_sv = [Dataset(f"{path_coarse}output/crossyz/{ixe+2:04}/sv{isv+1:03}.{ixe+2:04}.{iexpnr_coarse:03}.nc","r") for isv in range(nsv)]
        for it in range(nt): # Interpolate and write per time step
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_thl.variables['thlyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.thleast[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_qt.variables['qtyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.qteast[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_e12.variables['e120yz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.e12east[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zt_coarse,nc_u.variables['uyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.ueast[it+1,:,:] = f(grid.y+y0,grid.z)
            f = scInt.interp2d(ym_coarse[:-1],zt_coarse,nc_v.variables['vyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.veast[it+1,:,:] = f(grid.yh+y0,grid.z)
            f = scInt.interp2d(yt_coarse,zm_coarse[:-1],nc_w.variables['wyz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.weast[it+1,:,:] = f(grid.y+y0,grid.zh)
            if(nsv>0):
                for isv in range(nsv):
                    f = scInt.interp2d(yt_coarse,zt_coarse,nc_sv[isv].variables[f"sv{isv+1:03}"][it,:,:],bounds_error=False,kind='linear')
                    boundaryInp.sveast[isv,it+1,:,:] = f(grid.y+y0,grid.z)
        nc_thl.close()
        nc_qt.close()
        nc_e12.close()
        nc_u.close()
        nc_v.close()
        nc_w.close()
        if(nsv>0): [nc_sv[isv].close() for isv in range(nsv)]
        print('finished east')
        # South boundary
        nc_thl   = Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/thlxz.{iy0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_qt   = Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/qtxz.{iy0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_e12   = Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/e120xz.{iy0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_u   = Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/uxz.{iy0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_v   = Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/vxz.{iy0+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_w   = Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/wxz.{iy0+2:04}.{iexpnr_coarse:03}.nc","r")
        if(nsv>0): nc_sv = [Dataset(f"{path_coarse}output/crossxz/{iy0+2:04}/sv{isv+1:03}.{iy0+2:04}.{iexpnr_coarse:03}.nc","r") for isv in range(nsv)]
        for it in range(nt): # Interpolate and write per time step
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_thl.variables['thlxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.thlsouth[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_qt.variables['qtxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.qtsouth[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_e12.variables['e120xz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.e12south[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xm_coarse[:-1],zt_coarse,nc_u.variables['uxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.usouth[it+1,:,:] = f(grid.xh+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_v.variables['vxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.vsouth[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zm_coarse[:-1],nc_w.variables['wxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.wsouth[it+1,:,:] = f(grid.x+x0,grid.zh)
            if(nsv>0):
                for isv in range(nsv):
                    f = scInt.interp2d(xt_coarse,zt_coarse,nc_sv[isv].variables[f"sv{isv+1:03}"][it,:,:],bounds_error=False,kind='linear')
                    boundaryInp.svsouth[isv,it+1,:,:] = f(grid.x+x0,grid.z)
        nc_thl.close()
        nc_qt.close()
        nc_e12.close()
        nc_u.close()
        nc_v.close()
        nc_w.close()
        if(nsv>0): [nc_sv[isv].close() for isv in range(nsv)]
        print('finished south')
        # North boundary
        nc_thl   = Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/thlxz.{iye+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_qt   = Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/qtxz.{iye+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_e12   = Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/e120xz.{iye+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_u   = Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/uxz.{iye+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_v   = Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/vxz.{iye+2:04}.{iexpnr_coarse:03}.nc","r")
        nc_w   = Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/wxz.{iye+2:04}.{iexpnr_coarse:03}.nc","r")
        if(nsv>0): nc_sv = [Dataset(f"{path_coarse}output/crossxz/{iye+2:04}/sv{isv+1:03}.{iye+2:04}.{iexpnr_coarse:03}.nc","r") for isv in range(nsv)]
        for it in range(nt): # Interpolate and write per time step
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_thl.variables['thlxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.thlnorth[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_qt.variables['qtxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.qtnorth[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_e12.variables['e120xz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.e12north[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xm_coarse[:-1],zt_coarse,nc_u.variables['uxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.unorth[it+1,:,:] = f(grid.xh+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zt_coarse,nc_v.variables['vxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.vnorth[it+1,:,:] = f(grid.x+x0,grid.z)
            f = scInt.interp2d(xt_coarse,zm_coarse[:-1],nc_w.variables['wxz'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.wnorth[it+1,:,:] = f(grid.x+x0,grid.zh)
            if(nsv>0):
                for isv in range(nsv):
                    f = scInt.interp2d(xt_coarse,zt_coarse,nc_sv[isv].variables[f"sv{isv+1:03}"][it,:,:],bounds_error=False,kind='linear')
                    boundaryInp.svnorth[isv,it+1,:,:] = f(grid.x+x0,grid.z)
        nc_thl.close()
        nc_qt.close()
        nc_e12.close()
        nc_u.close()
        nc_v.close()
        nc_w.close()
        if(nsv>0): [nc_sv[isv].close() for isv in range(nsv)]
        print('finished north')
        # Top boundary
        nc_thl   = Dataset(f"{path_coarse}output/crossxy/{ktot:04}/thlxy.{ktot:04}.{iexpnr_coarse:03}.nc","r")
        nc_qt   = Dataset(f"{path_coarse}output/crossxy/{ktot:04}/qtxy.{ktot:04}.{iexpnr_coarse:03}.nc","r")
        nc_e12   = Dataset(f"{path_coarse}output/crossxy/{ktot:04}/e120xy.{ktot:04}.{iexpnr_coarse:03}.nc","r")
        nc_u   = Dataset(f"{path_coarse}output/crossxy/{ktot:04}/uxy.{ktot:04}.{iexpnr_coarse:03}.nc","r")
        nc_v   = Dataset(f"{path_coarse}output/crossxy/{ktot:04}/vxy.{ktot:04}.{iexpnr_coarse:03}.nc","r")
        nc_w   = Dataset(f"{path_coarse}output/crossxy/{ktot:04}/wxy.{ktot:04}.{iexpnr_coarse:03}.nc","r")
        if(nsv>0): nc_sv = [Dataset(f"{path_coarse}output/crossxy/{ktot:04}/sv{isv+1:03}.{ktot:04}.{iexpnr_coarse:03}.nc","r") for isv in range(nsv)]
        for it in range(nt): # Interpolate and write per time step
            f = scInt.interp2d(xt_coarse,yt_coarse,nc_thl.variables['thlxy'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.thltop[it+1,:,:] = f(grid.x+x0,grid.y+y0)
            f = scInt.interp2d(xt_coarse,yt_coarse,nc_qt.variables['qtxy'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.qttop[it+1,:,:] = f(grid.x+x0,grid.y+y0)
            f = scInt.interp2d(xt_coarse,yt_coarse,nc_e12.variables['e120xy'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.e12top[it+1,:,:] = f(grid.x+x0,grid.y+y0)
            f = scInt.interp2d(xm_coarse[:-1],yt_coarse,nc_u.variables['uxy'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.utop[it+1,:,:] = f(grid.xh+x0,grid.y+y0)
            f = scInt.interp2d(xt_coarse,ym_coarse[:-1],nc_v.variables['vxy'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.vtop[it+1,:,:] = f(grid.x+x0,grid.yh+y0)
            f = scInt.interp2d(xt_coarse,yt_coarse,nc_w.variables['wxy'][it,:,:],bounds_error=False,kind='linear')
            boundaryInp.wtop[it+1,:,:] = f(grid.x+x0,grid.y+y0)
            if(nsv>0):
                for isv in range(nsv):
                    f = scInt.interp2d(xt_coarse,yt_coarse,nc_sv[isv].variables[f"sv{isv+1:03}"][it,:,:],bounds_error=False,kind='linear')
                    boundaryInp.svtop[isv,it+1,:,:] = f(grid.x+x0,grid.y+y0)
        nc_thl.close()
        nc_qt.close()
        nc_e12.close()
        nc_u.close()
        nc_v.close()
        nc_w.close()
        if(nsv>0): [nc_sv[isv].close() for isv in range(nsv)]
        print('finished top')
        # Close netcdf file
        boundaryInp.exit()
        del boundaryInp
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")

    # -------- Create initial heterogeneous input fields -------- #
    if(linithetero):
        start_time = datetime.now()
        print("Start creation of initfields.inp."+str(iexpnr_fine).zfill(3)+".nc")
        # Read input data
        nc = Dataset(f"{path_coarse}input/initfields.inp.{iexpnr_coarse:03}.nc",'r')
        xt_coarse = nc.variables['xt'][:]; dx_coarse = xt_coarse[1]-xt_coarse[0]
        xm_coarse = nc.variables['xm'][:]
        yt_coarse = nc.variables['yt'][:]; dy_coarse = yt_coarse[1]-yt_coarse[0]
        ym_coarse = nc.variables['ym'][:]
        zt        = nc.variables['zt'][:]
        zm        = nc.variables['zm'][:]
        # Get corner indices fine domain in coarse domain
        ix0 = np.argwhere(xm_coarse-x0==0)[0][0]
        iy0 = np.argwhere(ym_coarse-y0==0)[0][0]
        ixe = np.argwhere(xm_coarse-(x0+grid.xh[-1])==0)[0][0]
        iye = np.argwhere(ym_coarse-(y0+grid.yh[-1])==0)[0][0]
        # Create netcdf file
        heteroInp = DALES.HeteroInp(grid, iexpnr=iexpnr_fine, path=f"{path_fine}input/")
        # Interpolate data and write data per height level
        for k in range(ktot):
            f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['thl0'][k,:,:],bounds_error=False,kind='linear')
            heteroInp.thl[k,:,:]=f(grid.x+x0,grid.y+y0)
            f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['qt0'][k,:,:],bounds_error=False,kind='linear')
            heteroInp.qt[k,:,:]=f(grid.x+x0,grid.y+y0)
            f = scInt.interp2d(xm_coarse,yt_coarse,nc.variables['u0'][k,:,:],bounds_error=False,kind='linear')
            heteroInp.u[k,:,:]=f(grid.xh+x0,grid.y+y0)
            f = scInt.interp2d(xt_coarse,ym_coarse,nc.variables['v0'][k,:,:],bounds_error=False,kind='linear')
            heteroInp.v[k,:,:]=f(grid.x+x0,grid.yh+y0)
            f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['w0'][k,:,:],bounds_error=False,kind='linear')
            heteroInp.w[k,:,:]=f(grid.x+x0,grid.y+y0)
            f = scInt.interp2d(xt_coarse,yt_coarse,nc.variables['e120'][k,:,:],bounds_error=False,kind='linear')
            heteroInp.e12[k,:,:]=f(grid.x+x0,grid.y+y0)
        nc.close()
        heteroInp.exit()
        del heteroInp
        end_time = datetime.now()
        hours = (end_time-start_time).days*24+(end_time-start_time).seconds//3600
        minutes = ((end_time-start_time).seconds//60)%60
        seconds = ((end_time-start_time).seconds-hours*3600-minutes*60)
        print(f"Finished in {hours:02}:{minutes:02}:{seconds:02}")
        
if __name__ == '__main__':
    main()
