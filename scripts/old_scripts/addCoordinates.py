# Georeferences a DALES netcdf output file using the coordinate transformation
# given in the HARMONIE reference file. For both the coarse and fine simulation output.
import numpy as np
from netCDF4 import Dataset
import os
def main():
    # -------------------------- Input -------------------------- #
    # General
    referenceFile='../harmonie/huss_his_BES_BES_harm43h22tg3_fERA5_exp0_1hr_202002010000-202002102300.nc' # Reference file containing coordinate transformation
    variablePath = '../simulations/512x384_nested/' # Path to simulation
    variableFile = 'crossxy/0006/thlxy.0006' # Path to variable (exclude ".iexpnr.nc")
    targetPath   ='../simulations/512x384_nested/qgis/data/simulation/' # Path to save output
    variableName = 'thlxy' # Variable name in netcdf file
    simulationType = '' # If there are multiple simulations present in the specified simulation folder specify type here. Otherwise '' is fine.
    refDate    = '2020-02-05 00:00:00' # Reference date to add in netcdf output file
    # Coarse domain (Should correspond wih createBoundaryInput.py)
    iexpnr_coarse = 0                  # Experiment number
    itot_coarse   = 512                # Number of grid points in x-direction
    jtot_coarse   = 384                # Number of grid points in y-direction
    xsize_coarse  = itot_coarse/4*2500 # Size in x-direction
    ysize_coarse  = jtot_coarse/4*2500 # Size in y-direction
    lat0 = 13.3                        # Latitude domain centre
    lon0 = -57.8                       # Longitude domain centre
    # Fine domain (Should correspond wih createBoundary.py)
    lfine       = True                 # Do high resolution domain
    iexpnr_fine = 1                    # Experiment number
    itot_fine   = 512                  # Number of grid points in x-direction (I)
    jtot_fine   = 384                  # Number of grid points in y-direction
    xsize_fine  = itot_fine/16*2500    # Size in x-direction
    ysize_fine  = jtot_fine/16*2500    # Size in y-direction
    x0_fine = 512*2500/4/4 # x-coordinate lower left corner fine simulation in coordinate system of coarse simulation
    y0_fine = 384*2500/4/4 # y-coordinate lower left corner fine simulation in coordinate system of coarse simulation
    # HARMONIE domain
    itot_harm  = int(xsize_coarse/2500)
    jtot_harm  = int(ysize_coarse/2500)

    # ------------------- Open Harmonie data -------------------- #
    nc = Dataset(referenceFile,'r')
    lat_harm  = nc.variables['lat'][:,:]
    lon_harm  = nc.variables['lon'][:,:]
    x_harm    = nc.variables['x'][:]
    y_harm    = nc.variables['y'][:]
    projParam = nc.variables['Lambert_Conformal']
    standard_parallel = projParam.standard_parallel*np.pi/180
    latRef  = projParam.latitude_of_projection_origin*np.pi/180
    lonRef  = projParam.longitude_of_central_meridian*np.pi/180
    x0_harm = projParam.false_easting
    y0_harm = projParam.false_northing
    R_earth = projParam.earth_radius
    nc.close()

    # ----- Transform x/y -> lat/lon for coarse simulation ------ #
    # copy original file to destination folder and extend name with _georef
    start=0
    while(variableFile.find('/',start)>0):
        start = variableFile.find('/',start)+1
    targetFile=f"{variableFile[start:]}.{iexpnr_coarse:03d}_georef.nc"
    command=f"cp {variablePath}coarse/output/{simulationType}{variableFile}.{iexpnr_coarse:03d}.nc {targetPath}coarse/{simulationType}{targetFile}"
    os.system(command)
    # Read coarse dimensions
    nc_coarse = Dataset(f"{targetPath}coarse/{simulationType}{targetFile}",'r+')
    for dim in nc_coarse.dimensions.values():
        if(dim.name=='xt' or dim.name=='xm'):
            xName = dim.name
        if(dim.name=='yt' or dim.name=='ym'):
            yName = dim.name
    x_coarse = nc_coarse.variables[xName][:]
    y_coarse = nc_coarse.variables[yName][:]
    # get coordinates in reference frame harmonie file
    i0l   = np.argwhere((lat_harm-lat0)**2+(lon_harm-lon0)**2 == np.min((lat_harm-lat0)**2+(lon_harm-lon0)**2))[0]
    iy0   = i0l[0]-int(jtot_harm/2)
    ix0   = i0l[1]-int(itot_harm/2)
    x0_coarse = x_harm[ix0]
    y0_coarse = y_harm[iy0]
    # Do transformation
    xR_coarse = x_coarse + x0_coarse - x0_harm # Get 'real' x distance in local rectilinear coordinate system
    yR_coarse = y_coarse + y0_coarse - y0_harm # Get 'real' y distance in local rectilinear coordinate system
    n     = np.sin(standard_parallel)
    F     = (np.cos(standard_parallel)*np.tan(np.pi/4+standard_parallel/2)**n)/n
    rho0  = R_earth*F*np.tan(np.pi/2-(np.pi/4+latRef/2))**n
    rho   = np.sign(n)*np.sqrt(xR_coarse[None,:]**2+(rho0-yR_coarse[:,None])**2)
    theta = np.arctan(xR_coarse[None,:]/(rho0-yR_coarse[:,None]))
    lat_coarse  = (2*np.arctan((R_earth*F/rho)**(1/n))-np.pi/2)/np.pi*180
    lon_coarse = (lonRef+theta/n)/np.pi*180
    # Write to file
    if 'lat' not in list(nc_coarse.variables):
        nclat = nc_coarse.createVariable('lat','f4',('yt','xt'))
        nclat.standard_name = 'Latitude'
        nclat.long_name = 'Latitude'
        nclat.units = 'degrees_north'
        nclat._CoordinateAxisType = 'Lat'
    else:
        nclat = nc_coarse.variables['lat']
    if 'lon' not in list(nc_coarse.variables):
        nclon = nc_coarse.createVariable('lon','f4',('yt','xt'))
        nclon.standard_name = 'Longitude'
        nclon.long_name = 'Longitude'
        nclon.units = 'degrees_east'
        nclon._CoordinateAxisType = 'Lon'
    else:
        nclon = nc_coarse.variables['lon']
    if 'Lambert_Conformal' not in list(nc_coarse.variables): # copy transformation info
        ncin = Dataset(referenceFile,'r')
        varin = ncin.variables['Lambert_Conformal']
        ncLambert = nc_coarse.createVariable('Lambert_Conformal','S1')
        ncLambert.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        ncin.close()
    nctime = nc_coarse.variables['time']
    nctime.units = f"seconds since {refDate}" ;
    nctime[:] = nctime[:]-2*nctime[0]+nctime[1]
    ncx = nc_coarse.variables[xName]
    ncx.axis = 'X'
    ncx.standard_name = 'projection_x_coordinate'
    ncx.long_name = 'X Coordinate Of Projection'
    ncx.units = 'm'
    if(ncx[0]<ncx[1]-ncx[0]):
        ncx[:] = ncx[:]+x0_coarse
    ncy = nc_coarse.variables[yName]
    ncy.axis = 'Y'
    ncy.standard_name = 'projection_y_coordinate'
    ncy.long_name = 'Y Coordinate Of Projection'
    ncy.units = 'm'
    if(ncy[0]<ncy[1]-ncy[0]):
        ncy[:] = ncy[:]+y0_coarse
    nc_coarse.variables[variableName].coordinates = "lon lat"
    nc_coarse.variables[variableName].grid_mapping = "Lambert_Conformal"
    nc_coarse.variables[variableName].cell_methods = "time: point"
    nclat[:,:] = lat_coarse
    nclon[:,:] = lon_coarse
    nc_coarse.close()

    # ------ Transform x/y -> lat/lon for fine simulation ------- #
    if(lfine):
        # copy original file to destination folder and extend name with _georef
        start=0
        while(variableFile.find('/',start)>0):
            start = variableFile.find('/',start)+1
        targetFile=f"{variableFile[start:]}.{iexpnr_fine:03d}_georef.nc"
        command=f"cp {variablePath}fine/output/{variableFile}.{iexpnr_fine:03d}.nc {targetPath}/fine/{targetFile}"
        os.system(command)
        nc_fine = Dataset(f"{targetPath}fine/{targetFile}",'r+')
        for dim in nc_coarse.dimensions.values():
            if(dim.name=='xt' or dim.name=='xm'):
                xName = dim.name
            if(dim.name=='yt' or dim.name=='ym'):
                yName = dim.name
        x_fine = nc_fine.variables[xName][:]
        y_fine = nc_fine.variables[yName][:]
        # Do transformation
        x_fineR = x_fine + x0_coarse + x0_fine - x0_harm # Get 'real' x distance in local rectilinear coordinate system
        y_fineR = y_fine + y0_coarse + y0_fine - y0_harm # Get 'real' y distance in local rectilinear coordinate system
        n     = np.sin(standard_parallel)
        F     = (np.cos(standard_parallel)*np.tan(np.pi/4+standard_parallel/2)**n)/n
        rho0  = R_earth*F*np.tan(np.pi/2-(np.pi/4+latRef/2))**n
        rho   = np.sign(n)*np.sqrt(x_fineR[None,:]**2+(rho0-y_fineR[:,None])**2)
        theta = np.arctan(x_fineR[None,:]/(rho0-y_fineR[:,None]))
        lat_fine  = (2*np.arctan((R_earth*F/rho)**(1/n))-np.pi/2)/np.pi*180
        lon_fine = (lonRef+theta/n)/np.pi*180
        # Write to file
        if 'lat' not in list(nc_fine.variables):
            nclat = nc_fine.createVariable('lat','f4',('yt','xt'))
            nclat.standard_name = 'Latitude'
            nclat.long_name = 'Latitude'
            nclat.units = 'degrees_north'
            nclat._CoordinateAxisType = 'Lat'
        else:
            nclat = nc_fine.variables['lat']
        if 'lon' not in list(nc_fine.variables):
            nclon = nc_fine.createVariable('lon','f4',('yt','xt'))
            nclon.standard_name = 'Longitude'
            nclon.long_name = 'Longitude'
            nclon.units = 'degrees_east'
            nclon._CoordinateAxisType = 'Lon'
        else:
            nclon = nc_fine.variables['lon']
        if 'Lambert_Conformal' not in list(nc_fine.variables): # copy transformation info
            ncin = Dataset(referenceFile,'r')
            varin = ncin.variables['Lambert_Conformal']
            ncLambert = nc_fine.createVariable('Lambert_Conformal','S1')
            ncLambert.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            ncin.close()
        nctime = nc_coarse.variables['time']
        nctime.units = f"seconds since {refDate}" ;
        nctime[:] = nctime[:]-2*nctime[0]+nctime[1]
        ncx = nc_fine.variables[xName]
        ncx.axis = 'X'
        ncx.standard_name = 'projection_x_coordinate'
        ncx.long_name = 'X Coordinate Of Projection'
        ncx.units = 'm'
        if(ncx[0]<ncx[1]-ncx[0]):
            ncx[:] = ncx[:]+x0_coarse+x0_fine
        ncy = nc_fine.variables[yName]
        ncy.axis = 'Y'
        ncy.standard_name = 'projection_y_coordinate'
        ncy.long_name = 'Y Coordinate Of Projection'
        ncy.units = 'm'
        if(ncy[0]<ncy[1]-ncy[0]):
            ncy[:] = ncy[:]+y0_coarse+y0_fine
        nc_fine.variables[variableName].coordinates = "lon lat"
        nc_fine.variables[variableName].grid_mapping = "Lambert_Conformal"
        nc_fine.variables[variableName].cell_methods = "time: point"
        nclat[:,:] = lat_fine
        nclon[:,:] = lon_fine
        nc_fine.close()

if __name__ == '__main__':
    main()
