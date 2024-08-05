# Crops the AUS2200 data to the required time and spatial extends. 
# Transforms sigma coordinates into height levels.
# Transforms wgs84 to local utm coordinates
# Transforms AUS2200 prognostic variables to DALES prognostic variables
import glob
import numpy as np
import xarray as xr
import pyproj
from pyproj import Transformer
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import dask.array as da
import dask.delayed as ddelay
from numba import jit
import sys
# Constants
# Simulation constants (modglobal)
p0   = 1e5          # Reference pressure
Rd   = 287.04       # Gas constant for dry air
Rv   = 461.5        # Gas constant for water vapor
cp   = 1004.        # Specific heat at constant pressure (dry air)
Lv   = 2.53e6       # Latent heat for vaporisation
grav = 9.81         # Gravitational constant
kappa= 0.4          # Von Karman constant
def prep_aus2200(input,grid):
  # Create transform to go from lat lon to local rectilinear grid
  transform = Transform([input['lon_sw'],input['lon_sw']+grid.xsize/100000],\
                        [input['lat_sw'],input['lat_sw']+grid.ysize/100000])
  x_sw,y_sw = transform.latlon_to_xy(input['lat_sw'],input['lon_sw'])
  # Translate to make southwest corner of DALES origin
  transform.update_parameters(false_easting=transform.parameters['false_easting']-x_sw,\
                              false_northing=transform.parameters['false_northing']-y_sw)
  x_sw,y_sw = transform.latlon_to_xy(input['lat_sw'],input['lon_sw'])
  # Get domain box in WGS84
  lat_ne,lon_ne = transform.xy_to_latlon(grid.xsize,grid.ysize)
  lat_se,lon_se = transform.xy_to_latlon(grid.xsize,0)
  lat_nw,lon_nw = transform.xy_to_latlon(0,grid.ysize)
  lat_min       = np.array([input['lat_sw'],lat_se,lat_ne,lat_nw]).min()
  lat_max       = np.array([input['lat_sw'],lat_se,lat_ne,lat_nw]).max()
  lon_min       = np.array([input['lon_sw'],lon_se,lon_ne,lon_nw]).min()
  lon_max       = np.array([input['lon_sw'],lon_se,lon_ne,lon_nw]).max()
  buffer        = 0.05
  # Variables to read
  # filenames = glob.glob(datadir+"u/*/*.nc")
  # variables3D   = ['ua','va','wa','ta','hus','clw']
  # variablesSurf = ['ps','tas','huss']
  # Get model level files
  varCodes = ['fld_s00i002','fld_s00i003','fld_s00i150','fld_s00i004','fld_s00i010','fld_s00i408','fld_s16i004']
  # varNames = ['u','v','w','theta','qv']
  varNames = ['u','v','w','th','qv','p','ta']
  varCodes_surf = ['fld_s16i222','fld_s03i236','fld_s03i237'] 
  varNames_surf = ['ps','Ts','qt']
  varCodes_cld = ['fld_s00i254']
  varNames_cld = ['ql']
  varCode_mask  = 'fld_s00i030'
  if('synturb' in input):
    varCodes   = varCodes+['fld_s03i473']
    varNames  = varNames+['tke']
    varCodes_surf = varCodes_surf+['fld_s00i025','fld_s03i460_0','fld_s03i461_0','fld_s03i217']
    varNames_surf = varNames_surf+['zi','ustar','vstar','wthls']
  def read_variables(filenames,varCodes,varNames,transform,lsurf=False):
    data = []
    mask = []
    for filename in filenames:
      with xr.open_dataset(filename) as ds:
        if(len(mask)==0 and lsurf):
          mask = ds[varCode_mask].squeeze(drop=True)
          dims = mask.dims
          mask = mask.sel({dims[-2]:slice(lat_min-buffer,lat_max+buffer),\
                         dims[-1]:slice(lon_min-buffer,lon_max+buffer)})
          mask = mask.rename('land_mask')
          mask = wgs84_to_utm(mask,2200,2200,grid.xsize,grid.ysize,transform,byChunks=False).round()
        ds = ds.sel(time=slice(input['start'],input['end']))
        if(len(ds['time'])<1): continue
        ds_new = []
        for ivar in range(len(varCodes)):
          var = ds[varCodes[ivar]]
          dims = var.dims
          var = var.sel({dims[-2]:slice(lat_min-buffer,lat_max+buffer),\
                         dims[-1]:slice(lon_min-buffer,lon_max+buffer)})
          if not(lsurf): 
            var = var.swap_dims({dims[-3]:dims[-3].replace('model_','').replace('number','height')})
            var = var.rename({var.dims[-3]:'z'})
            if(dims[-3]!='theta_level_height'): var = var.interp(z=ds['theta_level_height'].values)
          dims = var.dims
          var = var.rename(varNames[ivar])
          if(dims[0]!='time'):
            var = var.rename({dims[0]:'time'})
            var = var.assign_coords({'time':ds.time.values})    
          if(dims[-2]!='lat'): var = var.rename({dims[-2]:'lat'})
          if(dims[-1]!='lon'): var = var.rename({dims[-1]:'lon'})
          var = wgs84_to_utm(var,2200,2200,grid.xsize,grid.ysize,transform,byChunks=False)
          ds_new.append(var)
        ds_new = xr.merge(ds_new)
      data.append(ds_new)
    data = xr.concat(data[:],'time')
    return data,mask
  def get_ncName(filename):
    return filename.split('/')[-1]
  # Read model level data
  filenames = glob.glob(input['inpath']+"*/aus2200/d0198/RA3/um/umnsa_mdl_*.nc")
  filenames.sort(key=get_ncName)
  data,_ = read_variables(filenames,varCodes,varNames,transform,lsurf=False)
  # Read cloud data
  filenames = glob.glob(input['inpath']+"*/aus2200/d0198/RA3/um/umnsa_cldrad_*.nc")
  filenames.sort(key=get_ncName)
  data_cld,_ = read_variables(filenames,varCodes_cld,varNames_cld,transform,lsurf=False)
  # Read surface data
  filenames = glob.glob(input['inpath']+"*/aus2200/d0198/RA3/um/umnsa_slv_*.nc")
  filenames.sort(key=get_ncName)
  datas,mask = read_variables(filenames,varCodes_surf,varNames_surf,transform,lsurf=True)
  datas = datas.assign({'transform' : xr.DataArray([],name='Transverse_Mercator',attrs=transform.parameters)})
  datas = datas.assign({'u' : xr.zeros_like(datas.qt),
                        'v' : xr.zeros_like(datas.qt),
                        'w' : xr.zeros_like(datas.qt)})
  datas = datas.expand_dims({'z': np.array([0.])},axis=1)
  # Get qt
  qt = (data['qv']+data_cld['ql']).rename('qt')
  data = data.assign({'qt':qt})
  # Calculate exnr function and surface thl if first simulation, otherwise load exnr.inp.xxx
  if(input['time0']==input['start']): # Calculate exnr function
    tas_exnr = datas['Ts'].isel({'time': 0},drop=True).sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values
    ps_exnr  = datas['ps'].isel({'time': 0},drop=True).sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values
    exnrs    = (ps_exnr/p0)**(Rd/cp)
    thls_exnr= tas_exnr/exnrs
    rhobf    = calcBaseprof(data['z'].values,thls_exnr,ps_exnr,pref0=p0)
    p_exnr   = rhobf*Rd*data['ta'][0,:].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values*\
           ( 1+(Rv/Rd-1)*data['qt'][0,:].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values-\
             Rv/Rd*data_cld['ql'][0,:].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values ) # Ideal gas law
    exnr     = (p_exnr/p0)**(Rd/cp)
  else: # Read exnr function
    f = open(input['exnr_file'],'r')
    line0 = f.readline()
    f.close()
    thls_exnr = float(line0.split(',')[1].split('thls = ')[-1])
    ps_exnr = float(line0.split(',')[2].split('ps = ')[-1])
    exnr = np.loadtxt(input['exnr_file'],skiprows=2)
    exnrs = exnr[0,1]
    exnr = exnr[1:,1]
  exnr     = xr.DataArray(exnr,
                          dims = ['z'],
                          coords={'z': data['z']},
                          name = 'exnr',
                          attrs = {'thls': thls_exnr, 'ps': ps_exnr})
  data     = data.assign({'exnr': exnr})
  datas    = datas.assign({'exnr': xr.DataArray(exnrs,
                          dims = ['z'],
                          coords={'z': [0.]},
                          name = 'exnr',
                          attrs = {'thls': thls_exnr, 'ps': ps_exnr})})
  # Get thl
  thl = (data['ta']/data['exnr']-Lv*data_cld['ql']/(cp*data['exnr'])).rename('thl')
  data = data.assign({'thl':thl}).drop_vars(['th','qv','ta','p'])
  datas    = datas.assign({'thl':datas.Ts/exnrs})
  datas = datas.drop(labels=['ps','Ts'])
  if('synturb' in input): 
    rhobf    = calcBaseprof(data.z.values,thls_exnr,ps_exnr,pref0=p0)
    rhobs    = rhobf[0]-data.z[0].values*(rhobf[1]-rhobf[0])/(data.z[1].values-data.z[0].values)
    datas['wthls'] = datas['wthls']/(exnrs*rhobs*cp)
    datas = datas.assign({'tke' : xr.zeros_like(datas.qt)})
  data = xr.concat([datas,data],dim='z')
  data = data.assign({'transform' : xr.DataArray([],name='Transverse_Mercator',attrs=transform.parameters)})
  if('synturb' in input):
    data['ustar'] = data['ustar'].isel(z=0,drop=True)
    data['vstar'] = data['vstar'].isel(z=0,drop=True)
    data['zi'] = data['zi'].isel(z=0,drop=True).mean(dim=['x','y'])
    data['wthls'] = data['wthls'].isel(z=0,drop=True)
  data = data.assign({'land_mask':mask})
  if(input['lsave_source']): data.to_netcdf(f"{input['outpath']}aus2200.nc")
  return data,transform
  # Read orography data
  # with xr.open_dataset(input['fileOrog']) as ds:
  #   ds  = ds.rename({ds['orog'].dims[-2]: 'lat', ds['orog'].dims[-1]: 'lon'})
  #   orog = ds.orog[0,:,:].sel(lat=slice(lat_min-buffer,lat_max+buffer),lon=slice(lon_min-buffer,lon_max+buffer))
  # # Get 3D variables on rectilinear grid
  # for var in variables3D:
  #   with xr.open_mfdataset(f"{input['inpath3D']}{var}_*.nc",chunks={"time": input['tchunk']}) as ds:
  #     ds  = ds.rename({ds[var].dims[-2]: 'lat', ds[var].dims[-1]: 'lon'})
  #     data.append(ds[var].sel(time=slice(input['start'], input['end']),\
  #                                 lat=slice(lat_min-buffer,lat_max+buffer),\
  #                                 lon=slice(lon_min-buffer,lon_max+buffer)))
  #     data[-1] = InterpolateZ(orog,data[-1],timeDim='time',otherDims={'lon':'lon','lat':'lat'},level_choice='auto',quiet=False,byChunks=True).load()
  #     data[-1] = wgs84_to_utm(data[-1].chunk({'time':input['tchunk']}),2200,transform,byChunks=True).load() 
    # var,orog = loadData(pathDataLev,varName,fileOrog,timeStart,timeEnd,latS,latN,lonW,lonE,buffer,tchunk)
    # var_zint = InterpolateZ(orog,var,timeDim='time',otherDims={'lon':'lon','lat':'lat'},level_choice='auto',quiet=False,byChunks=True).load()
    # var_zint_utm = wgs84_to_utm(var_zint.chunk({'time':tchunk}),dx,dy,latS,latN,lonW,lonE,byChunks=True).load()
    # var_zint_utm.to_netcdf(path=f"{pathWrite}{varName}.nc")
  
  # Get surface variables on rectilinear grid
  # for var in variablesSurf:
  #   print(varName)
  #   var,_ = loadData(pathDataSurf,varName,fileOrog,timeStart,timeEnd,latS,latN,lonW,lonE,buffer,tchunk)
  #   var_utm = wgs84_to_utm(var,dx,dy,latS,latN,lonW,lonE,byChunks=True).load()
  #   var_utm.to_netcdf(path=f"{pathWrite}{varName}.nc")

  # Convert to DALES prognostic variables
# Load data and clip to window
# def loadData(path,varName,fileOrog,timeStart,timeEnd,latS,latN,lonW,lonE,buffer,tchunk):
#   import xarray as xr
#   import os
#   var = []
#   for file in sorted(os.listdir(f"{path}")):
#     if(file.startswith(f"{varName}_1hr_") and file.endswith('.nc')):
#       with xr.open_dataset(f"{path}{file}",chunks={"time": tchunk}) as ds:
#         ds  = ds.rename({ds[varName].dims[-2]: 'lat', ds[varName].dims[-1]: 'lon'})
#         var.append(ds[varName].sel(lat=slice(latS-buffer,latN+buffer),lon=slice(lonW-buffer,lonE+buffer)))
#   var = xr.concat(var[:],'time')
#   var = var.sel(time=slice(timeStart, timeEnd))
#   # with xr.open_mfdataset(f"{path}{varName}_*.nc",chunks={"time": tchunk},data_vars=[varName],preprocess=preprocess) as ds:
#   #   ds  = ds.rename({ds[varName].dims[-2]: 'lat', ds[varName].dims[-1]: 'lon'})
#   #   var = ds[varName].sel(time=slice(timeStart, timeEnd),lat=slice(latS-buffer,latN+buffer),lon=slice(lonW-buffer,lonE+buffer))
#   with xr.open_dataset(fileOrog) as ds:
#     ds  = ds.rename({ds['orog'].dims[-2]: 'lat', ds['orog'].dims[-1]: 'lon'})
#     orog = ds.orog[0,:,:].sel(lat=slice(latS,latN),lon=slice(lonW,lonE))
#   return var,orog   

# Interpolate data to geometric height levels
def InterpolateZ(orog,var,timeDim='time',otherDims={'lon':'lon','lat':'lat'},level_choice='auto',quiet=False,byChunks=True):
  '''
      Interpolate model level data onto geometric height surfaces.
      For this, we use equation (4.1) in Davies et al, doi:10.1256/qj.04.101:
      New z-levels are the same as the hybrid height surfaces suggest.
      ToDos: 
      - Currently uses spline interpolation for every grid cell. One could probably use wrf-python to interpolate more efficiently
      - Currently does not assure that points below the surface are missing or 0 (or some other default values)
      INPUTS:
          orog:         orography. expects 2D field in meters.
          var:          the 3D variable to interpolate. Expects an xarray.DataArray
          timeDim:      name of the time dimension of var.
          otherDims:    names of any other dimensions of var which are left as is
          level_choice: is the variable on theta ('theta') or rho ('rho') hybrid levels? the function guesses if 'auto'. Can also be a list of desired levels in meters.
          quiet:        flag for verbosity.
          byChunks:     if data is read in time chunks, for example via xr.open_mfdataset(), one probably wants to set this to True.
      OUTPUTS:
          outVar: same as var, but vertically interpolated onto 'z' dimension
      Written by Martin Jucker (coding@martinjucker.com). This version as developed during CLEX AUS2200 Hackathon 2023-03-17.
          
  '''
  #
  import numpy as np
  import xarray as xr
  import dask.array as da
  import dask.delayed as ddelay
  #
  if timeDim != 'time':
      var = var.rename({timeDim : 'time'})
      var.swap_dims({timeDim : 'time'},inplace=True)
  #
  #nDims = len(var.shape)
  # legacy 
  model = 'nest'
  # these numbers come from the file etc/vert_levs/L70_40km
  z_top_of_model = 40000.00
  first_constant_r_rho_level = 62
  eta_theta=np.array([
    0.0000000E+00,   0.1250000E-03,   0.5416666E-03,   0.1125000E-02,   0.1875000E-02, 
    0.2791667E-02,   0.3875000E-02,   0.5125000E-02,   0.6541667E-02,   0.8125000E-02, 
    0.9875000E-02,   0.1179167E-01,   0.1387500E-01,   0.1612500E-01,   0.1854167E-01, 
    0.2112500E-01,   0.2387500E-01,   0.2679167E-01,   0.2987500E-01,   0.3312500E-01, 
    0.3654167E-01,   0.4012500E-01,   0.4387500E-01,   0.4779167E-01,   0.5187500E-01, 
    0.5612501E-01,   0.6054167E-01,   0.6512500E-01,   0.6987500E-01,   0.7479167E-01, 
    0.7987500E-01,   0.8512500E-01,   0.9054167E-01,   0.9612500E-01,   0.1018750E+00, 
    0.1077917E+00,   0.1138750E+00,   0.1201250E+00,   0.1265417E+00,   0.1331250E+00, 
    0.1398750E+00,   0.1467917E+00,   0.1538752E+00,   0.1611287E+00,   0.1685623E+00, 
    0.1761954E+00,   0.1840590E+00,   0.1921980E+00,   0.2006732E+00,   0.2095645E+00, 
    0.2189729E+00,   0.2290236E+00,   0.2398690E+00,   0.2516917E+00,   0.2647077E+00, 
    0.2791699E+00,   0.2953717E+00,   0.3136506E+00,   0.3343919E+00,   0.3580330E+00, 
    0.3850676E+00,   0.4160496E+00,   0.4515977E+00,   0.4924007E+00,   0.5392213E+00, 
    0.5929016E+00,   0.6543679E+00,   0.7246365E+00,   0.8048183E+00,   0.8961251E+00, 
    0.1000000E+01
  ])
  eta_rho = np.array([
    0.6249999E-04,   0.3333333E-03,   0.8333333E-03,   0.1500000E-02,   0.2333333E-02, 
    0.3333333E-02,   0.4500000E-02,   0.5833333E-02,   0.7333333E-02,   0.9000000E-02, 
    0.1083333E-01,   0.1283333E-01,   0.1500000E-01,   0.1733333E-01,   0.1983333E-01, 
    0.2250000E-01,   0.2533333E-01,   0.2833333E-01,   0.3150000E-01,   0.3483333E-01, 
    0.3833333E-01,   0.4200000E-01,   0.4583333E-01,   0.4983333E-01,   0.5400000E-01, 
    0.5833334E-01,   0.6283334E-01,   0.6750000E-01,   0.7233334E-01,   0.7733333E-01, 
    0.8250000E-01,   0.8783333E-01,   0.9333333E-01,   0.9900000E-01,   0.1048333E+00, 
    0.1108333E+00,   0.1170000E+00,   0.1233333E+00,   0.1298333E+00,   0.1365000E+00, 
    0.1433333E+00,   0.1503334E+00,   0.1575020E+00,   0.1648455E+00,   0.1723789E+00, 
    0.1801272E+00,   0.1881285E+00,   0.1964356E+00,   0.2051189E+00,   0.2142687E+00, 
    0.2239982E+00,   0.2344463E+00,   0.2457803E+00,   0.2581997E+00,   0.2719388E+00, 
    0.2872708E+00,   0.3045112E+00,   0.3240212E+00,   0.3462124E+00,   0.3715503E+00, 
    0.4005586E+00,   0.4338236E+00,   0.4719992E+00,   0.5158110E+00,   0.5660614E+00, 
    0.6236348E+00,   0.6895022E+00,   0.7647274E+00,   0.8504717E+00,   0.9480625E+00
  ])
  if len(eta_theta) in var.shape:
    zDim = var.shape.index(len(eta_theta))
    input_levels = 'theta'
    if not quiet: print('assuming theta levels')
  elif len(eta_rho) in var.shape:
    zDim = var.shape.index(len(eta_rho))
    input_levels = 'rho'
    if not quiet: print('assuming rho levels')
  else:
    raise ValueError('cannot determine whether I should use theta or rho levels: var.shape, len(eta_theta, len(eta_rho): '+str(var.shape)+', '+str(len(eta_theta))+' ,'+str(len(eta_rho)))
  # now get ready to use equation (4.1) in Davies et al, doi:10.1256/qj.04.101
  if input_levels == 'rho':
    eta   = eta_rho
  elif input_levels == 'theta':
    eta   = eta_theta
  # also get ready to define output z-coordinate
  if isinstance(level_choice,str):
    if level_choice == 'auto':
      level_choice = input_levels
    if level_choice == 'rho':
      etaO = eta_rho
    elif level_choice == 'theta':
      etaO = eta_theta
    else:
      raise ValueError("level_choice must be 'rho' or 'theta', but is "+level_choice)
  # if level_choice = 'auto', hybrid_height is the same thing as the coordinate 'hybrid_height' in the files
  hybrid_height = eta*z_top_of_model
  if isinstance(level_choice,list):
    hybrid_height_out = np.array(level_choice)
  else:
    hybrid_height_out = etaO*z_top_of_model
  # now define above which level there is no more deformation
  etaI  = eta[first_constant_r_rho_level]
  # assuming no topography, my geometric height is hybrid_height
  #  with topography, it's slightly different
  if len(orog.shape)==3: orog=np.squeeze(orog)
  actual_height = hybrid_height[:,np.newaxis,np.newaxis] + orog.values[np.newaxis,:]*(1-eta[:,np.newaxis,np.newaxis]/etaI)**2
  #
  # now make sure we use geometric height above etaI
  noCorr = eta >= etaI
  actual_height[noCorr,:,:] = hybrid_height[noCorr,np.newaxis,np.newaxis]
  ######
  # here comes the complicated part, where we have to be careful with the chunking
  newDims = []
  for k in var.dims:
    if k in ['time','lat','lon']:
      newDims.append(k)
    else:
      zName = k
      newDims.append('z')
  newShape = list(var.shape)
  zInd = var.get_axis_num(zName)
  newShape[zInd] = len(hybrid_height_out)
  newCoords = {'time':var.coords['time'],
                'z'   :xr.DataArray(hybrid_height_out,coords={'z':hybrid_height_out}),
                'lat' :var.coords['lat'],
                'lon' :var.coords['lon']}
  tInd = var.get_axis_num('time')
  if byChunks:
    tchunks = var.chunks[tInd]
    tlen = len(tchunks)
  else:
    tlen = var.time.shape[0]
    tchunks = np.ones((tlen,))
  ## now comes the non-exact part: re-grid actual_height onto hybrid_height for all fields
  # get the indices which correspond to the horizontal dimensions
  #now, whereever there is no orography, the actual_height is exactly hybrid_height
  # now do the interpolation
  from scipy import interpolate
  def GetSlice(var,ts,te):
      return var.isel(time = slice(ts,te)).values
  outVar = []
  ts = 0
  for t in range(tlen):
    te = ts + int(tchunks[t])
    thisVar = ddelay(GetSlice)(var,ts,te)
    splne = ddelay(interpolate.interp1d)(hybrid_height,thisVar,kind=3,axis=zInd)
    thisVar = ddelay(splne(hybrid_height_out))
    outVar.append(thisVar)
    ts = te
  # now convert into xarray
  outVar = ddelay(np.concatenate)(outVar[:],axis=0)
  #print 'creating DataArray'
  outVar = xr.DataArray(
    da.from_delayed(outVar,newShape,dtype=float),
      coords = newCoords,
      dims   = newDims,
      name   = var.name,
      attrs  = var.attrs
  )
  outVar.z.attrs['axis']='Z'
  outVar.z.attrs['units']='m'
  outVar.z.attrs['standard_name']='geometric_height'
  outVar.z.attrs['long_name'] ='geometric height'
  outVar.attrs['long_name'] = var.standard_name.replace('_',' ')
  del(outVar.attrs['coordinates'])
  del(outVar.attrs['um_stash_source'])
  del(outVar.lat.attrs['bounds'])
  del(outVar.lon.attrs['bounds'])
  if 'src' in var.coords:
    outVar.coords['src'] = var['src']
  if timeDim != 'time':
    outVar = outVar.rename({'time' : timeDim})
  return outVar

# Convert data from lat lon grid to x y grid (wgs84 -> utm)
def wgs84_to_utm(var,dx,dy,xsize,ysize,transform,byChunks=False):
  Lon,Lat = np.meshgrid(var.lon,var.lat)
  X,Y = transform.latlon_to_xy(Lat,Lon)
  var.coords['x'] = (('lat','lon'), X)
  var.coords['y'] = (('lat','lon'), Y)
  var.x.attrs['axis']='X'
  var.x.attrs['units']='m'
  var.x.attrs['standard_name']='projection_x_coordinate'
  var.x.attrs['long_name'] =f"X Coordinate Of Projection"
  var.y.attrs['axis']='Y'
  var.y.attrs['units']='m'
  var.y.attrs['standard_name']='projection_y_coordinate'
  var.y.attrs['long_name'] =f"Y Coordinate Of Projection"
  # Find rectangular grid
  x_int = np.arange(0,xsize+2*dx,dx,dtype=np.double)
  y_int = np.arange(0,ysize+2*dy,dy,dtype=np.double)
  X_int,Y_int = np.meshgrid(x_int,y_int)
  # Find lat and lon for new variable
  newLat,newLon = transform.xy_to_latlon(X_int,Y_int)
  # Set dimensions and coordinates new variable
  Ndims = len(var.shape)
  if(Ndims==2): # land_mask or orography
    newCoords = {'y'   :y_int,
                  'x'   :x_int,
                  'lat' :(["y", "x"], newLat),
                  'lon' :(["y", "x"], newLon)
    }
    newDims = ['y','x']
    newShape = [len(y_int),len(x_int)]
  elif(Ndims==3): # Surface variable
    newCoords = {'time' :var.coords['time'],
                  'y'   :y_int,
                  'x'   :x_int,
                  'lat' :(["y", "x"], newLat),
                  'lon' :(["y", "x"], newLon)
    }
    newDims = ['time','y','x']
    newShape = [len(var.time),len(y_int),len(x_int)]

  elif(Ndims==4): # 3D variable
    newCoords = {'time' :var.coords['time'],
                'z'     :var.coords['z'],
                'y'     :y_int,
                'x'     :x_int,
                'lat' :(["y", "x"], newLat),
                'lon' :(["y", "x"], newLon)
    }
    newDims = ['time','z','y','x']
    newShape = [len(var.time),len(var.z),len(y_int),len(x_int)]
  else:
    sys.exit('Unvalid dimensions for interpolation')
  # Preallocate variables
  varInt = []
  ts = 0
  if(Ndims>2):
    # Take time chunk size into account
    tInd = var.get_axis_num('time')
    if byChunks:
      tchunks = var.chunks[tInd]
      tlen = len(tchunks)
    else:
      tlen = var.time.shape[0]
      tchunks = np.ones((tlen,))
  # Do interpolation
  def interpolation(var,xint,yint,ts,te):
    from scipy import interpolate
    Ndims = len(var.shape)
    if(Ndims==2): # land mask or orography
      varOut =  np.reshape(
        interpolate.griddata( 
        (var.x.values.flatten(),var.y.values.flatten()), 
        var.values.flatten(), 
        (xint.flatten(),yint.flatten()) 
        ),
      np.shape(xint))
    elif(Ndims==3): # Surface variable
      varOut = []
      for it in np.arange(ts,te):
        varOut.append( np.reshape(
          interpolate.griddata( 
          (var.x.values.flatten(),var.y.values.flatten()), 
          var[it,:,:].values.flatten(), 
          (xint.flatten(),yint.flatten()) 
          ),
        np.shape(xint)) )
      varOut = np.stack(varOut[:],axis=0)
    elif(Ndims==4): # 3D variable
      varOut = []
      for it in np.arange(ts,te):
        varTmp = []
        for k in range(var.shape[1]):
          varTmp.append(
            np.reshape(
              interpolate.griddata( 
              (var.x.values.flatten(),var.y.values.flatten()), 
              var[it,k,:,:].values.flatten(), 
              (xint.flatten(),yint.flatten()) 
              ),
            np.shape(xint)) 
          )
        varOut.append(np.stack(varTmp[:],axis=0))
      varOut = np.stack(varOut[:],axis=0)
    else:
      sys.exit('Invalid number of dimensions for interpolation')
    return varOut
  if Ndims == 2: # No time dimension
    varInt = ddelay(interpolation)(var,X_int,Y_int,0,0)
  else:
    for it in range(tlen): # Interpolation per time step
      te = ts + int(tchunks[it])
      varInt.append(
        ddelay(interpolation)(var,X_int,Y_int,ts,te)
      )
      ts = te
    varInt = ddelay(np.concatenate)(varInt[:],axis=0)
  # Convert to xarray
  varInt = xr.DataArray(
    da.from_delayed(varInt,newShape,dtype=float),
    coords = newCoords,
    dims   = newDims,
    name   = var.name,
    attrs  = var.attrs
  )
  
  varInt.lat.attrs = var.lat.attrs
  varInt.lon.attrs = var.lon.attrs
  varInt.x.attrs['axis']='X'
  varInt.x.attrs['units']='m'
  varInt.x.attrs['standard_name']='projection_x_coordinate'
  varInt.x.attrs['long_name'] =f"X Coordinate Of Projection"
  varInt.y.attrs['axis']='Y'
  varInt.y.attrs['units']='m'
  varInt.y.attrs['standard_name']='projection_y_coordinate'
  varInt.y.attrs['long_name'] =f"Y Coordinate Of Projection"
  return varInt

class Transform:
  def __init__(self,lon,lat):
    # Get the UTM zone corresponding to the window
    utm_crs_list = query_utm_crs_info(
      datum_name="WGS 84",
      area_of_interest=AreaOfInterest(
          west_lon_degree=min(lon),
          south_lat_degree=min(lat),
          east_lon_degree=max(lon),
          north_lat_degree=max(lat),
      ),
    )
    self.crs_latlon = 'epsg:4326'
    self.crs_utm = 'epsg:'+utm_crs_list[0].code
    wkt = CRS.from_epsg(utm_crs_list[0].code).to_wkt()
    k_0 = float(wkt.split('\"Scale factor at natural origin\",')[-1].split(',')[0])
    lat_0 = float(wkt.split('\"Latitude of natural origin\",')[-1].split(',')[0])
    lon_0 = float(wkt.split('\"Longitude of natural origin\",')[-1].split(',')[0])
    false_easting = float(wkt.split('\"False easting\",')[-1].split(',')[0])
    false_northing = float(wkt.split('\"False northing\",')[-1].split(',')[0])
    self.update_parameters(k_0=k_0,lat_0=lat_0,lon_0=lon_0,false_easting=false_easting,false_northing=false_northing)

  def update_parameters(self,k_0=None,lat_0=None,lon_0=None,false_easting=None,false_northing=None):
    if k_0 == None: k_0 = self.parameters['k_0']
    if lat_0 == None: lat_0 = self.parameters['lat_0']
    if lon_0 == None: lon_0 = self.parameters['lon_0']
    if false_easting == None : false_easting = self.parameters['false_easting']
    if false_northing == None : false_northing = self.parameters['false_norhting']
    self.parameters = dict(ellps='WGS84',k_0=k_0,lat_0=lat_0,lon_0=lon_0,false_easting=false_easting,false_northing=false_northing,proj4=\
    f"+proj=tmerc +ellps=WGS84 +k_0={k_0} +lat_0={lat_0} +lon_0={lon_0} +x_0={false_easting} +y_0={false_northing}")
    self.latlon_to_xy_transform = Transformer.from_crs(self.crs_latlon,self.parameters['proj4'])
    self.xy_to_latlon_transform = Transformer.from_crs(self.parameters['proj4'],self.crs_latlon)

  def latlon_to_xy(self,lat,lon):
    return self.latlon_to_xy_transform.transform(lat,lon)
  
  def xy_to_latlon(self,x,y):
    return self.xy_to_latlon_transform.transform(x,y)

def calcBaseprof(zt,thls,ps,pref0=1e5):
  # constants
  lapserate=np.array([-6.5/1000.,0.,1./1000,2.8/1000])
  zmat=np.array([11000.,20000.,32000.,47000.])
  grav=9.81
  rd=287.04
  cp=1004.
  zsurf=0
  k1=len(zt)
  # Preallocate
  pmat=np.zeros(4)
  tmat=np.zeros(4)
  rhobf=np.zeros(k1)
  pb=np.zeros(k1)
  tb=np.zeros(k1)
  # DALES code
  tsurf=thls*(ps/pref0)**(rd/cp)
  pmat[0]=np.exp((np.log(ps)*lapserate[0]*rd+np.log(tsurf+zsurf*lapserate[0])*grav-
    np.log(tsurf+zmat[0]*lapserate[0])*grav)/(lapserate[0]*rd))
  tmat[0]=tsurf+lapserate[0]*(zmat[0]-zsurf);
  for j in np.arange(1,4):
      if(abs(lapserate[j])<1e-10):
          pmat[j]=np.exp((np.log(pmat[j-1])*tmat[j-1]*rd+zmat[j-1]*grav-zmat[j]*grav)/(tmat[j-1]*rd))
      else:
          pmat[j]=np.exp((np.log(pmat[j-1])*lapserate[j]*rd+np.log(tmat[j-1]+zmat[j-1]*lapserate[j])*grav-np.log(tmat[j-1]+zmat[j]*lapserate[j])*grav)/(lapserate[j]*rd))
      tmat[j]=tmat[j-1]+lapserate[j]*(zmat[j]-zmat[j-1]);

  for k in range(k1):
      if(zt[k]<zmat[0]):
          pb[k]=np.exp((np.log(ps)*lapserate[0]*rd+np.log(tsurf+zsurf*lapserate[0])*grav-np.log(tsurf+zt[k]*lapserate[0])*grav)/(lapserate[0]*rd))
          tb[k]=tsurf+lapserate[0]*(zt[k]-zsurf)
      else:
          j=0
          while(zt[k]>=zmat[j]):
            j=j+1
          tb[k]=tmat[j-1]+lapserate[j]*(zt[k]-zmat[j-1])
          if(abs(lapserate[j])<1e-99):
              pb[k]=np.exp((np.log(pmat[j-1])*tmat[j-1]*rd+zmat[j-1]*grav-zt[k]*grav)/(tmat[j-1]*rd))
          else:
              pb[k]=np.exp((np.log(pmat[j-1])*lapserate[j]*rd+np.log(tmat[j-1]+zmat[j-1]*lapserate[j])*grav-np.log(tmat[j-1]+zt[k]*lapserate[j])*grav)/(lapserate[j]*rd))
      rhobf[k]=pb[k]/(rd*tb[k]) # dry estimate
  return rhobf