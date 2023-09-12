def main():
  # %% Paths to aus2200 level data, aus2200 surface data, and path to write transformed data to
  pathDataLev  = '/g/data/hh5/tmp/WACI-Hackathon-2023/AUS2200/data/lev/1hr/'
  pathDataSurf = '/g/data/hh5/tmp/WACI-Hackathon-2023/AUS2200/data/surf/1hr/'
  fileOrog     = pathDataSurf+'orog_1hr_20220222_20220307.nc'
  pathWrite    = '/home/565/fl2086/aus2200_DALES/aus2200/utm/'
  # lat/lon box to transform
  latS   = -22
  latN   = -20
  lonW   =  146
  lonE   =  150
  buffer = 0.1
  dx     = 2200
  dy     = 2200
  # Time window to transform
  timeStart = '2022-03-05 00:00:00'
  timeEnd   = '2022-03-06 00:00:00' # included
  # Variables to do
  varsLev  = []
  varsSurf = ['hfss','hfls','tauu','tauv']
  # Time chunking
  tchunk = 1

  for varName in varsLev:
    print(varName)
    var,orog = loadData(pathDataLev,varName,fileOrog,timeStart,timeEnd,latS,latN,lonW,lonE,buffer,tchunk)
    var_zint = InterpolateZ(orog,var,timeDim='time',otherDims={'lon':'lon','lat':'lat'},level_choice='auto',quiet=False,byChunks=True).load()
    var_zint_utm = wgs84_to_utm(var_zint.chunk({'time':tchunk}),dx,dy,latS,latN,lonW,lonE,byChunks=True).load()
    var_zint_utm.to_netcdf(path=f"{pathWrite}{varName}.nc")
  
  for varName in varsSurf:
    print(varName)
    var,_ = loadData(pathDataSurf,varName,fileOrog,timeStart,timeEnd,latS,latN,lonW,lonE,buffer,tchunk)
    var_utm = wgs84_to_utm(var,dx,dy,latS,latN,lonW,lonE,byChunks=True).load()
    var_utm.to_netcdf(path=f"{pathWrite}{varName}.nc")

# Load data and clip to window
def loadData(path,varName,fileOrog,timeStart,timeEnd,latS,latN,lonW,lonE,buffer,tchunk):
  import xarray as xr
  import os
  var = []
  for file in sorted(os.listdir(f"{path}")):
    if(file.startswith(f"{varName}_1hr_") and file.endswith('.nc')):
      with xr.open_dataset(f"{path}{file}",chunks={"time": tchunk}) as ds:
        ds  = ds.rename({ds[varName].dims[-2]: 'lat', ds[varName].dims[-1]: 'lon'})
        var.append(ds[varName].sel(lat=slice(latS-buffer,latN+buffer),lon=slice(lonW-buffer,lonE+buffer)))
  var = xr.concat(var[:],'time')
  var = var.sel(time=slice(timeStart, timeEnd))
  # with xr.open_mfdataset(f"{path}{varName}_*.nc",chunks={"time": tchunk},data_vars=[varName],preprocess=preprocess) as ds:
  #   ds  = ds.rename({ds[varName].dims[-2]: 'lat', ds[varName].dims[-1]: 'lon'})
  #   var = ds[varName].sel(time=slice(timeStart, timeEnd),lat=slice(latS-buffer,latN+buffer),lon=slice(lonW-buffer,lonE+buffer))
  with xr.open_dataset(fileOrog) as ds:
    ds  = ds.rename({ds['orog'].dims[-2]: 'lat', ds['orog'].dims[-1]: 'lon'})
    orog = ds.orog[0,:,:].sel(lat=slice(latS,latN),lon=slice(lonW,lonE))
  return var,orog   

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
def wgs84_to_utm(var,dx,dy,latS,latN,lonW,lonE,byChunks=False):
  import numpy as np
  import xarray as xr
  from pyproj import Transformer
  from pyproj import CRS
  from pyproj.aoi import AreaOfInterest
  from pyproj.database import query_utm_crs_info
  import dask.array as da
  import dask.delayed as ddelay
  # Get the UTM zone corresponding to the window
  utm_crs_list = query_utm_crs_info(
    datum_name="WGS 84",
    area_of_interest=AreaOfInterest(
        west_lon_degree=var.lon.min(),
        south_lat_degree=var.lat.min(),
        east_lon_degree=var.lon.max(),
        north_lat_degree=var.lat.max(),
    ),
  )
  target_crs = CRS.from_epsg(utm_crs_list[0].code)
  print(f"Target crs {target_crs}")
  # Construct transform and transform coordinates
  source_crs = 'epsg:4326'
  latlon_to_utm = Transformer.from_crs(source_crs,target_crs)
  Lon,Lat = np.meshgrid(var.lon,var.lat)
  X,Y = latlon_to_utm.transform(Lat,Lon)
  var.coords['x'] = (('lat','lon'), X)
  var.coords['y'] = (('lat','lon'), Y)
  var.x.attrs['axis']='X'
  var.x.attrs['units']='m'
  var.x.attrs['standard_name']='projection_x_coordinate'
  var.x.attrs['long_name'] =f"X Coordinate Of Projection {target_crs}"
  var.y.attrs['axis']='Y'
  var.y.attrs['units']='m'
  var.y.attrs['standard_name']='projection_y_coordinate'
  var.y.attrs['long_name'] =f"Y Coordinate Of Projection {target_crs}"
  # Find rectangular grid
  Lon,Lat = np.meshgrid(np.array([latS,latN]),np.array([lonW,lonE]))
  X,Y = latlon_to_utm.transform(Lon,Lat)
  x_int = np.arange(X.min(),X.max()+dx,dx)
  y_int = np.arange(Y.min(),Y.max()+dy,dy)
  X_int,Y_int = np.meshgrid(x_int,y_int)
  # Find lat and lon for new variable
  utm_to_latlon = Transformer.from_crs(target_crs,source_crs)
  newLat,newLon = utm_to_latlon.transform(X_int,Y_int)
  # Set dimensions and coordinates new variable
  if(len(var.shape)==3): # Surface variable
    newCoords = {'time' :var.coords['time'],
                  'y'   :y_int,
                  'x'   :x_int,
                  'lat' :(["y", "x"], newLat),
                  'lon' :(["y", "x"], newLon)
    }
    newDims = ['time','y','x']
    newShape = [len(var.time),len(y_int),len(x_int)]
    lsurf = True
  else: # 3D variable
    newCoords = {'time' :var.coords['time'],
                'z'     :var.coords['z'],
                'y'     :y_int,
                'x'     :x_int,
                'lat' :(["y", "x"], newLat),
                'lon' :(["y", "x"], newLon)
    }
    newDims = ['time','z','y','x']
    newShape = [len(var.time),len(var.z),len(y_int),len(x_int)]
    lsurf = False
  # Preallocate variables
  varInt = []
  ts = 0
  # Take time chunk size into account
  tInd = var.get_axis_num('time')
  if byChunks:
    tchunks = var.chunks[tInd]
    tlen = len(tchunks)
  else:
    tlen = var.time.shape[0]
    tchunks = np.ones((tlen,))
  # Do interpolation
  def interpolation(var,xint,yint,ts,te,lsurf=False):
    from scipy import interpolate
    if(lsurf): # Surface variable
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
    else: # 3D variable
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
    return varOut
  for it in range(tlen): # Interpolation per time step
    te = ts + int(tchunks[it])
    varInt.append(
      ddelay(interpolation)(var,X_int,Y_int,ts,te,lsurf=lsurf)
    )
    ts = te
  # Convert to xarray
  varInt = ddelay(np.concatenate)(varInt[:],axis=0)
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
  varInt.x.attrs['long_name'] =f"X Coordinate Of Projection {target_crs}"
  varInt.y.attrs['axis']='Y'
  varInt.y.attrs['units']='m'
  varInt.y.attrs['standard_name']='projection_y_coordinate'
  varInt.y.attrs['long_name'] =f"Y Coordinate Of Projection {target_crs}"
  return varInt

# %%
if __name__ == "__main__": main()
# %%
