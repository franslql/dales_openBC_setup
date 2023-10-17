# Crops the cosmo data to the required time and spatial extends. 
# Transforms wgs84 to local utm coordinates
# Transforms cosmo prognostic variables to DALES prognostic variables
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
# Constants
# Simulation constants (modglobal)
p0   = 1e5          # Reference pressure
Rd   = 287.04       # Gas constant for dry air
Rv   = 461.5        # Gas constant for water vapor
cp   = 1004.        # Specific heat at constant pressure (dry air)
Lv   = 2.53e6       # Latent heat for vaporisation
grav = 9.81         # Gravitational constant
kappa= 0.4          # Von Karman constant
def prep_cosmo(input,grid):
  # Create transform to go from lat lon to local rectilinear grid
  transform = Transform([input['lon_sw'],input['lon_sw']+grid.xsize/100000],\
                        [input['lat_sw'],input['lat_sw']+grid.ysize/100000])
  x_sw,y_sw = transform.latlon_to_xy(input['lat_sw'],input['lon_sw'])
  # Translate to make southwest corner of DALES origin
  transform.update_parameters(x_0=transform.parameters['x_0']-x_sw,\
                              y_0=transform.parameters['y_0']-y_sw)
  # Get domain box in WGS84
  lat_ne,lon_ne = transform.xy_to_latlon(grid.xsize,grid.ysize)
  lat_se,lon_se = transform.xy_to_latlon(grid.xsize,0)
  lat_nw,lon_nw = transform.xy_to_latlon(0,grid.ysize)
  lat_min       = np.array([input['lat_sw'],lat_se,lat_ne,lat_nw]).min()
  lat_max       = np.array([input['lat_sw'],lat_se,lat_ne,lat_nw]).max()
  lon_min       = np.array([input['lon_sw'],lon_se,lon_ne,lon_nw]).min()
  lon_max       = np.array([input['lon_sw'],lon_se,lon_ne,lon_nw]).max()
  buffer        = 0.05
  def read_variables(filenames,varCodes,varNames,transform,lsurf=False):
    data = []
    for filename in filenames:
      with xr.open_dataset(filename) as ds:
        ds = ds.sel(time=slice(input['start'],input['end']))
        if(len(ds['time'])<1): continue
        ds_new = []
        for varCode,varName in zip(varCodes,varNames):
          var = ds[varCode]
          dims = var.dims
          var = var.sel({dims[-2]:slice(lat_min-buffer,lat_max+buffer),\
                         dims[-1]:slice(lon_min-buffer,lon_max+buffer)})
          dims = var.dims
          var = var.rename(varName)
          if not lsurf: var = var.rename({dims[-3]:'z'})
          var = wgs84_to_utm(var,2000,2000,grid.xsize,grid.ysize,transform,byChunks=False)
          ds_new.append(var)
        ds_new = xr.merge(ds_new)
      data.append(ds_new)
    data = xr.concat(data[:],'time')
    return data
  # Read 3D variables
  varCodes = ['U','V','W','T','P','QV','QC']
  varNames = ['u','v','w','t','p','qv','ql']
  filenames = glob.glob(input['inpath']+"levels/*.nc")
  filenames.sort()
  data = read_variables(filenames,varCodes,varNames,transform)
  data = data.assign({'transform' : xr.DataArray([],name='Transverse_Mercator',attrs=transform.parameters)})
  # Read surface variables
  varCodes = ['PS','T_S']
  varNames = ['p','t']
  filenames = glob.glob(input['inpath']+"surface/*.nc")
  filenames.sort()
  datas = read_variables(filenames,varCodes,varNames,transform,lsurf=True)
  datas = datas.assign({'transform' : xr.DataArray([],name='Transverse_Mercator',attrs=transform.parameters)})
  datas = datas.assign({'u' : xr.zeros_like(datas.t),
                        'v' : xr.zeros_like(datas.t),
                        'w' : xr.zeros_like(datas.t),
                        'qv': data.qv.isel(z=0),
                        'ql': xr.zeros_like(datas.t)})
  datas = datas.expand_dims({'z': np.array([0.])},axis=1)
  data = xr.concat([datas,data],dim='z')
  # Calculate qt
  data = data.assign({'qt': data['qv']+data['ql']})
  # Calculate base profiles and exnr function
  tas_exnr = data['t'].isel({'time': 0, 'z': 0},drop=True).sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values
  ps_exnr  = data['p'].isel({'time': 0, 'z': 0},drop=True).sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values
  exnrs    = (ps_exnr/p0)**(Rd/cp)
  thls_exnr= tas_exnr/exnrs
  rhobf    = calcBaseprof(data.z.values,thls_exnr,ps_exnr,pref0=p0)
  p_exnr   = rhobf[1:]*Rd*data['t'][0,1:].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values*\
           ( 1+(Rv/Rd-1)*data['qt'][0,1:].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values-\
             Rv/Rd*data['ql'][0,1:].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).values ) # Ideal gas law
  exnr     = (p_exnr/p0)**(Rd/cp)
  exnr     = xr.DataArray(np.concatenate([exnrs[None],exnr]),
                          dims = ['z'],
                          coords={'z': data.z},
                          name = 'exnr',
                          attrs = {'thls': thls_exnr, 'ps': ps_exnr})
  data     = data.assign({'exnr': exnr})
  # Calculate liquid potential temperature and total specific humidity
  thl      = data['t']/exnr-Lv*data['ql']/(cp*exnr)
  data     = data.assign({'thl': thl})
  # Organize data, rename variables and drop non DALES prognostic variables
  data = data.drop(['t','p','ql','qv'])\
    .assign({'transform' : xr.DataArray([],name='Transverse_Mercator',attrs=transform.parameters)})
  return data,transform

# Convert data from lat lon grid to x y grid (wgs84 -> utm)
def wgs84_to_utm(var,dx,dy,xsize,ysize,transform,byChunks=False):
  X,Y = transform.latlon_to_xy(var.lat,var.lon)
  var.coords['x'] = (('rlat','rlon'), X)
  var.coords['y'] = (('rlat','rlon'), Y)
  var.x.attrs['axis']='X'
  var.x.attrs['units']='m'
  var.x.attrs['standard_name']='projection_x_coordinate'
  var.x.attrs['long_name'] =f"X Coordinate Of Projection"
  var.y.attrs['axis']='Y'
  var.y.attrs['units']='m'
  var.y.attrs['standard_name']='projection_y_coordinate'
  var.y.attrs['long_name'] =f"Y Coordinate Of Projection"
  # Find rectangular grid
  x_int = np.arange(0,xsize+2*dx,dx)
  y_int = np.arange(0,ysize+2*dy,dy)
  X_int,Y_int = np.meshgrid(x_int,y_int)
  # Find lat and lon for new variable
  newLat,newLon = transform.xy_to_latlon(X_int,Y_int)
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
    x_0 = float(wkt.split('\"False easting\",')[-1].split(',')[0])
    y_0 = float(wkt.split('\"False northing\",')[-1].split(',')[0])
    self.update_parameters(k_0=k_0,lat_0=lat_0,lon_0=lon_0,x_0=x_0,y_0=y_0)

  def update_parameters(self,k_0=None,lat_0=None,lon_0=None,x_0=None,y_0=None):
    k_0 = k_0 if k_0!=None else self.parameters['k_0']
    lat_0 = lat_0 if lat_0!=None else self.parameters['lat_0']
    lon_0 = lon_0 if lon_0!=None else self.parameters['lon_0']
    x_0 = x_0 if x_0!=None else self.parameters['x_0']
    y_0 = y_0 if y_0!=None else self.parameters['y_0']
    self.parameters = dict(ellps='WGS84',k_0=k_0,lat_0=lat_0,lon_0=lon_0,x_0=x_0,y_0=y_0,proj4=\
    f"+proj=tmerc +ellps=WGS84 +k_0={k_0} +lat_0={lat_0} +lon_0={lon_0} +x_0={x_0} +y_0={y_0}")
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