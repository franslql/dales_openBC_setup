# Crops the HARMONIE data to the required time and spatial extends. 
# Transforms pressure coordinates into height levels.
# Transforms HARMONIE prognostic variables to DALES prognostic variables
import numpy as np
import xarray as xr
from pyproj import Transformer
def prep_harmonie(input,grid):
  variables = ['ua','va','wa','ta','hus','clw','ps','tas','huss']
  data = {}
  # Open data and crop data
  for var in variables:
    with xr.open_mfdataset(f"{input['inpath']}var*.nc",chunks={"time": tchunk}) as ds:
      # Read transform information and transform lat/lon of southwest corner to harmonie x/y
      transform = Transform(ds['Lambert_Conformal'].attrs)
      x_sw,y_sw = transform.latlon_to_xy(input['lat_sw'],input['lon_sw'])
      # Crop data to time and spatial range, using harmonie spatial resolution as buffer
      dx = (ds['x'].values[1]-ds['x'][0]).values; dy = (ds['y'][1]-ds['y'][0]).values
      data[var] = ds[var].sel(time=slice(input['start'], input['end']),x=slice(x_sw-dx,x_sw+grid.xsize+dx),y=slice(y_sw-dy,y_sw+grid.ysize+dy))
    # Set south west corner of DALES as origin
    data[var]['x'] = data[var]['x']-x_sw; data[var]['y']-y_sw
    # Change coordinate transform accordingly
    transform.parameters['false_easting'] = transform.parameters['false_easting']-x_sw
    transform.parameters['false_northing']= transform.parameters['false_northing']-y_sw
    proj4 = ''
    for param in transform.parameters['proj4'][1:].split('+'):
      line = '+'+param
      if 'x_0' in param: line = f"+x_0={transform.parameters['false_easting']} "
      if 'y_0' in param: line = f"+y_0={transform.parameters['false_northing']} "
      proj4 = proj4+line
    transform.parameters['proj4']=proj4.rstrip()
  # Interpolate pressure levels to regular height levels
  for var in variables:
    p = calc_pressure(fields)
    z = calc_height(p,fields)
    fields = interpolate_height(z,fields)
  # Change to DALES prognostic variables

def calc_pressure(fields):
  # Calculate pressure with model coefficients given in H43_65lev.txt
  coeff = np.loadtxt('H43_65lev.txt')
  ph = coeff[:,1,None,None]+(ps[None,:,:]*coeff[:,2,None,None])
  p = 0.5*(ph[1:,:,:]+ph[:-1,:,:])
  return p[::-1,:,:]

def calc_height(p,fields):
  # Transform pressure levels into height levels
  p   = np.concatenate((fields['ps'],p),axis=)
  rho = p/(Rd*T*(1+(Rv/Rd-1)*qt-Rv/Rd*ql))
  rhoh= (rho[:-1,:,:]+rho[1:,:,:])/2
  z   = np.zeros(np.shape(T))
  for k in np.arange(1,np.shape(u)[0]):
    z[k,:,:] = z[k-1,:,:]-(p[k,:,:]-p[k-1,:,:])/(rhoh[k-1,:,:]*grav)
  return z

def interpolate_height(fields,var,grid)
  

class Transform:
  def __init__(self,parameters):
    self.parameters = parameters
    self.crs_latlon = 'epsg:4326'
    # Construct transformation objects
    self.latlon_to_xy_transform = Transformer.from_crs(self.crs_latlon,self.parameters['proj4'])
    self.xy_to_latlon_transform = Transformer.from_crs(self.parameters['proj4'],self.crs_latlon)

  def latlon_to_xy(self,lat,lon):
    return self.latlon_to_xy_transform.transform(lat,lon)

  def xy_to_latlon(self,x,y):
    return self.xy_to_latlon_transform.transform(x,y)

with xr.open_mfdataset(f"/perm/nmfl/eureca/harmonie/ta_*.nc",chunks={"time": 1}) as ds:
  x = ds['x'].values
  y = ds['y'].values
  lat = ds['lat'].values
  lon = ds['lon'].values
  transform_info = ds['Lambert_Conformal'].attrs
  transform = Transform(transform_info)
