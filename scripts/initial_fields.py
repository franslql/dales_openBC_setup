# Interpolate initial field to DALES grid
# Creates initfields.inp.xxx.nc
import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd
def initial_fields(input,grid,data,transform):
  time0 = pd.to_datetime(str(data['time'][0].values)).strftime('%Y-%m-%d %H:%M:%S')
  data = data.isel({'time': 0},drop=True).drop(['lat','lon']).chunk({'z':1})
  # Interpolate data to DALES staggered grid
  u0   = data['u'].interp(z=grid.zt, y=grid.yt, x=grid.xm, assume_sorted=True).rename({'z': 'zt', 'y': 'yt', "x": 'xm'}).rename('u0')
  v0   = data['v'].interp(z=grid.zt, y=grid.ym, x=grid.xt, assume_sorted=True).rename({'z': 'zt', 'y': 'ym', "x": 'xt'}).rename('v0')
  w0   = data['w'].interp(z=grid.zm, y=grid.yt, x=grid.xt, assume_sorted=True).rename({'z': 'zm', 'y': 'yt', "x": 'xt'}).rename('w0')
  thl0 = data['thl'].interp(z=grid.zt, y=grid.yt, x=grid.xt, assume_sorted=True).rename({'z': 'zt', 'y': 'yt', "x": 'xt'}).rename('thl0')
  qt0  = data['qt'].interp(z=grid.zt, y=grid.yt, x=grid.xt, assume_sorted=True).rename({'z': 'zt', 'y': 'yt', "x": 'xt'}).rename('qt0')
  e120 = (xr.ones_like(thl0)*input['e12']).rename('e120')
  u0.attrs.clear(); v0.attrs.clear(); w0.attrs.clear(); thl0.attrs.clear() ;qt0.attrs.clear()
  # Calculate lat lon for cell centers
  X,Y  = np.meshgrid(grid.xt,grid.yt)
  lat,lon = transform.xy_to_latlon(X,Y)
  lat = xr.DataArray(lat,
                     dims = ['yt','xt'],
                     coords={'yt': grid.yt, 'xt': grid.xt},
                     name = 'lat'
                     )
  lon = xr.DataArray(lon,
                     dims = ['yt','xt'],
                     coords={'yt': grid.yt, 'xt': grid.xt},
                     name = 'lon'
                     )
  # Create dataset and add initial fields
  initfields = xr.merge([u0,v0,w0,thl0,qt0,e120,lat,lon],combine_attrs='drop')
  # Add transform
  initfields = initfields.assign({'transform' : data['transform']})
  # Add variable attributes
  initfields['xt'] = initfields['xt'].assign_attrs({'longname': 'West-East displacement of cell centers','units': 'm'})
  initfields['xm'] = initfields['xm'].assign_attrs({'longname': 'West-East displacement of cell edges','units': 'm'})
  initfields['yt'] = initfields['yt'].assign_attrs({'longname': 'South-North displacement of cell centers','units': 'm'})
  initfields['ym'] = initfields['ym'].assign_attrs({'longname': 'South-North displacement of cell edges','units': 'm'})
  initfields['zt'] = initfields['zt'].assign_attrs({'longname': 'Vertical displacement of cell centers','units': 'm'})
  initfields['zm'] = initfields['zm'].assign_attrs({'longname': 'Vertical displacement of cell edges','units': 'm'})
  initfields['u0'] = initfields['u0'].assign_attrs({'longname': 'Initial West-East velocity','units': 'm/s'})
  initfields['v0'] = initfields['v0'].assign_attrs({'longname': 'Initial South-North velocity','units': 'm/s'})
  initfields['w0'] = initfields['w0'].assign_attrs({'longname': 'Initial vertical velocity','units': 'm/s'})
  initfields['thl0'] = initfields['thl0'].assign_attrs({'longname': 'Initial liquid water potential temperature','units': 'K'})
  initfields['qt0'] = initfields['qt0'].assign_attrs({'longname': 'Initial total water specific humidity','units': 'kg/kg'})
  initfields['e120'] = initfields['e120'].assign_attrs({'longname': 'Initial square root of turbulent kinetic energy','units': 'm/s'})
  initfields['lat'] = initfields['lat'].assign_attrs({'longname': 'Latitude of cell centers','units': 'degrees_north'}) 
  initfields['lon'] = initfields['lon'].assign_attrs({'longname': 'Longitude of cell centers','units': 'degrees_east'}) 
  # Add global attributes
  initfields = initfields.assign_attrs({'title': f"initfields.inp.{input['iexpnr']:03d}.nc",
                                        'history': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                        'author': input['author'],
                                        'time0': time0})
  # Save data
  initfields.to_netcdf(path=input['outpath']+initfields.attrs['title'], mode='w', format="NETCDF4")
  return initfields