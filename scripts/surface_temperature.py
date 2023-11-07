import numpy as np
import xarray as xr
from datetime import datetime
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import dask
import glob
def surface_temperature(input,grid,data,transform):
  def get_ncName(filename):
    return filename.split('/')[-1]
  filenames = glob.glob(f"{input['tskin']['ERA5_path']}*/*.nc")
  filenames.sort(key=get_ncName)
  # Get domain box in WGS84
  lat_ne,lon_ne = transform.xy_to_latlon(grid.xsize,grid.ysize)
  lat_se,lon_se = transform.xy_to_latlon(grid.xsize,0)
  lat_nw,lon_nw = transform.xy_to_latlon(0,grid.ysize)
  lat_min       = np.array([input['lat_sw'],lat_se,lat_ne,lat_nw]).min()
  lat_max       = np.array([input['lat_sw'],lat_se,lat_ne,lat_nw]).max()
  lon_min       = np.array([input['lon_sw'],lon_se,lon_ne,lon_nw]).min()
  lon_max       = np.array([input['lon_sw'],lon_se,lon_ne,lon_nw]).max()
  buffer        = 0.25
  with xr.open_mfdataset(filenames,chunks={"time": input['tchunk']}) as ds:
    tskin = ds.skt.sel({'time':slice(input['start'],input['end']),\
                        ds.skt.dims[-2]:slice(lat_max+buffer,lat_min-buffer),\
                        ds.skt.dims[-1]:slice(lon_min-buffer,lon_max+buffer)})\
                        /data.exnr.sel(z=0).values
  # Transform from lat lon to rectilinear grid
  if(tskin.dims[-2]!='lat'): tskin = tskin.rename({tskin.dims[-2]:'lat'})
  if(tskin.dims[-1]!='lon'): tskin = tskin.rename({tskin.dims[-1]:'lon'})
  lat_era = tskin.lat.values
  lon_era = tskin.lon.values
  Lon_era,Lat_era = np.meshgrid(lon_era,lat_era)
  X_era,Y_era = transform.latlon_to_xy(Lat_era,Lon_era)
  # Get time information from era 5 data
  ts = tskin['time'].values.astype('datetime64[s]')
  dts = (ts-np.datetime64(input['time0'],'s'))/np.timedelta64(1, 's')
  # Compute the triangulation for interpolation (stays the same for every time step)
  tri = Delaunay(list(zip(X_era.flatten(),Y_era.flatten())))
  X,Y = np.meshgrid(grid.xt,grid.yt)
  Lat,Lon = transform.xy_to_latlon(X,Y)
  # Preallocate variables for time loop and set new shape and coordiantes
  its = 0
  tskin_int = []
  new_shape = [tskin.sizes['time'],grid.jtot,grid.itot]
  new_coords = {'time': dts,
                'yt': grid.yt,
                'xt': grid.xt}
  new_dims = ['time','yt','xt']
  # Interpolate data from unstructured grid to structured rectilinear grid per time step
  for tchunk in tskin.chunks[0]:
    ite = its+tchunk
    data_slice = dask.delayed(load_data)(tskin,{'time': np.arange(its,ite)},drop=False)
    tskin_int.append(dask.delayed(interp_tskin)(data_slice,tri,X,Y))
    its = ite
  tskin_int = dask.delayed(np.concatenate)(tskin_int,axis=0)
  tskin_int = xr.DataArray(dask.array.from_delayed(tskin_int,new_shape,dtype=float),
                            dims=new_dims,
                            coords=new_coords,
                            name = 'tskin').chunk({'time': input['tchunk']})
  Lat = xr.DataArray(Lat,
                     dims = ['yt','xt'],
                     coords={'yt': grid.yt, 'xt': grid.xt},
                     name = 'lat'
                     )
  Lon = xr.DataArray(Lon,
                     dims = ['yt','xt'],
                     coords={'yt': grid.yt, 'xt': grid.xt},
                     name = 'lon'
                     )
  tskin_int = xr.merge([tskin_int,Lat,Lon],combine_attrs='drop')
  # Add transform
  tskin_int = tskin_int.assign({'transform' : xr.DataArray(data="",attrs=data['transform'].attrs)})
  # Set attributes
  tskin_int['time'] = tskin_int['time'].assign_attrs({'longname': 'Time', 'units': f"seconds since {input['time0']}"})
  tskin_int['xt'] = tskin_int['xt'].assign_attrs({'standard_name':'projection_x_coordinate','longname': 'X Coordinate Of Projection','units': 'm','axis':'X'})
  tskin_int['yt'] = tskin_int['yt'].assign_attrs({'standard_name':'projection_y_coordinate','longname': 'Y Coordinate Of Projection','units': 'm','axis':'Y'})
  tskin_int['lat'] = tskin_int['lat'].assign_attrs({'standard_name':'Latitude','longname': 'Latitude','units': 'degrees_north','_CoordinateAxisType':'Lat'}) 
  tskin_int['lon'] = tskin_int['lon'].assign_attrs({'standard_name':'Longitude','longname': 'Longitude','units': 'degrees_east','_CoordinateAxisType':'Lon'}) 
  tskin_int['tskin'] = tskin_int['tskin'].assign_attrs({'longname': 'Skin temperature', 'units': 'K','coordinates':'lon lat','grid_mapping':'Transverse_Mercator','cell_methods':'time: point'})
  tskin_int = tskin_int.assign_attrs({'title': f"tskin.inp.{input['iexpnr']:03d}.nc",
                                        'history': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                        'author': input['author'],
                                        'time0': input['time0'],
                                        'exnrs': data.exnr.sel(z=0).values})
  tskin_int.to_netcdf(path=input['outpath']+tskin_int.attrs['title'], mode='w', format="NETCDF4")
  return tskin_int

def surface_temperature_fine(input,grid):
  with xr.open_mfdataset(f"{input['inpath_coarse']}tskin.inp.*.nc",chunks={"time": input['tchunk']}) as ds:
    tskin_fine = ds.sel(time=slice(input['start'], input['end'])).interp(xt=grid.xt+input['x_offset'],yt=grid.yt+input['y_offset'],assume_sorted=True)
    # Adjust transform
    tskin_fine['transform'].attrs['false_easting'] = tskin_fine['transform'].attrs['false_easting']-input['x_offset']
    tskin_fine['transform'].attrs['false_northing'] = tskin_fine['transform'].attrs['false_northing']-input['y_offset']
    proj4 = ''
    for param in tskin_fine['transform'].attrs['proj4'][1:].split('+'):
      line = '+'+param
      if 'x_0' in param: line = f"+x_0={tskin_fine['transform'].attrs['false_easting']} "
      if 'y_0' in param: line = f"+y_0={tskin_fine['transform'].attrs['false_northing']} "
      proj4 = proj4+line
    tskin_fine['transform'].attrs['proj4']=proj4.rstrip()
    # Set time information
    ts = tskin_fine['time'].values.astype('datetime64[s]')
    dts = (ts-np.datetime64(input['time0'],'s'))/np.timedelta64(1, 's')
    # Set coordinates
    tskin_fine = tskin_fine.assign_coords({'time': dts, 'xt':grid.xt,'yt':grid.yt})
    # Add global attributes
    tskin_fine = tskin_fine.assign_attrs({'title': f"tskin.inp.{input['iexpnr']:03d}.nc",
                                          'history': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                          'author': input['author'],
                                          'time0': input['time0']})
    tskin_fine.to_netcdf(path=input['outpath']+tskin_fine.attrs['title'], mode='w', format="NETCDF4")
  return tskin_fine

def load_data(var,index,drop=False):
  return var.isel(index,drop=drop).values

def interp_tskin(data,tri,x,y):
  data_int = np.zeros([np.shape(data)[0],*np.shape(x)])
  for it in range(np.shape(data)[0]):
    interpolator = LinearNDInterpolator(tri, data[it,:,:].flatten())
    data_int[it,:,:] = interpolator(x,y)
  return data_int