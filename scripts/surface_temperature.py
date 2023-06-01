import numpy as np
import xarray as xr
from datetime import datetime
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import dask
def surface_temperature(input,grid,data,transform):
  with xr.open_mfdataset(f"{input['ERA5_path']}*.nc",chunks={"time": input['tchunk']}) as ds:
    tskin = ds.T_SKIN.sel(time=slice(input['start'],input['end']))/data.exnr.sel(z=0).values
  # Transform from lat lon to rectilinear grid
  lat_era = tskin.lat.values
  lon_era = tskin.lon.values
  Lon_era,Lat_era = np.meshgrid(lon_era,lat_era)
  X_era,Y_era = transform.latlon_to_xy(Lat_era,Lon_era)
  # Get time information from era 5 data
  ts = tskin['time'].values.astype('datetime64[s]')
  dts = (ts-ts[0])/np.timedelta64(1, 's')
  time0 = pd.to_datetime(str(tskin['time'][0].values)).strftime('%Y-%m-%d %H:%M:%S')
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
  tskin_int = tskin_int.assign({'transform' : data['transform']})
  # Set attributes
  tskin_int['time'] = tskin_int['time'].assign_attrs({'longname': 'Time', 'units': f"seconds since {time0}"})
  tskin_int['xt'] = tskin_int['xt'].assign_attrs({'longname': 'West-East displacement of cell centers','units': 'm'})
  tskin_int['yt'] = tskin_int['yt'].assign_attrs({'longname': 'South-North displacement of cell centers','units': 'm'})
  tskin_int['lat'] = tskin_int['lat'].assign_attrs({'longname': 'Latitude of cell centers','units': 'degrees_north'}) 
  tskin_int['lon'] = tskin_int['lon'].assign_attrs({'longname': 'Longitude of cell centers','units': 'degrees_east'}) 
  tskin_int['tskin'] = tskin_int['tskin'].assign_attrs({'longname': 'Skin temperature', 'units': 'K'})
  tskin_int = tskin_int.assign_attrs({'title': f"tskin.inp.{input['iexpnr']:03d}.nc",
                                        'history': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                        'author': input['author'],
                                        'time0': time0,
                                        'exnrs': data.exnr.sel(z=0).values})
  tskin_int.to_netcdf(path=input['outpath']+tskin_int.attrs['title'], mode='w', format="NETCDF4")
  return tskin_int

def load_data(var,index,drop=False):
  return var.isel(index,drop=drop).values

def interp_tskin(data,tri,x,y):
  data_int = np.zeros([np.shape(data)[0],*np.shape(x)])
  for it in range(np.shape(data)[0]):
    interpolator = LinearNDInterpolator(tri, data[it,:,:].flatten())
    data_int[it,:,:] = interpolator(x,y)
  return data_int