import numpy as np
import xarray as xr
from datetime import datetime
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
from datetime import datetime
import dask
def surface_temperature_cosmo(input,grid,data,transform):
  # Interpolate to DALES grid
  tskin_int = data['thl'].sel(z=0,drop=True).interp(y=grid.yt, x=grid.xt).rename('tskin').rename({'x':'xt','y':'yt'})
  print(tskin_int)
  # Adjust time variable to seconds since initial field
  ts = tskin_int['time'].values.astype('datetime64[s]')
  dts = (ts-np.datetime64(input['time0'],'s'))/np.timedelta64(1, 's')
  tskin_int = tskin_int.assign_coords({'time':('time', dts)})
  tskin_int['time'].attrs.clear()
  # Add transform information
  tskin_int = tskin_int.to_dataset().assign({'transform' : data['transform']})
  # Set attributes
  tskin_int['time'] = tskin_int['time'].assign_attrs({'longname': 'Time', 'units': f"seconds since {input['time0']}"})
  tskin_int['xt'] = tskin_int['xt'].assign_attrs({'longname': 'West-East displacement of cell centers','units': 'm'})
  tskin_int['yt'] = tskin_int['yt'].assign_attrs({'longname': 'South-North displacement of cell centers','units': 'm'})
  tskin_int['lat'] = tskin_int['lat'].assign_attrs({'longname': 'Latitude of cell centers','units': 'degrees_north'}) 
  tskin_int['lon'] = tskin_int['lon'].assign_attrs({'longname': 'Longitude of cell centers','units': 'degrees_east'}) 
  tskin_int['tskin'] = tskin_int['tskin'].assign_attrs({'longname': 'Skin temperature', 'units': 'K'})
  tskin_int = tskin_int.assign_attrs({'title': f"tskin.inp.{input['iexpnr']:03d}.nc",
                                        'history': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                        'author': input['author'],
                                        'time0': input['time0'],
                                        'exnrs': data.exnr.sel(z=0).values})
  tskin_int.to_netcdf(path=input['outpath']+tskin_int.attrs['title'], mode='w', format="NETCDF4")
  return tskin_int
