import xarray as xr
from datetime import datetime,timezone
import numpy as np

def surface_roughness(input,grid,data,transform):
  mask = (data['land_mask'].interp(y=grid.yt, x=grid.xt, assume_sorted=True).rename({'y': 'yt', "x": 'xt'}).round().values).astype(int)
  z0 = np.where(mask==1,input['z0']['z0_land'],input['z0']['z0_ocean'])
  # Get lat lon
  X,Y = np.meshgrid(grid.xt,grid.yt)
  Lat,Lon = transform.xy_to_latlon(X,Y)
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
  # Convert back to xarray
  z0 = xr.DataArray(z0,
                    dims=['yt','xt'],
                    coords={'yt': grid.yt,'xt': grid.xt},
                    name = 'z0')
  mask = xr.DataArray(mask,
                      dims=['yt','xt'],
                      coords={'yt': grid.yt,'xt': grid.xt},
                      name = 'land_mask')
  # Add together
  z0 = xr.merge([z0,mask,Lat,Lon],combine_attrs='drop')
  # # Add transform
  z0 = z0.assign({'transform' : xr.DataArray(data="",attrs=data['transform'].attrs)})
  # Add metadata
  z0['xt'] = z0['xt'].assign_attrs({'standard_name':'projection_x_coordinate','longname': 'X Coordinate Of Projection','units': 'm','axis':'X'})
  z0['yt'] = z0['yt'].assign_attrs({'standard_name':'projection_y_coordinate','longname': 'Y Coordinate Of Projection','units': 'm','axis':'Y'})
  z0['lat'] = z0['lat'].assign_attrs({'standard_name':'Latitude','longname': 'Latitude','units': 'degrees_north','_CoordinateAxisType':'Lat'}) 
  z0['lon'] = z0['lon'].assign_attrs({'standard_name':'Longitude','longname': 'Longitude','units': 'degrees_east','_CoordinateAxisType':'Lon'}) 
  z0['z0'] = z0['z0'].assign_attrs({'longname': 'Surface roughness', 'units': 'm','coordinates':'lon lat','grid_mapping':'Transverse_Mercator'})
  z0['land_mask'] = z0['land_mask'].assign_attrs({'longname': 'Land mask', 'units': '-','coordinates':'lon lat','grid_mapping':'Transverse_Mercator'})
  z0 = z0.assign_attrs({'title': f"z0.inp.{input['iexpnr']:03d}.nc",
                        'history': f"Created on {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                        'author': input['author']})
  z0.to_netcdf(path=input['outpath']+z0.attrs['title'], mode='w', format="NETCDF4")
  return z0

def surface_roughness_fine(input,grid):
  with xr.open_mfdataset(f"{input['inpath_coarse']}z0.inp.*.nc",chunks={"time": input['tchunk']}) as ds:
    z0_fine = ds.sel(time=slice(input['start'], input['end'])).interp(xt=grid.xt+input['x_offset'],yt=grid.yt+input['y_offset'],assume_sorted=True)
    # Adjust transform
    z0_fine['transform'].attrs['false_easting'] = z0_fine['transform'].attrs['false_easting']-input['x_offset']
    z0_fine['transform'].attrs['false_northing'] = z0_fine['transform'].attrs['false_northing']-input['y_offset']
    proj4 = ''
    for param in z0_fine['transform'].attrs['proj4'][1:].split('+'):
      line = '+'+param
      if 'x_0' in param: line = f"+x_0={z0_fine['transform'].attrs['false_easting']} "
      if 'y_0' in param: line = f"+y_0={z0_fine['transform'].attrs['false_northing']} "
      proj4 = proj4+line
    z0_fine['transform'].attrs['proj4']=proj4.rstrip()
    # Set time information
    ts = z0_fine['time'].values.astype('datetime64[s]')
    dts = (ts-np.datetime64(input['time0'],'s'))/np.timedelta64(1, 's')
    # Set coordinates
    z0_fine = z0_fine.assign_coords({'time': dts, 'xt':grid.xt,'yt':grid.yt})
    # Add global attributes
    z0_fine = z0_fine.assign_attrs({'title': f"z0.inp.{input['iexpnr']:03d}.nc",
                                          'history': f"Created on {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                          'author': input['author'],
                                          'time0': input['time0']})
    z0_fine.to_netcdf(path=input['outpath']+z0_fine.attrs['title'], mode='w', format="NETCDF4")
  return z0_fine

def get_ncName(filename):
    return filename.split('/')[-1]


