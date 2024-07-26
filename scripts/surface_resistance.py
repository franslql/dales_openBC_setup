import xarray as xr
from datetime import datetime,timezone
import numpy as np

def surface_resistance(input,grid,data,transform):
  mask = (data['land_mask'].interp(y=grid.yt, x=grid.xt, assume_sorted=True).rename({'y': 'yt', "x": 'xt'}).round().values).astype(int)
  rs = np.where(mask==1,input['rs']['rs_land'],input['rs']['rs_ocean'])
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
  rs = xr.DataArray(rs,
                    dims=['yt','xt'],
                    coords={'yt': grid.yt,'xt': grid.xt},
                    name = 'rs')
  mask = xr.DataArray(mask,
                      dims=['yt','xt'],
                      coords={'yt': grid.yt,'xt': grid.xt},
                      name = 'land_mask')
  # Add together
  rs = xr.merge([rs,mask,Lat,Lon],combine_attrs='drop')
  # # Add transform
  rs = rs.assign({'transform' : xr.DataArray(data="",attrs=data['transform'].attrs)})
  # Add metadata
  rs['xt'] = rs['xt'].assign_attrs({'standard_name':'projection_x_coordinate','longname': 'X Coordinate Of Projection','units': 'm','axis':'X'})
  rs['yt'] = rs['yt'].assign_attrs({'standard_name':'projection_y_coordinate','longname': 'Y Coordinate Of Projection','units': 'm','axis':'Y'})
  rs['lat'] = rs['lat'].assign_attrs({'standard_name':'Latitude','longname': 'Latitude','units': 'degrees_north','_CoordinateAxisType':'Lat'}) 
  rs['lon'] = rs['lon'].assign_attrs({'standard_name':'Longitude','longname': 'Longitude','units': 'degrees_east','_CoordinateAxisType':'Lon'}) 
  rs['rs'] = rs['rs'].assign_attrs({'longname': 'Surface composite resistance', 'units': 's/m','coordinates':'lon lat','grid_mapping':'Transverse_Mercator'})
  rs['land_mask'] = rs['land_mask'].assign_attrs({'longname': 'Land mask', 'units': '-','coordinates':'lon lat','grid_mapping':'Transverse_Mercator'})
  rs = rs.assign_attrs({'title': f"rs.inp.{input['iexpnr']:03d}.nc",
                        'history': f"Created on {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                        'author': input['author']})
  rs.to_netcdf(path=input['outpath']+rs.attrs['title'], mode='w', format="NETCDF4")
  return rs

def surface_resistance_fine(input,grid):
  with xr.open_mfdataset(f"{input['inpath_coarse']}rs.inp.*.nc") as ds:
    rs_fine = ds.interp(xt=grid.xt+input['x_offset'],yt=grid.yt+input['y_offset'],assume_sorted=True)
    # Adjust transform
    rs_fine['transform'].attrs['false_easting'] = rs_fine['transform'].attrs['false_easting']-input['x_offset']
    rs_fine['transform'].attrs['false_northing'] = rs_fine['transform'].attrs['false_northing']-input['y_offset']
    proj4 = ''
    for param in rs_fine['transform'].attrs['proj4'][1:].split('+'):
      line = '+'+param
      if 'x_0' in param: line = f"+x_0={rs_fine['transform'].attrs['false_easting']} "
      if 'y_0' in param: line = f"+y_0={rs_fine['transform'].attrs['false_northing']} "
      proj4 = proj4+line
    rs_fine['transform'].attrs['proj4']=proj4.rstrip()
    # Set coordinates
    rs_fine = rs_fine.assign_coords({'xt':grid.xt,'yt':grid.yt})
    # Add global attributes
    rs_fine = rs_fine.assign_attrs({'title': f"rs.inp.{input['iexpnr']:03d}.nc",
                                          'history': f"Created on {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                          'author': input['author'],
                                          'time0': input['time0']})
    rs_fine.to_netcdf(path=input['outpath']+rs_fine.attrs['title'], mode='w', format="NETCDF4")
  return rs_fine

def get_ncName(filename):
    return filename.split('/')[-1]


