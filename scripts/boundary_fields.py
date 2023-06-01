# Interpolate fields to DALES domain boundary
# Creates openboundaries.inp.xxx.nc
import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd 
def boundary_fields(input,grid,data):
  data = data.drop(['lat','lon'])
  # West boundary
  uwest   = data['u'].interp({'z': grid.zt, 'y': grid.yt, 'x': grid.xm[0]}).rename({'z': 'zt', 'y': 'yt'}).rename('uwest').drop(['x'])
  vwest   = data['v'].interp({'z': grid.zt, 'y': grid.ym, 'x': grid.xm[0]}).rename({'z': 'zt', 'y': 'ym'}).rename('vwest').drop(['x'])
  wwest   = data['w'].interp({'z': grid.zm, 'y': grid.yt, 'x': grid.xm[0]}).rename({'z': 'zm', 'y': 'yt'}).rename('wwest').drop(['x'])
  thlwest = data['thl'].interp({'z': grid.zt, 'y': grid.yt, 'x': grid.xm[0]}).rename({'z': 'zt', 'y': 'yt'}).rename('thlwest').drop(['x'])
  qtwest  = data['qt'].interp({'z': grid.zt, 'y': grid.yt, 'x': grid.xm[0]}).rename({'z': 'zt', 'y': 'yt'}).rename('qtwest').drop(['x'])
  e12west = (xr.ones_like(thlwest)*input['e12']).rename('e12west')
  uwest.attrs.clear(); vwest.attrs.clear(); wwest.attrs.clear(); thlwest.attrs.clear(); qtwest.attrs.clear()
  # East boundary
  ueast   = data['u'].interp({'z': grid.zt, 'y': grid.yt, 'x': grid.xm[-1]}).rename({'z': 'zt', 'y': 'yt'}).rename('ueast').drop(['x'])
  veast   = data['v'].interp({'z': grid.zt, 'y': grid.ym, 'x': grid.xm[-1]}).rename({'z': 'zt', 'y': 'ym'}).rename('veast').drop(['x'])
  weast   = data['w'].interp({'z': grid.zm, 'y': grid.yt, 'x': grid.xm[-1]}).rename({'z': 'zm', 'y': 'yt'}).rename('weast').drop(['x'])
  thleast = data['thl'].interp({'z': grid.zt, 'y': grid.yt, 'x': grid.xm[-1]}).rename({'z': 'zt', 'y': 'yt'}).rename('thleast').drop(['x'])
  qteast  = data['qt'].interp({'z': grid.zt, 'y': grid.yt, 'x': grid.xm[-1]}).rename({'z': 'zt', 'y': 'yt'}).rename('qteast').drop(['x'])
  e12east = (xr.ones_like(thleast)*input['e12']).rename('e12east')
  ueast.attrs.clear(); veast.attrs.clear(); weast.attrs.clear(); thleast.attrs.clear(); qteast.attrs.clear()
  # South boundary
  usouth   = data['u'].interp({'z': grid.zt, 'y': grid.ym[0], 'x': grid.xm}).rename({'z': 'zt', 'x': 'xm'}).rename('usouth').drop(['y'])
  vsouth   = data['v'].interp({'z': grid.zt, 'y': grid.ym[0], 'x': grid.xt}).rename({'z': 'zt', 'x': 'xt'}).rename('vsouth').drop(['y'])
  wsouth   = data['w'].interp({'z': grid.zm, 'y': grid.ym[0], 'x': grid.xt}).rename({'z': 'zm', 'x': 'xt'}).rename('wsouth').drop(['y'])
  thlsouth = data['thl'].interp({'z': grid.zt, 'y': grid.ym[0], 'x': grid.xt}).rename({'z': 'zt', 'x': 'xt'}).rename('thlsouth').drop(['y'])
  qtsouth  = data['qt'].interp({'z': grid.zt, 'y': grid.ym[0], 'x': grid.xt}).rename({'z': 'zt', 'x': 'xt'}).rename('qtsouth').drop(['y'])
  e12south = (xr.ones_like(thlsouth)*input['e12']).rename('e12south')
  usouth.attrs.clear(); vsouth.attrs.clear(); wsouth.attrs.clear(); thlsouth.attrs.clear(); qtsouth.attrs.clear()
  # North boundary
  unorth   = data['u'].interp({'z': grid.zt, 'y': grid.ym[-1], 'x': grid.xm}).rename({'z': 'zt', 'x': 'xm'}).rename('unorth').drop(['y'])
  vnorth   = data['v'].interp({'z': grid.zt, 'y': grid.ym[-1], 'x': grid.xt}).rename({'z': 'zt', 'x': 'xt'}).rename('vnorth').drop(['y'])
  wnorth   = data['w'].interp({'z': grid.zm, 'y': grid.ym[-1], 'x': grid.xt}).rename({'z': 'zm', 'x': 'xt'}).rename('wnorth').drop(['y'])
  thlnorth = data['thl'].interp({'z': grid.zt, 'y': grid.ym[-1], 'x': grid.xt}).rename({'z': 'zt', 'x': 'xt'}).rename('thlnorth').drop(['y'])
  qtnorth  = data['qt'].interp({'z': grid.zt, 'y': grid.ym[-1], 'x': grid.xt}).rename({'z': 'zt', 'x': 'xt'}).rename('qtnorth').drop(['y'])
  e12north = (xr.ones_like(thlnorth)*input['e12']).rename('e12north')
  unorth.attrs.clear(); vnorth.attrs.clear(); wnorth.attrs.clear(); thlnorth.attrs.clear(); qtnorth.attrs.clear()
  # Top boundary
  utop   = data['u'].interp({'z': grid.zm[-1], 'y': grid.yt, 'x': grid.xm}).rename({'y': 'yt', 'x': 'xm'}).rename('utop').drop(['z'])
  vtop   = data['v'].interp({'z': grid.zm[-1], 'y': grid.ym, 'x': grid.xt}).rename({'y': 'ym', 'x': 'xt'}).rename('vtop').drop(['z'])
  wtop   = data['w'].interp({'z': grid.zm[-1], 'y': grid.yt, 'x': grid.xt}).rename({'y': 'yt', 'x': 'xt'}).rename('wtop').drop(['z'])
  thltop = data['thl'].interp({'z': grid.zm[-1], 'y': grid.yt, 'x': grid.xt}).rename({'y': 'yt', 'x': 'xt'}).rename('thltop').drop(['z'])
  qttop  = data['qt'].interp({'z': grid.zm[-1], 'y': grid.yt, 'x': grid.xt}).rename({'y': 'yt', 'x': 'xt'}).rename('qttop').drop(['z'])
  e12top = (xr.ones_like(thltop)*input['e12']).rename('e12top')
  utop.attrs.clear(); vtop.attrs.clear(); wtop.attrs.clear(); thltop.attrs.clear(); qttop.attrs.clear()
  # Add fields to dataset
  openboundaries = xr.merge([uwest, vwest, wwest, thlwest, qtwest, e12west,
                              ueast, veast, weast, thleast, qteast, e12east,
                              usouth,vsouth,wsouth,thlsouth,qtsouth,e12south,
                              unorth,vnorth,wnorth,thlnorth,qtnorth,e12north,
                              utop,  vtop,  wtop,  thltop,  qttop,  e12top],
                              combine_attrs='drop')
  # Adjust time variable to seconds since initial field
  ts = openboundaries['time'].values.astype('datetime64[s]')
  dts = (ts-ts[0])/np.timedelta64(1, 's')
  openboundaries = openboundaries.assign_coords({'time':('time', dts)})
  openboundaries['time'].attrs.clear()
  time0 = pd.to_datetime(str(data['time'][0].values)).strftime('%Y-%m-%d %H:%M:%S') 
  # Add variable attributes
  openboundaries['time'] = openboundaries['time'].assign_attrs({'longname': 'Time', 'units': f"seconds since {time0}"})
  openboundaries['xt'] = openboundaries['xt'].assign_attrs({'longname': 'West-East displacement of cell centers','units': 'm'})
  openboundaries['xm'] = openboundaries['xm'].assign_attrs({'longname': 'West-East displacement of cell edges','units': 'm'})
  openboundaries['yt'] = openboundaries['yt'].assign_attrs({'longname': 'South-North displacement of cell centers','units': 'm'})
  openboundaries['ym'] = openboundaries['ym'].assign_attrs({'longname': 'South-North displacement of cell edges','units': 'm'})
  openboundaries['zt'] = openboundaries['zt'].assign_attrs({'longname': 'Vertical displacement of cell centers','units': 'm'})
  openboundaries['zm'] = openboundaries['zm'].assign_attrs({'longname': 'Vertical displacement of cell edges','units': 'm'})
  variables = ['u','v','w','thl','qt','e12']
  units     = ['m/s','m/s','m/s','K','kg/kg','m/s']
  long_names= ['West-East velocity at ',
                'South-North velocity at ',
                'Vertical velocity at ',
                'Liquid water potential temperature at ',
                'Total water specific humidity at ',
                'Square root of turbulent kinetic energy at ']
  for ivar in range(len(variables)):
    var = variables[ivar]
    unit = units[ivar]
    long_name = long_names[ivar]
    for boundary in ['West','East','South','North','top']:
      openboundaries[var+boundary.lower()] = openboundaries[var+boundary.lower()]\
      .assign_attrs({'longname': long_name+boundary+' boundary', 'units': unit})
  # Add global attributes
  time0 = pd.to_datetime(str(data['time'][0].values)).strftime('%Y-%m-%d %H:%M:%S') 
  openboundaries = openboundaries.assign_attrs({'title': f"openboundaries.inp.{input['iexpnr']:03d}.nc",
                                        'history': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                        'author': input['author'],
                                        'time0': time0})
  openboundaries.to_netcdf(path=input['outpath']+openboundaries.attrs['title'], mode='w', format="NETCDF4")
  return openboundaries