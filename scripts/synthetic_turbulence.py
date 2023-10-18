import numpy as np
import xarray as xr
from datetime import datetime
import pandas as pd 
def synthetic_turbulence(input,grid,data,transform):
  zi_min = 500
  grav = 9.81
  T0 = 293
  zi = np.maximum(data['zi'],zi_min)
  # Get velocity variances from TKE assuming isotropic turbulence
  tke_prof = data['tke'].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y']).interp(z=grid.zt, assume_sorted=True).rename({'z': 'zt'})
  e  = 2/3*tke_prof
  u2 = e.rename('u2').expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])
  v2 = e.rename('v2').expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])
  w2 = e.rename('w2').expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])
  # Get uw and vw from ustar and vstar assuming linear profile in BL
  mask = (u2.coords['zt']<=zi)
  ustar = data['ustar'].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y'])
  vstar = data['vstar'].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y'])
  uw = xr.where(mask,ustar**2*(1-u2.coords['zt']/zi),0.).rename('uw')\
    .transpose('time','zt')\
    .expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])\
    .chunk({'time': input['tchunk']})
  vw = xr.where(mask,vstar**2*(1-u2.coords['zt']/zi),0.).rename('vw')\
    .transpose('time','zt')\
    .expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])\
    .chunk({'time': input['tchunk']})
  # Get uv from requirement that matrix needs to be positive and take uv closest to zero
  uv_min = -np.sqrt((uw*vw/e)**2+e**2-uw**2-vw**2)-uw*vw/e
  uv_min = uv_min+0.01*abs(uv_min)
  uv_max =  np.sqrt((uw*vw/e)**2+e**2-uw**2-vw**2)-uw*vw/e
  uv_max = uv_max-0.01*abs(uv_max)
  uv     = np.minimum(np.maximum(uv_min,0.),uv_max).rename('uv')\
    .transpose('time','zt','ypatch','xpatch')
  uv     = xr.where(np.isnan(uv),0.,uv)
  # Get wthls
  wthls = data['wthls'].sel(x=slice(0,grid.xsize),y=slice(0,grid.ysize)).mean(dim=['x','y'])
  wthl  = xr.where(mask,wthls*(1-1.2*u2.coords['zt']/zi),0.).rename('wthl')\
    .transpose('time','zt')\
    .expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])\
    .chunk({'time': input['tchunk']})
  # Get thl2
  wstar = (grav/T0*wthls*zi)**(1/3)
  Tstar = wthls/wstar
  thl2  = xr.where(mask,( Tstar*(u2.coords['zt']/zi)**(-1/3) )**2*np.minimum(u2.coords['zt']/u2.coords['zt'][3],1), 0.)\
    .rename('thl2')\
    .transpose('time','zt')\
    .expand_dims(dim={'ypatch': [grid.ysize/2], 'xpatch': [grid.xsize/2]},axis=[2,3])\
    .chunk({'time': input['tchunk']})
  # Set qt wqt to zero
  wqt = xr.zeros_like(u2).rename('wqt')
  qt2 = xr.zeros_like(u2).rename('qt2')
  synturb_prof = xr.merge([u2,v2,w2,uv,uw,vw,thl2,wthl,qt2,wqt],combine_attrs='drop')
  variables = ['u2','v2','w2','uv','uw','vw','thl2','wthl','qt2','wqt']
  units     = ['m2/s2','m2/s2','m2/s2','m2/s2','m2/s2','m2/s2','K2','K m/s','kg2/kg2','kg/kg m/s']
  long_names= ['Variance West-East velocity at ',
               'Variance South-North velocity at ',
               'Variance Vertical velocity at ',
               'Covariance West-East and South-North velocity at ',
               'Covariance West-East and vertical velocity at ',
               'Covariance South-North and vertical velocity at ',
               'Variance liquid water potential temperature at ',
               'Covariance liquid water potential temperature and vertical velocity at ',
               'Variance total water specific humidity at ',
               'Covariance total water specific humidity and vertical velocity at ']
  synturb = xr.Dataset()
  xrzeros = xr.DataArray(np.zeros((u2.shape[0],1,1)),
    coords={'time': u2['time'], 'ypatch': u2['ypatch'], 'xpatch': u2['xpatch']},
    dims=['time','ypatch','xpatch'])
  # Add profiles to lateral boundaries
  for ivar in range(len(variables)):
    var = variables[ivar]
    unit = units[ivar]
    long_name = long_names[ivar]
    for boundary in ['West','East','South','North']:
      if(boundary == 'West' or boundary == 'East'): 
        synturb = synturb.assign({var+boundary.lower(): synturb_prof[var].isel(xpatch=0,drop=True)})
      if(boundary == 'South' or boundary == 'North'):
        synturb = synturb.assign({var+boundary.lower(): synturb_prof[var].isel(ypatch=0,drop=True)})
      synturb[var+boundary.lower()]=synturb[var+boundary.lower()]\
        .assign_attrs({'longname': long_name+boundary+' boundary', 'units': unit})
    # Add top values (set to zero)
    boundary = 'top'
    synturb = synturb.assign({var+boundary: xr.zeros_like(xrzeros)})
    synturb[var+boundary.lower()]=synturb[var+boundary.lower()]\
        .assign_attrs({'longname': long_name+boundary+' boundary', 'units': unit})
  # Adjust time variable to seconds since initial field
  ts = synturb['time'].values.astype('datetime64[s]')
  dts = (ts-np.datetime64(input['time0'],'s'))/np.timedelta64(1, 's')
  synturb = synturb.assign_coords({'time':('time', dts)})
  synturb['time'].attrs.clear()
  # Add global attributes
  synturb['time'] = synturb['time'].assign_attrs({'longname': 'Time', 'units': f"seconds since {input['time0']}"})
  synturb['xpatch'] = synturb['xpatch'].assign_attrs({'longname': 'West-East displacement of synthetic turbulence input patches','units': 'm'})
  synturb['ypatch'] = synturb['ypatch'].assign_attrs({'longname': 'South-North displacement of synthetic turbulence input patches','units': 'm'})
  synturb = synturb.assign_attrs({'title': f"openboundaries.inp.{input['iexpnr']:03d}.nc",
                                        'history_synturb': f"Created on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                        'author_synturb': input['author'],
                                        'time0_synturb': input['time0']})
  synturb.to_netcdf(path=input['outpath']+synturb.attrs['title'], mode='a', format="NETCDF4")
  return synturb