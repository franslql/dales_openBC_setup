# Horizontal spatial Gaussian filter for input data
import numpy as np
import xarray as xr

def gaussian_filter(data,input):
  variables=['u','v','w','thl','qt']
  dx = (data['x'][1]-data['x'][0]).values
  sigma = input['filter']['sigma']
  start = input['filter']['time_start']
  end   = input['filter']['time_end']
  Nwindow = int(np.ceil(4*sigma/dx))
  xwindow = np.arange(-Nwindow,Nwindow+1)*dx
  XW,YW = np.meshgrid(xwindow,xwindow)
  weights = np.exp(-(XW**2/(2*sigma**2)+YW**2/(2*sigma**2)))
  weights = weights/np.sum(weights)
  window = xr.DataArray(weights,coords={'ywindow': xwindow, 'xwindow': xwindow},dims=['ywindow','xwindow'])
  mask = data['time'].isin(data['time'].sel(time=slice(start,end)))
  vars=[]
  for var in variables:
    var_in = data[var].isel(time=mask).chunk({'time':1, 'z':1})
    var_padded  = var_in.pad({'x':Nwindow,'y':Nwindow},mode='symmetric')
    var_rolling = var_padded.rolling({'y':int(Nwindow*2+1),'x':int(Nwindow*2+1)},center=True).construct(y='ywindow',x='xwindow')
    var_filtered= var_rolling.dot(window).isel(x=np.arange(Nwindow,data.sizes['x']+Nwindow),y=np.arange(Nwindow,data.sizes['y']+Nwindow))
    var_unfiltered = data[var].isel(time=~mask)
    vars.append(xr.concat([var_filtered,var_unfiltered],dim='time').rename(var).sortby('time'))
  vars=xr.merge(vars)
  data_filtered = xr.merge([vars.chunk({'time':1, 'z':1}),data.chunk({'time':1, 'z':1})],compat="override")
  return data_filtered.chunk({'time':1 ,'z': 'auto'})
