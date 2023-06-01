# Creates DALES grid
import numpy as np
class GridDales:
  def __init__(self, input):
    self.xsize = input['xsize']
    self.ysize = input['ysize']

    self.itot = input['itot']
    self.jtot = input['jtot']
    self.kmax = input['kmax']

    self.dx  = input['xsize']/self.itot
    self.dy  = input['ysize']/self.jtot

    self.dz0   = input['dz0']
    self.alpha = input['alpha']

    self.xt = np.arange(0.5*self.dx, self.xsize, self.dx)
    self.yt = np.arange(0.5*self.dy, self.ysize, self.dy)

    self.xm = np.arange(0, self.xsize+self.dx, self.dx)
    self.ym = np.arange(0, self.ysize+self.dy, self.dy)

    if self.alpha!=0: # Stretched height grid
      self.dz = np.zeros(self.kmax)
      self.zt = np.zeros(self.kmax)
      self.zm = np.zeros(self.kmax+1)
      self.dz[:]  = self.dz0 * (1 + self.alpha)**np.arange(self.kmax)
      self.zm[1:] = np.cumsum(self.dz)
      self.zt[:]   = 0.5 * (self.zm[1:] + self.zm[:-1])
      self.zsize  = self.zm[-1]
    else: # Equidistant height grid
      self.dz = np.ones(self.kmax)*self.dz0
      self.zsize = self.kmax*self.dz0
      self.zt  = np.arange(0.5*self.dz0, self.zsize, self.dz0)
      self.zm = np.arange(0, self.zsize+self.dz0, self.dz0)