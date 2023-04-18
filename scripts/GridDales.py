# Creates DALES grid
import numpy as np
class Grid:
    def __init__(self, xsize, ysize, itot, jtot, ktot, dz0, alpha):
        self.xsize = xsize
        self.ysize = ysize

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        self.dx = xsize/itot
        self.dy = ysize/jtot

        self.xt = np.arange(0.5*self.dx, self.xsize, self.dx)
        self.yt = np.arange(0.5*self.dy, self.ysize, self.dy)

        self.xm = np.arange(0, self.xsize+self.dx, self.dx)
        self.ym = np.arange(0, self.ysize+self.dy, self.dy)

        if alpha!=0: # Stretched height grid
            self.dz = np.zeros(ktot)
            self.zt = np.zeros(ktot)
            self.zm = np.zeros(ktot+1)
            self.dz[:]  = dz0 * (1 + alpha)**np.arange(ktot)
            self.zm[1:] = np.cumsum(self.dz)
            self.zt[:]   = 0.5 * (self.zh[1:] + self.zh[:-1])
            self.zsize  = self.zh[-1]
        else: # Equidistant height grid
            self.dz = np.ones(ktot)*dz0
            self.zsize = ktot*dz0
            self.zt  = np.arange(0.5*dz0, self.zsize, dz0)
            self.zm = np.arange(0, self.zsize+dz0, dz0)