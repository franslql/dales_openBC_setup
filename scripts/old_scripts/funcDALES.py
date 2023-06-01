import numpy as np
from netCDF4 import Dataset
from datetime import datetime

def writeProfInp(zt,thl,qt,u,v,tke,iexpnr=0,path='',description='Initial profiles'):
    date = datetime.now().strftime("%d/%m/%Y")
    file = open(f"{path}prof.inp.{iexpnr:03}", 'w')
    file.write(f"# {description} {date}\n")
    file.write('# zf thl qt u v tke \n')
    for i in range(len(zt)):
        file.write(f"{zt[i]} {thl[i]} {qt[i]} {u[i]} {v[i]} {tke[i]} \n")
    file.close()
    return 0

def writeLscaleInp(zt,ug,vg,wfls,dqtdxls,dqtdyls,dqtdtls,dthlrad,iexpnr=0,path='',description='Large scale forcing input'):
    date = datetime.now().strftime("%d/%m/%Y")
    file = open(f"{path}lscale.inp.{iexpnr:03}", 'w')
    file.write(f"# {description} {date}\n")
    file.write('# height ug vg wfls dqtdxls dqtdyls dqtdtls dthlrad \n')
    for i in range(len(zt)):
        file.write(f"{zt[i]} {ug[i]} {vg[i]} {wfls[i]} {dqtdxls[i]} {dqtdyls[i]} {dqtdtls[i]} {dthlrad[i]} \n")
    file.close()
    return 0

def calcBaseprof(zt,thls,ps,pref0=1e5):
    # constants
    lapserate=np.array([-6.5/1000.,0.,1./1000,2.8/1000])
    zmat=np.array([11000.,20000.,32000.,47000.])
    grav=9.81
    rd=287.04
    cp=1004.
    zsurf=0
    k1=len(zt)
    # Preallocate
    pmat=np.zeros(4)
    tmat=np.zeros(4)
    rhobf=np.zeros(k1)
    pb=np.zeros(k1)
    tb=np.zeros(k1)
    # DALES code
    tsurf=thls*(ps/pref0)**(rd/cp)
    pmat[0]=np.exp((np.log(ps)*lapserate[0]*rd+np.log(tsurf+zsurf*lapserate[0])*grav-
      np.log(tsurf+zmat[0]*lapserate[0])*grav)/(lapserate[0]*rd))
    tmat[0]=tsurf+lapserate[0]*(zmat[0]-zsurf);
    for j in np.arange(1,4):
        if(abs(lapserate[j])<1e-10):
            pmat[j]=np.exp((np.log(pmat[j-1])*tmat[j-1]*rd+zmat[j-1]*grav-zmat[j]*grav)/(tmat[j-1]*rd))
        else:
            pmat[j]=np.exp((np.log(pmat[j-1])*lapserate[j]*rd+np.log(tmat[j-1]+zmat[j-1]*lapserate[j])*grav-np.log(tmat[j-1]+zmat[j]*lapserate[j])*grav)/(lapserate[j]*rd))
        tmat[j]=tmat[j-1]+lapserate[j]*(zmat[j]-zmat[j-1]);

    for k in range(k1):
        if(zt[k]<zmat[0]):
            pb[k]=np.exp((np.log(ps)*lapserate[0]*rd+np.log(tsurf+zsurf*lapserate[0])*grav-np.log(tsurf+zt[k]*lapserate[0])*grav)/(lapserate[0]*rd))
            tb[k]=tsurf+lapserate[0]*(zt[k]-zsurf)
        else:
            j=0
            while(zt[k]>=zmat[j]):
              j=j+1
            tb[k]=tmat[j-1]+lapserate[j]*(zt[k]-zmat[j-1])
            if(abs(lapserate[j])<1e-99):
                pb[k]=np.exp((np.log(pmat[j-1])*tmat[j-1]*rd+zmat[j-1]*grav-zt[k]*grav)/(tmat[j-1]*rd))
            else:
                pb[k]=np.exp((np.log(pmat[j-1])*lapserate[j]*rd+np.log(tmat[j-1]+zmat[j-1]*lapserate[j])*grav-np.log(tmat[j-1]+zt[k]*lapserate[j])*grav)/(lapserate[j]*rd))
        rhobf[k]=pb[k]/(rd*tb[k]) # dry estimate
    return rhobf

class Grid:
    def __init__(self, xsize, ysize, itot, jtot, ktot, dz, dz0, alpha, stretched):

        # Calculate and store DALES grid
        self.xsize = xsize
        self.ysize = ysize

        self.itot = itot
        self.jtot = jtot
        self.ktot = ktot

        self.dx = xsize/itot
        self.dy = ysize/jtot

        self.x = np.arange(0.5*self.dx, self.xsize, self.dx)
        self.y = np.arange(0.5*self.dy, self.ysize, self.dy)

        self.xh = np.arange(0, self.xsize+self.dx, self.dx)
        self.yh = np.arange(0, self.ysize+self.dy, self.dy)

        if stretched:
            self.dz = np.zeros(ktot)
            self.z = np.zeros(ktot)
            self.zh = np.zeros(ktot+1)
            self.dz[:]  = dz0 * (1 + alpha)**np.arange(ktot)
            self.zh[1:] = np.cumsum(self.dz)
            self.z[:]   = 0.5 * (self.zh[1:] + self.zh[:-1])
            self.zsize  = self.zh[-1]
        else:
            self.dz = np.ones(ktot)*dz
            self.zsize = ktot*dz
            self.z  = np.arange(0.5*dz, self.zsize, dz)
            self.zh = np.arange(0, self.zsize+dz, dz)

class OpenBoundariesInp:
    def __init__(self, grid, iexpnr=0, path='', nsv=0):
        self.iexpnr = iexpnr
        self.path = path
        self.name = f"openboundaries.inp.{iexpnr:03}.nc"
        self.nc = Dataset(self.path+self.name,'w')
        self.nc.createDimension('time',None)
        self.nc.createDimension('zt',grid.ktot)
        self.nc.createDimension('yt',grid.jtot)
        self.nc.createDimension('xt',grid.itot)
        self.nc.createDimension('zm',grid.ktot+1)
        self.nc.createDimension('ym',grid.jtot+1)
        self.nc.createDimension('xm',grid.itot+1)
        self.time = self.nc.createVariable('time','f4',('time'))
        self.zt = self.nc.createVariable('zt','f4',('zt'))
        self.yt = self.nc.createVariable('yt','f4',('yt'))
        self.xt = self.nc.createVariable('xt','f4',('xt'))
        self.zm = self.nc.createVariable('zm','f4',('zm'))
        self.ym = self.nc.createVariable('ym','f4',('ym'))
        self.xm = self.nc.createVariable('xm','f4',('xm'))
        self.uwest = self.nc.createVariable('uwest','f4',('time','zt','yt'))
        self.vwest = self.nc.createVariable('vwest','f4',('time','zt','ym'))
        self.wwest = self.nc.createVariable('wwest','f4',('time','zm','yt'))
        self.thlwest = self.nc.createVariable('thlwest','f4',('time','zt','yt'))
        self.qtwest = self.nc.createVariable('qtwest','f4',('time','zt','yt'))
        self.e12west = self.nc.createVariable('e12west','f4',('time','zt','yt'))
        self.ueast = self.nc.createVariable('ueast','f4',('time','zt','yt'))
        self.veast = self.nc.createVariable('veast','f4',('time','zt','ym'))
        self.weast = self.nc.createVariable('weast','f4',('time','zm','yt'))
        self.thleast = self.nc.createVariable('thleast','f4',('time','zt','yt'))
        self.qteast = self.nc.createVariable('qteast','f4',('time','zt','yt'))
        self.e12east = self.nc.createVariable('e12east','f4',('time','zt','yt'))
        self.usouth = self.nc.createVariable('usouth','f4',('time','zt','xm'))
        self.vsouth = self.nc.createVariable('vsouth','f4',('time','zt','xt'))
        self.wsouth = self.nc.createVariable('wsouth','f4',('time','zm','xt'))
        self.thlsouth = self.nc.createVariable('thlsouth','f4',('time','zt','xt'))
        self.qtsouth = self.nc.createVariable('qtsouth','f4',('time','zt','xt'))
        self.e12south = self.nc.createVariable('e12south','f4',('time','zt','xt'))
        self.unorth = self.nc.createVariable('unorth','f4',('time','zt','xm'))
        self.vnorth = self.nc.createVariable('vnorth','f4',('time','zt','xt'))
        self.wnorth = self.nc.createVariable('wnorth','f4',('time','zm','xt'))
        self.thlnorth = self.nc.createVariable('thlnorth','f4',('time','zt','xt'))
        self.qtnorth = self.nc.createVariable('qtnorth','f4',('time','zt','xt'))
        self.e12north = self.nc.createVariable('e12north','f4',('time','zt','xt'))
        self.utop = self.nc.createVariable('utop','f4',('time','yt','xm'))
        self.vtop = self.nc.createVariable('vtop','f4',('time','ym','xt'))
        self.wtop = self.nc.createVariable('wtop','f4',('time','yt','xt'))
        self.thltop = self.nc.createVariable('thltop','f4',('time','yt','xt'))
        self.qttop = self.nc.createVariable('qttop','f4',('time','yt','xt'))
        self.e12top = self.nc.createVariable('e12top','f4',('time','yt','xt'))
        self.xt[:] = grid.x
        self.yt[:] = grid.y
        self.zt[:] = grid.z
        self.xm[:] = grid.xh
        self.ym[:] = grid.yh
        self.zm[:] = grid.zh

        if(nsv>0):
            self.nc.createDimension('isv',nsv)
            self.isv = self.nc.createVariable('isv','i4',('isv'))
            self.isv[:] = np.arange(1,nsv+1)
            self.svwest = self.nc.createVariable('svwest','f4',('isv','time','zt','yt'))
            self.sveast = self.nc.createVariable('sveast','f4',('isv','time','zt','yt'))
            self.svsouth= self.nc.createVariable('svsouth','f4',('isv','time','zt','xt'))
            self.svnorth= self.nc.createVariable('svnorth','f4',('isv','time','zt','xt'))
            self.svtop  = self.nc.createVariable('svtop','f4',('isv','time','yt','xt'))

    def writeTime(self,t,it):
        self.time[it]=t

    def writeWest(self,thl,qt,u,v,w,e12,it):
        self.thlwest[it,:,:] = thl
        self.qtwest[it] = qt
        self.uwest[it,:,:] = u
        self.vwest[it,:,:] = v
        self.wwest[it,:,:] = w
        self.e12west[it,:,:] = e12

    def writeEast(self,thl,qt,u,v,w,e12,it):
        self.thleast[it,:,:] = thl
        self.qteast[it] = qt
        self.ueast[it,:,:] = u
        self.veast[it,:,:] = v
        self.weast[it,:,:] = w
        self.e12east[it,:,:] = e12

    def writeSouth(self,thl,qt,u,v,w,e12,it):
        self.thlsouth[it,:,:] = thl
        self.qtsouth[it] = qt
        self.usouth[it,:,:] = u
        self.vsouth[it,:,:] = v
        self.wsouth[it,:,:] = w
        self.e12south[it,:,:] = e12

    def writeNorth(self,thl,qt,u,v,w,e12,it):
        self.thlnorth[it,:,:] = thl
        self.qtnorth[it] = qt
        self.unorth[it,:,:] = u
        self.vnorth[it,:,:] = v
        self.wnorth[it,:,:] = w
        self.e12north[it,:,:] = e12

    def writeTop(self,thl,qt,u,v,w,e12,it):
        self.thltop[it,:,:] = thl
        self.qttop[it,:,:] = qt
        self.utop[it,:,:] = u
        self.vtop[it,:,:] = v
        self.wtop[it,:,:] = w
        self.e12top[it,:,:] = e12

    def exit(self):
        self.nc.close()

class HeteroInp:
    def __init__(self, grid, iexpnr=0, path=''):
        # Create netcdf file
        self.nc = Dataset(path+'initfields.inp.'+str(iexpnr).zfill(3)+'.nc','w')
        self.nc.createDimension('zt',grid.ktot)
        self.nc.createDimension('yt',grid.jtot)
        self.nc.createDimension('xt',grid.itot)
        self.nc.createDimension('zm',grid.ktot+1)
        self.nc.createDimension('ym',grid.jtot+1)
        self.nc.createDimension('xm',grid.itot+1)
        self.zt = self.nc.createVariable('zt','f4',('zt'))
        self.yt = self.nc.createVariable('yt','f4',('yt'))
        self.xt = self.nc.createVariable('xt','f4',('xt'))
        self.zm = self.nc.createVariable('zm','f4',('zm'))
        self.ym = self.nc.createVariable('ym','f4',('ym'))
        self.xm = self.nc.createVariable('xm','f4',('xm'))
        self.u = self.nc.createVariable('u0','f4',('zt','yt','xm'))
        self.v = self.nc.createVariable('v0','f4',('zt','ym','xt'))
        self.w = self.nc.createVariable('w0','f4',('zm','yt','xt'))
        self.thl = self.nc.createVariable('thl0','f4',('zt','yt','xt'))
        self.qt = self.nc.createVariable('qt0','f4',('zt','yt','xt'))
        self.e12 = self.nc.createVariable('e120','f4',('zt','yt','xt'))
        self.zt[:] = grid.z
        self.yt[:] = grid.y
        self.xt[:] = grid.x
        self.zm[:] = grid.zh
        self.ym[:] = grid.yh
        self.xm[:] = grid.xh

    def writeFields(self,thl,qt,u,v,w,e12):
        self.u[:,:,:]   = u
        self.v[:,:,:]   = v
        self.w[:,:,:]   = w
        self.thl[:,:,:] = thl
        self.qt[:,:,:]  = qt
        self.e12[:,:,:] = e12

    def exit(self):
        # Close netcdf file
        self.nc.close()

class SynTurbInp:
    def __init__(self,xpatch,ypatch,zpatch,iexpnr=0,path=''):
        # Create turbulence fields and dimensions
        self.nc = Dataset(path+'openboundaries.inp.'+str(iexpnr).zfill(3)+'.nc','r+')
        # Add variables to netcdf file
        dimensions = np.array([])
        for dim in self.nc.dimensions.values():
            dimensions = np.append(dimensions,dim.name)
        if(not('zpatch' in dimensions)):
            self.nc.createDimension('zpatch',len(zpatch))
            nczpatch = self.nc.createVariable('zpatch','f4',('zpatch'))
        else:
            self.nc.dimensions['zpatch'].size=len(zpatch)
        nczpatch[:] = zpatch
        if(not('ypatch' in dimensions)):
            self.nc.createDimension('ypatch',len(ypatch))
            ncypatch = self.nc.createVariable('ypatch','f4',('ypatch'))
        else:
            ncypatch = self.nc.variables['ypatch']
        ncypatch[:] = ypatch
        if(not('xpatch' in dimensions)):
            self.nc.createDimension('xpatch',len(xpatch))
            ncxpatch = self.nc.createVariable('xpatch','f4',('xpatch'))
        else:
            ncxpatch = self.nc.variables['xpatch']
        ncxpatch[:] = xpatch
        variables = np.array([])
        for var in self.nc.variables.values():
            variables = np.append(variables,var.name)
        self.u2west   = self.createVariableCheck(variables,f"u2west",'f4',('time','zpatch','ypatch'))
        self.v2west   = self.createVariableCheck(variables,f"v2west",'f4',('time','zpatch','ypatch'))
        self.w2west   = self.createVariableCheck(variables,f"w2west",'f4',('time','zpatch','ypatch'))
        self.uvwest   = self.createVariableCheck(variables,f"uvwest",'f4',('time','zpatch','ypatch'))
        self.uwwest   = self.createVariableCheck(variables,f"uwwest",'f4',('time','zpatch','ypatch'))
        self.vwwest   = self.createVariableCheck(variables,f"vwwest",'f4',('time','zpatch','ypatch'))
        self.thl2west = self.createVariableCheck(variables,f"thl2west",'f4',('time','zpatch','ypatch'))
        self.qt2west  = self.createVariableCheck(variables,f"qt2west",'f4',('time','zpatch','ypatch'))
        self.wthlwest = self.createVariableCheck(variables,f"wthlwest",'f4',('time','zpatch','ypatch'))
        self.wqtwest  = self.createVariableCheck(variables,f"wqtwest",'f4',('time','zpatch','ypatch'))
        self.u2east   = self.createVariableCheck(variables,f"u2east",'f4',('time','zpatch','ypatch'))
        self.v2east   = self.createVariableCheck(variables,f"v2east",'f4',('time','zpatch','ypatch'))
        self.w2east   = self.createVariableCheck(variables,f"w2east",'f4',('time','zpatch','ypatch'))
        self.uveast   = self.createVariableCheck(variables,f"uveast",'f4',('time','zpatch','ypatch'))
        self.uweast   = self.createVariableCheck(variables,f"uweast",'f4',('time','zpatch','ypatch'))
        self.vweast   = self.createVariableCheck(variables,f"vweast",'f4',('time','zpatch','ypatch'))
        self.thl2east = self.createVariableCheck(variables,f"thl2east",'f4',('time','zpatch','ypatch'))
        self.qt2east  = self.createVariableCheck(variables,f"qt2east",'f4',('time','zpatch','ypatch'))
        self.wthleast = self.createVariableCheck(variables,f"wthleast",'f4',('time','zpatch','ypatch'))
        self.wqteast  = self.createVariableCheck(variables,f"wqteast",'f4',('time','zpatch','ypatch'))
        self.u2south   = self.createVariableCheck(variables,f"u2south",'f4',('time','zpatch','xpatch'))
        self.v2south   = self.createVariableCheck(variables,f"v2south",'f4',('time','zpatch','xpatch'))
        self.w2south   = self.createVariableCheck(variables,f"w2south",'f4',('time','zpatch','xpatch'))
        self.uvsouth   = self.createVariableCheck(variables,f"uvsouth",'f4',('time','zpatch','xpatch'))
        self.uwsouth   = self.createVariableCheck(variables,f"uwsouth",'f4',('time','zpatch','xpatch'))
        self.vwsouth   = self.createVariableCheck(variables,f"vwsouth",'f4',('time','zpatch','xpatch'))
        self.thl2south = self.createVariableCheck(variables,f"thl2south",'f4',('time','zpatch','xpatch'))
        self.qt2south  = self.createVariableCheck(variables,f"qt2south",'f4',('time','zpatch','xpatch'))
        self.wthlsouth = self.createVariableCheck(variables,f"wthlsouth",'f4',('time','zpatch','xpatch'))
        self.wqtsouth  = self.createVariableCheck(variables,f"wqtsouth",'f4',('time','zpatch','xpatch'))
        self.u2north   = self.createVariableCheck(variables,f"u2north",'f4',('time','zpatch','xpatch'))
        self.v2north   = self.createVariableCheck(variables,f"v2north",'f4',('time','zpatch','xpatch'))
        self.w2north   = self.createVariableCheck(variables,f"w2north",'f4',('time','zpatch','xpatch'))
        self.uvnorth   = self.createVariableCheck(variables,f"uvnorth",'f4',('time','zpatch','xpatch'))
        self.uwnorth   = self.createVariableCheck(variables,f"uwnorth",'f4',('time','zpatch','xpatch'))
        self.vwnorth   = self.createVariableCheck(variables,f"vwnorth",'f4',('time','zpatch','xpatch'))
        self.thl2north = self.createVariableCheck(variables,f"thl2north",'f4',('time','zpatch','xpatch'))
        self.qt2north  = self.createVariableCheck(variables,f"qt2north",'f4',('time','zpatch','xpatch'))
        self.wthlnorth = self.createVariableCheck(variables,f"wthlnorth",'f4',('time','zpatch','xpatch'))
        self.wqtnorth  = self.createVariableCheck(variables,f"wqtnorth",'f4',('time','zpatch','xpatch'))
        self.u2top   = self.createVariableCheck(variables,f"u2top",'f4',('time','ypatch','xpatch'))
        self.v2top   = self.createVariableCheck(variables,f"v2top",'f4',('time','ypatch','xpatch'))
        self.w2top   = self.createVariableCheck(variables,f"w2top",'f4',('time','ypatch','xpatch'))
        self.uvtop   = self.createVariableCheck(variables,f"uvtop",'f4',('time','ypatch','xpatch'))
        self.uwtop   = self.createVariableCheck(variables,f"uwtop",'f4',('time','ypatch','xpatch'))
        self.vwtop   = self.createVariableCheck(variables,f"vwtop",'f4',('time','ypatch','xpatch'))
        self.thl2top = self.createVariableCheck(variables,f"thl2top",'f4',('time','ypatch','xpatch'))
        self.qt2top  = self.createVariableCheck(variables,f"qt2top",'f4',('time','ypatch','xpatch'))
        self.wthltop = self.createVariableCheck(variables,f"wthltop",'f4',('time','ypatch','xpatch'))
        self.wqttop  = self.createVariableCheck(variables,f"wqttop",'f4',('time','ypatch','xpatch'))

    def createVariableCheck(self,varnames,varName,type,dimensions):
        if(varName in varnames):
            return self.nc.variables[varName]
        else:
            return self.nc.createVariable(varName,type,dimensions)

    def exit(self):
        # Close netcdf file
        self.nc.close()
