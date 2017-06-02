# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys

datadir_path = '/media/data1/VOCES/Runs/MATfiles/OFES_Monthly/'
ztfile = datadir_path+'ofes_Tdepth.txt'
zwfile = datadir_path+'ofes_Wdepth.txt'
ncdir_path = '/media/data1/VOCES/Runs/OFES_Monthly_nc/'
ncfile = ncdir_path+'grid_ofes.nc'

def load_matfile(filename):
    # load mat file
    f = loadmat(filename)
    temp = np.expand_dims(f["data_sa_mon_t"][:,:],axis=0)
    w = np.expand_dims(f["data_sa_mon_w"][:,:],axis=0)
    lat = f["lat"][:]
    lon = f["long"][:]

    return(temp,w,lat,lon)

def initiate_ncfile(output,xlen,ylen,zlen):
    f = netcdf.netcdf_file(output, 'w')
    f.history = 'OFES grid'
    f.createDimension('lon',xlen)
    f.createDimension('lat',ylen)
    f.createDimension('z',zlen)

    return(f)

def writevar1D(ncfile,var,varname,dim,varunits):
    varname = ncfile.createVariable(varname,'f',(dim,))
    varname[:]=var
    varname.units = varunits

def writevar2D(f,var,varname,dim1,dim2,varunits):
    varname = f.createVariable(varname,'f',(dim1,dim2,))
    varname[:,:]=var
    varname.units = varunits

def writevar3D(f,var,varname,dim1,dim2,dim3,varunits):
    varname = f.createVariable(varname,'f',(dim1,dim2,dim3,))
    varname[:,:]=var
    varname.units = varunits




# Load files
[temp,w,lat,lon] = load_matfile(datadir_path+'ofes_yr_1992_mon_01_swas.mat')
deptht = np.loadtxt(ztfile)
depthw = np.loadtxt(zwfile)
f = loadmat(datadir_path+'ofes_depth_swas.mat')
bathy = np.squeeze(np.expand_dims(f["h"],axis=0),0)

# create mask from temperature fields
tmask = np.where(np.isnan(temp),1,0)  #np.where(temp>=0,1,0)
wmask = np.where(np.isnan(w),1,0)     #np.where(w==0.,0,1)
bathy = np.where(np.isnan(bathy),0,bathy)     #np.where(w==0.,0,1)

# write nc
ylen=lat.shape[1]
xlen=lon.shape[1]
zlen = len(deptht)

output = ncfile
cmd = "rm "+ output
os.system(cmd)

f = initiate_ncfile(output,xlen,ylen,zlen)
writevar1D(f,lon[0,:],'lon','lon','degrees west')
writevar1D(f,lat[0,:],'lat','lat','degrees south')
writevar1D(f,lon[0,:],'longitude','lon','degrees west')
writevar1D(f,lat[0,:],'latitude','lat','degrees south')
writevar1D(f,deptht,'z','z','metres')
writevar1D(f,deptht,'depth_t','z','metres')
writevar1D(f,depthw,'depth_w','z','metres')
writevar2D(f,bathy,'bathy','lat','lon','metres')
writevar3D(f,tmask,'tmask','z','lat','lon','boolean')
writevar3D(f,wmask,'wmask','z','lat','lon','boolean')

f.close()

