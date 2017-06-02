# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal

### FUNCTIONS ###
def load_matfile(filename):
    # load mat file
    f = loadmat(filename)
    ssh = np.expand_dims(f["data_sa_mon_e"][:,:],axis=0)
    sal = np.expand_dims(f["data_sa_mon_s"][:,:],axis=0)
    temp = np.expand_dims(f["data_sa_mon_t"][:,:],axis=0)
    u = np.expand_dims(f["data_sa_mon_u"][:,:],axis=0)
    v = np.expand_dims(f["data_sa_mon_v"][:,:],axis=0)
    w = np.expand_dims(f["data_sa_mon_w"][:,:],axis=0)
    lat = f["lat"][:]
    lon = f["long"][:]

    return(ssh,sal,temp,u,v,w,lat,lon)

def initiate_ncfile(output,xlen,ylen,zlen,tlen):
    f = netcdf.netcdf_file(output, 'w')
    f.history = 'OFES monthly mean nc file'
    f.createDimension('lon',xlen)
    f.createDimension('lat',ylen)
    f.createDimension('zt',zlen)
    f.createDimension('zw',zlen)
    f.createDimension('time',tlen)

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

def writevar4D(f,var,varname,dim1,dim2,dim3,dim4,varunits):
    varname = f.createVariable(varname,'f',(dim1,dim2,dim3,dim4,))
    varname[:,:]=var
    varname.units = varunits

def time_jd(yyyy,mm,dd,yyyy_orig,mm_orig,dd_orig):
    a,b = gcal2jd(yyyy_orig,mm_orig,dd_orig)
    jd_orig = a+b+0.5

    a,b = gcal2jd(yyyy,mm,dd)
    jd_date = (a+b+0.5) - jd_orig

    return (jd_orig, jd_date)

    
### MAIN ###
def main(argv):
    year = sys.argv[1]
    month = sys.argv[2]
    day = 15

    print(year,month)
    # Paths 
    datadir_path = '/media/data1/VOCES/Runs/MATfiles/OFES_Monthly/'
    matfile = datadir_path+'ofes_yr_'+year+'_mon_'+month+'_swas.mat'
    ztfile = datadir_path+'ofes_Tdepth.txt'
    zwfile = datadir_path+'ofes_Wdepth.txt'
    ncdir_path = '/media/data1/VOCES/Runs/OFES_Monthly_nc/'
    ncfile = ncdir_path+'OFES_y'+year+'m'+month+'.nc'

    # Load files
    [ssh,sal,temp,u,v,w,lat,lon] = load_matfile(matfile)
    deptht = np.loadtxt(ztfile)
    depthw = np.loadtxt(zwfile)
    
    # write nc
    ylen=lat.shape[1]
    xlen=lon.shape[1]
    zlen = len(deptht)
    tlen = 1
    time_orig,date_file = time_jd(year,month,day,1980,01,01)

    output = ncfile
    cmd = "rm "+ output
    os.system(cmd)
    
    f = initiate_ncfile(output,xlen,ylen,zlen,tlen)
    
    writevar1D(f,lon[0,:],'lon','lon','degrees west')
    writevar1D(f,lat[0,:],'lat','lat','degrees south')
    writevar1D(f,deptht,'zt','zt','metres')
    writevar1D(f,date_file,'time','time','days since 1980-01-01')
    writevar1D(f,depthw,'zw','zw','metres')
    writevar3D(f,ssh,'zeta','time','lat','lon','cm')
    writevar4D(f,sal,'salt','time','zt','lat','lon','PSU')
    writevar4D(f,temp,'temp','time','zt','lat','lon','degree C')
    writevar4D(f,u,'u','time','zt','lat','lon','cm/s')
    writevar4D(f,v,'v','time','zt','lat','lon','cm/s')
    writevar4D(f,w,'w','time','zw','lat','lon','cm/s')
    
    f.close()
    cmd='ncks --mk_rec_dmn time -O '+output+' '+output
    os.system(cmd)

main(sys.argv)
