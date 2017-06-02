# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal
import modeling_tools

### FUNCTIONS ###
def load_matfile(filename):
    # load mat file
    f = loadmat(filename)
    salt = np.expand_dims(f["salt"][:,:],axis=0)
    temp = np.expand_dims(f["temp"][:,:],axis=0)
    taux = np.expand_dims(f["taux"][:,:],axis=0)
    tauy = np.expand_dims(f["tauy"][:,:],axis=0)
    u = np.expand_dims(f["u"][:,:],axis=0)
    #u = np.zeros(temp.shape)
    #u[:,:,:-1,:] = tmp
    v = np.expand_dims(f["v"][:,:],axis=0)
    #v = np.zeros(temp.shape)
    #v[:,:-1,:,:] = tmp
    ubar = np.expand_dims(f["ubar"][:,:],axis=0)
    vbar = np.expand_dims(f["vbar"][:,:],axis=0)
    w = np.expand_dims(f["w"][:,:],axis=0)
    zeta = np.expand_dims(f["zeta"][:,:],axis=0)

    return(salt,temp,taux,tauy,u,v,ubar,vbar,w,zeta)

def load_grid(gridname):
    f = netcdf.netcdf_file(gridname, 'r')
    h = f.variables['h'].data
    lat_rho = f.variables['lat_rho'].data
    lon_rho = f.variables['lon_rho'].data
    lat_u = f.variables['lat_u'].data
    lon_u = f.variables['lon_u'].data
    lat_v = f.variables['lat_v'].data
    lon_v = f.variables['lon_v'].data
    f.close()
    return(h,lat_rho,lon_rho,lat_u,lon_u,lat_v,lon_v)

def initiate_ncfile(output,lon,lat,zlen,tlen):
    ylen=lat.shape[0]
    xlen=lon.shape[1]
    f = netcdf.netcdf_file(output, 'w')
    f.history = 'Combes and Matano 2014 - Monthly mean nc file'
    f.createDimension('lon',xlen)
    f.createDimension('lat',ylen)
    f.createDimension('z',zlen)
    f.createDimension('time',tlen)

    return(f)

def change_tzyx2tyxz(var):
    d0 = var.shape[0]
    d1 = var.shape[1]
    d2 = var.shape[2]
    d3 = var.shape[3]

    var = np.swapaxes(var,3,1)
    var = np.swapaxes(var,2,3)
    return(var)



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
    datadir_path = '/media/data1/VOCES/Runs/MATfiles/Combes_monthly/'
    matfile = datadir_path+year+'/'+month+'.mat'
    gridfile = datadir_path+'roms_agrif_grd.nc'
    ncdir_path = '/media/data1/VOCES/Runs/Combes_monthly_nc/'
    ncfile_T = ncdir_path+'CMM_y'+year+'m'+month+'_T.nc'
    ncfile_U = ncdir_path+'CMM_y'+year+'m'+month+'_U.nc'
    ncfile_V = ncdir_path+'CMM_y'+year+'m'+month+'_V.nc'

    print(matfile)
    # Load files
    [salt,temp,taux,tauy,u,v,ubar,vbar,w,zeta] = load_matfile(matfile)
    [h,lat_rho,lon_rho,lat_u,lon_u,lat_v,lon_v] = load_grid(gridfile)
    zlen = 40
    tlen = 1
    time_orig,time_file = time_jd(year,month,day,1980,01,01)
   
    # T file
    ## Rearrange T variables (t,z,y,x)-->(t,y,x,z)
    temp = change_tzyx2tyxz(temp)
    salt = change_tzyx2tyxz(salt)
    w = change_tzyx2tyxz(w)
    

    z = modeling_tools.zlevs_1d(h, zeta, 6, 0, 10, zlen, 'r', 2)
    
    ## Inverse z axis and harmonize size of matrices
    temp = temp[:,::-1,:,:]
    salt = salt[:,::-1,:,:]
    w = w[:,::-1,:,:]
    z = z[::-1,:,:]

    ## write T file
    output = ncfile_T
    cmd = "rm "+ output
    os.system(cmd)
    f = initiate_ncfile(output,lon_rho,lat_rho,zlen,tlen)
    writevar1D(f,lon_rho[0,:],'lon','lon','degrees west')
    writevar1D(f,lat_rho[:,0],'lat','lat','degrees south')
    writevar1D(f,lon_rho[0,:],'longitude','lon','degrees west')
    writevar1D(f,lat_rho[:,0],'latitude','lat','degrees south')
    writevar1D(f,time_file,'time','time','days since 1980-01-01')
    writevar3D(f,zeta,'zeta','time','lat','lon','cm')
    writevar3D(f,z,'depth','z','lat','lon','m')
    writevar4D(f,salt,'salt','time','z','lat','lon','PSU')
    writevar4D(f,temp,'temp','time','z','lat','lon','degree C')
    writevar4D(f,w,'w','time','z','lat','lon','cm/s')
    f.close()
    cmd='ncks --mk_rec_dmn time -O '+output+' '+output
    os.system(cmd)

    # write U file
    ## Rearrange U variables (t,z,y,x)-->(t,y,x,z)
    u = change_tzyx2tyxz(u)
    u = u[:,::-1,:,:]
    z_u = z[:,:,:-1]

    ## write U file
    output = ncfile_U
    cmd = "rm "+ output
    os.system(cmd)
    f = initiate_ncfile(output,lon_u,lat_u,zlen,tlen)
    writevar1D(f,lon_u[0,:],'lon','lon','degrees west')
    writevar1D(f,lat_u[:,0],'lat','lat','degrees south')
    writevar1D(f,lon_u[0,:],'longitude','lon','degrees west')
    writevar1D(f,lat_u[:,0],'latitude','lat','degrees south')
    writevar1D(f,time_file,'time','time','days since 1980-01-01')
    writevar3D(f,taux,'taux','time','lat','lon','N/m2')
    writevar4D(f,u,'u','time','z','lat','lon','m/s')
    writevar3D(f,ubar,'ubar','time','lat','lon','m/s')
    writevar3D(f,z_u,'depth_u','z','lat','lon','m')
    f.close()
    cmd='ncks --mk_rec_dmn time -O '+output+' '+output
    os.system(cmd)

    # write V file
    ## Rearrange V variables (t,z,y,x)-->(t,y,x,z)
    v = change_tzyx2tyxz(v)
    v = v[:,::-1,:,:]
    z_v = z[:,:-1,:]

    ## write V file
    output = ncfile_V
    cmd = "rm "+ output
    os.system(cmd)
    f = initiate_ncfile(output,lon_v,lat_v,zlen,tlen)
    writevar1D(f,lon_v[0,:],'lon','lon','degrees west')
    writevar1D(f,lat_v[:,0],'latitude','lat','degrees south')
    writevar1D(f,lon_v[0,:],'longitude','lon','degrees west')
    writevar1D(f,lat_v[:,0],'lat','lat','degrees south')
    writevar1D(f,time_file,'time','time','days since 1980-01-01')
    writevar3D(f,tauy,'tauy','time','lat','lon','N/m2')
    writevar4D(f,v,'v','time','z','lat','lon','m/s')
    writevar3D(f,vbar,'vbar','time','lat','lon','m/s')
    writevar3D(f,z_v,'depth_v','z','lat','lon','m')
    f.close()
    cmd='ncks --mk_rec_dmn time -O '+output+' '+output
    os.system(cmd)


main(sys.argv)
