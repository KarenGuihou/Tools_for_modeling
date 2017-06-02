# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal
import math

### FUNCTIONS ###
def load_ncvar(filename, var, filetype):
    """
    Read a variable from a Netcdf file
    --
    Input   filename, variable, filetype (nc/hdf5)
    Output  variable
    """
    if filetype == 'nc':
        f = netcdf.netcdf_file(filename, 'r')
        data = f.variables[var].data
        f.close()
        return data
    elif filetype == 'hdf5':
        f = h5py.File(filename, 'r')
        variable = (f[var])[:]
        f.close()
        return variable
    else:
        print('unknown filetype. Is it something else that hdf5 or nc?')

def initiate_ncfile(output,lon,lat,zlen,tlen):
    """
    Initiate a 4D netcdf file. 
    --
    Input   filename, xvar, yvar, zlen, tlen
    Output  file
    """
    if filetype == 'nc':
    ylen=lat.shape[0]
    xlen=lon.shape[1]
    f = netcdf.netcdf_file(output, 'w')
    #f.history = 'Combes and Matano 2014 - Monthly mean nc file'
    f.createDimension('lon',xlen)
    f.createDimension('lat',ylen)
    f.createDimension('z',zlen)
    f.createDimension('time',tlen)

    return(f)

def writevar1D(ncfile,var,varname,dim,varunits):
    """
    Write a 1D variable to a netcdf file
    --
    Input   filename, variable, name of the variable in the ncfile, dimension, unit
    Output  (none)
    """
    if filetype == 'nc':
    varname = ncfile.createVariable(varname,'f',(dim,))
    varname[:]=var
    varname.units = varunits

def writevar2D(f,var,varname,dim1,dim2,varunits):
    """
    Write a 2D variable to a netcdf file
    --
    Input   filename, variable, name of the variable in the ncfile, dimension1, dimension2, unit
    Output  (none)
    """
    varname = f.createVariable(varname,'f',(dim1,dim2,))
    varname[:,:]=var
    varname.units = varunits

def writevar3D(f,var,varname,dim1,dim2,dim3,varunits):
    """
    Write a 3D variable to a netcdf file
    --
    Input   filename, variable, name of the variable in the ncfile, dim1, dim2, dim3, unit
    Output  (none)
    """
    varname = f.createVariable(varname,'f',(dim1,dim2,dim3,))
    varname[:,:]=var
    varname.units = varunits

def writevar4D(f,var,varname,dim1,dim2,dim3,dim4,varunits):
    """
    Write a 4D variable to a netcdf file
    --
    Input   filename, variable, name of the variable in the ncfile, dim1, dim2, dim3, dim4, unit
    Output  (none)
    """
    varname = f.createVariable(varname,'f',(dim1,dim2,dim3,dim4,))
    varname[:,:]=var
    varname.units = varunits

def get_e3u(depth):
    """
    Generate celle thickness from 3D depth grid
    --
    Input   depth
    Output  cell thickness
    """
    e3u = np.zeros_like(depth)
    e3u[0,:,:] = depth[0,:,:]*-1
    e3u[1:,:,:] = depth[:-1,:,:]-depth[1:,:,:]
    return(e3u)

def uv_on_Tpoint(u,v):  #,lon_u,lat_u,lon_v,lat_v,lon_rho,lat_rho):
    """
    Calculate u,v at T points (arakawa-C grid)
    Final size [1:-1,1:-1]
    u,v are 4D var (t,z,x,y)
    --
    Input   Uu, Vv
    Output  Ut, Vt
    """
    u_rhopt = (u[:,:,1:-1,1:] + u[:,:,1:-1,:-1]) / 2
    v_rhopt = (v[:,:,1:,1:-1] + v[:,:,:-1,1:-1]) / 2
    return(u_rhopt,v_rhopt)

def do_kdtree(combined_x_y_arrays, points):
    """
    Get the index in an array closest to a given value
    --
    Input   2D array, list of points
    Ouput   Indexes
    """
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    (dist, indexes) = mytree.query(points)
    return indexes
