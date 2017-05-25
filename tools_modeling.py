# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal

### FUNCTIONS ###
def load_ncvar(filename, var, filetype):
    """
    Read a variable from a NEMO output (netcdf 3 or 4)
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

def readMODELnc(filename, var):
    """ 
    Read a variable from a NEMO output (netcdf 3 or 4)
    """

    f = netcdf.netcdf_file(filename, 'r')
    data = f.variables[var].data
    f.close()

    return data

def calculate_dist_2_coords(lat1,lon1,lat2,lon2):
    """
    Inspired by http://andrew.hedges.name/experiments/haversine/ 
    and http://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    Input lat/lon must be in decimal degrees.
    """
    from math import sin, cos, sqrt, atan2, radians
    R = 6373.0 # approximate radius of earth in km
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return(distance)

def get_e3u(depth):
    """
    Generate celle thickness from depth grid
    """
    e3u = np.zeros_like(depth)
    e3u[0,:,:] = depth[0,:,:]*-1
    e3u[1:,:,:] = depth[:-1,:,:]-depth[1:,:,:]
    return(e3u)

def get_transport_section(u,depth,lat,lon,ii_o,ii_e,jj_o,jj_e,t):
    """ 
    Calculation of transport across a given section. 
	T = (u * depth * length_section)/10^6
	"""
    e3u = get_e3u(depth)
    length_section = calculate_dist_2_coords(lat[jj_o],lon[ii_o],lat[jj_e-1],lon[ii_e-1])*1e3
    transport_along_section = np.multiply(u[t,:,jj_o:jj_e,ii_o:ii_e],e3u[:,jj_o:jj_e,ii_o:ii_e])
    transport = np.sum(transport_along_section)*length_section/1e6
    return(transport)

def do_kdtree(combined_x_y_arrays, points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    (dist, indexes) = mytree.query(points)
    return indexes

def uv_on_Tpoint(u,v):  #,lon_u,lat_u,lon_v,lat_v,lon_rho,lat_rho):
    """
    Calculate u,v at T points (arakawa-C grid)
    Final size [1:-1,1:-1]
    u,v are 4D var (t,z,x,y)
    """
    u_rhopt = (u[:,:,1:-1,1:] + u[:,:,1:-1,:-1]) / 2
    v_rhopt = (v[:,:,1:,1:-1] + v[:,:,:-1,1:-1]) / 2
    return(u_rhopt,v_rhopt)

def calculation_bearing(latdeg,londeg):
    """
    Calculation of the bearing between 2 coordinates. 
    WARNING: calculation at coord(i) between coord(i-1) and coord(i+1) 
    input: lat, lon in degrees
    output; theta in degrees
    formula: theta = atan2(sin(delta_lbda).cos(phi2),cos(phi1).sin(phi2)-sin(phi1).cos(phi2).cos(delta_lbda))
    """
    # we pass lat, lon in radians
    lat = np.radians(latdeg)
    lon = np.radians(londeg)
    theta = np.zeros(len(lat)-2)
    for ii in range(1,len(lat)-1):
        n = ii-1
        m = ii+1
        delta_lbda = abs(lon[n]-lon[m])
        part1 = np.multiply(np.sin(delta_lbda),np.cos(lat[m]))
        part2 = np.multiply(np.cos(lat[n]),np.sin(lat[m]))
        part3 = np.multiply(np.multiply(np.sin(lat[n]),np.cos(lat[m])),np.cos(delta_lbda))
        theta[n] = np.arctan2(part1,part2-part3)
    thetadeg = np.degrees(theta)
    # thetadeg varies between -180 and 180. We want from 0 t 360
    #bearing = (thetadeg + 360) % 360
    bearing = thetadeg
    return(bearing)

