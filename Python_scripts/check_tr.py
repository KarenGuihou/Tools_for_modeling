# coding: utf-8

from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys 

import tools_modeling
import tools_calcul

from jdcal import gcal2jd, jd2gcal
from scipy.spatial import distance
from math import sin, cos, sqrt, atan2, radians

def select_paths(model,axe):
    '''
    load the lon,lat,depth for each model
    ~~~
    Input: model, axe
    '''
    if model == 'nemo':
        path_data = '/media/workspace/VOCES/Runs/NEMO_ORCA/'
        path_bathy = path_data + 'orca12_2000_swas_uv.nc'
        lon_name = 'lonr'
        lat_name = 'latr'
        zlev_name = 'ZW'
        if axe == 'zonal':
            varname = 'u'
        elif axe == 'merid':
            varname = 'v'
        lon = tools_modeling.load_ncvar(path_bathy,lon_name,'nc')[:,0]
        lat = tools_modeling.load_ncvar(path_bathy,lat_name,'nc')[0,:]
    elif model == 'cmm':
        path_data = '/media/workspace/VOCES/Runs/Combes_monthly_nc/'
        if axe == 'zonal':
            path_bathy = path_data + 'CMM_y2000m01_U.nc'
            lon_name = 'longitude'
            lat_name = 'latitude'
            zlev_name = 'depth_u'
            varname = 'U'
        elif axe == 'merid':
            path_bathy = path_data + 'CMM_y2000m01_V.nc'
            lon_name = 'longitude'
            lat_name = 'latitude'
            zlev_name = 'depth_v'
            varname = 'V'
        lon = tools_modeling.load_ncvar(path_bathy,lon_name,'nc')
        lat = tools_modeling.load_ncvar(path_bathy,lat_name,'nc')
    depth = tools_modeling.load_ncvar(path_bathy,zlev_name,'nc')
    if model == 'nemo':
        depth3d = np.zeros((len(depth),len(lat),len(lon)))
        for z in range(0,len(depth)):
            depth3d[z,:,:] = depth[z]
        depth = depth3d

    return(path_data,lon,lat,depth,varname)

def ind_axes(axe,i0,i1,j0,j1):
    if axe == 'zonal':
        i0 = i0-1
        j0 = j0-1
        j1 = j1+1
        plen = j1-j0
    else:
        i0 = i0-1
        i1 = i1+1
        j0 = j0-1
        plen = i1-i0
    print(i0,i1,j0,j1)
    return(i0,i1,j0,j1,plen)

def load_vel(model,path_data,yyyy,mm,varname,i0,i1,j0,j1):
    if model == 'nemo':
        filename=path_data+'orca12_'+str(yyyy)+'_swas_uv.nc'
        u = np.ma.masked_invalid(tools_modeling.load_ncvar(filename,varname,'nc'))[mm,:,j0:j1,i0:i1]
    elif model == 'cmm':
        mstr = "%02d" % (mm+1,)
        filename=path_data+'CMM_y'+str(yyyy)+'m'+mstr+'_'+varname.upper()+'.nc'
        u = tools_modeling.load_ncvar(filename,varname.lower(),'nc')[0,:,j0:j1,i0:i1]
    return(u)
    f.close()

def write_tr(loc,model,transport,transport_corrected):
    output = 'Transport_MS_'+loc+'_'+model+'.nc'
    f = netcdf.netcdf_file(output, 'w')
    f.history = 'Monthly transport from'+model+'at '+loc
    f.createDimension('month',transport.shape[1])
    f.createDimension('year',transport.shape[0])
    f.createDimension('npts',transport.shape[2])
    tools_modeling.writevar3D(f,transport,'transport','year','month','npts','Sv')
    tools_modeling.writevar3D(f,transport_corrected,'transport_QC','year','month','npts','Sv')
    f.close()

def get_transport(u,depth,lat,lon):
    """ 
    Calculation of transport across a given section (axe=0, along zonal, axe=1 along merid.)
    T = (u * depth * length_section)/10^6
    --
    Input   u,axe,depth,lat,lon,indice_i_orig,indice_i_end,indice_j_orig,indice_j_end
    Output  transport (Sv)
    """
    dz = tools_modeling.get_e3u(depth)
    print('dz----')
    print(dz.shape)
    print('u----')
    print(u.shape)
    if len(lon) == 1:
        plen = len(lat)
        transport = np.zeros((plen))
        for jj in range(0,plen-1):
            print(u.shape)
            print(dz.shape)
            length_per_grid_pt = tools_calcul.calculate_dist_2_coords(lat[jj],lon,lat[jj+1],lon)
            transport[jj] = np.sum(u[:,jj,:]*dz[:,jj,:])*length_per_grid_pt/1e6
    else:
        plen = len(lon)
        transport = np.zeros((plen))
        for jj in range(0,plen-1):
            length_per_grid_pt = tools_calcul.calculate_dist_2_coords(lat,lon[jj],lat,lon[jj+1])
            print(u.shape)
            print(dz.shape)
            transport[jj] = np.sum(u[:,:,jj]*dz[:,:,jj])*length_per_grid_pt/1e6
    print(transport)
    return(transport)

def calculate_tr(loc, model,varname, i0,i1,j0,j1,plen,path_data,filevarname,lon,lat,depth):
    #Define indices
    years = range(1980,1982)
    months = range(0,03)
    indm = 0
    transport = np.zeros((len(years),len(months),plen))
    # loop on mm and yyyy
    for mm in months:
        mstr = "%02d" % (mm,)
        indy = 0
        for yyyy in years:
            print(mm,yyyy)
            usec      = load_vel(model,path_data,yyyy,mm,varname,i0,i1,j0,j1)
            depthsec  = depth[:,j0:j1,i0:i1]
            latsec    = lat[j0:j1]
            lonsec    = lon[i0:i1]
            transport[indy,indm,:] = get_transport(usec,depthsec,latsec,lonsec)
            indy+=1
        indm+=1
    transport_corrected = np.where(transport >= np.mean(transport,axis=0) + np.std(transport,axis=0)*2,0,transport)
    transport_corrected = np.where(transport <= np.mean(transport,axis=0) - np.std(transport,axis=0)*2,0,transport_corrected)
    return(transport,transport_corrected)

def launch_cal(model,axe,zone,i0,i1,j0,j1):
    print(model,axe,zone)
    path_data,lon,lat,depth,varname = select_paths(model,axe)
    i0,i1,j0,j1,plen = ind_axes(axe,i0,i1,j0,j1)
    transport,transport_corrected = calculate_tr(zone,model,varname,i0,i1,j0,j1,plen,path_data,varname,lon,lat,depth)
    write_tr(zone,model,transport,transport_corrected)


launch_cal('cmm' ,'zonal','MSatl'   ,155,155,162,168)
launch_cal('cmm' ,'merid','MScent'  ,129,133,150,150)
