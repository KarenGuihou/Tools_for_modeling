# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal
import math
import tools_modeling

### FUNCTIONS ###
def change_tzyx2tyxz(var):
    """
    Swap the axes of an array
    --
    Input   v(t,z,y,x)
    Ouput   v(t,y,x,z)
    """
    d0 = var.shape[0]
    d1 = var.shape[1]
    d2 = var.shape[2]
    d3 = var.shape[3]

    var = np.swapaxes(var,3,1)
    var = np.swapaxes(var,2,3)
    return(var)

def time_jd(yyyy,mm,dd,yyyy_orig,mm_orig,dd_orig):
    """
    Return a calendar time in julian days, from a origin.
    --
    Input   yyyy,mm,dd,yyyy_orig,mm_orig,dd_orig
    Output  jd_orig, jd_date
    """
    a,b = gcal2jd(yyyy_orig,mm_orig,dd_orig)
    jd_orig = a+b+0.5

    a,b = gcal2jd(yyyy,mm,dd)
    jd_date = (a+b+0.5) - jd_orig

    return (jd_orig, jd_date)

def calculate_dist_2_coords(lat1,lon1,lat2,lon2):
    """
    Calculate the distance between 2 coordinates using the Haversine formula. In metres
    Inspired by http://andrew.hedges.name/experiments/haversine/ 
    and http://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    --
    Input   lat1,lon1,lat2,lon2 (must be in decimal degrees)
    Output  distance
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
    distance = R * c * 1e3

    return(distance)

def get_transport(u,depth,lat,lon):
    """ 
    Calculation of transport across a given section (axe=0, along zonal, axe=1 along merid.)
    T = (u * depth * length_section)/10^6
    --
    Input   u,axe,depth,lat,lon,indice_i_orig,indice_i_end,indice_j_orig,indice_j_end
    Output  transport (Sv)
    """
    dz = tools_modeling.get_dz(depth)
    if len(lon) == 1:
        plen = len(lat)
        transport = np.zeros((plen))
        for jj in range(0,plen-1):
            length_per_grid_pt = calculate_dist_2_coords(lat[jj],lon,lat[jj+1],lon)
            transport[jj] = np.sum(u[:,jj,:]*dz[:,jj,:])*length_per_grid_pt/1e6
    elif len(lat) == 1:
        plen = len(lon)
        transport = np.zeros((plen))
        for jj in range(0,plen-1):
            length_per_grid_pt = calculate_dist_2_coords(lat,lon[jj],lat,lon[jj+1])
            transport[jj] = np.sum(u[:,:,jj]*dz[:,:,jj])*length_per_grid_pt/1e6
    elif len(lon) == len(lat):
        plen = len(lat)
        transport = np.zeros((plen))
        for jj in range(0,plen-1):
            length_per_grid_pt = calculate_dist_2_coords(lat[jj],lon[jj],lat[jj+1],lon[jj+1])
            transport[jj] = np.sum(u[:,jj]*dz[:,jj])*length_per_grid_pt/1e6
    return(transport)

def do_kdtree(combined_x_y_arrays, points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    (dist, indexes) = mytree.query(points)
    return indexes

def calculation_bearing_geodesiclib(latdeg,londeg):
    """
    Calculation of the bearing between 2 coordinates. 
    WARNING: calculation at coord(i) between coord(i-1) and coord(i+1) 
    formula: theta = atan2(sin(delta_lbda).cos(phi2),cos(phi1).sin(phi2)-sin(phi1).cos(phi2).cos(delta_lbda))
    --
    Input   lat, lon (in degrees)
    Output  theta (in degrees)
    """
    from geographiclib.geodesic import Geodesic
    geod = Geodesic.WGS84
    lat = latdeg
    lon = londeg
    theta = np.zeros(len(lat)-2)
    for ii in range(1,len(lat)-1):
        n = ii-1
        m = ii+1
        l = geod.InverseLine(lat[n],lon[n],lat[m],lon[m])
        theta[n] = l.azi1
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[0],lon[0],lat[2],lon[2],ii,theta[0]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[1],lon[1],lat[3],lon[3],ii,theta[1]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[118],lon[118],lat[120],lon[120],ii,theta[118]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[500],lon[500],lat[503],lon[503],ii,theta[500]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[503],lon[503],lat[505],lon[505],ii,theta[503]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[545],lon[545],lat[547],lon[547],ii,theta[545]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[1044],lon[1044],lat[1046],lon[1046],ii,theta[1044]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=%d) = %4.1f' % (lat[647],lon[647],lat[649],lon[649],ii,theta[647]))
    bearing = theta
    return(bearing)

def calculation_bearing(latdeg,londeg):
    """
    Calculation of the bearing between 2 coordinates from Haversine formula
    WARNING: calculation at coord(i) between coord(i-1) and coord(i+1) 
    formula: theta = atan2(sin(delta_lbda).cos(phi2),cos(phi1).sin(phi2)-sin(phi1).cos(phi2).cos(delta_lbda))
    --
    Input   lat, lon (degrees)
    Output  theta (degrees)
    """
    # we pass lat, lon in radians
    lat = np.radians(latdeg)
    lon = np.radians(londeg)
    theta = np.zeros(len(lat)-2)
    for ii in range(1,len(lat)-1):
        n = ii-1
        m = ii+1
        delta_lbda = lon[m]-lon[n]
        part1 = np.multiply(np.sin(delta_lbda),np.cos(lat[m]))
        part2 = np.multiply(np.cos(lat[n]),np.sin(lat[m]))
        part3 = np.multiply(np.multiply(np.sin(lat[n]),np.cos(lat[m])),np.cos(delta_lbda))
        theta[n] = np.arctan2(part1,part2-part3)
    thetadeg = np.degrees(theta)
    # thetadeg varies between -180 and 180. We want from 0 t 360
    bearing = (thetadeg + 360) % 360
    #lat = np.deg2rad(lat)
    #lon = np.deg2rad(lon)
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=0) = %4.1f' % (lat[0],lon[0],lat[2],lon[2],bearing[0]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=1) = %4.1f' % (lat[1],lon[1],lat[3],lon[3],bearing[1]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=118) = %4.1f' % (lat[118],lon[118],lat[120],lon[120],bearing[118]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=500) = %4.1f' % (lat[500],lon[500],lat[503],lon[503],bearing[500]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=503) = %4.1f' % (lat[503],lon[503],lat[505],lon[505],bearing[503]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=545) = %4.1f' % (lat[545],lon[545],lat[547],lon[547],bearing[545]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=1044) = %4.1f' % (lat[1044],lon[1044],lat[1046],lon[1046],bearing[1044]))
    #print('Azimuth between (%5.2fN,%5.2fE) and (%5.2fN,%5.2fE) (n=647) = %4.1f' % (lat[647],lon[647],lat[649],lon[649],bearing[647]))
    return(bearing)

def projection_U_across(bearing,u_iso,v_iso):
    """
    Projection of Ux, Vy on an axis perpendicular to a given direction "as" (e.g. across-shelf)
    Uas is positif at the right of the direction
    --
    Input   angle between north and new direction (0-360 degrees), Ux, Vy
    Output  Ux on new axis,Vy on new axis, Uas on new axis
    """
    # Pass the angles in radian and initiate variables
    Xas = np.zeros((u_iso.shape[0],u_iso.shape[1],u_iso.shape[2]))
    Yas = np.zeros((u_iso.shape[0],u_iso.shape[1],u_iso.shape[2]))
    bearing_rad = np.deg2rad(bearing)
    count90 = 0
    count180 = 0
    count270 = 0
    count360 = 0
    for n in range(len(bearing_rad)):
        # 4 possible cases, given the angle of the axis. 
        # 0       <=  theta   < pi/2
        # pi/2    <=  theta   < pi
        # pi      <=  theta   < pi*3/2
        # pi*3/2  <=  theta   < 2pi
        if bearing_rad[n]>=0 and bearing_rad[n]<math.pi/2:
            Xas[:,:,n] =    np.multiply(u_iso[:,:,n],np.cos(bearing_rad[n]))    # x' = x * cos(theta)
            Yas[:,:,n] = -1*np.multiply(v_iso[:,:,n],np.sin(bearing_rad[n]))    # y' = -y * sin(theta)
            count90 += 1
        elif bearing_rad[n]>=math.pi/2 and bearing_rad[n]<math.pi:
            Xas[:,:,n] =    np.multiply(u_iso[:,:,n],np.cos(bearing_rad[n])) # x' = u cos(theta) = -u sin(alpha+pi/2)
            Yas[:,:,n] = -1*np.multiply(v_iso[:,:,n],np.sin(bearing_rad[n])) # y' = -v sin(theta) = -v cos(alpha+pi/2)
            count180 += 1
        elif bearing_rad[n]>=(math.pi) and bearing_rad[n]<(math.pi *3/2):
            Xas[:,:,n] =    np.multiply(u_iso[:,:,n],np.cos(bearing_rad[n])) # x' =  u cos(theta) = -u cos(alpha+pi)
            Yas[:,:,n] = -1*np.multiply(v_iso[:,:,n],np.sin(bearing_rad[n])) # y' = -v sin(theta) =  v sin(alpha+pi)
            count270 += 1
        elif bearing_rad[n]>=(math.pi *3/2) and bearing_rad[n]<(math.pi *2):
            Xas[:,:,n] =    np.multiply(u_iso[:,:,n],np.cos(bearing_rad[n]))    # x' = u cos(theta) = u sin(alpha+3pi/2)
            Yas[:,:,n] = -1*np.multiply(v_iso[:,:,n],np.sin(bearing_rad[n]))    # y' = -v sin(theta) = v cos(alpha+3pi/2)
            count360 += 1
        else:
            print('pb here at n = %2d, bearing = %4.2d' %(n,bearing_rad[n]))
    count = count90 + count180 + count270 + count360
    if (count-1) != n:
        print('WARNING: some count missing. count = %4d and n= %4d' % (count,n))
    Uas = np.sum((Xas,Yas),axis=0)
    #ind = ntest
    #print('Azimuth ==> %4.3f' % (bearing_rad[ind]))
    #print('u_iso = %5.2f -> Xas = %5.2f' %(u_iso[0,1,ind],Xas[0,1,ind]))
    #print('v_iso = %5.2f -> Yas = %5.2f ==> Uas = %5.2f' % (v_iso[0,1,ind],Yas[0,1,ind],Uas[0,1,ind]))
    #print('Total number of treated points = %4d (%4d + %4d + %4d + %4d)' % (count,count90,count180,count270,count360))
    return(Xas,Yas,Uas)

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

def smooth_var(var,nlen,box):
    """
    Simple tool to smooth data, using a running window
    --
    Input   variable, lenght of the variable, size of the running window
    Output  Smoothed variable
    """
    var_smoothed = var * 1 #np.zeros_like(var)
    ind = box/2
    for n in range(ind,nlen-ind):
        if len(var.shape) == 1:
            var_smoothed[n] =  np.mean(var[n-ind:n+ind+1])            #(var[n-1]+var[n]+var[n+1])/3
        elif len(var.shape) == 2:
            var_smoothed[:,n] = np.mean(var[:,n-ind:n+ind+1],axis=1)    #(var[:,n-1]+var[:,n]+var[:,n+1])/3
        elif len(var.shape) == 3:
            var_smoothed[:,:,n] = np.mean(var[:,:,n-ind:n+ind+1],axis=2)  #(var[:,:,n-1]+var[:,:,n]+var[:,:,n+1])/3
        else:
            print('WARNING Case not coded')
    return(var_smoothed)

def sort_index(var):
    """
    Return a sorted variable and the corresponding indexes
    --
    Input   var
    Output  var_sorted, index
    """
    sortedvar = sorted(var)
    sortedindex = [b[0] for b in sorted(enumerate(var),key=lambda i:i[1])]
    return sortedvar, sortedindex
