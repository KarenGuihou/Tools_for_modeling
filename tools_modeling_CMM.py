# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal

def load_grid(gridname,filetype):
    """
    Read the grid file from Combes & Matano
    --
    Input   filename
    Ouput   h,lat_rho,lon_rho,lat_u,lon_u,lat_v,lon_v
    """
    if filetype == 'nc':
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

def zlevs_1d(h, zeta, theta_s, theta_b, hc, N, type, vtransform):
    """ 
    clone from zlevs_1d.m (ROMSTOOLS_v3.1_03_02_2014)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%
	%  function z = zlevs_1d(h, zeta, theta_s, theta_b, hc, N, type, vtransform)
	%
	%  this function compute the depth of rho or w points for ROMS
	%
	%  On Input:
	%
	%    type    'r': rho point 'w': w point
	%    oldnew     : old (Song, 1994) or new s-coord (Sasha, 2006)
	%
	%  On Output:
	%
	%    z=z(k)       Depths (m) of RHO- or W-points (1D matrix).
	% 
	%  Further Information:  
	%  http://www.brest.ird.fr/Roms_tools/
	%  
	%  This file is part of ROMSTOOLS
	%
	%  ROMSTOOLS is free software; you can redistribute it and/or modify
	%  it under the terms of the GNU General Public License as published
	%  by the Free Software Foundation; either version 2 of the License,
	%  or (at your option) any later version.
	%
	%  ROMSTOOLS is distributed in the hope that it will be useful, but
	%  WITHOUT ANY WARRANTY; without even the implied warranty of
	%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	%  GNU General Public License for more details.
	%
	%  You should have received a copy of the GNU General Public License
	%  along with this program; if not, write to the Free Software
	%  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
	%  MA  02111-1307  USA
	%
	%  Copyright (c) 2002-2006 by Pierrick Penven 
	%  e-mail:Pierrick.Penven@ird.fr  
	%
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    M,L = h.shape

    # Set S-Curves in domain [-1 < sc < 0] at vertical W- and RHO-points.
    cff1 = 1./np.sinh(theta_s)
    cff2 = 0.5/np.tanh(0.5*theta_s)
    
    if type=='w':
        sc = sc = (np.arange(0,N,dtype=np.float) - N)/N
        N = N + 1
    else:
        sc = (np.arange(0,N,dtype=np.float)-N-0.5) / N
    
    Cs = (1.-theta_b) * cff1 * np.sinh(theta_s * sc) + theta_b * (cff2 * np.tanh(theta_s * (sc + 0.5)) - 0.5)

    # Create S-coordinate system: based on model topography h(i,j),
    # fast-time-averaged free-surface field and vertical coordinate
    # transformation metrics compute evolving depths of of the three-
    # dimensional model grid.

    z = np.zeros((N,M,L))
    if (vtransform==1):
        print('--- using old s-coord')
        hinv=1./h
        cff=hc*(sc-Cs)
        cff1=Cs
        cff2=sc+1
        for k in range(0,N):
            z0=cff[k]+cff1[k]*h
            z[k,:,:]=z0+zeta*(1+z0*hinv)
    elif (vtransform==2):
        print('--- using new s-coord')
        hinv=1./(h+hc)
        cff=hc*sc
        cff1=Cs
        for k in range(0,N):
            z[k,:,:]=zeta+(zeta+h)*(cff[k]+cff1[k]*h)*hinv
    else:
        print('wrong argument in zlevs_1d')

    return(z)

