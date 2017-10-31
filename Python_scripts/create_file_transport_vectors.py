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


def write_tr(name,transport,time):
    output = name+'_vector.nc'
    f = netcdf.netcdf_file(output, 'w')
    f.history = 'Monthly transport'
    f.createDimension('time',transport.shape[0])
    f.createDimension('npts',transport.shape[1])
    tools_modeling.writevar2D(f,transport,'transport','time','npts','Sv')
    tools_modeling.writevar1D(f,time,'date','time','days since 01-01-1980')
    f.close()

def launch_cal(name):
    filename = name+'.nc'
    transport = tools_modeling.load_ncvar(filename,'transport','nc')
    years = range(1980,2007)
    months = range(1,13)
    ylen = transport.shape[0]
    mlen = transport.shape[1]
    nlen = transport.shape[2]
    date = np.zeros((ylen*mlen))
    print(date.shape)
    print(date[0])
    indy = 0
    compteur = 0
    while compteur < ylen*mlen-1:
        print(compteur)
        compteur += 1
        date[compteur] = date[compteur-1]+365/12
    transport_vec = np.reshape(transport,(ylen*mlen,nlen))
    write_tr(name,transport_vec,date)


#launch_cal('Transport_leMaire_cmm')
#launch_cal('Transport_MSatl_cmm')
#launch_cal('Transport_LeMaire_nemo')
#launch_cal('Transport_MSatl_nemo')
launch_cal('Transport_S1_cmm')
launch_cal('Transport_S2_cmm')
launch_cal('Transport_S3_cmm')



