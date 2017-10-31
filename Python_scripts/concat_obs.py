# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys
from jdcal import gcal2jd, jd2gcal
import tools_modeling
import tools_calcul


path = '/media/data1/VOCES/Observations/MagellanStrait/MS_coriolis/Orig_files/'
fic = os.listdir(path)

temp_all = 0
salt_all = 0
depth_all = 0
latitude_all = 0
longitude_all = 0
juld_all = 0

# extract for each file T,S,lat,lon,depth,time
for n in fic:
    inputfile = path + n
    print('processing %s' % (file,))
    temp = tools_modeling.load_ncvar(inputfile,'TEMP','nc')
    juld = tools_modeling.load_ncvar(inputfile,'JULD','nc')
    latitude = tools_modeling.load_ncvar(inputfile,'LATITUDE','nc')
    longitude = tools_modeling.load_ncvar(inputfile,'LONGITUDE','nc')
    salt = tools_modeling.load_ncvar(inputfile,'PSAL','nc')
    depth = tools_modeling.load_ncvar(inputfile,'DEPH','nc')
    for tt in range(0,temp.shape[0]):
        for xx in range(0,temp.shape[1]):
            if temp[tt,xx] != 99999.0:
                temp_all = np.append(temp_all,temp[tt,xx])
                latitude_all = np.append(latitude_all,latitude[tt])
                longitude_all = np.append(longitude_all,longitude[tt])
                juld_all = np.append(juld_all,juld[tt])
                if isinstance(salt,int):
                    salt_all = np.append(salt_all,0)
                elif salt[tt,xx] == 99999.0:
                    salt_all = np.append(salt_all,0)
                else:
                    salt_all = np.append(salt_all,salt[tt,xx])
                if isinstance(depth,int):
                    depth_all = np.append(depth_all,0)
                else:
                    depth_all = np.append(depth_all,depth[tt,xx])

# remove first item (0)
temp_all = temp_all[1:]
salt_all = salt_all[1:]
depth_all = depth_all[1:]
latitude_all = latitude_all[1:]
longitude_all = longitude_all[1:]
juld_all = juld_all[1:]

# Sort by date
[juld_sort, index_sort] = tools_calcul.sort_index(juld_all)
temp_sort = temp_all[index_sort]
salt_sort = salt_all[index_sort]
latitude_sort = latitude_all[index_sort]
longitude_sort = longitude_all[index_sort]
depth_sort = depth_all[index_sort]

# Write output
output = 'Coriolis_MS.nc'
f = netcdf.netcdf_file(output, 'w')
f.history = 'Coriolis data from 1980 to 2006, in the Magellan Strait'
f.createDimension('time',len(juld_all))
tools_modeling.writevar1D(f,temp_sort,'temp','time','degrees C')
tools_modeling.writevar1D(f,salt_sort,'psal','time','PSU')
tools_modeling.writevar1D(f,latitude_sort,'latitude','time','degrees N')
tools_modeling.writevar1D(f,longitude_sort,'longitude','time','degrees E')
tools_modeling.writevar1D(f,depth_sort,'depth','time','metres')
tools_modeling.writevar1D(f,juld_sort,'juld','time','days since 1950-01-01 00:00:00 UTC')
