# coding: utf-8
from scipy.io import loadmat
import numpy as np
from scipy.io import netcdf
import os
import sys 
from jdcal import gcal2jd, jd2gcal
import sys
import tools_modeling
import tools_calcul
import tools_modeling_CMM

##### Functions #####
def select_paths_grid(model,iso):
    '''
    load the lon,lat,depth for each model
    ~~~
    Input: model
    '''
    if model == 'nemo':
        path_data = '/media/workspace/VOCES/Runs/NEMO_ORCA/'
        path_bathy = path_data + 'orca12_2000_01_swas_uv.nc'
        lon_name = 'lonr'
        lat_name = 'latr'
        zlev_name = 'Z'
        lon = tools_modeling.load_ncvar(path_bathy,lon_name,'nc')[0,:]
        lat = tools_modeling.load_ncvar(path_bathy,lat_name,'nc')[:,0]
        depth = tools_modeling.load_ncvar(path_bathy,zlev_name,'nc')
        depth3d = np.zeros((len(depth),len(lat),len(lon)))
        for z in range(0,len(depth)):
            depth3d[z,:,:] = depth[z]
        depth = depth3d
        uname = 'u'
        vname = 'v'
    elif model == 'cmm':
        path_data = '/media/workspace/VOCES/Runs/Combes_monthly_nc/'
        path_bathy = '/media/workspace/VOCES/Runs/Combes_grid/roms_agrif_grd.nc'
        lon_name = 'lon_rho'
        lat_name = 'lat_rho'
        zlev_name = 'h'
        uname = 'u'
        vname = 'v'
        lon = tools_modeling.load_ncvar(path_bathy,lon_name,'nc')[1:-1,1:-1]
        lat = tools_modeling.load_ncvar(path_bathy,lat_name,'nc')[1:-1,1:-1]
        depth = tools_modeling.load_ncvar(path_bathy,zlev_name,'nc')[1:-1,1:-1]
        output_root = 'ncfiles/CMM_'+str(iso)+'_' # % (y,m,isobath))

    return(path_data,lon,lat,depth,uname,vname,output_root)

def get_var_along_iso(var,tlen,zlen,nlen,indx,indy,smoothing):
    if len(var.shape) == 2:
        var_iso = np.zeros(nlen)
        for n in range(nlen):
            var_iso[n] = var[int(indx[n]),int(indy[n])]
    elif len(var.shape) == 3:
        var_iso = np.zeros((zlen,nlen))
        for n in range(nlen):
            var_iso[:,n] = var[:,int(indx[n]),int(indy[n])]
    elif len(var.shape) == 4:
        var_iso = np.zeros((tlen,zlen,nlen))
        for t in range(tlen):
            for n in range(nlen):
                var_iso[t,:,n] = var[t,:,int(indx[n]),int(indy[n])]
    else:
        print('WARNING Case not coded')
    var_iso=  np.ma.masked_where(var_iso == 0, var_iso)
    if smoothing == 1:
        var_iso = tools_calcul.smooth_var(var_iso,nlen,5)
    return(var_iso)

#def get_ind_iso(nlen,ind_iso):
#    indx = np.zeros(nlen)
#    indy = np.zeros(nlen)
#    for n in range(len(ind_iso[:,0])):
#        indx[n] = int(ind_iso[n,0])
#        indy[n] = int(ind_iso[n,1])
#    return(indx,indy)

def get_iso_contour(depth,isobath,fig):
    sys.path.append('/media/workspace/VOCES/Utils/scikit-image/')
    from skimage import measure
    ind_iso = np.rint(max(measure.find_contours(depth,isobath),key=len)) 
    nlen = len(ind_iso[:,0])
    if 1 == fig :    # fig
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(h, interpolation='nearest', cmap=plt.cm.gray)
        ax.plot(ind_iso[:, 1], ind_iso[:, 0], linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    indx = np.zeros(nlen)
    indy = np.zeros(nlen)
    for n in range(len(ind_iso[:,0])):
        indx[n] = int(ind_iso[n,0])
        indy[n] = int(ind_iso[n,1])

    return(indx,indy,nlen)

## Get the variables along the isobath
def get_coordinates_contour(depth,isobath,lon,lat):
    indx,indy,nlen = get_iso_contour(depth,isobath,0)
    lat_iso = get_var_along_iso(lat,0,0,nlen,indx,indy,1)
    lon_iso = get_var_along_iso(lon,0,0,nlen,indx,indy,1)
    h_iso = get_var_along_iso(depth,0,0,nlen,indx,indy,1)
    return(lat_iso,lon_iso,h_iso,indx,indy,nlen)

def load_var(model,path_data,mstr,ystr):
    if model == 'cmm':
        filenameu = path_data + 'CMM_y'+ystr+'m'+mstr+'_U.nc'
        filenamev = path_data + 'CMM_y'+ystr+'m'+mstr+'_V.nc'
        filenamet = path_data + 'CMM_y'+ystr+'m'+mstr+'_T.nc'
        print(filenamet)
        #output = ('CMM_y%4dm%02d_isobath%04d.nc' % (y,m,isobath))
        # load files
        u = tools_modeling.load_ncvar(filenameu,'u','nc')
        v = tools_modeling.load_ncvar(filenamev,'v','nc')
        temp = tools_modeling.load_ncvar(filenamet,'temp','nc')
        salt = tools_modeling.load_ncvar(filenamet,'salt','nc')
        u_rho,v_rho = tools_modeling.uv_on_Tpoint(u,v)
        time = tools_modeling.load_ncvar(filenamet,'time','nc')
        depth_lev = tools_modeling.load_ncvar(filenamet,'depth','nc')
        return(u_rho,v_rho,temp,salt,time,depth_lev)

def writeNC_along_contour(output_root,ystr,mstr,bearing,u,v,lon,lat,Uas,Xas,Yas,temp,salt,dz,depth,time,depth_lev,transport):
    output = output_root+'y'+ystr+'m'+mstr+'.nc' # % (y,m,isobath))
    xlenwr = len(bearing)
    tlenwr = u.shape[0]
    zlenwr = u.shape[1]
    coord = np.array([lon[:],lat[:]])

    f = netcdf.netcdf_file(output, 'w')
    f.history = 'Velocities across the isobath'
    f.createDimension('x',xlenwr)
    f.createDimension('time',tlenwr)
    f.createDimension('depth',zlenwr)
    f.createDimension('nax',2)

    tools_modeling.writevar3D(f,Uas,'Uacross','time','depth','x','cm/s')
    tools_modeling.writevar3D(f,temp,'temp','time','depth','x','Celsius degrees')
    tools_modeling.writevar3D(f,salt,'salt','time','depth','x','PSU')
    tools_modeling.writevar3D(f,u,'u','time','depth','x','cm/s')
    tools_modeling.writevar3D(f,v,'v','time','depth','x','cm/s')
    tools_modeling.writevar3D(f,dz,'dz','time','depth','x','metres')
    tools_modeling.writevar3D(f,Xas,'Xas','time','depth','x','cm/s')
    tools_modeling.writevar3D(f,Yas,'Yas','time','depth','x','cm/s')
    tools_modeling.writevar2D(f,coord[:,1:-1],'coord','nax','x','Degrees West')
    tools_modeling.writevar2D(f,transport,'transport','time','x','Sv')
    tools_modeling.writevar2D(f,depth_lev[:,1:-1],'depth','depth','x','Sv')
    tools_modeling.writevar1D(f,bearing,'angle_isobath','x','degrees to North')
    tools_modeling.writevar1D(f,lon[1:-1],'lon','x','Degrees West')
    tools_modeling.writevar1D(f,lat[1:-1],'lat','x','Degrees North')
    tools_modeling.writevar1D(f,depth[1:-1],'h','x','metres')
    tools_modeling.writevar1D(f,time,'t','time','days since 1980-01-01')
    f.close()

def write_tr(transport,trindy,trindm,lon,lat):
    output = 'Transport_shelfbreak.nc' # % (y,m,isobath))
    xlenwr = transport.shape[0]
    mlenwr = trindm
    ylenwr = trindy

    f = netcdf.netcdf_file(output, 'w')
    f.history = 'Transport along the shelf'
    f.createDimension('x',xlenwr)
    f.createDimension('year',ylenwr)
    f.createDimension('month',mlenwr)

    tools_modeling.writevar3D(f,transport,'transport','x','year','month','cm/s')
    tools_modeling.writevar1D(f,lon[1:-1],'lon','x','Degrees West')
    tools_modeling.writevar1D(f,lat[1:-1],'lat','x','Degrees North')
    f.close()


def loop_on_yyyymm(depth,lon,lat,years,months,path_data,model,indx,indy,nlen,output_root):
    transport = np.zeros((nlen-2,len(years),len(months)))
    trindy = 0
    for y in years:
        trindm = 0
        for m in months:
            print('Processing ',y,m)
            mstr = "%02d" % (m+1,)
            ystr = "%04d" % (y+1,)
            #u,v,temp,salt,depth,time = load_var(model,path_data,mstr,ystr):
            u,v,temp,salt,time,depth_lev = load_var(model,path_data,mstr,ystr)
            # get dimensions
            tlen = u.shape[0]
            zlen = u.shape[1]
            xlen = u.shape[2]
            ylen = u.shape[3]
        
            # extract along-isobath var
            #depth_iso = get_var_along_iso(depth,0,zlen,nlen,indx,indy,1)
            u_iso = get_var_along_iso(u,tlen,zlen,nlen,indx,indy,1)[:,:,1:-1]
            v_iso = get_var_along_iso(v,tlen,zlen,nlen,indx,indy,1)[:,:,1:-1]
            temp_iso = get_var_along_iso(temp,tlen,zlen,nlen,indx,indy,1)[:,:,1:-1]
            salt_iso = get_var_along_iso(salt,tlen,zlen,nlen,indx,indy,1)[:,:,1:-1]
            depth_iso = get_var_along_iso(depth_lev,0,zlen,nlen,indx,indy,1)
            dz_iso = tools_modeling.get_dz(depth_iso)[:,1:-1]
            
            # Across-Shelf velocity
            bearing = tools_calcul.calculation_bearing(lat,lon)
            Xas,Yas,Uas = tools_calcul.projection_U_across(bearing,u_iso,v_iso)

            # Transport
            transport[:,trindy,trindm] = tools_calcul.get_transport(Uas[0,:,:],depth_iso[:,1:-1],lat[1:-1],lon[1:-1])

            # Writing result to a file
            writeNC_along_contour(output_root,ystr,mstr,bearing,u_iso,v_iso,lon,lat,Uas,Xas,Yas,temp_iso,salt_iso,dz_iso,depth,time,depth_iso,transport[:,trindy,trindm])
            trindm+=1
        trindy+=1
    write_tr(transport,trindy,trindm,lon,lat)

def main(model,years,months):
    path_data,lon,lat,depth,uname,vname,output_root = select_paths_grid(model,isobath)
    lat_iso,lon_iso,depth_iso,indx,indy,nlen = get_coordinates_contour(depth,isobath,lon,lat)
    loop_on_yyyymm(depth_iso,lon_iso,lat_iso,years,months,path_data,model,indx,indy,nlen,output_root)

isobath = 200
years = range(1979,2006)
months = range(0,12)
main('cmm',years,months)
### TO DO  main(nemo)
