'''
The idea of this code is to identify boundary points and their orientation. 
This information can then be used to calculate cross boundary fluxes. 
@author: James Harle
'''
#External Imports
import numpy as np
import scipy.spatial as sp
from netCDF4 import Dataset
import matplotlib.pyplot as plt

class Boundary:
    # Bearings for overlays
    _NORTH = [1,-1,1,-1,2,None,1,-1]
    _SOUTH = [1,-1,1,-1,None,-2,1,-1]
    _EAST = [1,-1,1,-1,1,-1,2,None]
    _WEST = [1,-1,1,-1,1,-1,None,-2]
    
    def __init__(self, msk_file, fileH):
        """Generates and order the indices for NEMO Boundary and returns a 
        Grid object with indices and end points
        
        Keyword arguments:
        boundary_mask -- boundary mask
        Attributes:
        bdy_i -- index
        bdy_r -- r index 
        """
        nc_msk = Dataset(msk_file,mode='r')
        nc_msk_var = nc_msk.variables
        boundary_mask = nc_msk_var["mask"][:,:]        
        nc_msk.close()
        bdy_msk = boundary_mask.copy()
        ndif_b = -1
        ndif_n = 0
        
        while ndif_b < ndif_n:
            
            ndif_b = ndif_n
            
            # remove any narrow strips or isolated channels from the mask
            msk = np.pad(bdy_msk,((1,1),(1,1)), 'constant', constant_values=(-1))
            grid_ind = np.logical_and.reduce((msk[1:-1,  :-2] == -1,
                                              msk[1:-1, 1:-1] ==  1,
                                              msk[1:-1, 2:  ] == -1))
            bdy_msk[grid_ind] = -1
            msk = np.pad(bdy_msk,((1,1),(1,1)), 'constant', constant_values=(-1))
            grid_ind = np.logical_and.reduce((msk[ :-2, 1:-1] == -1,
                                              msk[1:-1, 1:-1] ==  1,
                                              msk[2:  , 1:-1] == -1))
            bdy_msk[grid_ind] = -1
    
            msk = np.pad(bdy_msk,((1,1),(1,1)), 'constant', constant_values=(-1))
            grid_ind = np.logical_and.reduce((msk[1:-1,  :-2] ==  1,
                                              msk[1:-1, 1:-1] == -1,
                                              msk[1:-1, 2:  ] ==  1))
            bdy_msk[grid_ind] =  1
            msk = np.pad(bdy_msk,((1,1),(1,1)), 'constant', constant_values=(-1))
            grid_ind = np.logical_and.reduce((msk[ :-2, 1:-1] ==  1,
                                              msk[1:-1, 1:-1] == -1,
                                              msk[2:  , 1:-1] ==  1))
            bdy_msk[grid_ind] =  1
            
            ndif_n = np.sum((bdy_msk[:] != boundary_mask[:]))
            
        # Create padded array for overlays
        msk = np.pad(bdy_msk,((1,1),(1,1)), 'constant', constant_values=(-1))
        # create index arrays of I and J coords
        igrid, jgrid = np.meshgrid(np.arange(bdy_msk.shape[1]), np.arange(bdy_msk.shape[0]))

        # first we start with the eastern points of the open boundary
        EBi, EBj = self._find_bdy(igrid, jgrid, msk, self._EAST)
        
        # create a 2D array index for the points that are on border
        tij = np.column_stack((EBi,EBj)) 
        #bdy_ie = np.transpose(tij, (1, 2, 0))
        bdy_ie = tij
        bdy_re = np.tile(2,(bdy_ie.shape[0],1))
        
        # next we look for western points of the open boundary
        WBi, WBj = self._find_bdy(igrid, jgrid, msk, self._WEST)
        
        # create a 2D array index for the points that are on border
        tij = np.column_stack((WBi-1,WBj))        
        #bdy_iw = np.transpose(tij, (1, 2, 0))
        bdy_iw = tij
        bdy_rw = np.tile(4,(bdy_iw.shape[0],1))

        # Create padded array for overlays
        msk = np.pad(bdy_msk,((1,1),(1,1)), 'constant', constant_values=(-1))
        # create index arrays of I and J coords
        igrid, jgrid = np.meshgrid(np.arange(bdy_msk.shape[1]), np.arange(bdy_msk.shape[0]))

        # first we start with the eastern points of the open boundary
        NBi, NBj = self._find_bdy(igrid, jgrid, msk, self._NORTH)
        
        # create a 2D array index for the points that are on border
        tij = np.column_stack((NBi,NBj))        
        #bdy_in = np.transpose(tij, (1, 2, 0))
        bdy_in = tij
        bdy_rn = np.tile(1,(bdy_in.shape[0],1))
        
        # next we look for western points of the open boundary
        SBi, SBj = self._find_bdy(igrid, jgrid, msk, self._SOUTH)
        
        # create a 2D array index for the points that are on border
        tij = np.column_stack((SBi,SBj-1))        

        bdy_is = tij
        bdy_rs = np.tile(3,(bdy_is.shape[0],1))
        
        bdy_i = np.concatenate((bdy_in, bdy_ie, bdy_is, bdy_iw))
        bdy_r = np.concatenate((bdy_rn, bdy_re, bdy_rs, bdy_rw))

        # offset the u and v points about the t point for ordering       
        nbidta = bdy_i[:,0]+0.5
        nbjdta = bdy_i[:,1]+0.5
        nbidta[np.ravel(np.logical_or.reduce((bdy_r[:] == 2, bdy_r[:] == 4)))] += 0.5
        nbjdta[np.ravel(np.logical_or.reduce((bdy_r[:] == 1, bdy_r[:] == 3)))] += 0.5
             
        id_order, end_pts = self._bdy_sections(nbidta,nbjdta)
        # tidy            
        self.bdy_i = bdy_i[id_order,:]
        self.bdy_r = bdy_r[id_order]
        self.msk_o = boundary_mask
        self.msk_n = bdy_msk
        self.end_p = end_pts
        
        # get the lons and lats of the open boundary points

        # Identify the sub region need to save on IO
        min_i = np.min(self.bdy_i[:,0])
        max_i = np.max(self.bdy_i[:,0])
        min_j = np.min(self.bdy_i[:,1])
        max_j = np.max(self.bdy_i[:,1])
        # Set up nc pointers and extract data
        nc_h   = Dataset(fileH,mode='r')
        nav_lon = nc_h.variables["glamt"][0,min_j:max_j+1,min_i:max_i+1]        
        nav_lat = nc_h.variables["gphit"][0,min_j:max_j+1,min_i:max_i+1]
        nc_h.close()

        # rejig the indices to reflect the reduced domain
        subx = self.bdy_i[:,0] - min_i
        suby = self.bdy_i[:,1] - min_j       
        ind  = self._sub2ind(nav_lon.squeeze().shape,subx,suby)
        
        nx      = max_i-min_i+1
        ny      = max_j-min_j+1
        nav_lon = np.reshape(nav_lon,(nx*ny))        
        nav_lat = np.reshape(nav_lat,(nx*ny)) 
        
        self.lon   = nav_lon[ind]
        self.lat   = nav_lat[ind]
        
    def _plt(self):
        """ Plot up boundary point on mask and orginal mask 
        """
        nbidta = self.bdy_i[:,0]+0.5
        nbjdta = self.bdy_i[:,1]+0.5
        nbidta[np.ravel(np.logical_or.reduce((self.bdy_r[:] == 2, self.bdy_r[:] == 4)))] += 0.5
        nbjdta[np.ravel(np.logical_or.reduce((self.bdy_r[:] == 1, self.bdy_r[:] == 3)))] += 0.5
        fig, ax = plt.subplots(1, 1)
        plt.pcolormesh(self.msk_o)
        ind = np.ravel(self.bdy_r[:] == 1)   
        plt.scatter(self.bdy_i[ind,0]+0.5,self.bdy_i[ind,1]+1,c='r',s=10)
        ind = np.ravel(self.bdy_r[:] == 2)   
        plt.scatter(self.bdy_i[ind,0]+1,self.bdy_i[ind,1]+0.5,c='m',s=10)
        ind = np.ravel(self.bdy_r[:] == 3)   
        plt.scatter(self.bdy_i[ind,0]+0.5,self.bdy_i[ind,1]+1,c='g',s=10)
        ind = np.ravel(self.bdy_r[:] == 4)   
        plt.scatter(self.bdy_i[ind,0]+1,self.bdy_i[ind,1]+0.5,c='y',s=10)
        fig, ax = plt.subplots(1, 1)
        plt.pcolormesh(self.msk_n)
        ind = np.ravel(self.bdy_r[:] == 1)   
        plt.scatter(self.bdy_i[ind,0]+0.5,self.bdy_i[ind,1]+1,c='r',s=10)
        ind = np.ravel(self.bdy_r[:] == 2)   
        plt.scatter(self.bdy_i[ind,0]+1,self.bdy_i[ind,1]+0.5,c='m',s=10)
        ind = np.ravel(self.bdy_r[:] == 3)   
        plt.scatter(self.bdy_i[ind,0]+0.5,self.bdy_i[ind,1]+1,c='g',s=10)
        ind = np.ravel(self.bdy_r[:] == 4)   
        plt.scatter(self.bdy_i[ind,0]+1,self.bdy_i[ind,1]+0.5,c='y',s=10)
        for t in np.arange(len(nbidta)):
            plt.text(nbidta[t],nbjdta[t],str(t))
        
    def _remove_duplicate_points(self, bdy_i, bdy_r):
        """ Removes the duplicate points in the bdy_i and return the bdy_i and bdy_r
        bdy_i -- bdy indexes
        bdy_r -- bdy rim values 
        """
        bdy_i2 = np.transpose(bdy_i, (1, 0))
        uniqind = self._unique_rows(bdy_i2)

        bdy_i = bdy_i2[uniqind]
        bdy_r = bdy_r[uniqind]
        return bdy_i, bdy_r
    
    def _remove_landpoints_open_ocean(self, mask, bdy_i, bdy_r):
        """ Removes the land points and open ocean points """        
        unmask_index = mask[bdy_i[:,1],bdy_i[:,0]] != 0 
        bdy_i = bdy_i[unmask_index, :]
        bdy_r = bdy_r[unmask_index]
        return bdy_i, bdy_r, unmask_index        
        
    def _find_bdy(self, I, J, mask, brg):
        """Finds the border indexes by checking the change from ocean to land.
        Returns the i and j index array where the shift happens.
        
        Keyword arguments:
        I -- I x direction indexes
        J -- J y direction indexes
        mask -- mask data
        brg -- mask index range
        """
        # subtract matrices to find boundaries, set to True
        m1 = mask[brg[0]:brg[1], brg[2]:brg[3]]
        m2 = mask[brg[4]:brg[5], brg[6]:brg[7]]
        overlay = np.subtract(m1,m2)
        # Create boolean array of bdy points in overlay
        bool_arr = overlay==2
        # index I or J to find bdies
        bdy_I = I[bool_arr]
        bdy_J = J[bool_arr]
        
        return bdy_I, bdy_J

    def _fill(self, mask, ref, brg):
        """  """
        tmp = mask[brg[4]:brg[5], brg[6]:brg[7]]
        ind = (ref - tmp) > 1
        ref[ind] = tmp[ind] + 1
        mask[brg[0]:brg[1], brg[2]:brg[3]] = ref

        return mask, ref
        
    def _flux(self, fileU, fileV, fileT, fileH, fileZ, msk_version):
        """ return the fluxes from the identified boundary points
        
        Requires:
        fileU - list of input zonal velocity files
        fileV - list of input meridional velocity files
        fileT - list of input sea surface height files
        fileU - str input of mesh_hgr
        fileU - str input of mesh_zgr
        
        """
        # Identify the sub region need to save on IO
        min_i = np.min(self.bdy_i[:,0])
        max_i = np.max(self.bdy_i[:,0])
        min_j = np.min(self.bdy_i[:,1])
        max_j = np.max(self.bdy_i[:,1])
         
        # Set up nc pointers and extract data
        
        nc_h = Dataset(fileH,mode='r')
        nc_z = Dataset(fileZ,mode='r')
        if msk_version == 1 :
            e3u = nc_z.variables["e3u"][0,:,min_j:max_j+1,min_i:max_i+1]        
            e3v = nc_z.variables["e3v"][0,:,min_j:max_j+1,min_i:max_i+1]
            gdept = nc_z.variables["gdept"][0,:,min_j:max_j+1,min_i:max_i+1]
        else:
            e3u = nc_z.variables["e3u_0"][0,:,min_j:max_j+1,min_i:max_i+1]
            e3v = nc_z.variables["e3v_0"][0,:,min_j:max_j+1,min_i:max_i+1]
            gdept = nc_z.variables["gdept_0"][0,:,min_j:max_j+1,min_i:max_i+1]
        e2u = nc_h.variables["e2u"][0,min_j:max_j+1,min_i:max_i+1]        
        e1v = nc_h.variables["e1v"][0,min_j:max_j+1,min_i:max_i+1]
        nc_h.close()
        nc_z.close()

        # rejig the indices to reflect the reduced domain
        subx = self.bdy_i[:,0] - min_i
        suby = self.bdy_i[:,1] - min_j 
        
        nx = max_i-min_i+1
        ny = max_j-min_j+1
        nz = e3u.shape[0]
        
        ind  = self._sub2ind(e2u.squeeze().shape,subx,suby)
        e3u    = np.reshape(e3u,(nz,nx*ny))
        e3v    = np.reshape(e3v,(nz,nx*ny))
        gdept  = np.reshape(gdept,(nz,nx*ny))
        e2u    = np.reshape(e2u,(nx*ny))
        e1v    = np.reshape(e1v,(nx*ny))
        indd = np.ravel(self.bdy_r[:])
        i1 = indd == 1
        i2 = indd == 2
        i3 = indd == 3
        i4 = indd == 4
        
        count = 0

        #flux = np.zeros((nz,len(ind),len(fileU)))        
        nc_u = Dataset(fileU[0],mode='r')
        nt   = len(nc_u.variables["vozocrtx"][:,0,0,0])
        nc_u.close()
        
        flux = np.zeros((nz,len(ind),nt*len(fileU)))        
        gdept_bdy = np.zeros((nz,len(ind)))        
        
        for l in range(len(fileU)):
            #print(fileU[l])
            
            nc_u = Dataset(fileU[l],mode='r')
            nc_v = Dataset(fileV[l],mode='r')
            nc_t = Dataset(fileT[l],mode='r')

            nt   = len(nc_u.variables["vozocrtx"][:,0,0,0])    
            #print(nt) 
            for t in range(nt):
           
                #print(t,min_j,max_j+1,min_i,max_i+1)
                u    = nc_u.variables["vozocrtx"][t,:,min_j:max_j+1,min_i:max_i+1]        
                v    = nc_v.variables["vomecrty"][t,:,min_j:max_j+1,min_i:max_i+1] 
                #h    = nc_t.variables["sossheig"][t,min_j:max_j+1,min_i:max_i+1] 
                u    = np.reshape(u,(nz,nx*ny)) 
                v    = np.reshape(v,(nz,nx*ny))
                #h    = np.reshape(h,(nx*ny))
                
                flux[:,i1,count] = -v[:,ind[i1]]*e1v[ind[i1]]*e3v[:,ind[i1]]
                flux[:,i2,count] = -u[:,ind[i2]]*e2u[ind[i2]]*e3u[:,ind[i2]]
                flux[:,i3,count] =  v[:,ind[i3]]*e1v[ind[i3]]*e3v[:,ind[i3]]
                flux[:,i4,count] =  u[:,ind[i4]]*e2u[ind[i4]]*e3u[:,ind[i4]]
                
                gdept_bdy[:,i1] =  gdept[:,ind[i1]]
                gdept_bdy[:,i2] =  gdept[:,ind[i2]]
                gdept_bdy[:,i3] =  gdept[:,ind[i3]]
                gdept_bdy[:,i4] =  gdept[:,ind[i4]]
                
                flux[0,i1,count] -= -v[0,ind[i1]]*e1v[ind[i1]]#*h[ind[i1]]
                flux[0,i2,count] -= -u[0,ind[i2]]*e2u[ind[i2]]#*h[ind[i2]]
                flux[0,i3,count] -=  v[0,ind[i3]]*e1v[ind[i3]]#*h[ind[i3]]
                flux[0,i4,count] -=  u[0,ind[i4]]*e2u[ind[i4]]#*h[ind[i4]]
                
                count += 1
            
            nc_u.close()
            nc_v.close()
            nc_t.close()
                
        self.flux = flux
        self.depth = gdept_bdy
            
    def _nflux(self, fileN, fileT, fileH, fileZ):
        """ return the fluxes from the identified boundary points
        
        Requires:
        fileN - list of input diadT files
        fileT - list of input sea surface height files
        fileH - str input of mesh_hgr
        fileZ - str input of mesh_zgr
        
        """
        # Identify the sub region need to save on IO
        min_i = np.min(self.bdy_i[:,0])-1
        max_i = np.max(self.bdy_i[:,0])+1
        min_j = np.min(self.bdy_i[:,1])-1
        max_j = np.max(self.bdy_i[:,1])+1
         
        # Set up nc pointers and extract data
        
        nc_h = Dataset(fileH,mode='r')
        nc_z = Dataset(fileZ,mode='r')
        e3u = nc_z.variables["e3u_0"][0,:,min_j:max_j+1,min_i:max_i+1]        
        e3v = nc_z.variables["e3v_0"][0,:,min_j:max_j+1,min_i:max_i+1]
        e2u = nc_h.variables["e2u"][0,min_j:max_j+1,min_i:max_i+1]        
        e1v = nc_h.variables["e1v"][0,min_j:max_j+1,min_i:max_i+1]
        nc_h.close()
        nc_z.close()

        # rejig the indices to reflect the reduced domain
        subx = self.bdy_i[:,0] - min_i
        suby = self.bdy_i[:,1] - min_j 
        
        nx = max_i-min_i+1
        ny = max_j-min_j+1
        nz = e3u.shape[0]
        
        ind  = self._sub2ind(e2u.squeeze().shape,subx  ,suby  )
        indu = self._sub2ind(e2u.squeeze().shape,subx,suby  )
        indv = self._sub2ind(e2u.squeeze().shape,subx  ,suby)
        e3u    = np.reshape(e3u,(nz,nx*ny))
        e3v    = np.reshape(e3v,(nz,nx*ny))
        e2u    = np.reshape(e2u,(nx*ny))
        e1v    = np.reshape(e1v,(nx*ny))
        indd = np.ravel(self.bdy_r[:])
        i1 = indd == 1
        i2 = indd == 2
        i3 = indd == 3
        i4 = indd == 4
        
        count = 0

        #flux = np.zeros((nz,len(ind),len(fileU)))        
        nc_N = Dataset(fileN[0],mode='r')
        nt   = len(nc_N.variables["trcdia_uIN"][:,0,0,0])
        nc_N.close()
        
        flux = np.zeros((nz,len(ind),nt*len(fileN)))        
        tran = np.zeros((nz,len(ind),nt*len(fileN)))        
        
        for l in range(len(fileN)):
            
            nc_N = Dataset(fileN[l],mode='r')
            nc_t = Dataset(fileT[l],mode='r')

            nt   = len(nc_u.variables["vozocrtx"][:,0,0,0])    
            
            for t in range(nt):
            
                u    = nc_N.variables["trcdia_uIN"][t,:,min_j:max_j+1,min_i:max_i+1]        
                v    = nc_N.variables["trcdia_vIN"][t,:,min_j:max_j+1,min_i:max_i+1] 
                #h    = nc_t.variables["sossheig"][t,min_j:max_j+1,min_i:max_i+1] 
                u    = np.reshape(u,(nz,nx*ny)) 
                v    = np.reshape(v,(nz,nx*ny))
                #h    = np.reshape(h,(nx*ny))
                e3v[v==0] = np.nan
                e3u[u==0] = np.nan
                u[u==0] = np.nan
                v[v==0] = np.nan # need to do this in order to prevent skewing of fluxes when averaged back onto u-points                
                
                # nutrient fluxes are on t points so average back onto u and v points
                
                flux[:,i1,count] = -(v[:,ind[i1]]+v[:,indv[i1]])*0.5*e1v[ind[i1]]*e3v[:,ind[i1]]
                flux[:,i2,count] = -(u[:,ind[i2]]+u[:,indu[i2]])*0.5*e2u[ind[i2]]*e3u[:,ind[i2]]
                flux[:,i3,count] =  (v[:,ind[i3]]+v[:,indv[i3]])*0.5*e1v[ind[i3]]*e3v[:,ind[i3]]
                flux[:,i4,count] =  (u[:,ind[i4]]+u[:,indu[i4]])*0.5*e2u[ind[i4]]*e3u[:,ind[i4]]
                
                flux[0,i1,count] -= -(v[0,ind[i1]]+v[0,indv[i1]])*0.5*e1v[ind[i1]]#*h[ind[i1]]
                flux[0,i2,count] -= -(u[0,ind[i2]]+u[0,indu[i2]])*0.5*e2u[ind[i2]]#*h[ind[i2]]
                flux[0,i3,count] -=  (v[0,ind[i3]]+v[0,indv[i3]])*0.5*e1v[ind[i3]]#*h[ind[i3]]
                flux[0,i4,count] -=  (u[0,ind[i4]]+u[0,indu[i4]])*0.5*e2u[ind[i4]]#*h[ind[i4]]
                
                # nutrient transpppports
                

                
                #tran[:,i1,count] = -(v[:,ind[i1]]+v[:,indv[i1]])*0.5
                #tran[:,i2,count] = -(u[:,ind[i2]]+u[:,indu[i2]])*0.5
                #tran[:,i3,count] =  (v[:,ind[i3]]+v[:,indv[i3]])*0.5
                #tran[:,i4,count] =  (u[:,ind[i4]]+u[:,indu[i4]])*0.5
                
                tran[:,i1,count] = e1v[ind[i1]]*e3v[:,ind[i1]]
                tran[:,i2,count] = e2u[ind[i2]]*e3u[:,ind[i2]]
                tran[:,i3,count] = e1v[ind[i3]]*e3v[:,ind[i3]]
                tran[:,i4,count] = e2u[ind[i4]]*e3u[:,ind[i4]]
                
                
                count += 1
            
            nc_N.close()
            nc_t.close()

            
        self.nflux = flux
        self.ntran = tran
            
    def _sub2ind(self, shap, subx, suby):
        """subscript to index of a 1d array"""
        ind = (suby * shap[1]) + subx
        return ind
            
    def _unique_rows(self, t):
        """ This returns unique rows in the 2D array. 
        Returns indexes of unique rows in the input 2D array 
        t -- input 2D array
        """        
        tlist = t.tolist()
        sortt = []
        indx = zip(*sorted([(val, i) for i,val in enumerate(tlist)]))[1]
        indx = np.array(indx)
        for i in indx:
            sortt.append(tlist[i])
        del tlist
        for i,x in enumerate(sortt):
            if x == sortt[i-1]:
                indx[i] = -1
        
        return indx[indx != -1]

    def _bdy_sections(self,nbidta,nbjdta):
        """Extract individual bdy sections
    
        Keyword arguments:
        """
        
        # TODO Need to put a check in here to STOP if we have E-W wrap
        # as this is not coded yet
        
        # Define the outer most halo
        outer_rim_i = nbidta[:]
        outer_rim_j = nbjdta[:]
    
        # Set initial constants
    
        nbdy = len(outer_rim_i)
        count = 0
        flag = 0
        mark = 0
        source_tree = sp.cKDTree(zip(outer_rim_i, outer_rim_j)) 
        id_order = np.ones((nbdy,), dtype=np.int)*source_tree.n
        id_order[count] = 0 # use index 0 as the starting point 
        count += 1
        end_pts = {}
        nsec = 0
        
        # Search for individual sections and order
    
        while count <= nbdy:
            
            lcl_pt = zip([outer_rim_i[id_order[count-1]]],
                         [outer_rim_j[id_order[count-1]]])
            
            junk, an_id = source_tree.query(lcl_pt, k=3, distance_upper_bound=1.1)

            if an_id[0,1] in id_order:
                if (an_id[0,2] in id_order) or (an_id[0,2] == source_tree.n) : # we are now at an end point and ready to sequence a section
                    if flag == 0:
                        flag = 1  
                        #end_pts[nsec] = [id_order[count-1], id_order[count-1]] # make a note of the starting point
                        end_pts[nsec] = [count-1, count-1] # make a note of the starting point
                        id_order[mark] = id_order[count-1]
                        id_order[mark+1:] = source_tree.n # remove previous values
                        count = mark + 1
                        end_pts[nsec] = [count-1, count-1]
                    else:
                        i = 0
                        #end_pts[nsec][1] = id_order[count-1] # update the end point of the section
                        end_pts[nsec][1] = count-1 # update the end point of the section
                        nsec += 1
                        
                        while i in id_order:
                            i += 1
                            
                        if count < nbdy:
                            id_order[count] = i
                        flag = 0
                        mark = count
                        count += 1
    
                else: # lets add the next available point to the sequence
                    id_order[count] = an_id[0,2]
                    count += 1
            elif an_id[0,1] == nbdy: # it's an isolated point
                #end_pts[nsec] = [id_order[count-1], id_order[count-1]] # make a note of the starting point
                end_pts[nsec] = [count-1, count-1] # make a note of the starting point
                i = 0
                #end_pts[nsec][1] = id_order[count-1] # update the end point of the section
                end_pts[nsec][1] = count-1 # update the end point of the section
                nsec += 1
                        
                while i in id_order:
                    i += 1
                            
                if count < nbdy:
                    id_order[count] = i
                flag = 0
                mark = count
                count += 1        
                
            else: # lets add the next available point to the sequence
                id_order[count] = an_id[0,1]
                count += 1
            
        return id_order, end_pts
