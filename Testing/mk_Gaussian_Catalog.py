import sys, os, time
from scipy.interpolate import interp1d
from mcfit import P2xi, xi2P
from nbodykit.lab import *
from nbodykit import setup_logging, style, cosmology

from nbodykit.source.catalog import ArrayCatalog
from nbodykit.transform import StackColumns
from nbodykit.mockmaker import gaussian_real_fields, poisson_sample_to_points, gaussian_complex_fields
from nbodykit.cosmology.power.linear import LinearPower
from nbodykit.mpirng import MPIRandomState
from pmesh.pm import RealField, ComplexField, ParticleMesh
import mpsort

class GenerateGRF(object):
    
    def __init__(self, Nmesh, Pk_in=None, los=None, which_rsd=None, do_inv_phase=False,
                 mean_weight=1., cname=None, verbose=False):
        
        '''
            los, which_rsd:
            if los is not None, need to choose which method
            to generate rsd, currently available choices are:
              - "displaced_particles" (displace each particle position)
              - "k-space" (\delta_rsd(k)=bias*(1+f/bias*k_los)\delta_realspace(k))
              - "displaced_weights" (not work very well yet)
            mean_weight:
            default = 1 (has the meaning of ngal)
            if you want density contrast or (D-R), set it to be 0
        '''
        
        self.seed = numpy.random.randint(0,0xfffffff)
        alpha = 1 # leave it as 1
        self.nobj = int(587071/alpha)
        self.vol = 3.9e9/alpha
        self.lbox = int(self.vol**(1./3))
        self.nbar = self.nobj/self.vol
        self.Nmesh = Nmesh
        self.do_inv_phase = do_inv_phase
        self.bias = 2.2
        self.mean_weight = mean_weight
        self.verbose = verbose
        
        if cname is not None:
            self.cname = cname # dir and name to save the catalog
        if Pk_in is not None:
            self.Pk_in = Pk_in
        else:
            self.Pk_in = None
        if los is not None:
            self.los = numpy.array(los)
            self.which_rsd = which_rsd
        else:
            self.los = None

        print("#obj=", self.nobj, "vol=%f [Gpc/h]^3" %(self.vol/1e9),
              "lbox=%f [Mpc/h]"%self.lbox, "nbar=", self.nbar)
        
    def init_cosmo(self):
        h = 0.676
        Omega_nu = 0.00140971
        Omega0_m = 0.31
        Omega0_b = 0.022/h**2
        Omega0_cdm = Omega0_m - Omega0_b - Omega_nu
        n_s = 0.96
        sigma8 = 0.824

        self.cosmo = cosmology.Cosmology(h=h, Omega0_b=Omega0_b,
                                    Omega0_cdm=Omega0_cdm, n_s=n_s)
        self.cosmo.match(sigma8=sigma8)
        self.growth_rate = self.cosmo.scale_independent_growth_rate(0.57)
        
    def init_plin(self):
        h = 0.676
        Omega_nu = 0.00140971
        Omega0_m = 0.31
        Omega0_b = 0.022/h**2
        Omega0_cdm = Omega0_m - Omega0_b - Omega_nu
        n_s = 0.96
        sigma8 = 0.824

        cosmo = cosmology.Cosmology(h=h, Omega0_b=Omega0_b,
                                    Omega0_cdm=Omega0_cdm, n_s=n_s)
        cosmo.match(sigma8=sigma8)
        redshift = 0.57
        self.Plin = LinearPower(cosmo, redshift, transfer='CLASS')
        
    def init_Pk_interp(self):
        self.k_logspace = numpy.logspace(-3, 1.5, 200)
        if self.Pk_in is None:
            try:
                tmp = self.Plin
            except:
                print("please init linear Pk at first ...")
            Pk_gal = self.bias**2*self.Plin(self.k_logspace)
            self.Pk_gal_interp = interp1d(self.k_logspace, Pk_gal,
                                          kind='cubic', bounds_error=None,
                                          fill_value="extrapolate")
        else:
            kk_in = self.Pk_in[:,0]
            Pk_gal = self.Pk_in[:,1]
            self.Pk_gal_interp = interp1d(kk_in, Pk_gal,
                                          kind='cubic', bounds_error=None,
                                          fill_value="extrapolate")
            
        ## P_G_interp is used for generating lognormal mocks
        self.r, self.xi = P2xi(self.k_logspace)(self.Pk_gal_interp(self.k_logspace))
        self.xi_G = numpy.log(self.xi+1)
        self.k, self.P_G = xi2P(self.r)(self.xi_G)
        self.P_G_interp = interp1d(self.k_logspace, self.P_G, kind='cubic',
                                   bounds_error=None,
                                   fill_value="extrapolate")
        
    def mesh_grf(self):
        self.mesh_cat = LinearMesh(self.Pk_gal_interp, Nmesh=self.Nmesh,
                               BoxSize=self.lbox, seed=self.seed)
        

    def pos_pm_grf(self):

        self.pm = ParticleMesh(Nmesh=[self.Nmesh, self.Nmesh, self.Nmesh],
                          BoxSize=self.lbox)
        
        self.deltar, disp = gaussian_real_fields(pm=self.pm,
                            linear_power=self.Plin, compute_displacement=True,
                            unitary_amplitude=False, seed=self.seed)
        self.pos_grf, self.disp_grf = poisson_sample_to_points(delta=self.deltar,
                                      displacement=disp, pm=self.pm,
                                      nbar=self.nbar, bias=self.bias)

    def pos_pm_lognormal(self):

        self.pm = ParticleMesh(Nmesh=[self.Nmesh, self.Nmesh, self.Nmesh],
                          BoxSize=self.lbox)
        
        self.deltar, disp = gaussian_real_fields(pm=self.pm,
                       linear_power=self.P_G_interp, compute_displacement=True,
                       unitary_amplitude=False, seed=self.seed)
        self.pos_grf, self.disp_grf = self.poisson_sample_to_points_aa(delta=self.deltar,
                                      displacement=disp, pm=self.pm,
                                      nbar=self.nbar)

        
    def mk_catalog(self):
        pos = StackColumns(self.pos_grf[:,0], self.pos_grf[:,1], self.pos_grf[:,2])
        data = numpy.empty(len(self.pos_grf[:,0]), dtype=[('Position', ('f8', 3))])
        self.catalog = ArrayCatalog(data)
        self.catalog['Position'] = pos
        self.mesh_cat = self.catalog.to_mesh(resampler='tsc', Nmesh=self.Nmesh,
                        compensated=True, BoxSize=self.lbox, position='Position',
                        interlaced=True, weight='Weight')

    def mk_catalog_grf_weighted(self):
        
        self.pm = ParticleMesh(Nmesh=[self.Nmesh, self.Nmesh, self.Nmesh],
                          BoxSize=self.lbox)
        self.deltar, self.disp = gaussian_real_fields(pm=self.pm,
                            linear_power=self.Pk_gal_interp,
                            compute_displacement=True, seed=self.seed,
                            inverted_phase=False, unitary_amplitude=False)
        rand_cat = RandomCatalog(self.nobj) #, seed=42
        rand_cat['Position'] = rand_cat.rng.uniform(itemshape=(3,))*self.lbox
        self.field = self.deltar.copy() + 1.
        if self.los is None:
            self.field_3col = self.field.readout(rand_cat['Position'].compute(),
                                                      resampler='nnb')
            rand_cat['Weight'] =self.field_3col/(numpy.sum(self.field_3col)/self.nobj)-(1-self.mean_weight)
        
        if self.los is not None:
            self.idx_los = numpy.where(self.los==1)[0][0]
            if self.which_rsd == "displaced_particles":
                print(">> los=", self.los, "move particle position for displacement field")
                self.field_3col = self.field.readout(rand_cat['Position'].compute(),
                                                          resampler='nnb')
                rand_cat['Weight'] =self.field_3col/(numpy.sum(self.field_3col)/self.nobj)-(1-self.mean_weight)
                self.disp_los_1d = self.disp[self.idx_los].readout(rand_cat['Position'].compute(),
                                                           resampler='nnb') * self.growth_rate / self.bias
                pos = StackColumns(rand_cat['Position'][:,0].compute() + self.disp_los_1d * self.los[0],
                                   rand_cat['Position'][:,1].compute() + self.disp_los_1d * self.los[1],
                                   rand_cat['Position'][:,2].compute() + self.disp_los_1d * self.los[2])
                rand_cat['Position'] = pos
            elif self.which_rsd == "k-space":
                self.deltak, self.dispk = gaussian_complex_fields(pm=self.pm,
                                    linear_power=self.Pk_gal_interp,
                                    compute_displacement=True, seed=self.seed,
                                    inverted_phase=False, unitary_amplitude=False)
                kgrid = [kk.astype('f8') for kk in self.deltak.slabs.optx]
                knorm = numpy.sqrt(sum(kk**2 for kk in kgrid))
                knorm[knorm==0.] = numpy.inf
                kgrid = [k/knorm for k in kgrid]
                k_los = kgrid[self.idx_los]
                self.field_k = self.deltak.copy()
                self.field_k.value *= (1 + self.growth_rate/self.bias*k_los**2)
                self.field_rsd = self.field_k.c2r()
                self.field_rsd.value += 1.
                self.field_3col = self.field_rsd.readout(rand_cat['Position'].compute(),
                                                         resampler='nnb')
                rand_cat['Weight'] = self.field_3col/(numpy.sum(self.field_3col)/self.nobj)-(1-self.mean_weight)
            elif self.which_rsd == "displaced_weights":
                print(">> los=", self.los, "perform dual mesh")
                self.Nmesh2 = 1*self.Nmesh
                self = grf2_rsd2
                rand_cat['Weight_One'] = 1
                nbar_in_cell = self.nobj/self.Nmesh**3
                rand_cat_mesh = rand_cat.to_mesh(resampler='tsc', Nmesh=self.Nmesh2,
                                compensated=True, BoxSize=self.lbox, position='Position',
                                interlaced=True, weight='Weight_One')
                painted_mesh = rand_cat_mesh.paint(mode='real', Nmesh=self.Nmesh2)
                self.ngal_in_cell = (painted_mesh.value-1)*nbar_in_cell
                self.disp_mesh_los = -self.disp[self.idx_los]*self.growth_rate/self.bias
                self.ngal_in_cell_rsd = self.ngal_in_cell.copy()
                self.nw_in_cell = self.ngal_in_cell*self.deltar.value
                self.nw_in_cell_rsd = self.ngal_in_cell*self.deltar.value
                if self.idx_los == 0:
                    for ii in range(self.Nmesh2):
                        if ii%100==0: print(ii)
                        for jj in range(self.Nmesh2):
                            cell_idx = numpy.arange(self.Nmesh2)
                            nonzero_disp_idx, = numpy.where(abs(self.disp_mesh_los[:,ii,jj]) > 0)
                            cell_disp_idx = (cell_idx[nonzero_disp_idx] +
                                             self.disp_mesh_los[nonzero_disp_idx,ii,jj]).astype(int)
                            cell_disp_idx[cell_disp_idx >= self.Nmesh2] -= self.Nmesh2
                            cell_disp_idx[cell_disp_idx < 0]    += self.Nmesh2
                            self.ngal_in_cell_rsd[cell_disp_idx,ii,jj] += self.ngal_in_cell[cell_idx[nonzero_disp_idx],ii,jj]
                            self.ngal_in_cell_rsd[cell_idx[nonzero_disp_idx],ii,jj] -= self.ngal_in_cell[cell_idx[nonzero_disp_idx],ii,jj]
                            self.nw_in_cell_rsd[cell_disp_idx,ii,jj] += self.nw_in_cell[cell_idx[nonzero_disp_idx],ii,jj]
                            self.nw_in_cell_rsd[cell_idx[nonzero_disp_idx],ii,jj] -= self.nw_in_cell[cell_idx[nonzero_disp_idx],ii,jj]
                elif self.idx_los == 1:
                    for ii in range(self.Nmesh2):
                        if ii%100==0: print(ii)
                        for jj in range(self.Nmesh2):
                            cell_idx = numpy.arange(self.Nmesh2)
                            nonzero_disp_idx, = numpy.where(abs(self.disp_mesh_los[ii,:,jj]) > 0)
                            cell_disp_idx = (cell_idx[nonzero_disp_idx] +
                                             self.disp_mesh_los[ii,nonzero_disp_idx,jj]).astype(int)
                            cell_disp_idx[cell_disp_idx >= self.Nmesh2] -= self.Nmesh2
                            cell_disp_idx[cell_disp_idx < 0]    += self.Nmesh2
                            self.ngal_in_cell_rsd[ii,cell_disp_idx,jj] += self.ngal_in_cell[ii,cell_idx[nonzero_disp_idx],jj]
                            self.ngal_in_cell_rsd[ii,cell_idx[nonzero_disp_idx],jj] -= self.ngal_in_cell[ii,cell_idx[nonzero_disp_idx],jj]
                            self.nw_in_cell_rsd[ii,cell_disp_idx,jj] += self.nw_in_cell[ii,cell_idx[nonzero_disp_idx],jj]
                            self.nw_in_cell_rsd[ii,cell_idx[nonzero_disp_idx],jj] -= self.nw_in_cell[ii,cell_idx[nonzero_disp_idx],jj]
                elif self.idx_los == 2:
                    for ii in range(self.Nmesh2):
                        if ii%100==0: print(ii)
                        for jj in range(self.Nmesh2):
                            cell_idx = numpy.arange(self.Nmesh2)
                            nonzero_disp_idx, = numpy.where(abs(self.disp_mesh_los[ii,jj,:]) > 0)
                            cell_disp_idx = (cell_idx[nonzero_disp_idx] +
                                             self.disp_mesh_los[ii,jj,nonzero_disp_idx]).astype(int)
                            cell_disp_idx[cell_disp_idx >= self.Nmesh2] -= self.Nmesh2
                            cell_disp_idx[cell_disp_idx < 0]    += self.Nmesh2
                            self.ngal_in_cell_rsd[ii,jj,cell_disp_idx] += self.ngal_in_cell[ii,jj,cell_idx[nonzero_disp_idx]]
                            self.ngal_in_cell_rsd[ii,jj,cell_idx[nonzero_disp_idx]] -= self.ngal_in_cell[ii,jj,cell_idx[nonzero_disp_idx]]
                            self.nw_in_cell_rsd[ii,jj,cell_disp_idx] += self.nw_in_cell[ii,jj,cell_idx[nonzero_disp_idx]]
                            self.nw_in_cell_rsd[ii,jj,cell_idx[nonzero_disp_idx]] -= self.nw_in_cell[ii,jj,cell_idx[nonzero_disp_idx]]
                self.field_rsd = self.deltar.copy()
                self.field_rsd.value = self.nw_in_cell_rsd / self.ngal_in_cell_rsd + 1.
                self.field_rsd.value = numpy.nan_to_num(self.field_rsd.value, nan=1)
                self.field_rsd_3col = self.field_rsd.readout(rand_cat['Position'].compute(),
                                                             resampler='nnb')
                rand_cat['Weight'] =self.field_rsd_3col/(numpy.sum(self.field_rsd_3col)/self.nobj)-(1-self.mean_weight)
        
        if self.do_inv_phase:
            self.deltar_pinv, disp_pinv = gaussian_real_fields(pm=self.pm,
                                linear_power=self.Pk_gal_interp,
                                compute_displacement=True,
                                inverted_phase=True, seed=self.seed,
                                unitary_amplitude=False)
            self.field_pinv = (self.deltar_pinv.copy() + self.mean_weight)
            # use the same random catalog -- all particle positions
            # are the same but weighted differently.
            self.field_3col_pinv = self.field_pinv.readout(
                                        rand_cat['Position'].compute(),
                                        resampler='nnb')
            rand_cat['Weight_pinv'] =self.field_3col_pinv/(numpy.sum(self.field_3col_pinv)/self.nobj)
        self.catalog = rand_cat.copy()
        self.mesh_cat = self.catalog.to_mesh(resampler='tsc', Nmesh=self.Nmesh,
                        compensated=True, BoxSize=self.lbox, position='Position',
                        interlaced=True, weight='Weight')
        if self.do_inv_phase:
            self.mesh_cat_pinv = self.catalog.to_mesh(resampler='tsc', Nmesh=self.Nmesh,
                            compensated=True, BoxSize=self.lbox, position='Position',
                            interlaced=True, weight='Weight_pinv')

    def calc_pk(self, poles=None):
        kmin = numpy.pi/self.lbox*2
        if self.los is not None:
            if poles is None:
                print("in 2d mode, please specify poles")
                exit()
            self.Pk = FFTPower(self.mesh_cat, mode='2d', dk=0.005, kmin=kmin,
                               los=self.los, poles=poles)
        if poles is not None:
            self.Pk = FFTPower(self.mesh_cat, mode='2d', dk=0.005, kmin=kmin,
                               los=[0,0,1], poles=poles)
        else:
            self.Pk = FFTPower(self.mesh_cat, mode='1d', dk=0.005, kmin=kmin)
        if self.do_inv_phase:
            if self.los is not None:
                self.Pk_pinv = FFTPower(self.mesh_cat_pinv, mode='2d', dk=0.005, kmin=kmin,
                                        los=self.los, poles=poles)
            else:
                self.Pk_pinv = FFTPower(self.mesh_cat_pinv, mode='1d', dk=0.005, kmin=kmin)
        
    def save_catalog_ascii(self):
        nrows, ncols = self.catalog.csize, int(len(self.catalog.columns)-2+2)
        cat_ascii = numpy.zeros([nrows, ncols])
        cat_ascii[:,:3] = self.catalog['Position'][:,:3].compute()
        cat_ascii[:,3]  = self.catalog['Weight'].compute()
        if self.do_inv_phase:
            cat_ascii[:,4]  = self.catalog['Weight_pinv'].compute()
        
        numpy.savetxt(self.cname, cat_ascii)
        
    def run_mk_catalog_grf_weighted(self, do_save_catalog):
        
        if self.Pk_in is None:
            self.init_plin()
        if self.los is not None:
            self.init_cosmo()
        self.init_Pk_interp()
        self.mk_catalog_grf_weighted()
        if do_save_catalog:
            if self.cname is None:
                print("Please specify where to save catalog ...")
            self.save_catalog_ascii()
        
    def run_mk_catalog_lognormal(self, do_save_catalog):
        
        if self.Pk_in is None:
            self.init_plin()
        self.init_Pk_interp()
        self.pos_pm_lognormal()
        self.mk_catalog()
        if do_save_catalog:
            if self.cname is None:
                print("Please specify where to save catalog ...")
            self.save_catalog_ascii()
            
    def lognormal_transform_aa(self, density):

        toret = density.copy()
        sigma_G = numpy.mean((density.value.flatten())**2)
        toret[:] = numpy.exp(-sigma_G + density.value) #- 1
#         toret[:] /= toret.cmean(dtype='f8')

        return toret

    def assign_field_brute(self):
        '''
            to double check the "readout" function
        '''
        pos_round = numpy.round(self.catalog['Position'].compute())
        self.field_brute = numpy.zeros(len(pos_round))
        print(">> assign mesh field value to particle")
        for ii in range(len(pos_round)):
            if ii % 60000 == 0: print(ii)
            ipos = numpy.round(pos_round[ii]/(self.lbox/self.Nmesh))
            if (ipos.min()>=0) & (ipos.max() < self.Nmesh):
                self.field_brute[ii] = self.field.value[int(ipos[0]),
                                                        int(ipos[1]),
                                                        int(ipos[2])]
            else:
                ipos[numpy.where(ipos>self.Nmesh)] -= 1
                ipos[numpy.where(ipos<0)]   += 1


    def poisson_sample_to_points_aa(self, delta, displacement, pm, nbar, seed=None, logger=None):
        
        '''
            Code adapted from nbodykit to poisson sample points
            of a continuous field.
        
        '''

        comm = delta.pm.comm

        seed1, seed2 = numpy.random.RandomState(self.seed).randint(0, 0xfffffff, size=2)

        delta = self.lognormal_transform_aa(delta)
        self.delta_lognorm = delta.copy()

        if logger and pm.comm.rank == 0:
            logger.info("Lognormal transformation done")

        # mean number of objects per cell
        H = delta.BoxSize / delta.Nmesh
        overallmean = H.prod() * nbar

        # number of objects in each cell (per rank, as a RealField)
        cellmean = delta * overallmean

        # create a random state with the input seed
        rng = MPIRandomState(seed=seed1, comm=comm, size=delta.size)

        # generate poissons. Note that we use ravel/unravel to
        # maintain MPI invariane.
        Nravel = rng.poisson(lam=cellmean.ravel())
        N = delta.pm.create(type='real')
        N.unravel(Nravel)

        Ntot = N.csum()
        if logger and pm.comm.rank == 0:
            logger.info("Poisson sampling done, total number of objects is %d" % Ntot)

        pos_mesh = delta.pm.generate_uniform_particle_grid(shift=0.0)
        disp_mesh = numpy.empty_like(pos_mesh)

        # no need to do decompose because pos_mesh is strictly within the
        # local volume of the RealField.
        N_per_cell = N.readout(pos_mesh, resampler='nnb')
        for i in range(N.ndim):
            disp_mesh[:, i] = displacement[i].readout(pos_mesh, resampler='nnb')

        # fight round off errors, if any
        N_per_cell = numpy.int64(N_per_cell + 0.5)

        pos = pos_mesh.repeat(N_per_cell, axis=0)
        disp = disp_mesh.repeat(N_per_cell, axis=0)

        del pos_mesh
        del disp_mesh

        if logger and pm.comm.rank == 0:
            logger.info("catalog produced. Assigning in cell shift.")

        orderby = numpy.int64(pos[:, 0] / H[0] + 0.5)
        for i in range(1, delta.ndim):
            orderby[...] *= delta.Nmesh[i]
            orderby[...] += numpy.int64(pos[:, i] / H[i] + 0.5)

        # sort by ID to maintain MPI invariance.
        pos = mpsort.sort(pos, orderby=orderby, comm=comm)
        disp = mpsort.sort(disp, orderby=orderby, comm=comm)

        if logger and pm.comm.rank == 0:
            logger.info("sorting done")

        rng_shift = MPIRandomState(seed=seed2, comm=comm, size=len(pos))
        in_cell_shift = rng_shift.uniform(0, H[i], itemshape=(delta.ndim,))

        pos[...] += in_cell_shift
        pos[...] %= delta.BoxSize

        if logger and pm.comm.rank == 0:
            logger.info("catalog shifted.")

        return pos, disp    




