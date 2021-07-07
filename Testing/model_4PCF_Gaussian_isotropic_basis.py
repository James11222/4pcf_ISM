from nbodykit.lab import *
from nbodykit.cosmology.correlation import pk_to_xi, xi_to_pk
import mcfit
from mcfit import P2xi, xi2P

from scipy.integrate import quad, cumtrapz
from scipy.special import spherical_jn
from scipy.interpolate import interp1d
from sympy.physics.wigner import wigner_3j

class model_4PCF(object):
    
    def __init__(self, meta_data, do_rsd=False, r_in=None, xi_in=None, verbose=False):

        '''
          lls is a list specifies the angular momentum
          e.g.
          
          ['000',
           '011',
           '022',
           '033']
        '''
        
        self.meta_data = meta_data
        self.lls  = meta_data.ell_gaussian
        self.verbose = verbose
        self.do_rsd = do_rsd
        self.k_in = meta_data.k_in
        if r_in is None:
            self.r_in = meta_data.rbins_1d
        else:
            self.r_in = r_in
            self.redshift = meta_data.redshift
        
        if hasattr(self.meta_data, 'bias'):
            self.bias = self.meta_data.bias
        else:
            self.bias = 1
        
        if not self.do_rsd:
            if hasattr(self.meta_data, 'pk_in'):
                self.Pr = self.meta_data.pk_in
            else:
                self.Pr = None
            if hasattr(self.meta_data, 'twopcf_mean'):
                self.xir = self.meta_data.twopcf_mean
            else:
                self.xir = None
                
        elif self.do_rsd:
            if hasattr(self.meta_data, 'pk_in'):
                print("load existing Pr")
                self.Pr = meta_data.pk_in
            else:
                self.Pr = None
            self.pk_ell = {}
            self.xi_ell = {}
            for ii in [0, 2, 4]:
                if hasattr(self.meta_data, 'pk_zs_mean'):
                    try:
                       # print("load existing Pk%s"%ii)
                        self.pk_ell[ii] = self.meta_data.pk_zs_mean[ii]
                    except:
                        pass
                if hasattr(self.meta_data, 'xi_zs_mean'):
                    self.xi_ell[ii] = self.meta_data.xi_zs_mean[ii]         
            
    def run(self):
        
        self.init_arrs()
        self.init_2stat()
        self.get_jnbar()
        self.get_zeta_model()
        
    def init_arrs(self):
        
        kbin_min= 1e-3
        kbin_max= 5
        nbink = 1000
        if self.r_in is None:
            sbin_min= 8.5
            sbin_max = 170
            dsbin = 17
            self.rr= numpy.arange(sbin_min, sbin_max, dsbin)
        else:
            self.rr = self.r_in.copy()
        self.nbins = len(self.rr)
        self.kk = numpy.linspace(kbin_min, kbin_max, nbink)
        self.kk_log = numpy.logspace(numpy.log10(kbin_min), numpy.log10(kbin_max), nbink)
        self.dkk = self.kk[1]-self.kk[0]
        self.rrv, self.kkv = numpy.meshgrid(self.rr, self.kk)
        
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
        
    def init_2stat(self):
        
        if not self.do_rsd:
            if self.Pr is not None:
                print("load existing Pk")
                pk_interp = interp1d(self.k_in, self.Pr, kind='cubic', bounds_error=False, fill_value=0)
                self.Pk = pk_interp(self.kk)
                if self.xir is not None:
                    print("load existing xi")
                    self.r = self.r_in.copy()
                    self.xi_interp = interp1d(self.r_in, self.xir, kind='cubic', fill_value='extrapolate')
                else:
                    print("FT xi from existing Pk")
                    Pk_log = pk_interp(self.kk_log)
                    self.r, self.xi = P2xi(self.kk_log)(Pk_log)
                    self.xi_interp = interp1d(self.r, self.xi, kind='cubic', fill_value='extrapolate')

            elif self.Pkr is None:
                print("calculate linear Pk")
                print("FT xi from linear theory Pk")
                self.init_cosmo()
                Plin = cosmology.LinearPower(self.cosmo, redshift=self.redshift, transfer='CLASS')
                self.Pk = Plin(self.kk)
                Pk_log = Plin(self.kk_log)
                self.r, self.xi = P2xi(self.kk_log)(Pk_log)
                self.xi_interp = interp1d(self.r, self.xi, kind='cubic', fill_value='extrapolate')   
                
        elif self.do_rsd:
            self.xi_ell_interp = {}
            if (not bool(self.pk_ell)) or (not bool(self.xi_ell)) or (len(self.pk_ell)<3):
                if not hasattr(self, 'cosmo'):
                    print("init cosmo")
                    self.init_cosmo()
                growth_rate = self.cosmo.scale_independent_growth_rate(0.57)
                beta = growth_rate/self.bias
                kaiser_fac = {}
                kaiser_fac[0] = (1 + 2*beta/3 + beta**2/5) * self.bias**2
                kaiser_fac[2] = (4*beta/3 + 4*beta**2/7) * self.bias**2
                kaiser_fac[4] = (8*beta**2/35) * self.bias**2
                
            if not bool(self.pk_ell):
                self.Pk_ell = {}
                self.Pk_ell[1], self.Pk_ell[3], self.Pk_ell[5] = 0, 0, 0
                if self.Pr is None:
                    Plin = cosmology.LinearPower(self.cosmo, redshift=self.redshift, transfer='CLASS')
                    self.Pk = Plin(self.kk)
                    print("linear Pk for ell=[0,2,4]")
                else:
                    pk_interp = interp1d(self.k_in, self.Pr, kind='cubic', bounds_error=False, fill_value=0)
                    self.Pk = pk_interp(self.kk)
                    print("lieanr Kaiser + input Pr for ell=[0,2,4]")
                for ii in [0, 2, 4]:
                    if ii not in self.Pk_ell.keys():
                        self.Pk_ell[ii] = self.Pk * kaiser_fac[ii]
            elif len(self.pk_ell)==3:
                self.Pk_ell = {}
                self.Pk_ell[1], self.Pk_ell[3], self.Pk_ell[5] = 0, 0, 0
                for ii in [0, 2, 4]:
                    pk_interp = interp1d(self.k_in, self.pk_ell[ii], kind='cubic', bounds_error=False, fill_value=0)
                    self.Pk_ell[ii] = pk_interp(self.kk)       
                    print("load existing pk_%s"%ii)
            elif len(self.pk_ell)<3:
                self.Pk_ell = {}
                self.Pk_ell[1], self.Pk_ell[3], self.Pk_ell[5] = 0, 0, 0
                for ii in [0, 2, 4]:
                    try:
                        pk_interp = interp1d(self.k_in, self.pk_ell[ii], kind='cubic', bounds_error=False, fill_value=0)
                        self.Pk_ell[ii] = pk_interp(self.kk)       
                        print("load existing pk_%s"%ii)
                    except:
                        if self.Pr is None:
                            Plin = cosmology.LinearPower(self.cosmo, redshift=self.redshift, transfer='CLASS')
                            self.Pk = Plin(self.kk)
                            print("linear Pk%s"%ii)
                        else:
                            pk_interp = interp1d(self.k_in, self.Pr, kind='cubic', bounds_error=False, fill_value=0)
                            self.Pk = pk_interp(self.kk) 
                            print("lieanr Kaiser + input Pr for ell=%s"%ii)
                        self.Pk_ell[ii] = self.Pk * kaiser_fac[ii] 
                    
            if not bool(self.xi_ell):   
                if self.Pr is not None:
                    pk_interp = interp1d(self.k_in, self.Pr, kind='cubic', bounds_error=False, fill_value=0)
                    Pk_log = pk_interp(self.kk_log)
                    self.r, self.xi = P2xi(self.kk_log)(Pk_log)
                    self.xi_interp = interp1d(self.r, self.xi, kind='cubic', fill_value='extrapolate')
                    self.get_xibar()
                    for ii in [0, 2, 4]:
                        print("linear xi%s"%ii)
                        xi = self.xi_interp(self.r) 
                        if ii == 2:
                            xi -= self.xi_bar
                        elif ii == 4:
                            xi -= self.xi_bar_bar
                        self.xi_ell_interp[ii] = interp1d(self.r, xi * kaiser_fac[ii], kind='cubic', fill_value='extrapolate')
                if hasattr(self.meta_data, 'pk_zs_mean'):
                    print("FT pk_zs_mean to xi_ell")
                    for ii in [0, 2, 4]:
                        pk_interp = interp1d(self.k_in, self.pk_ell[ii], kind='cubic', bounds_error=False, fill_value=0)
                        Pk_log = pk_interp(self.kk_log)
                        self.r, self.xi = P2xi(self.kk_log)(Pk_log)
                        self.xi_ell_interp[ii] = interp1d(self.r, self.xi, kind='cubic', fill_value='extrapolate')
                     
            else:
                self.r = self.r_in.copy()
                for ii in [0, 2, 4]:
                    print("load existing xi%s"%ii)
                    self.xi_ell_interp[ii] = interp1d(self.r, self.xi_ell[ii], kind='cubic', fill_value='extrapolate')
            for ii in [1, 3, 5]:
                self.xi_ell_interp[ii] = interp1d(self.r, numpy.zeros_like(self.r), kind='cubic', fill_value='extrapolate')

                
    def get_xibar(self):
        ss = numpy.linspace(1e-2, 200, 1e3)
        ds = numpy.average(ss[1:]-ss[:-1])
        self.xi_bar = numpy.zeros(len(self.r))
        self.xi_bar_bar = numpy.zeros(len(self.r))
        for ii in range(len(self.r)):
            si = ss[ss < self.r[ii]]
            self.xi_bar[ii] = numpy.sum(self.xi_interp(si)*ds*si**2)/self.r[ii]**3*3
            self.xi_bar_bar[ii] = numpy.sum(self.xi_interp(si)*ds*si**4)/self.r[ii]**5*5
            
    def get_jnbar(self):

        self.jn_bar = {}
        nkbins = len(self.kk)
        half_width = (self.rr[1] - self.rr[0])*0.49

        for l in range(0,6):
            self.jn_bar[l] = numpy.zeros([len(self.rr), nkbins])
                        
        for ii, ir in enumerate(self.rr):
            u = numpy.linspace(ir-half_width, ir+half_width, 100)
            du = u[1] - u[0]
            uv, kv = numpy.meshgrid(u, self.kk, indexing='ij')
            norm = numpy.sum(uv*uv, axis=0)*du
            for l in range(0,6):
                ans = numpy.sum(uv*uv*spherical_jn(l, uv*kv), axis=0)*du
                ans /= norm
                self.jn_bar[l][ii,:] = ans
                
    def get_flll(self, ells, a, b, c, verbose=False):
        
        ells = numpy.array(ells)
        if not self.do_rsd:
            if len(set(ells)) > 1:
                ells_unit = numpy.ones_like(ells)
                ells_unit[ells==0] = 0
                ans = (self.Pk * 
                      (self.jn_bar[ells[0]][a])**ells_unit[0] *
                      (self.jn_bar[ells[1]][b])**ells_unit[1] *
                      (self.jn_bar[ells[2]][c])**ells_unit[2] *
                       self.kk**2)                  
            else:
                ans = (self.Pk * 
                      (self.jn_bar[ells[0]][a])**ells[0] *
                      (self.jn_bar[ells[1]][b]) *
                      (self.jn_bar[ells[2]][c]) *
                       self.kk**2)
        elif self.do_rsd:
            ans = (self.Pk_ell[ells[0]] * 
                  (self.jn_bar[ells[1]][a]) *
                  (self.jn_bar[ells[2]][b]) *
                   self.kk**2)                  
            
        if verbose:
            try:
                print("ells_unit", ells_unit)
            except:
                pass

        return ans

    def get_zeta_model(self):
    
        self.zetas_dict = {}
        self.zetas_dict_1d = {}
    
        for il in self.lls:
            self.zetas_dict[il] = numpy.zeros([self.nbins, self.nbins, self.nbins])

        for il in self.lls:
            ells = numpy.array([int(i) for i in il])
            for ii, ir1 in enumerate(self.rr):
                for jj, ir2 in enumerate(self.rr):
                    for mm, ir3 in enumerate(self.rr):
                        if not self.do_rsd:
                            y_int = numpy.sum(self.get_flll(ells, ii, jj, mm))*self.dkk/2./numpy.pi**2
                            rs = numpy.array([ir1, ir2, ir3])
                            if len(set(ells)) > 1:
                                self.zetas_dict[il][ii, jj, mm] = self.xi_interp(rs[ells==0])*y_int
                            else:
                                self.zetas_dict[il][ii, jj, mm] = 0
                                y_int = numpy.sum(self.get_flll(ells, ii, jj, mm))*self.dkk/2./numpy.pi**2
                                self.zetas_dict[il][ii, jj, mm] += self.xi_interp(ir1)*y_int 
                                y_int = numpy.sum(self.get_flll(ells, jj, mm, ii))*self.dkk/2./numpy.pi**2
                                self.zetas_dict[il][ii, jj, mm] += self.xi_interp(ir2)*y_int 
                                y_int = numpy.sum(self.get_flll(ells, mm, ii, jj))*self.dkk/2./numpy.pi**2
                                self.zetas_dict[il][ii, jj, mm] += self.xi_interp(ir3)*y_int 
                        elif self.do_rsd:
                            ells_perm = [ells[0], ells[1], ells[2]]
                            y_int1 = numpy.sum(self.get_flll(ells_perm, jj, mm, 0))*self.dkk/2./numpy.pi**2
                            y_int1 *= (self.xi_ell_interp[ells_perm[0]](ir1) *
                                       ((1j)**(ells_perm[1]-ells_perm[2])).real/(2*ells_perm[0]+1)**2.)
                            ells_perm = [ells[1], ells[0], ells[2]]
                            y_int2 = numpy.sum(self.get_flll(ells_perm, ii, mm, 0))*self.dkk/2./numpy.pi**2
                            y_int2 *= (self.xi_ell_interp[ells_perm[0]](ir2) *
                                       ((1j)**(ells_perm[1]-ells_perm[2])).real/(2*ells_perm[0]+1)**2.)
                            ells_perm = [ells[2], ells[0], ells[1]]
                            y_int3 = numpy.sum(self.get_flll(ells_perm, ii, jj, 0))*self.dkk/2./numpy.pi**2
                            y_int3 *= (self.xi_ell_interp[ells_perm[0]](ir3) *
                                       ((1j)**(ells_perm[1]-ells_perm[2])).real/(2*ells_perm[0]+1)**2.)
                            self.zetas_dict[il][ii, jj, mm] = (y_int1 + y_int2 + y_int3) # il is ell combo, ii, jj, mm is bin index

            if not self.do_rsd:
                phase = (4*numpy.pi)**1.5*(-1)**max(ells)*numpy.sqrt(2*max(ells)+1)
            elif self.do_rsd:
                threej = numpy.float64(wigner_3j(ells[0],ells[1],ells[2],0,0,0))
                phase = (4*numpy.pi)**1.5*numpy.sqrt((2*ells[0]+1)*(2*ells[1]+1)*(2*ells[2]+1))*threej
            #print(ells, max(ells))
            self.zetas_dict[il] *= phase

            self.zetas_1d = []
            bin_idx = []
            self.rbin_3d = []
            for ii in range(self.nbins):
                for jj in range(ii+1, self.nbins):
                    for mm in range(jj+1, self.nbins):
                        bin_idx.append([ii, jj, mm])
                        self.rbin_3d.append([self.rr[ii], self.rr[jj], self.rr[mm]])
                        self.zetas_1d.append(self.zetas_dict[il][ii, jj, mm])

            self.zetas_dict_1d[il] = numpy.array(self.zetas_1d)