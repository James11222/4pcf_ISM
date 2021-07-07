import numpy as np
from subprocess import call
import astropy.io.fits as pyf
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as fitter
import time

# plt.style.use("dark_background")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 22

class measure_4pcf(object):
    
    def __init__(self, density_field_data = None, save_dir=None, save_name=None, ell_max=5,
                 nbins=4, bin_spacing='LIN', ld_one_d=64):
        """
        This class allows us to measure the 4pcf from some input data field
        """
        self.ell_max = ell_max
        self.eps = 1e-15
        self.bin_min = 1-1e-5
        self.bin_max = 32+1e-5
        self.nbins = nbins
        if bin_spacing == 'LIN' or bin_spacing == 'INV' or bin_spacing == 'LOG':
            #We can toggle what binning we want to use using the bin_spacing argument
            switch = {
            'LIN' : np.linspace(self.bin_min, self.bin_max, self.nbins+1),
            'INV' : 1./np.linspace(1./self.bin_min, 1./self.bin_max, self.nbins+1),
            'LOG' : np.exp(np.linspace(np.log(self.bin_min), np.log(self.bin_max),                      self.nbins+1))}
        else:
            raise ValueError("""Please put a valid bin_spacing argument, acceptable options are: \n LIN \n INV \n LOG \n in string format.""")
        
        self.bin_edges = switch[bin_spacing]
        self.ld_one_d = ld_one_d
        
        if density_field_data is not None:
            self.density_field_data = density_field_data
        else:
            raise ValueError("Please include a density_field_data argument. Should be a density cube in the form of a numpy array")
        
        if save_name is not None:
            self.save_name = save_name
        else:
            raise ValueError("Please include a save_name argument")
            
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            raise ValueError("Please include a save_dir argument")
            
        ################################################################
        #                          UTILITIES
        ################################################################
            
    def create_XYZR(self):
        """
        This function adds the attributes 

        self.X
        self.Y
        self.Z
        self.R 

        to the object. This is effectively a helper function for
        self.create_radial_bins() and self.calc_and_save_YLMs()
        """

        x = np.linspace(-self.ld_one_d/2, self.ld_one_d/2-1 , self.ld_one_d)
        xsq = x*x
        m_center = np.where(x==0)[0][0]
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        X = -X
        Y = -Y
        Z = -Z
        Xsq, Ysq, Zsq = np.meshgrid(xsq, xsq, xsq,indexing='ij')
        #PRECOMPUTE POWERS OF ARRAYS AND COMBINATIONS (e.g. x - iy).
        Rsq = Xsq+Ysq+Zsq
        R = np.sqrt(Rsq)
        del Rsq
        zero_ind = np.where(R==0)
        R[zero_ind] = self.eps
        X[zero_ind] = self.eps
        Y[zero_ind] = self.eps
        Z[zero_ind] = self.eps
        self.X = X
        self.Y = Y
        self.Z = Z
        self.R = R

    def ylm_save(self, ylm, ell, m):
        """
        Helper function to save ylm
        """
        np.save(self.save_dir +'ylm_'+self.save_name+'_'+str(ell)+'_'+str(m)+'.npy',ylm)
        del ylm

    def ylm_transform_save(self, ylm_on_shell, ell, m, i):
        """
        Helper function to save ylm_FT
        """
        FT = np.fft.fftn(np.fft.fftshift(ylm_on_shell))
        np.save(self.save_dir + 'YLMtilde_'+self.save_name+'_'+str(ell)+'_'+str(m)+'_bin_'+str(i)+'.npy',FT)
        del FT

    def calc_and_save_YLMs(self):

        """
        COMPUTE YLMS SEQUENTIALLY AND SAVE.

        xmiydivr = e^(-iφ)sin(θ) = (x - iy)/r
        zdivr = cos(θ) = z/r

        xmidivrsq = e^(-2iφ)sin^2(θ) = [(x - iy)/r]^2
        zdivrsq = cos^2(θ) = [z/r]^2

        ..cu means cubed

        ..ft means to the fourth power

        ..fi means to the fifth power

        """
        if hasattr(self, 'X'):
            X = self.X
            Y = self.Y
            Z = self.Z
            R = self.R
        else:
            raise AssertionError("You need to run self.create_XYZR() first")

        #ell, m = 0,0
        y00 =.5*(1./np.pi)**.5*np.ones((self.ld_one_d,self.ld_one_d,self.ld_one_d))
        self.ylm_save(y00, 0, 0)
        del y00

        #ell, m = 1, -1
        xdivr = X/R
        del X
        ydivr = Y/R #we'll need these as individuals later anyway.
        del Y
        xmiydivr = xdivr - 1j*ydivr
        y1m1 = .5*np.sqrt(3./(2.*np.pi))*xmiydivr
        self.ylm_save(y1m1, 1, 1)
        del y1m1

        #ell, m = 1, 0
        zdivr = Z/R
        del Z
        y10 = .5*np.sqrt(3./np.pi)*zdivr
        self.ylm_save(y10, 1, 0)
        del y10

        #ell, m = 2, -2
        xmiydivrsq = xmiydivr*xmiydivr
        y2m2 = .25*np.sqrt(15./(2.*np.pi))*xmiydivrsq
        self.ylm_save(y2m2, 2, 2)
        del y2m2

        #ell, m = 2, -1
        y2m1 = .5*np.sqrt(15./(2.*np.pi))*xmiydivr*zdivr
        self.ylm_save(y2m1, 2, 1)
        del y2m1

        #ell, m = 2, 0
        xdivrsq = xdivr*xdivr
        ydivrsq = ydivr*ydivr
        zdivrsq = zdivr*zdivr
        y20 = .25*np.sqrt(5./np.pi)*(2.*zdivrsq-xdivrsq-ydivrsq)
        self.ylm_save(y20, 2, 0)
        del y20

        #ell, m = 3, -3
        xmiydivrcu = xmiydivr*xmiydivrsq
        y3m3 = .125*np.sqrt(35./np.pi)*xmiydivrcu
        self.ylm_save(y3m3, 3, 3)
        del y3m3

        #ell, m = 3, -2
        y3m2 = .25*np.sqrt(105./(2.*np.pi))*xmiydivrsq*zdivr
        self.ylm_save(y3m2, 3, 2)
        del y3m2

        #ell, m = 3, -1
        y3m1 = .125*np.sqrt(21./np.pi)*(xmiydivr*(4.*zdivrsq-xdivrsq-ydivrsq))
        self.ylm_save(y3m1, 3, 1)
        del y3m1

        #ell, m = 3, 0
        y30 = .25*np.sqrt(7./np.pi)*(zdivr*(2.*zdivrsq-3.*xdivrsq-3.*ydivrsq))
        self.ylm_save(y30, 3, 0)
        del y30

        #ell, m = 4, -4
        xmiydivrft = xmiydivr*xmiydivrcu
        y4m4 = .1875*np.sqrt(35./(2.*np.pi))*xmiydivrft
        self.ylm_save(y4m4, 4, 4)
        del y4m4

        #ell, m = 4, -3
        y4m3 = .375*np.sqrt(35./np.pi)*xmiydivrcu*zdivr
        self.ylm_save(y4m3, 4, 3)
        del y4m3

        #ell, m = 4, -2
        y4m2 = .375*np.sqrt(5./(2.*np.pi))*xmiydivrsq*(7.*zdivrsq-1)
        self.ylm_save(y4m2, 4, 2)
        del y4m2

        #ell, m = 4, -1
        y4m1 = .375*np.sqrt(5./np.pi)*xmiydivr*zdivr*(7.*zdivrsq-3.)
        self.ylm_save(y4m1, 4, 1)
        del y4m1

        #ell, m = 4, 0
        zdivrft = zdivrsq*zdivrsq
        y40 = .1875*np.sqrt(1./np.pi)*(35.*zdivrft-30.*zdivrsq+3.)
        self.ylm_save(y40, 4, 0)
        del y40

        #ell, m = 5, -5
        xmiydivrfi = xmiydivr*xmiydivrft
        y5m5 = (3./32.)*np.sqrt(77./np.pi)*xmiydivrfi
        self.ylm_save(y5m5, 5, 5)
        del y5m5

        #ell, m = 5, -4
        y5m4 = (3./16.)*np.sqrt(385./(2.*np.pi))*xmiydivrft*zdivr 
        self.ylm_save(y5m4, 5, 4)
        del y5m4

        #ell, m = 5, -3
        y5m3 = (1./32.)*np.sqrt(385./np.pi)*xmiydivrcu*(9.*zdivrsq-1.)
        self.ylm_save(y5m3, 5, 3)
        del y5m3

        #ell, m = 5, -2
        zdivrcu = zdivr*zdivrsq
        y5m2 = (1./8.)*np.sqrt(1155./(2.*np.pi))*xmiydivrsq*(3.*zdivrcu-zdivr)
        self.ylm_save(y5m2, 5, 2)
        del y5m2

        #ell, m = 5, -1
        y5m1 = (1./16.)*np.sqrt(165./(2.*np.pi))*xmiydivr*(21.*zdivrft-14.*zdivrsq+1.)
        self.ylm_save(y5m1, 5, 1)
        del y5m1

        #ell, m = 5, 0
        zdivrfi = zdivr*zdivrft
        y50 = (1./16.)*np.sqrt(11./np.pi)*(63.*zdivrfi-70.*zdivrcu+15.*zdivr)
        self.ylm_save(y50, 5, 0)
        del y50

    def create_radial_bins(self, save_bin_info=True):
        """
        This function creates and saves all the information corresponding to our radial bins
        """
        boundsandnumber = np.zeros((2, self.nbins+1))
        boundsandnumber[0,:] = self.bin_edges
        for i in range(self.nbins):
            boundsandnumber[1,i] = np.sum(np.logical_and(self.R >= self.bin_edges[i],
                                                       self.R < self.bin_edges[i+1]))

        self.boundsandnumber = boundsandnumber
        if save_bin_info:
            np.save(self.save_dir + 'bin_bounds_and_pixel_number_'+self.save_name+'.npy',boundsandnumber)

    def bin_spherical_harmonics(self,verbose=True):
        """
        This method applies the nbins to the spherical harmonics 
        """
        for ell in range(0, self.ell_max+1):
            for m in range(0, ell+1):
                if verbose:
                    print("ell, m = ", ell, m)

                #do one ylm at a time to save lots of accessing memory
                ylm = np.load(self.save_dir + 'ylm_'+self.save_name+'_'+str(ell)+'_'+str(m)+'.npy') 
                for i in range(self.nbins):
                    if verbose:
                        print("bin i = ", i)
                    #where is radius in bin?
                    rib = np.where((self.R >= self.bin_edges[i]) & (self.R < self.bin_edges[i+1]))
                    ylm_on_shell = np.zeros((self.ld_one_d, self.ld_one_d, self.ld_one_d)) + 0j
                    ylm_on_shell[rib] = ylm[rib]
                    del rib
                    self.ylm_transform_save(ylm_on_shell, ell, m, i)
                    del ylm_on_shell
                if 'ylm' in globals():
                    del ylm
                file_to_rm = self.save_dir + 'ylm_'+self.save_name+'_'+str(ell)+'_'+str(m)+'.npy'
                call(["rm", file_to_rm])

    def calc_ft_data(self, normalized=False):
        """
        This function takes the fourier transform of the data
        """
        data = self.density_field_data
        if normalized:
            data = (np.log(data) - np.mean(np.log(data)))/np.std(np.log(data))
        ft_data = np.fft.fftn(data)
        self.ft_data = ft_data

    def calc_a_lm_coeffs(self,verbose=True, kernel_name = None):
        """
        Calculate the a^b_lm(x) coefficients which is the convolution of the
        density field δ with the binned spherical harmonics. 
        """
        if kernel_name is not None:
            self.kernel_name = kernel_name
        else:
            raise AssertionError("You need to give a kernel_name argument value")

        if hasattr(self, 'ft_data'):
            #CONVOLUTION OF DATA AND SPH_KERNEL AT A GIVEN BIN
            for l in range(0, self.ell_max+1, 1):
                for m in range(0,l+1, 1):
                    for bin in range(0, self.nbins, 1):
                        if verbose:
                            print("l, m, bin =", l, m, bin)
                        #load ft of bsph_kernels
                        bsph_kernel = np.load(self.save_dir + 'YLMtilde_'+
                                              self.kernel_name+'_'+str(l)+'_'+str(m)+
                                            '_bin_'+str(bin)+'.npy')
                        conv = np.fft.ifftn(self.ft_data*bsph_kernel)
                        del bsph_kernel
                        #a_lm^b coefficients saved here
                        np.save(self.save_dir + self.save_name+
                                'conv_data_kernel_'+self.kernel_name+'_'+str(l)+'_'+str(m)+
                              '_bin_'+str(bin)+'.npy', conv)
                        del conv
        else:
            raise AssertionError("You need to run self.calc_ft_data() first")

    def calc_zeta(self, normalize=True):
        """
        This is the the big calculation for measuring the 4PCF coefficients.
        This code is adapted from Philcox et al. 2021 (Encore Paper)
        """

        def S(m):
            """
            A simple function defined in section 4.1.1 of
            Philcox et al. 2021 above algorithm 3. (Helper Function)
            """
            if m == 0:
                return 1/2
            return 1

        def complex_modulus(z):
            """
            assume z to be an array of complex numbers of the form z = a + ib
            (Helper Function)
            """
            a = np.real(z)
            b = np.imag(z)

            return np.sqrt(a**2 + b**2)

        #we define these local variables out of laziness to change my original non-object oriented code
        ell_max = self.ell_max
        nbins = self.nbins
        if hasattr(self, 'boundsandnumber') == False:
            raise AssertionError("You need to run self.create_radial_bins() first!")
        
        start = time.time()
        print("Executing 4PCF Calculation ...")

        zeta = np.zeros((ell_max+1, ell_max+1, ell_max+1,nbins, nbins, nbins)) + 0j
        CG_Coefficients = np.load("CG_Coeffs.npy")
        for l_1 in range(0,ell_max+1):
            for l_2 in range(0,ell_max+1):
                for l_3 in range(np.abs(l_1 - l_2), min(l_1 + l_2, ell_max)+1):
                    if (l_1 + l_2 + l_3)%2 != 0: # we don't assume this to be true for Turbulent ISM
                        continue
                    for m_1 in range(-l_1, l_1 + 1):
                        for m_2 in range(-l_2, l_2 + 1):
                            m_3 = -m_1 - m_2
                            if m_3 > l_3 or m_3 < 0:
                                continue
                            coupling_w = self.density_field_data * (-1)**(l_1 + l_2 + l_3) * CG_Coefficients[l_1,l_2,l_3,m_1,m_2,m_3]
                            for b_1 in range(0, nbins): #might be an error with indexing here
                                if m_1 < 0:
                                    a_lmb_1 = (-1)**m_1 * (np.load(self.save_dir + self.save_name+'conv_data_kernel_'+self.kernel_name+'_'+
                                            str(l_1)+'_'+str(-m_1)+'_bin_'+str(b_1)+'.npy').astype(np.complex128)).conjugate() 
                                else:
                                    a_lmb_1 = np.load(self.save_dir + self.save_name+'conv_data_kernel_'+self.kernel_name+'_'+str(l_1)+
                                                '_'+str(m_1)+'_bin_'+str(b_1)+'.npy').astype(np.complex128)
                                for b_2 in range(b_1, nbins):#might be an error with indexing here
                                    if m_2 < 0:
                                        a_lmb_2 = (-1)**m_1 * (np.load(self.save_dir + self.save_name+'conv_data_kernel_'+self.kernel_name+'_'+
                                                str(l_2)+'_'+str(-m_2)+'_bin_'+str(b_2)+'.npy').astype(np.complex128)).conjugate()              
                                    else:
                                        a_lmb_2 = np.load(self.save_dir + self.save_name+'conv_data_kernel_'+self.kernel_name+'_'+str(l_2)+
                                                    '_'+str(m_2)+'_bin_'+str(b_2)+'.npy').astype(np.complex128)
                                    for b_3 in range(b_2, nbins):#might be an error with indexing here
                                        a_lmb_3 = np.load(self.save_dir + self.save_name+'conv_data_kernel_'+self.kernel_name+'_'+str(l_3)+
                                                    '_'+str(m_3)+'_bin_'+str(b_3)+'.npy').astype(np.complex128)
                                        zeta[l_1, l_2, l_3, b_1, b_2, b_3] += np.sum(2 * S(m_3) * 
                                                                                coupling_w * 
                                                                                complex_modulus(a_lmb_1 * a_lmb_2 * a_lmb_3))



        finish=time.time()
        print("Finished Calculating 4PCF in {0:0.4f} seconds".format(finish-start))
        if normalize:
            """
            Normalize zeta^L_B (where L = {\ell_1, \ell_2, \ell_3} and B = {b_1, b_2, b_3}) hat
            coefficients from calc_zeta by dividing by bin volume
            """
            binvolume = self.boundsandnumber[1,0:nbins]
            normalize_coeff = (binvolume[:,None, None] * binvolume[None,:, None] * binvolume[None, None, :])
            self.zeta_normed_values = (zeta/(normalize_coeff*(self.ld_one_d**3)))
        else: 
            print("your zeta coefficients are not normalized. To do so, please run")
            self.zeta_values = zeta    


    def normalize_zeta(self):
        """
        Normalize zeta^L_B (where L = {\ell_1, \ell_2, \ell_3} and B = {b_1, b_2, b_3}) hat
        coefficients from calc_zeta by dividing by bin volume
        """
        if hasattr(self, 'zeta_values'):
            binvolume = self.boundsandnumber[1,0:self.nbins]
            normalize_coeff = (binvolume[:,None, None] * binvolume[None,:, None] * binvolume[None, None, :])
            self.zeta_normed_values = (zeta/(normalize_coeff*(self.ld_one_d**3)))
        else:
            raise AssertionError("You already normalized zeta!")
            
            
    def run_all(self,calc_ylm_flag=False, bin_ylm_flag=False, alm_flag=False, verbose_flag=True):
        """
        The flag arguments indicate what you have already calculated. 
        You may want to skip the ylm or alm creation steps if you've 
        already calculated them.
        """
        if verbose_flag:
            print("Creating XYZ Grids for radial bin and ylm creation ... \n")
        self.create_XYZR()
        if verbose_flag:
            print("Creating radial bins ... \n")
        self.create_radial_bins()
        if verbose_flag:
            print("taking the fourier transform of data ... \n")
        self.calc_ft_data()
        
        if calc_ylm_flag:
            if bin_ylm_flag:
                if alm_flag:
                    self.calc_zeta()
                else:
                    self.calc_a_lm_coeffs(verbose=verbose_flag, kernel_name=self.save_name)
                    self.calc_zeta()
            else:
                self.bin_spherical_harmonics(verbose=verbose_flag)
                if alm_flag:
                    self.calc_zeta()
                else:
                    self.calc_a_lm_coeffs(verbose=verbose_flag, kernel_name=self.save_name)
                    self.calc_zeta()
              
        else:
            self.calc_and_save_YLMs()
            if bin_ylm_flag:
                if alm_flag:
                    self.calc_zeta()
                else:
                    self.calc_a_lm_coeffs(verbose=verbose_flag, kernel_name=self.save_name)
                    self.calc_zeta()
            else:
                self.bin_spherical_harmonics(verbose=verbose_flag)
                if alm_flag:
                    self.calc_zeta()
                else:
                    self.calc_a_lm_coeffs(verbose=verbose_flag, kernel_name=self.save_name)
                    self.calc_zeta()
    
    def save_zeta(self):
        """
        a simple function to save the 4pcf data
        """
        if hasattr(self, 'zeta_normed_values'):
            print("Saving normalized zetas")
            np.save(self.save_dir + self.save_name + '_zeta_normed.npy', self.zeta_normed_values)
        elif hasattr(self, 'zeta_values') and not hasattr(self, 'zeta_normed_values'):
            print("Saving non-normalized zetas")
            np.save(self.save_dir + self.save_name + '_zeta_unnormed.npy', self.zeta_values)
        else:
            raise AssertionError("You need to run calc_zetas or add the attribute self.zeta_normed_values or self.zeta_values")

            
                
        ##################################################################
        #                        Plotting Methods
        ##################################################################

    def plot_YLMs(self):
            """
            If you want to see what all the calculated spherical harmonics look like,
            you should run this function.
            """
            plt.figure(figsize=(14,13))
            for ell in range(1, self.ell_max+1):
                for m in range(0, ell+1):
                    plt.subplot(self.ell_max, self.ell_max+1, (ell-1)*(self.ell_max+1)+m+1)
                    ylm = np.load(self.save_dir + 'ylm_'+self.save_name+'_'+str(ell)+'_'+str(m)+'.npy')
                    plt.imshow(ylm[:,self.ld_one_d//2,:].real, 
                               interpolation='none', vmin=-1, vmax=1, 
                               cmap='bwr',origin='lower')
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('l='+str(ell)+' m='+str(m))
            plt.suptitle('x-z slices of real part of spherical harmonics', fontsize='xx-large')
            plt.show()
            
    def plot_alm(self,ell=1):
        """
        We can visualize what the alm coefficients look like with this function
        """
        if hasattr(self, 'kernel_name') != True:
            self.kernel_name = self.save_name
            print("saving the kernel_name attribute as save_name")
        
        plt.figure(figsize=(14,7))
        plt.subplot(2,self.nbins+1,1)
        plt.imshow(self.density_field_data[:,self.ld_one_d//2,:])
        plt.title('unconvolved δ')
        plt.xlabel('x-axis')
        plt.ylabel('z-axis')
        plt.xticks([])
        plt.yticks([])

        for m in range(2):
            for bin in range(self.nbins):
                plt.subplot(2,self.nbins+1,m*(self.nbins+1)+bin+2)
                ylm_b = np.load(self.save_dir + self.save_name+'conv_data_kernel_'+self.kernel_name+'_'+str(ell)+
                                    '_'+str(m)+'_bin_'+str(bin)+'.npy').astype(np.complex128)
                plt.imshow(ylm_b[:,self.ld_one_d//2,:].real)
                if m == 0:
                    plt.title('bin '+str(bin))
                if bin == 0:
                    plt.ylabel('m='+str(m))
                plt.xticks([])
                plt.yticks([])
        plt.suptitle("$a_{\ell m}^b$ for $\\ell=$" + str(ell) + " (y = " + str(self.ld_one_d//2) + ")", fontsize='large') 
        plt.show()