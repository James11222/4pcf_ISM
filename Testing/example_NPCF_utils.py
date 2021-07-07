from nbodykit.lab import *

class GetCoeffCpp(object):

    '''
        get measured npcf coefficients 
    '''

    def __init__(self, fdir, npcf, imock=1, verbose=False):
        
        self.fdir = fdir
        self.npcf = npcf
        self.imock = imock
        self.verbose = verbose
    

    def calc_accepted_ells(self, ell_max = 5):
    
        index_dict = {}
        for l_1 in range(0,ell_max+1, 1):
            for l_2 in range(0,ell_max+1, 1):
                for l_3 in range(0,ell_max+1, 1):
                    if l_3 >= numpy.abs(l_1 - l_2) and l_3 < (min(l_1 + l_2, ell_max) + 1): 
                        if (l_1 + l_2 + l_3)%2 == 0 and not numpy.all(numpy.array([l_1,l_2,l_3]).astype(bool)):
                            index_dict[str(l_1) + str(l_2) + str(l_3)] = ("$\\ell_1, \ell_2, \ell_3$ = "
                                                   + str(l_1) + ',' + str(l_2) + ',' + str(l_3))
                        else:
                            index_dict[str(l_1) + str(l_2) + str(l_3)] = "Invalid Combination"
                    else:
                        index_dict[str(l_1) + str(l_2) + str(l_3)] = "Invalid Combination"

        ells = [k for k,v in index_dict.items() if v[0] != 'I']    
        self.ell_gaussian = list(ells)
        
    def load_mhd(self):
        """
        a method that takes in an input power spectrum and a 2pcf
        measurement. This function creates a meta_data object which 
        goes into the model_4PCF_Gaussian_isotropic_basis
        """
        configs = numpy.load(self.fdir + 'grf_configs.npy')
        configs = {'k_input':configs[0], 'Pk0_input':configs[1], 'volume':1574.0**3}
        mock_2pcf = numpy.load(self.fdir + "grf_2pcf.npy") # load in 2pcf coefficients
        self.pk_in = configs['Pk0_input'] # same structure of Pk_in from other file
        #self.calc_accepted_ells() # ell's sets I want to compare 
        #b1, b2, b3, multiplied by physical units will be r1, r2, r3
        #r_n is just the physical centers of b_n
        self.rbins_1d =  numpy.array(sorted(list(set(mock_2pcf[0] ).union(set(mock_2pcf[0])).union(set(mock_2pcf[0])))))
        self.k_in = configs['k_input'] # first/second column in Pk_in
        self.vol = configs['volume'] # volume of the total box
# 7        if which_space == 'real':
        self.twopcf_mean = mock_2pcf[1] #available from nbodykit 2pcf
    #twopcf_mean isnt the mean of one realization, it is the average of multiple
    #realizations and is should be the length of nbins!
        #self.twopcf_std = logn_2pcf['twopcf_std'] #available from nbodykit 2pcf
#         self.ell_gaussian = ['000',
#                              '011',
#                              '022',
#                              '033',
#                              '044',
#                              '055',
#                              '101',
#                              '110',
#                              '111',]
        self.calc_accepted_ells()
        
        
        
        
        
        
    def load_lognormal_4pcf_1000(self, which_space='real'):
        if which_space == 'real':
            logn_2pcf = numpy.load(self.fdir + "npcf_measurement/1000_lognormal_mocks_real_2pcf.npz")
            logn = numpy.load(self.fdir + "npcf_measurement/1000_lognormal_mocks_real_4pcf.npz")
            self.pk_in = logn['Pk0_input']
        elif which_space == 'zs':
            logn = numpy.load(self.fdir + "npcf_measurement/1000_lognormal_mocks_rsd_4pcf.npz")
            self.pk_in = numpy.load(self.fdir + "npcf_measurement/1000_lognormal_mocks_real_4pcf.npz")['Pk0_input']
            self.pk_zs_mean = {}
            self.pk_zs_mean[0] = logn['Pk0_input']
            self.pk_zs_mean[2] = logn['Pk2_input']
            self.pk_zs_mean[4] = logn['Pk4_input']
        self.ell1, self.ell2, self.ell3 = logn['ell1'], logn['ell2'], logn['ell3']
        self.ells = [i for i in zip(logn['ell1'], logn['ell2'], logn['ell3'])]
        tt1 = [i for i in self.ells if 0 in i]
        tt2 = [i for i in self.ells if 0 not in i]
        self.ell_gaussian = []
        for ii, iell in enumerate(tt1):
            tmp = "".join([str(i) for i in iell])
            self.ell_gaussian.append(tmp)
        self.ell_non_gaussian = []
        for ii, iell in enumerate(tt2):
            tmp = "".join([str(i) for i in iell])
            self.ell_non_gaussian.append(tmp)
        self.bins = [i for i in zip(logn['bins1'], logn['bins2'], logn['bins3'])]
        self.rbins = [i for i in zip(logn['r1'], logn['r2'], logn['r3'])]
        self.rbins_1d = numpy.array(sorted(list(set(logn['r1']).union(set(logn['r2'])).union(set(logn['r3'])))))
        self.fourpcf_full = logn['fourpcf_all']
        self.fourpcf_full_mean = numpy.mean(logn['fourpcf_all'], axis=0)
        self.fourpcf_c = logn['fourpcf_all']-logn['disc_fourpcf_all']
        self.fourpcf_c_mean = numpy.mean(logn['fourpcf_all']-logn['disc_fourpcf_all'], axis=0)
        self.fourpcf_dc = logn['disc_fourpcf_all']
        self.fourpcf_dc_mean = numpy.mean(logn['disc_fourpcf_all'], axis=0)
        self.k_in   = logn['k_input']
        self.pk0_in = logn['Pk0_input']
        self.vol = logn['volume']
        self.shot_noise = 1./(logn['Ngal']/logn['volume'])
        self.redshift = 2.0
        if which_space == 'real':
            self.twopcf_mean = logn_2pcf['twopcf_mean']
            self.twopcf_std = logn_2pcf['twopcf_std']
            
            
    def get_coeff_boss_cpp(self):
        table = numpy.genfromtxt(self.fdir + f'npcf_measurement/boss_cmass.zeta_{self.npcf:d}pcf.txt',
                       skip_header=int(7+(self.npcf-3)), filling_values=999)
        nrows, ncols = table.shape
        if self.npcf == 3:
            norders = self.npcf - 2
        elif self.npcf == 4:
            norders = self.npcf - 1
        elif self.npcf == 5:
            norders = self.npcf
        elif self.npcf == 6:
            norders = self.npcf + 1
        table = numpy.genfromtxt(self.fdir + f'npcf_measurement/boss_cmass.zeta_{self.npcf:d}pcf.txt',
                        skip_header=int(7+(self.npcf-3)), filling_values=999, 
                        dtype=[('orders', ('int', int(norders))), 
                            ('zetas', ('f8', int(ncols-norders)))])
        self.bins = numpy.genfromtxt(self.fdir + f'npcf_measurement/boss_cmass.zeta_{self.npcf:d}pcf.txt',
                       skip_header=int(7+(self.npcf-3)-5), filling_values=999,
                 max_rows=int(self.npcf-1)).astype(int)
        self.bins = numpy.array([ii for ii in zip(*self.bins)])
        self.ells = []
        self.zetas = {}
        try:
            for ii, il in enumerate(table['orders']):     
                tmp = [str(s) for s in list(il)]
                self.ells.append("".join(tmp))
                self.zetas["".join(tmp)] = table[ii][1]
        except:
            for ii in table['orders']:
                tmp = int(self.npcf-1) * str(ii)
                self.ells.append("".join(tmp))   
                self.zetas["".join(tmp)] = table[ii][1]


if __name__ == '__main__':

    fdir = "/Users/mianoctobers/Projects/NPCF/results/"
    npcf = 4
    zetas = GetCoeffCpp(fdir, npcf)