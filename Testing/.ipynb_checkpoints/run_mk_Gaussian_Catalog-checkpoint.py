import numpy
from mk_Gaussian_Catalog import GenerateGRF

Nmesh = 64
cdir =  '/Users/jamessunseri/desktop/uf_reu/4pcf_ISM/Testing/' #changing these from Jiamin's original code, saving directory
pname = None # changing these from Jiamin's original code
do_rsd = False

if do_rsd:
    for ii in range(2,3):
        cname = cdir + f'GRF_rsd001_mesh_weightedk_{ii:04d}.txt'
        print("save catalog to", cname)
        Pk_in = numpy.genfromtxt(pname)
        grf = GenerateGRF(Nmesh, Pk_in=Pk_in, los=[0,0,1], which_rsd="k-space", do_inv_phase=False,
                 mean_weight=0., cname=cname, verbose=False)
        grf.run_mk_catalog_grf_weighted(do_save_catalog=True)
else:
    for ii in range(10,11):
        # cname = cdir + f'GRF_mesh_weighted_{ii:04d}.txt'
        cname = cdir + 'deltar.npy'
        print("save catalog to", cname)
        # Pk_in = numpy.genfromtxt(pname)
        Pk_in = None
        grf = GenerateGRF(Nmesh, Pk_in=Pk_in, do_inv_phase=True, cname=cname)
        grf.run_mk_catalog_grf_weighted(do_save_catalog=True)
        #grf.calc_pk()
