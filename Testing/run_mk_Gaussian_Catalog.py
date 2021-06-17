import numpy
from mk_Gaussian_Catalog import GenerateGRF

Nmesh = 512
cdir = '/Users/mianoctobers/Projects/NPCF/data/'
pname = '/Users/mianoctobers/Projects/NPCF/pk/Plin_z0d57_CosmoBoss.txt'
do_rsd = True

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
        cname = cdir + f'GRF_mesh_weighted_{ii:04d}.txt'
        print("save catalog to", cname)
        Pk_in = numpy.genfromtxt(pname)
        grf = GenerateGRF(Nmesh, Pk_in=Pk_in, do_inv_phase=True, cname=cname)
        grf.run_mk_catalog_grf_weighted(do_save_catalog=True)
        #grf.calc_pk() 
