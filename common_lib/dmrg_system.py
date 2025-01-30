import numpy as np

import tenpy
from tenpy.models.hubbard import FermiHubbardChain
from tenpy.models.lattice import Square
from tenpy.models import CouplingMPOModel
from tenpy.networks.site import FermionSite, BosonSite, SpinHalfFermionSite, spin_half_species
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.algorithms import dmrg, tebd, tdvp
from tenpy.algorithms.mps_common import DensityMatrixMixer, SubspaceExpansion


class QWZSquare(CouplingMPOModel):
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N', str)
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_lattice(self, model_params):
        site = self.init_sites(model_params)
        Lx = model_params.get('Lx', 4)
        Ly = model_params.get('Ly', 4)
        bc_x = model_params.get('bc_x', 'open', str)
        bc_y = model_params.get('bc_y', 'open', str)
        return Square(Lx=Lx, Ly=Ly, site=site, bc = [bc_x, bc_y])

    def init_terms(self, model_params):
        # read out parameters
        t = model_params.get("t", 1.) 
        Uv = model_params.get("U", 1.)
        m = model_params.get("m", 1.)
        bias = model_params.get("bias", 0.)
        # add terms
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Uv, u, 'NuNd')
            self.add_onsite(m, u, "Nu")
            self.add_onsite(-m, u, "Nd")
        for i1, i2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(t, i1, "Cdu", i2, "Cu", dx, plus_hc=True)
            self.add_coupling(-t, i1, "Cdd", i2, "Cd", dx, plus_hc=True)
            if np.array_equal(dx, [1,0]):
                self.add_coupling(1j * t if complex else t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                self.add_coupling(1j * t if complex else -t, i1, "Cdd", i2, "Cu", dx, plus_hc=True)
            if np.array_equal(dx, [0,1]):
                self.add_coupling(t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                self.add_coupling(-t, i1, "Cdd", i2, "Cu", dx, plus_hc=True) 
        self.add_local_term(bias, [("Nd", [0,0,0])], 0)   




class FermiHubbardSquare(CouplingMPOModel):
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N', str)
        #cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        site = FermionSite(conserve=cons_N)
        return site

    def init_lattice(self, model_params):
        site = self.init_sites(model_params)
        Lx = model_params.get('Lx', 4)
        Ly = model_params.get('Ly', 4)
        bc_x = model_params.get('bc_x', 'open', str)
        bc_y = model_params.get('bc_y', 'open', str)
        return Square(Lx=Lx, Ly=Ly, site=site, bc = [bc_x, bc_y])

    def init_terms(self, model_params):
        # read out parameters
        t = model_params.get("t", 1.) 
        Uv = model_params.get("U", 1.)
        bias = model_params.get("bias", 0.)
        # add terms
        for i1, i2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(t, i1, "Cd", i2, "C", dx, plus_hc=True)
            self.add_coupling(Uv, i1, "N", i2, "N", dx, plus_hc=False)
        self.add_local_term(bias, [("N", [0,0,0])], 0)   