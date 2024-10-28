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
import tenpy.linalg.np_conserved as npc
import h5py
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import h5py
from tenpy.tools import hdf5_io

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--L", type=int, default=2, help="Side of the square")
parser.add_argument("--Lx", type=int, default=2, help="Side of the rectangle in the x direction (if L is not specified)")
parser.add_argument("--Ly", type=int, default=2, help="Side of the rectangle in the y direction (if L is not specified)")
parser.add_argument("--m", type=float, default=5.0, help="mass term in the Hamiltonian")
parser.add_argument("--t", type=float, default=1.0, help="hopping term in the Hamiltonian")
parser.add_argument("--U", type=float, default=0.2, help="interaction term in the Hamiltonian")
parser.add_argument("--output_dir" , type=str, default="data/", help="output directory")

args = parser.parse_args()


L = args.L
Lx = args.Lx
Ly = args.Ly
m = args.m
t = args.t
U = args.U
output_dir = args.output_dir

print(f"Initial parameters: m = {m}, t = {t}, U = {U}")
outputFilename = output_dir + f"dmrg_log_L={L}_t={t}_m={m}_U={U}.h5"

if L != -1:
    Lx = L
    Ly = L

# Create the Fermi-Hubbard Model
model_params = {
    't': t,                 # Nearest-neighbor hopping strength
    'U': U,                 # On-site Hubbard interaction
    'm': m,                 # Mass gap
    'Lx': Lx,
    'Ly': Ly,
    'cons_Sz': None,
    'cons_N': 'N',
}

dmrg_params = {
    "mixer": DensityMatrixMixer,
    "mixer_params": {
        "amplitude": 0.3,
        "decay": 2,
        "disable_after": 50
    },
    "trunc_params": {
        "chi_max": 500, #bond dimension
        "svd_min": 1*10**-10
    },
    "max_E_err": 0.0001, #energy convergence step threshold
    "max_S_err": 0.0001, #entropy convergence step threshold
    "max_sweeps": 2000  #may or may not be enough to converge
}

class FermiHubbardSquare(CouplingMPOModel):
    
    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N', str)
        cons_Sz = model_params.get('cons_Sz', 'Sz', str)
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        return site

    def init_lattice(self, model_params):
        site = self.init_sites(model_params)
        Lx = model_params.get('Lx', 4)
        Ly = model_params.get('Ly', 4)
        return Square(Lx=Lx, Ly=Ly, site=site)

    def init_terms(self, model_params):
        # read out parameters
        t = model_params.get("t", 1.) 
        Uv = model_params.get("U", 1.)
        m = model_params.get("m", 1)
        # add terms
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(Uv, u, 'NuNd')
            self.add_onsite(m, u, "Nu")
            self.add_onsite(-m, u, "Nd")
        for i1, i2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(t, i1, "Cdu", i2, "Cu", dx, plus_hc=True)
            self.add_coupling(-t, i1, "Cdd", i2, "Cd", dx, plus_hc=True)
            if np.array_equal(dx, [1,0]):
                self.add_coupling(1j * t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                self.add_coupling(1j * t, i1, "Cdd", i2, "Cu", dx, plus_hc=True)
            if np.array_equal(dx, [0,1]):
                self.add_coupling(t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                self.add_coupling(-t, i1, "Cdd", i2, "Cu", dx, plus_hc=True)        


model = FermiHubbardSquare(model_params)
H_mpo = model.calc_H_MPO()
print(f'MPO bond dimensions: {H_mpo.chi}')




product_state = [ 'down' for n in range(Lx * Ly) ]
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)

E0, psi = engine.run()
print(f'Ground state energy: {E0}')

data = {"psi": psi,  # e.g. an MPS
        "E0": E0,  # ground state energy
        "model": model,
        "sweepstats": engine.sweep_stats,
        "parameters": {"L": Lx * Ly, "t": t, "U": U, "m": m}
}

with h5py.File(outputFilename, 'w') as f:
    hdf5_io.save_to_hdf5(f, data)