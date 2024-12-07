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
import time
from tenpy.tools import hdf5_io

from argparse import ArgumentParser

import sys, os
sys.path.append('/home1/wttai/machine_learning/common_lib')
from dmrg_correlation_helper import compute_corr_results

parser = ArgumentParser()

parser.add_argument("--L", type=int, default=2, help="Side of the square")
parser.add_argument("--Lx", type=int, default=2, help="Side of the rectangle in the x direction (if L is not specified)")
parser.add_argument("--Ly", type=int, default=2, help="Side of the rectangle in the y direction (if L is not specified)")
parser.add_argument("--N" , type=int, default=-1, help="Number of particles (default to half-filling)")
parser.add_argument("--N_frac", type=float, default=-1, help="Fraction of particles (default to half-filling)")
parser.add_argument("--m", type=float, default=5.0, help="mass term in the Hamiltonian")
parser.add_argument("--t", type=float, default=1.0, help="hopping term in the Hamiltonian")
parser.add_argument("--U", type=float, default=0.2, help="interaction term in the Hamiltonian")
parser.add_argument("--chi_max", type=int, default=1000, help="maximum bond dimension")
parser.add_argument("--pbc",  dest="pbc", help="periodic boundary conditions", action="store_true")
parser.add_argument("--output_dir" , type=str, default="data/", help="output directory")
parser.add_argument("--bias", type=float, default=1e-5, help="bias term in the Hamiltonian")

args = parser.parse_args()


L = args.L
Lx = args.Lx
Ly = args.Ly
m = args.m
t = args.t
U = args.U
N = args.N
N_frac = args.N_frac
chi_max = args.chi_max
pbc = args.pbc
output_dir = args.output_dir
bias = args.bias

if L != -1:
    Lx = L
    Ly = L


if N != -1 and N_frac != -1:
    raise ValueError("Cannot specify both N and N_frac")
if N == -1 and N_frac == -1:
    N = Lx * Ly
if N_frac != -1:
    N = int(2 * N_frac * Lx * Ly)

if N > Lx * Ly:
    raise ValueError("Current code is not good for more than half-filling")

print(f"Initial parameters: m = {m}, t = {t}, U = {U}")
outputFilename = output_dir + f"dmrg_log_{"pbc" if pbc else "obc"}_L={L}_N={N}_t={t}_m={m}_U={U}.h5"


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
                self.add_coupling(1j * t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                self.add_coupling(1j * t, i1, "Cdd", i2, "Cu", dx, plus_hc=True)
            if np.array_equal(dx, [0,1]):
                self.add_coupling(t, i1, "Cdu", i2, "Cd", dx, plus_hc=True)
                self.add_coupling(-t, i1, "Cdd", i2, "Cu", dx, plus_hc=True) 
        self.add_local_term(bias, [("Nd", [0,0,0])], 0)       


# Create the Fermi-Hubbard Model
model_params = {
    't': t,                 # Nearest-neighbor hopping strength
    'U': U,                 # On-site Hubbard interaction
    'm': m,                 # Mass gap
    'Lx': Lx,
    'Ly': Ly,
    'cons_Sz': None,
    'cons_N': 'N',
    'bc_x': 'periodic' if pbc else 'open',
    'bc_y': 'periodic' if pbc else 'open', 
    'bias': bias
}

dmrg_params = {
    "mixer": DensityMatrixMixer,
    "mixer_params": {
        "amplitude": 0.3,
        "decay": 2,
        "disable_after": 50
    },
    "trunc_params": {
        "chi_max": chi_max, #bond dimension
        "svd_min": 1*10**-10
    },
    "max_E_err": 0.0001, #energy convergence step threshold
    "max_S_err": 0.0001, #entropy convergence step threshold
    "max_sweeps": 2000  #may or may not be enough to converge
}

model = FermiHubbardSquare(model_params)
H_mpo = model.calc_H_MPO()
print(f'MPO bond dimensions: {H_mpo.chi}')

product_state = ['up' if n < N else 'empty' for n in range(Lx*Ly)]#['empty' if n < Lx * Ly // 2 else 'up' for n in range(Lx * Ly)]
#product_state = [ 'down' for n in range(Lx * Ly) ]
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)

E0, psi = engine.run()
print(f'Ground state energy: {E0}')

correlation_types = {
    "uu": ("Nu", "Nu"),
    "ud": ("Nu", "Nd"),
    "dd": ("Nd", "Nd")
}

# Compute and store correlation results
corrs_results = {}
for label, corr_type in correlation_types.items():
    corrs_results[label] = compute_corr_results(corr_type, psi, Lx, Ly)



data = {"psi": psi,  # e.g. an MPS
        "E0": E0,  # ground state energy
        "model": model,
        "chi": psi.chi,
        "sweepstats": engine.sweep_stats,
        "corrs_results": corrs_results
}

metaData = {
    'L': L,
    'N': N,
    'm': m,
    't': t,
    'U': U,
    'bias': bias,
    'pbc': pbc,
    "mixer_params": {
        "amplitude": 0.3,
        "decay": 2,
        "disable_after": 50
    },
    "trunc_params": {
        "chi_max": chi_max, #bond dimension
        "svd_min": 1*10**-10
    },
    "max_E_err": 0.0001, #energy convergence step threshold
    "max_S_err": 0.0001, #entropy convergence step threshold
    "max_sweeps": 2000,  #may or may not be enough to converge
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

mergedData = {
    'metadata': metaData,
    'data': data
}

with h5py.File(outputFilename, 'w') as f:
    hdf5_io.save_to_hdf5(f, mergedData)