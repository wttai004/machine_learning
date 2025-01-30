import numpy as np

from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd, tdvp
from tenpy.algorithms.mps_common import DensityMatrixMixer, SubspaceExpansion
import h5py
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
import h5py
import time
import json, fcntl
from tenpy.tools import hdf5_io

from argparse import ArgumentParser

import sys, os
sys.path.append('/home1/wttai/machine_learning/common_lib')
from dmrg_correlation_helper import compute_corr_results
from dmrg_system import QWZSquare

parser = ArgumentParser()

# parser.add_argument("--L", type=int, default=2, help="Side of the square")
# parser.add_argument("--Lx", type=int, default=-1, help="Side of the rectangle in the x direction (if L is not specified)")
# parser.add_argument("--Ly", type=int, default=-1, help="Side of the rectangle in the y direction (if L is not specified)")
parser.add_argument("--L", type=int, default=2, help="Side of the square")
parser.add_argument("--L2", type=int, default=-1, help="Side of the rectangle (if using rectangle)")

parser.add_argument("--N" , type=int, default=-1, help="Number of particles (default to half-filling)")
parser.add_argument("--N_frac", type=float, default=-1, help="Fraction of particles (default to half-filling)")
parser.add_argument("--m", type=float, default=5.0, help="mass term in the Hamiltonian")
parser.add_argument("--t", type=float, default=1.0, help="hopping term in the Hamiltonian")
parser.add_argument("--U", type=float, default=0.2, help="interaction term in the Hamiltonian")
parser.add_argument("--chi_max", type=int, default=1000, help="maximum bond dimension")
parser.add_argument("--pbc",  dest="pbc", help="periodic boundary conditions", action="store_true")
parser.add_argument("--output_dir" , type=str, default="data/", help="output directory")
parser.add_argument("--bias", type=float, default=1e-5, help="bias term in the Hamiltonian")
parser.add_argument("--real", dest="real", help="use real Hamiltonian and model", action="store_false")

parser.add_argument("--create_database", dest="create_database", help="create a database", action="store_true")
parser.add_argument("--database_name", type=str, default="database", help="database directory")
parser.add_argument("--job_id", type=int, default=0, help="job id")

args = parser.parse_args()


L = args.L
L2 = L if args.L2 == -1 else args.L2
Lx = L
Ly = L2

m = args.m
t = args.t
U = args.U
N = args.N
N_frac = args.N_frac
chi_max = args.chi_max
pbc = args.pbc
output_dir = args.output_dir
bias = args.bias
create_database = args.create_database
database_name = args.database_name
job_id = args.job_id
complex = args.real

if N != -1 and N_frac != -1:
    raise ValueError("Cannot specify both N and N_frac")
if N == -1 and N_frac == -1:
    N = Lx * Ly
if N_frac != -1:
    N = int(2 * N_frac * Lx * Ly)

if N > Lx * Ly:
    raise ValueError("Current code is not good for more than half-filling")

print(f"Initial parameters: m = {m}, t = {t}, U = {U}", flush = True)

if L2 != L:
    physicalSystemDir = f"L={L}_L2={L2}_N={N}_t={t}_m={m}_U={U}_{"pbc" if pbc else "obc"}{"" if complex else "_real"}/"
else:
    physicalSystemDir = f"L={L}_N={N}_t={t}_m={m}_U={U}_{"pbc" if pbc else "obc"}{"" if complex else "_real"}/"

outputFilename=output_dir + physicalSystemDir + f"dmrg_log_chi_max={chi_max}.h5"

print(f"Output file: {outputFilename}", flush = True)

os.makedirs(output_dir + physicalSystemDir, exist_ok=True)


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

model = QWZSquare(model_params)
H_mpo = model.calc_H_MPO()
print(f'MPO bond dimensions: {H_mpo.chi}')

product_state = ['up' if n < N else 'empty' for n in range(Lx*Ly)]#['empty' if n < Lx * Ly // 2 else 'up' for n in range(Lx * Ly)]
#product_state = [ 'down' for n in range(Lx * Ly) ]
psi = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)

engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)

E0, psi = engine.run()
print(f'Ground state energy: {E0}')

correlation_types = {
    "ss": ("Nu", "Nu"),
    "sp": ("Nu", "Nd"),
    "pp": ("Nd", "Nd")
}

# Compute and store correlation results
corrs_results = {}
for label, corr_type in correlation_types.items():
    corrs_results[label] = compute_corr_results(corr_type, psi, Lx, Ly, pbc = pbc)



data = {"psi": psi,  # e.g. an MPS
        "E0": E0,  # ground state energy
        "model": model,
        "chi": psi.chi,
        "sweepstats": engine.sweep_stats,
        "corrs_results": corrs_results
}

metaData = {
    'L': L,
    'L2': L2,
    'N': N,
    'm': m,
    't': t,
    'U': U,
    'bias': bias,
    'pbc': pbc,
    'complex': complex,
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



if create_database:
    database_location = output_dir + database_name + ".json"
    
    print(f"Creating database at {database_location}", flush = True)
    #os.makedirs(database_name, exist_ok=True)

    if not os.path.exists(database_location):
        with open(database_location, 'w') as f:
            json.dump([], f)

    # Append the new job data
    with open(database_location, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        # Load existing data
        data = json.load(f)
        
        job_data = {
            'job_id': job_id,
            'metadata': metaData,
            #'data': runtimeData,
            'outputFilename': outputFilename
        }
        # Append the new entry
        data.append(job_data)
        
        # Write back to file
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

        #fcntl.flock(f, fcntl.LOCK_UN)

    print(f"Database created at {database_location}", flush = True)
    exit(0)
