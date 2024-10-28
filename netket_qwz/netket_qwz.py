import netket as nk
import netket.experimental as nkx
from scipy.sparse.linalg import eigsh
import numpy as np
import scipy.sparse.linalg
import jax
import jax.numpy as jnp
import json
import time
import matplotlib.pyplot as plt

import sys, os
sys.path.append('/Users/wttai/Documents/Jupyter/machine_learning/common_lib')
sys.path.append('/home1/wttai/machine_learning/common_lib')
from models import get_qwz_graph, get_qwz_Ham, cdag, c, nc
from networks import *
from helper import get_ed_data

print("Program running...", flush = True) 

home_dir = os.path.expanduser('~')
##### PROGRAM START #####
from argparse import ArgumentParser

# parse command-line arguments
parser = ArgumentParser()

parser.add_argument("--L", type=int, default=2, help="Side of the square")
parser.add_argument("--N" , type=int, default=-1, help="Number of particles")
parser.add_argument("--m", type=float, default=5.0, help="mass term in the Hamiltonian")
parser.add_argument("--t", type=float, default=1.0, help="hopping term in the Hamiltonian")
parser.add_argument("--U", type=float, default=0.2, help="interaction term in the Hamiltonian")
parser.add_argument("--n_iter", type=int, default=300, help="number of iterations")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--diag_shift", type=float, default=0.01, help="diagonal shift in the Stochastic Reconfiguration method")
parser.add_argument("--n_discard_per_chain", type=int, default=16, help="number of samples to discard per chain")
parser.add_argument("--n_samples", type=int, default=2**12, help="number of samples")
parser.add_argument("--model", type=str, default="slater", help="model to use: slater or nj")
parser.add_argument("--pbc",  dest="pbc", help="periodic boundary conditions", action="store_true")
parser.add_argument("--n_hidden", type=int, default=8, help="number of hidden units in the Neural Jastrow/Backflow model")
parser.add_argument("--n_hidden_layers", type=int, default=1, help="number of hidden layers in the Neural Jastrow/Backflow model")
parser.add_argument("--n_iter_trial", type=int, default=100, help="number of iterations attempted to check convergence")
parser.add_argument("--max_restarts", type=int, default=3, help="maximum number of restarts")
parser.add_argument("--output_dir", type=str, default= "/home1/wttai/machine_learning/netket_qwz/data/", help="output directory")
args = parser.parse_args()

L = args.L
if args.N == -1:
    N = L ** 2
else:
    N = args.N
m = args.m
t = args.t
U = args.U
n_hidden = args.n_hidden
n_hidden_layers = args.n_hidden_layers
n_iter = args.n_iter
n_iter_trial = args.n_iter_trial
learning_rate = args.learning_rate
diag_shift = args.diag_shift
n_discard_per_chain = args.n_discard_per_chain
n_samples = args.n_samples
model = args.model
pbc = args.pbc
max_restarts = args.max_restarts
outputDir = args.output_dir

maxVariance = 20
restart_count = 0  # Counter to track restarts
converged = False  # Flag to check if the run converged

#outputDir = "/home1/wttai/machine_learning/netket_qwz/data/"

print("NetKet version: ", nk.__version__)

print(f"Initial parameters: m = {m}, t = {t}, U = {U}")
print(f"Particle number = {N}, L = {L}, pbc = {pbc}")

graph, hi = get_qwz_graph(L, N = N, pbc = pbc)
complex=True


H = get_qwz_Ham(hi, graph, m = m, t = t, U = U)

s, p = 1, -1

def corr_func(i):
    return nc(hi, i, s) * nc(hi, 0, s) + nc(hi, i, p) * nc(hi, 0, p)
    #return cdag_(hi, i, s) * c_(hi, 0, s) + cdag_(hi, i, p) * c_(hi, 0, p)

corrs = {}
for i in range(N):
    corrs[f"nc{i}nc0"] = corr_func(i)

if model == "slater":
    print("Using Slater determinant wave function")

    # Create the Slater determinant model
    model = LogSlaterDeterminant(hi, complex = complex)
    outputFilename=outputDir+f"slater_log_L={L}_N={N}_t={t}_m={m}_U={U}"

elif model == "nj":
    print("Using Neural Jastrow-Slater wave function")
    # Create a Neural Jastrow Slater wave function 
    model = LogNeuralJastrowSlater(hi, hidden_units=n_hidden, complex = complex, num_hidden_layers=n_hidden_layers)
    #outputFilename=outputDir+f"data/nj_log_L={L}_t={t}_m={m}_U={U}_n_hidden={n_hidden}"
    outputFilename=outputDir+f"nj_log_L={L}_N={N}_t={t}_m={m}_U={U}_n_hidden={n_hidden}_n_hidden_layers={n_hidden_layers}"

elif model == "nb":
    print("Using Neural Backflow wave function")
    model = LogNeuralBackflow(hi, hidden_units=n_hidden, complex = complex, num_hidden_layers=n_hidden_layers)
    #outputFilename=outputDir+f"data/nb_log_L={L}_t={t}_m={m}_U={U}_n_hidden={n_hidden}"
    outputFilename=outputDir+f"nb_log_L={L}_N={N}_t={t}_m={m}_U={U}_n_hidden={n_hidden}_n_hidden_layers={n_hidden_layers}"

else:
    raise ValueError("Invalid model type")

# Create the Slater determinant model
model = LogSlaterDeterminant(hi, complex=complex)

# Define the Metropolis-Hastings sampler
#sa = nk.sampler.ExactSampler(hi)
sa = nk.sampler.MetropolisLocal(hi)

# Define the optimizer
op = nk.optimizer.Sgd(learning_rate=learning_rate)

# Define a preconditioner
preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=complex)

# Function to run the VMC simulation
def run_simulation(n_iter = 50, gs = -1):
    # Create the VMC (Variational Monte Carlo) driver
    if gs == -1:
        vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
        gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)
    
    # Construct the logger to visualize the data later on
    log = nk.logging.RuntimeLog()
    
    # Run the optimization for a short number of iterations (e.g., 50)
    gs.run(n_iter=n_iter, out=log, obs=corrs)
    
    return gs, log

# Main loop for checking convergence and restarting if needed
if max_restarts != -1:
    while restart_count < max_restarts and not converged:
        gs, log = run_simulation(n_iter = n_iter_trial)
        
        print(log['Energy']['Variance'])
        # Check if the standard deviation of the energy at the last iteration is too high
        if log['Energy']['Variance'][-1] > maxVariance:
            print(f"Bad convergence detected. Restarting attempt {restart_count + 1} of {max_restarts}...")
            restart_count += 1
        else:
            converged = True
            print("Good convergence. Continuing with the full run...")
    # If the loop exits without good convergence, raise an exception
    if not converged:
        raise Exception("Failed to converge after 3 attempts. Aborting the run.")

# If converged, run the full simulation
print("Starting full simulation...")
# You can extend this part to run the full simulation for more iterations
gs, log = run_simulation(n_iter = n_iter, gs = -1 if max_restarts == -1 else gs)  # Re-run with the full iteration count
print("Full simulation completed.")


print("All done!")

print(f"Saving into {outputFilename}")

log.serialize(outputFilename)

runtimeData = json.load(open(outputFilename + ".json"))

metaData = {
    'L': L,
    'm': m,
    't': t,
    'U': U,
    'n_hidden': n_hidden,
    'n_iter': n_iter,
    'learning_rate': learning_rate,
    'diag_shift': diag_shift,
    'n_discard_per_chain': n_discard_per_chain,
    'n_samples': n_samples, 
    'pbc': pbc,
    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
}

mergedData = {
    'metadata': metaData,
    'data': runtimeData
}

with open(outputFilename + ".json", 'w') as f:
    json.dump(mergedData, f, indent=4)