import netket as nk
import netket.experimental as nkx
from scipy.sparse.linalg import eigsh
import numpy as np
import scipy.sparse.linalg
import jax
import jax.numpy as jnp
import json
import matplotlib.pyplot as plt

import sys, os
sys.path.append('/Users/wttai/Documents/Jupyter/machine_learning/common_lib')
sys.path.append('/home1/wttai/machine_learning/common_lib')
from models import get_qwz_graph, get_qwz_Ham, cdag_, c_, nc_
from networks import *
from helper import get_ed_data

print("Program running...", flush = True) 

home_dir = os.path.expanduser('~')
##### PROGRAM START #####
from argparse import ArgumentParser

# parse command-line arguments
parser = ArgumentParser()

parser.add_argument("--L", type=int, default=4, help="Side of the square")
parser.add_argument("--m", type=float, default=5.0, help="mass term in the Hamiltonian")
parser.add_argument("--t", type=float, default=1.0, help="hopping term in the Hamiltonian")
parser.add_argument("--U", type=float, default=0.2, help="interaction term in the Hamiltonian")
parser.add_argument("--n_iter", type=int, default=300, help="number of iterations")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--diag_shift", type=float, default=0.05, help="diagonal shift in the Stochastic Reconfiguration method")
parser.add_argument("--n_discard_per_chain", type=int, default=16, help="number of samples to discard per chain")
parser.add_argument("--n_samples", type=int, default=2**12, help="number of samples")
parser.add_argument("--model", type=str, default="slater", help="model to use: slater or nj")
parser.add_argument("--pbc",  dest="pbc", help="periodic boundary conditions", action="store_true")
args = parser.parse_args()

L = args.L
N = L ** 2
m = args.m
t = args.t
U = args.U
n_iter = args.n_iter
learning_rate = args.learning_rate
diag_shift = args.diag_shift
n_discard_per_chain = args.n_discard_per_chain
n_samples = args.n_samples
model = args.model
pbc = args.pbc

print("NetKet version: ", nk.__version__)

graph, hi = get_qwz_graph(L, N = N, pbc = pbc)
s = 0
p = 1

print(f"Initial parameters: m = {m}, t = {t}, U = {U}")

H = get_qwz_Ham(hi, graph, m = m, t = t, U = U)


def corr_func(i):
    return nc_(hi, i, s) * nc_(hi, 0, s) + nc_(hi, i, p) * nc_(hi, 0, p)
    #return cdag_(hi, i, s) * c_(hi, 0, s) + cdag_(hi, i, p) * c_(hi, 0, p)

corrs = {}
for i in range(N):
    corrs[f"cdag{i}c0"] = corr_func(i)

if model == "slater":
    print("Using Slater determinant wave function")

    # Create the Slater determinant model
    model = LogSlaterDeterminant(hi)
    outputFilename=f"data/slater_log_L={L}_t={t}_m={m}_U={U}"

elif model == "nj":
    print("Using Neural Jastrow-Slater wave function")

    # Create a Neural Jastrow Slater wave function 
    model = LogNeuralJastrowSlater(hi, hidden_units=N)
    outputFilename=f"data/nj_log_L={L}_t={t}_m={m}_U={U}"

else:
    raise ValueError("Invalid model type")
# Define the Metropolis-Hastings sampler
sa = nk.sampler.MetropolisExchange(hi, graph=graph)

vstate = nk.vqs.MCState(sa, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)

# Define the optimizer
op = nk.optimizer.Sgd(learning_rate=learning_rate)

# Define a preconditioner
preconditioner = nk.optimizer.SR(diag_shift=diag_shift)

# Create the VMC (Variational Monte Carlo) driver
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)

# Construct the logger to visualize the data later on
slater_log=nk.logging.RuntimeLog()

# Run the optimization for 300 iterations
gs.run(n_iter=n_iter, out=outputFilename, obs = corrs)


print("All done!")