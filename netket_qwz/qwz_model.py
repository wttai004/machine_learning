import netket as nk
import netket.experimental as nkx
from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc
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
from models import get_qwz_graph, get_qwz_Ham, cdag_, c_
from networks import *
from helper import get_ed_data

print("NetKet version: ", nk.__version__)

L = 4  # Side of the square
N = L ** 2

graph, hi = get_qwz_graph(L)

m = 4.1
t = 1.0
U = 0.2
s = 0
p = 1

print("Initial parameters: m = 4.1, t = 1.0, U = 0.2")

H = get_qwz_Ham(hi, graph, m = m, t = t, U = U)


def corr_func(i):
    return cdag_(hi, i, s) * c_(hi, 0, s) + cdag_(hi, i, p) * c_(hi, 0, p)

corrs = {}
for i in range(N):
    corrs[f"cdag{i}c0"] = corr_func(i)

print("Using Slater determinant wave function")

n_iter = 300
# Create the Slater determinant model
model = LogSlaterDeterminant(hi)

# Define the Metropolis-Hastings sampler
sa = nk.sampler.MetropolisExchange(hi, graph=graph)

vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)

# Define the optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Define a preconditioner
preconditioner = nk.optimizer.SR(diag_shift=0.05)

# Create the VMC (Variational Monte Carlo) driver
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)

# Construct the logger to visualize the data later on
slater_log=nk.logging.RuntimeLog()

# Run the optimization for 300 iterations
gs.run(n_iter=n_iter, out=f"data/slater_log_L={L}_m={m}", obs = corrs)

### Neural Jastrow-Slater wave function


print("Using Neural Jastrow-Slater wave function")

# Create a Neural Jastrow Slater wave function 
model = LogNeuralJastrowSlater(hi, hidden_units=N)

# Define a Metropolis exchange sampler
sa = nk.sampler.MetropolisExchange(hi, graph=graph)

# Define an optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Create a variational state
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)

# Create a Variational Monte Carlo driver
preconditioner = nk.optimizer.SR(diag_shift=0.01)
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)

# Construct the logger to visualize the data later on
nj_log=nk.logging.RuntimeLog()

# Run the optimization for 300 iterations
gs.run(n_iter=n_iter, out=f"data/nj_log_L={L}_m={m}", obs = corrs)

#### m is now 3.9

print("m is now 3.9")


m = 3.9
t = 1.0
U = 0.2
s = 0
p = 1

H = get_qwz_Ham(hi, graph, m = m, t = t, U = U)

def corr_func(i):
    return cdag_(hi, i, s) * c_(hi, 0, s) + cdag_(hi, i, p) * c_(hi, 0, p)

corrs = {}
for i in range(N):
    corrs[f"cdag{i}c0"] = corr_func(i)

n_iter = 300

print("Using Slater determinant wave function")
# Create the Slater determinant model
model = LogSlaterDeterminant(hi)

# Define the Metropolis-Hastings sampler
sa = nk.sampler.MetropolisExchange(hi, graph=graph)

vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)

# Define the optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Define a preconditioner
preconditioner = nk.optimizer.SR(diag_shift=0.05)

# Create the VMC (Variational Monte Carlo) driver
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)

# Construct the logger to visualize the data later on
slater_log=nk.logging.RuntimeLog()

# Run the optimization for 300 iterations
gs.run(n_iter=n_iter, out=f"data/slater_log_L={L}_m={m}", obs = corrs)

### Neural Jastrow-Slater wave function

print("Using Neural Jastrow-Slater wave function")

# Create a Neural Jastrow Slater wave function 
model = LogNeuralJastrowSlater(hi, hidden_units=N)

# Define a Metropolis exchange sampler
sa = nk.sampler.MetropolisExchange(hi, graph=graph)

# Define an optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Create a variational state
vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)

# Create a Variational Monte Carlo driver
preconditioner = nk.optimizer.SR(diag_shift=0.01)
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)

# Construct the logger to visualize the data later on
nj_log=nk.logging.RuntimeLog()

# Run the optimization for 300 iterations
gs.run(n_iter=n_iter, out=f"data/nj_log_L={L}_m={m}", obs = corrs)

print("All done!")