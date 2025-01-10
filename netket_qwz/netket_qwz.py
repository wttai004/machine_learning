import netket as nk
import netket.experimental as nkx
from scipy.sparse.linalg import eigsh
import numpy as np
import scipy.sparse.linalg
import jax
import jax.numpy as jnp
import json, fcntl
import time
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

import sys, os
sys.path.append('/Users/wttai/Documents/Jupyter/machine_learning/common_lib')
sys.path.append('/home1/wttai/machine_learning/common_lib')
#from models_old import get_qwz_graph, get_qwz_Ham, get_qwz_exchange_graph, cdag, c, nc
from netket_qwz_system import NetketQWZSystem
from networks import *
from helper import get_ed_data

print("Program running...", flush = True) 

home_dir = os.path.expanduser('~')
##### PROGRAM START #####
from argparse import ArgumentParser

# parse command-line arguments
parser = ArgumentParser()

parser.add_argument("--L", type=int, default=2, help="Side of the square")
parser.add_argument("--L2", type=int, default=-1, help="Side of the rectangle (if using rectangle)")
parser.add_argument("--N" , type=int, default=-1, help="Number of particles")
parser.add_argument("--N_frac", type=float, default=-1, help="Fraction of particles (default to half-filling)")
parser.add_argument("--m", type=float, default=5.0, help="mass term in the Hamiltonian")
parser.add_argument("--t", type=float, default=1.0, help="hopping term in the Hamiltonian")
parser.add_argument("--U", type=float, default=0.2, help="interaction term in the Hamiltonian")
parser.add_argument("--n_iter", type=int, default=10, help="number of iterations")
parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--diag_shift", type=float, default=0.01, help="diagonal shift in the Stochastic Reconfiguration method")
parser.add_argument("--n_discard_per_chain", type=int, default=16, help="number of samples to discard per chain")
parser.add_argument("--n_samples", type=int, default=2**12, help="number of samples")
parser.add_argument("--model", type=str, default="slater", help="model to use: slater or nj")
parser.add_argument("--pbc",  dest="pbc", help="periodic boundary conditions", action="store_true")
parser.add_argument("--n_hidden", type=int, default=8, help="number of hidden units in the Neural Jastrow/Backflow model")
parser.add_argument("--n_hidden_layers", type=int, default=1, help="number of hidden layers in the Neural Jastrow/Backflow model")
parser.add_argument("--n_iter_trial", type=int, default=100, help="number of iterations attempted to check convergence")
parser.add_argument("--max_restarts", type=int, default=-1, help="maximum number of restarts")
parser.add_argument("--output_dir", type=str, default= "/home1/wttai/machine_learning/netket_qwz/data/", help="output directory")
parser.add_argument("--bias", type=float, default=1e-5, help="bias term in the Hamiltonian")
parser.add_argument("--n_runs", type=int, default=1, help="number of runs")
parser.add_argument("--real", dest="real", help="use real Hamiltonian and model", action="store_false")
parser.add_argument("--test_exact_energy", dest="test_exact_energy", help="test the exact energy (only for small systems)", action="store_true")
parser.add_argument("--exact_sampling", dest="exact_sampling", help="use exact sampling instead of Metropolis exchange (only for small systems)", action="store_true")

parser.add_argument("--create_database", dest="create_database", help="create a database", action="store_true")
parser.add_argument("--database_name", type=str, default="database", help="database directory")
parser.add_argument("--job_id", type=int, default=0, help="job id")
args = parser.parse_args()

L = args.L
L2 = L if args.L2 == -1 else args.L2
N = args.N
N_frac = args.N_frac
if N != -1 and N_frac != -1:
    raise ValueError("Cannot specify both N and N_frac")
if N == -1 and N_frac == -1:
    N = L * L2
if N_frac != -1:
    N = int(2 * N_frac * L * L2)

if N > L * L2:
    raise ValueError("Current code is not good for more than half-filling")

m = args.m
t = args.t
U = args.U
bias = args.bias
n_hidden = args.n_hidden
n_hidden_layers = args.n_hidden_layers
n_iter = args.n_iter
n_iter_trial = args.n_iter_trial
learning_rate = args.learning_rate
diag_shift = args.diag_shift
n_discard_per_chain = args.n_discard_per_chain
n_samples = args.n_samples
model_name = args.model
pbc = args.pbc
max_restarts = args.max_restarts
n_runs = args.n_runs
outputDir = args.output_dir
create_database = args.create_database
database_name = args.database_name
job_id = args.job_id
complex = args.real
test_exact_energy = args.test_exact_energy
exact_sampling = args.exact_sampling

maxVariance = 50

#outputDir = "/home1/wttai/machine_learning/netket_qwz/data/"

print("NetKet version: ", nk.__version__, flush = True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"Starting run at {timestamp}", flush = True)

print(f"Initial parameters: m = {m}, t = {t}, U = {U}", flush = True)
if L2 != L:
    print(f"Rectangle dimensions: L = {L}, L2 = {L2}", flush = True)
else:
    print(f"Square side: L = {L}", flush = True)
print(f"Particle number = {N}, pbc = {pbc}", flush = True)

s, p = 1, -1
#test_exact_energy = True

args = {'U': U, 't': t, 'm': m, 'bias': bias, 'complex': complex}

def run_simulation(run_id = 1):
    print("_" * 50) 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_time = datetime.now()
    print(f"Starting run {run_id} of {n_runs} at {timestamp}", flush = True)

    restart_count = 0  # Counter to track restarts
    converged = False  # Flag to check if the run converged

    system = NetketQWZSystem(L, L2 = L2, N = N, pbc = pbc, args = args)

    hi = system.hi
    exchange_graph = system.get_exchange_graph()

    Ham = system.get_qwz_hamiltonian()

    if test_exact_energy or exact_sampling:
        if hi.n_states > 1e5:
            warnings.warn(f"The Hilbert space dimension might be too large for exact methods: {hi.n_states}", UserWarning)


    if test_exact_energy:
        evals, evecs = get_ed_data(Ham, k = 6)
        E_gs = evals[0]
        print(f"Exact ground state energy: {E_gs}", flush = True)



    s, p = 1, -1
    #dummy_array = [0 for i in range(L**2)]
    corrs = {'ss': [0 for i in range(L*L2)], 'sp': [0 for i in range(L*L2)], 'pp': [0 for i in range(L*L2)]}

    for i in range(L*L2):
        corrs['pp'][i] = system.corr_func(i, p, p)
        corrs['sp'][i] = system.corr_func(i, s, p)
        corrs['ss'][i] = system.corr_func(i, s, s)
        #corrs[f"nc{i}nc0"] = corr_func(i)
    
    if L2 != L:
        physicalSystemDir = f"L={L}_L2={L2}_N={N}_t={t}_m={m}_U={U}_{"pbc" if pbc else "obc"}{"" if complex else "_real"}/"
    else:
        physicalSystemDir = f"L={L}_N={N}_t={t}_m={m}_U={U}_{"pbc" if pbc else "obc"}{"" if complex else "_real"}/"

    if model_name == "slater":
        print("Using Slater determinant wave function", flush = True)
        # Create the Slater determinant model
        model = LogSlaterDeterminant(hi, complex = complex)
        outputFilename = (outputDir + physicalSystemDir + 
                          f"slater_log_" 
                          + (f"exact_sampling" if exact_sampling else f"n_samples={n_samples}") 
                          + f"_run={run_id}")

    elif model_name == "nj":
        print("Using Neural Jastrow-Slater wave function", flush = True)
        # Create a Neural Jastrow Slater wave function 
        model = LogNeuralJastrowSlater(hi, hidden_units=n_hidden, complex = complex, num_hidden_layers=n_hidden_layers)
        outputFilename = (outputDir + physicalSystemDir + 
                          f"nj_log_n_hidden={n_hidden}_n_hidden_layers={n_hidden_layers}_" 
                          + (f"exact_sampling" if exact_sampling else f"n_samples={n_samples}") 
                          + f"_run={run_id}")

    elif model_name == "nb":
        print("Using Neural Backflow wave function", flush = True)
        model = LogNeuralBackflow(hi, hidden_units=n_hidden, complex = complex, num_hidden_layers=n_hidden_layers)
        outputFilename = (outputDir + physicalSystemDir + 
                          f"nb_log_n_hidden={n_hidden}_n_hidden_layers={n_hidden_layers}_" 
                          + (f"exact_sampling" if exact_sampling else f"n_samples={n_samples}") 
                          + f"_run={run_id}")

    else:
        raise ValueError("Invalid model type")

    os.makedirs(outputDir + physicalSystemDir, exist_ok=True)

    print(f"Output will be saved to {outputFilename}", flush=True)

    # Create the Slater determinant model
    model = LogSlaterDeterminant(hi, complex=complex)

    # Define the Metropolis-Hastings sampler
    if exact_sampling:
        sa = nk.sampler.ExactSampler(hi)
    else:
        sa = nk.sampler.MetropolisExchange(hi, graph=exchange_graph)

    # Define the optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)

    # Define a preconditioner
    preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=complex)

    # Function to run the VMC simulation
    def run_simulation(n_iter = 50, gs = -1):
        # Create the VMC (Variational Monte Carlo) driver
        if gs == -1:
            vstate = nk.vqs.MCState(sa, model, n_samples=2**12, n_discard_per_chain=16)
            gs = nk.VMC(Ham, op, variational_state=vstate, preconditioner=preconditioner)
        
        # Construct the logger to visualize the data later on
        log = nk.logging.RuntimeLog()
        
        # Run the optimization for a short number of iterations (e.g., 50)
        gs.run(n_iter=n_iter, out=log, obs=corrs)
        
        return gs, log

    # Main loop for checking convergence and restarting if needed
    if max_restarts != -1:
        while restart_count < max_restarts and not converged:
            gs, log = run_simulation(n_iter = n_iter_trial)
            
            #print(log['Energy']['Variance'])
            # Check if the standard deviation of the energy at the last iteration is too high
            if log['Energy']['Variance'][-1] > maxVariance:
                print(f"Bad convergence detected. Restarting attempt {restart_count + 1} of {max_restarts}...", flush = True)
                restart_count += 1
            else:
                converged = True
                print("Good convergence. Continuing with the full run...", flush = True)
        # If the loop exits without good convergence, raise an exception
        if not converged:
            raise Exception("Failed to converge after 3 attempts. Aborting the run.", flush = True)

    # If converged, run the full simulation
    print("Starting full simulation...", flush = True)
    # You can extend this part to run the full simulation for more iterations
    gs, log = run_simulation(n_iter = n_iter, gs = -1 if max_restarts == -1 else gs)  # Re-run with the full iteration count
    print("Full simulation completed.", flush = True)


    print("All done!", flush = True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"Run completed at {timestamp}", flush = True)

    print(f"Ground state energy: {log['Energy']['Mean'][-1]}", flush = True)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()

    print(f"Elapsed time in seconds: {elapsed_time.total_seconds()} seconds", flush = True)

    print(f"Saving into {outputFilename}", flush = True)

    log.serialize(outputFilename)

    runtimeData = json.load(open(outputFilename + ".json"))

    metaData = {
        'L': L,
        'L2': L2,
        'N': N,
        'm': m,
        't': t,
        'U': U,
        'bias': bias,
        'complex': complex,
        'real': not complex,
        'n_hidden': n_hidden,
        'n_hidden_layers': n_hidden_layers,
        'n_iter_trial': n_iter_trial,
        'n_iter': n_iter,
        'learning_rate': learning_rate,
        'diag_shift': diag_shift,
        'n_discard_per_chain': n_discard_per_chain,
        'n_samples': n_samples, 
        'exact_sampling': exact_sampling,
        'pbc': pbc,
        'model': model_name,
        'run_id': run_id,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'elapsed_time': elapsed_seconds,
    }

    mergedData = {
        'metadata': metaData,
        'data': runtimeData
    }

    with open(outputFilename + ".json", 'w') as f:
        json.dump(mergedData, f, indent=4)



    if create_database:
        database_location = outputDir + database_name + ".json"
        
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
        #exit(0)


for i in range(n_runs):
    run_simulation(i + 1)