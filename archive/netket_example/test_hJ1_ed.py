import numpy as np
import netket as nk

# Define the system size
L = 4
graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=False)

# Define the Hilbert space based on this graph
hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

# Define the Sz and SzSz operators
Szs = sum([nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)])/hi.size
Szcorr = sum([nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j) for (i, j) in graph.edges()])/len(graph.edges())


# Function to create the B-field Hamiltonian
def create_hfield_hamiltonian(hilbert, hfield):
    return sum([hfield * nk.operator.spin.sigmaz(hilbert, i) for i in range(hilbert.size)])

# Define the Heisenberg Hamiltonian
ha_heisenberg = nk.operator.Heisenberg(hilbert=hi, graph=graph)

# Function to compute ground-state energy and expectation values
def compute_expectation_values(hfield):
    # Create the total Hamiltonian
    ha_hfield = create_hfield_hamiltonian(hi, hfield)
    ha = ha_heisenberg + ha_hfield
    
    # Compute the ground-state energy and wavefunction
    evals, evecs = nk.exact.lanczos_ed(ha, k=1, compute_eigenvectors=True)
    exact_gs_energy = evals[0]
    gs = evecs[:, 0]
    
    # Compute the expectation values
    Sz_expectation = gs @ (Szs @ gs)
    SzSz_expectation = gs @ (Szcorr @ gs)
    
    return exact_gs_energy, Sz_expectation, SzSz_expectation

# Range of magnetic field values to explore
hfield_values = np.linspace(-2, 2, 11)
expectation_values = []

# Loop over magnetic field values and compute expectation values
for hfield in hfield_values:
    energy, Sz, SzSz = compute_expectation_values(hfield)
    expectation_values.append((hfield, energy, Sz, SzSz))

# Convert the results to a numpy array for easier handling
expectation_values = np.array(expectation_values)

# Print the results
for (hfield, energy, Sz, SzSz) in expectation_values:
    print(f"hfield={hfield:.2f}, E0={energy:.6f}, <Sz>={Sz:.6f}, <SzSz>={SzSz:.6f}")

# Optionally, plot the results if you have matplotlib installed
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(expectation_values[:, 0], expectation_values[:, 2], label="<Sz>")
plt.xlabel("hfield")
plt.ylabel("<Sz>")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(expectation_values[:, 0], expectation_values[:, 3], label="<SzSz>")
plt.xlabel("hfield")
plt.ylabel("<SzSz>")
plt.legend()

plt.tight_layout()
plt.show()
