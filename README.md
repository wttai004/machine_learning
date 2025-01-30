# Neural Quantum State simulation on fermionic systems

(Ongoing work—for personal reference)

This project is a work in progress that applies Neural Quantum States (NQS) to fermionic systems to obtain its ground state. The code simulates the [Qi-Wu-Zhang model](https://arxiv.org/abs/cond-mat/0505308) on a 2D lattice and computes its ground state energy and correlation functions. Currently, the simulation works well for $U = 0$ but demonstrates questionable convergence for higher $U$. Further work is needed for diagnosis.

Data is not uploaded here due to storage constraints.

Organization:

common_lib: contains helper functions for simulation

dmrg_qwz: simulation using Density Matrix Renormalization Group (DMRG) implemented in [TenPy](https://tenpy.readthedocs.io/en/latest/) to benchmark ground state behavior.

netket_qwz: simulation using NQS, implemented using [NetKet](https://www.netket.org). Currently, it implements a Slater Determinant model for benchmarking, as well as the Neural Jastrow and Neural Backflow Ansätze. For the latter two, a feed-forward neural network with tunable layer count and hidden neurons is implemented. 
