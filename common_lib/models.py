import netket as nk
import netket.experimental as nkx
from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc

s = 0
p = 1

def get_qwz_graph(L, L2 = -1, N = -1, pbc = False):
    # Define the lattice
    if L2 == -1:
        L2 = L
    def index(row, col):
        return (row * L + col) 
    edge_colors = []

    # Generate horizontal edges
    for row in range(L2):
        for col in range(L-1):
            edge_colors.append([index(row, col), index(row, col+1), 1])
        # Wrap-around edge in each row
        if pbc:
            edge_colors.append([index(row, L-1), index(row, 0), 1])

    # Generate vertical edges
    for col in range(L):
        for row in range(L2-1):
            edge_colors.append([index(row, col), index(row+1, col), 2])
        # Wrap-around edge in each column
        if pbc:
            edge_colors.append([index(L2-1, col), index(0, col), 2])
    # Define the netket graph object
    graph = nk.graph.Graph(edges=edge_colors)

    N_f = graph.n_nodes

    if N == -1:
        N = L * L2
    hi = nkx.hilbert.SpinOrbitalFermions(N_f, s=1/2, n_fermions=N)

    return graph, hi

def get_qwz_exchange_graph(L, L2 = -1, pbc = False):
    # This defines the exchange graph for the purpose of the Metropolis Exchange algorithm
    # Unlike the get_qwz_graph, this explicitly considers both orbitals and orbital exchanges (represented as spin flip). 
    if L2 == -1:
        L2 = L
    def index(row, col, s):
        return s * L * L2 + (row * L + col)
    edges = []

    # Generate horizontal edges
    for spin in range(2):
        for row in range(L2):
            for col in range(L-1):
                edges.append([index(row, col, spin), index(row, col+1, spin)])
            # Wrap-around edge in each row
            if pbc:
                edges.append([index(row, L-1, spin), index(row, 0, spin)])

        # Generate vertical edges
        for col in range(L):
            for row in range(L2-1):
                edges.append([index(row, col, spin), index(row+1, col, spin)])
            # Wrap-around edge in each column
            if pbc:
                edges.append([index(L2-1, col, spin), index(0, col, spin)])

    # Add spin flip edges
    for row in range(L2):
        for col in range(L):
            edges.append([index(row, col, 0), index(row, col, 1)])
    # Define the netket graph object
    graph = nk.graph.Graph(edges=edges)
    return graph

def get_hubbard_Ham(hi, graph, t = 1.0, U = 1.0):
    #Note that here, s and p actually means the spins 
    H = 0.0 + 0.0j
    for (i, j) in graph.edges():
        for spin in [-1, 1]:
            H += t * (cdag(hi, i, spin) * c(hi, j, spin) + cdag(hi, j, spin) * c(hi, i, spin))

    for i in graph.nodes():
        H += U * (nc(hi, i, 1) + nc(hi, i, -1) - 1) * (nc(hi, i, 1) + nc(hi, i, -1) - 1)
    return H

def get_qwz_Ham(hi, graph, m = 1.0, t = 1.0, U = 1.0):
    s, p = 1, -1

    H = 0.0 + 0.0j

    for i in graph.nodes():
        H += m * (nc(hi, i, s) - nc(hi, i, p))
        H += U * nc(hi, i, s) * nc(hi, i, p)

    for (i, j), color in zip(graph.edges(), graph.edge_colors):
        H +=  t * (cdag(hi, i, s) * c(hi, j, s) + cdag(hi, j, s) * c(hi, i, s) - cdag(hi, i, p) * c(hi, j, p) - cdag(hi ,j, p) * c(hi ,i, p))
        if color == 1:
            # x direction hopping
            H += 1j * t * (cdag(hi, i, s) * c(hi, j, p) - cdag(hi, j, p) * c(hi, i, s)
                        + cdag(hi, i, p) * c(hi, j, s) - cdag(hi, j, s) * c(hi, i, p))
        elif color == 2:
            # y direction hopping
            H += t * (cdag(hi, i, s) * c(hi, j, p) + cdag(hi, j, p) * c(hi, i, s)
                     - cdag(hi, i, p) * c(hi, j, s) - cdag(hi, j, s) * c(hi, i, p))
        else:
            raise ValueError("Invalid color")
    return H

def get_debug_Ham(hi, graph, m = 1.0, t = 1.0, U = 1.0):
    #This is the qwz hamiltonian but modified so that it is real
    #It probably means nothing physically relevant
    s, p = 1, -1

    H = 0.0 + 0.0j

    for i in graph.nodes():
        H += m * (nc(hi, i, s) - nc(hi, i, p))
        H += U * nc(hi, i, s) * nc(hi, i, p)

    for (i, j), color in zip(graph.edges(), graph.edge_colors):
        H +=  t * (cdag(hi, i, s) * c(hi, j, s) + cdag(hi, j, s) * c(hi, i, s) - cdag(hi, i, p) * c(hi, j, p) - cdag(hi ,j, p) * c(hi ,i, p))
        if color == 1:
            # x direction hopping
            H += t * (cdag(hi, i, s) * c(hi, j, p) + cdag(hi, j, p) * c(hi, i, s)
                        - cdag(hi, i, p) * c(hi, j, s) - cdag(hi, j, s) * c(hi, i, p))
        elif color == 2:
            # y direction hopping
            H += t * (cdag(hi, i, s) * c(hi, j, p) + cdag(hi, j, p) * c(hi, i, s)
                     - cdag(hi, i, p) * c(hi, j, s) - cdag(hi, j, s) * c(hi, i, p))
        else:
            raise ValueError("Invalid color")
    return H
