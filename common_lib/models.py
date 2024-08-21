import netket as nk
import netket.experimental as nkx
from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc

s = 0
p = 1

def get_qwz_graph(L):
    def index(row, col):
        return (row * L + col) 
    edge_colors = []
    s = 0
    p = 1

    # Generate horizontal edges
    for row in range(L):
        for col in range(L-1):
            edge_colors.append([index(row, col), index(row, col+1), 1])
        # Wrap-around edge in each row
        edge_colors.append([index(row, L-1), index(row, 0), 1])

    # Generate vertical edges
    for col in range(L):
        for row in range(L-1):
            edge_colors.append([index(row, col), index(row+1, col), 2])
        # Wrap-around edge in each column
        edge_colors.append([index(L-1, col), index(0, col), 2])
    # Define the netket graph object
    graph = nk.graph.Graph(edges=edge_colors)

    N_f = graph.n_nodes

    hi = nkx.hilbert.SpinOrbitalFermions(N_f * 2, s=None, n_fermions=N_f)

    return graph, hi

def c_(hi, i, o):
    #Each site has two orbitals o = 0, 1
    return c(hi, 2*i+o)

def cdag_(hi, i, o):
    return cdag(hi, 2*i+o)

def nc_(hi, i, o):
    return nc(hi, 2*i+o)


m = 4.1
t = 1.0
U = 0.2
s = 0
p = 1
def get_qwz_Ham(hi, graph, m = 1.0, t = 1.0, U = 1.0):

    H = 0.0 + 0.0j

    for i in graph.nodes():
        H += m * (nc_(hi, i, 0) - nc_(hi, i, 1))
        H += U * nc_(hi, i, s) * nc_(hi, i, p)

    for (i, j), color in zip(graph.edges(), graph.edge_colors):
        H +=  t/2 * (cdag_(hi, i, s) * c_(hi, j, s) + cdag_(hi, j, s) * c_(hi, i, s) - cdag_(hi, i, p) * c_(hi, j, p) - cdag_(hi ,j, p) * c_(hi ,i, p))
        if color == 1:
            # x direction hopping
            H += 1j * t/2 * (cdag_(hi, i, s) * c_(hi, j, p) - cdag_(hi, j, p) * c_(hi, i, s)
                        + cdag_(hi, i, p) * c_(hi, j, s) - cdag_(hi, j, s) * c_(hi, i, p))
        elif color == 2:
            # y direction hopping
            H += t/2 * (cdag_(hi, i, s) * c_(hi, j, p) - cdag_(hi, i, p) * c_(hi, j, s) 
                    + cdag_(hi, j, p) * c_(hi, i, s) - cdag_(hi, j, s) * c_(hi, i, p))
        
        else:
            raise ValueError("Invalid color")
    return H
