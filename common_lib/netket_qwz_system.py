import netket as nk
import netket.experimental as nkx
from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc
import numpy as np

s = 0
p = 1

class NetketQWZSystem:
    def __init__(self, L, args, L2=-1, N=-1, pbc=False):
        self.L = L
        self.args = args
        self.L2 = L2 if L2 != -1 else L
        self.N = N if N != -1 else L * self.L2
        self.pbc = pbc
        self.graph, self.hi = self._define_graph_and_hilbert()

    def _index(self, row, col, spin=0):
        return spin * self.L * self.L2 + (row * self.L + col)

    def _define_graph_and_hilbert(self):
        edge_colors = []
        for row in range(self.L2):
            for col in range(self.L - 1):
                edge_colors.append([self._index(row, col), self._index(row, col + 1), 1])
            if self.pbc:
                edge_colors.append([self._index(row, self.L - 1), self._index(row, 0), 3])

        for col in range(self.L):
            for row in range(self.L2 - 1):
                edge_colors.append([self._index(row, col), self._index(row + 1, col), 2])
            if self.pbc:
                edge_colors.append([self._index(self.L2 - 1, col), self._index(0, col), 4])

        graph = nk.graph.Graph(edges=edge_colors)
        hilbert = nkx.hilbert.SpinOrbitalFermions(graph.n_nodes, s=1 / 2, n_fermions=self.N)
        return graph, hilbert

    def get_exchange_graph(self):
        edges = []
        for spin in range(2):
            for row in range(self.L2):
                for col in range(self.L - 1):
                    edges.append([self._index(row, col, spin), self._index(row, col + 1, spin)])
                if self.pbc:
                    edges.append([self._index(row, self.L - 1, spin), self._index(row, 0, spin)])
            for col in range(self.L):
                for row in range(self.L2 - 1):
                    edges.append([self._index(row, col, spin), self._index(row + 1, col, spin)])
                if self.pbc:
                    edges.append([self._index(self.L2 - 1, col, spin), self._index(0, col, spin)])
        for row in range(self.L2):
            for col in range(self.L):
                edges.append([self._index(row, col, 0), self._index(row, col, 1)])

        return nk.graph.Graph(edges=edges)

    def get_hubbard_hamiltonian(self):
        H = 0.0 + 0.0j
        t = self.args['t']
        U = self.args['U']
        for (i, j) in self.graph.edges():
            for spin in [-1, 1]:
                H += t * (cdag(self.hi, i, spin) * c(self.hi, j, spin) + cdag(self.hi, j, spin) * c(self.hi, i, spin))
        for i in self.graph.nodes():
            H += U * (nc(self.hi, i, 1) + nc(self.hi, i, -1) - 1)**2
        return H

    def get_qwz_hamiltonian(self):
        H = 0.0 + 0.0j
        s, p = 1, -1

        m = self.args['m']
        t = self.args['t']
        U = self.args['U']
        bias = self.args['bias']

        for i in self.graph.nodes():
            H += m * (nc(self.hi, i, s) - nc(self.hi, i, p))
            H += U * nc(self.hi, i, s) * nc(self.hi, i, p)
        for (i, j), color in zip(self.graph.edges(), self.graph.edge_colors):
            H += t * (cdag(self.hi, i, s) * c(self.hi, j, s) + cdag(self.hi, j, s) * c(self.hi, i, s) 
                      - cdag(self.hi, i, p) * c(self.hi, j, p) - cdag(self.hi, j, p) * c(self.hi, i, p))
            if color == 1:
                H += 1j * t * (cdag(self.hi, i, s) * c(self.hi, j, p) - cdag(self.hi, j, p) * c(self.hi, i, s)
                            + cdag(self.hi, i, p) * c(self.hi, j, s) - cdag(self.hi, j, s) * c(self.hi, i, p))
            elif color == 2:
                H += t * (cdag(self.hi, i, s) * c(self.hi, j, p) + cdag(self.hi, j, p) * c(self.hi, i, s)
                          -cdag(self.hi, i, p) * c(self.hi, j, s) - cdag(self.hi, j, s) * c(self.hi, i, p))

            elif color == 3:
                # x direction hopping for periodic boundary condition
                H += 1j * t * (cdag(self.hi, j, s) * c(self.hi, i, p) - cdag(self.hi, i, p) * c(self.hi, j, s)
                    + cdag(self.hi, j, p) * c(self.hi, i, s) - cdag(self.hi, i, s) * c(self.hi, j, p))
                
            elif color == 4:
                # y direction hopping for periodic boundary condition
                H += t * (cdag(self.hi, j, s) * c(self.hi, i, p) + cdag(self.hi, i, p) * c(self.hi, j, s)
                        - cdag(self.hi, j, p) * c(self.hi, i, s) - cdag(self.hi, i, s) * c(self.hi, j, p))
            else:
                raise ValueError(f"Invalid edge color {color}")
        H += bias * nc(self.hi, 0, p)
        return H

    def corr_func(self, i, o0, o1):
        delta_x, delta_y = i // self.L, i % self.L
        sum_corr = 0
        for x in range(self.L):
            for y in range(self.L):
                site0 = self._index(x, y)
                if not self.pbc and (x + delta_x >= self.L or y + delta_y >= self.L):
                    continue
                site1 = self._index((x + delta_x) % self.L, (y + delta_y) % self.L)
                sum_corr += nc(self.hi, site0, o0) * nc(self.hi, site1, o1)
        return sum_corr
