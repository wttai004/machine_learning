import netket as nk
import netket.experimental as nkx
from netket.experimental.operator.fermion import destroy as c
from netket.experimental.operator.fermion import create as cdag
from netket.experimental.operator.fermion import number as nc
import numpy as np
import scipy
from abc import ABC, abstractmethod

class NetketSystem:
    def __init__(self, L, args, L2=-1, N=-1, pbc=False):
        self.L = L
        self.args = args
        self.L2 = L2 if L2 != -1 else L
        self.N = N if N != -1 else L * self.L2
        self.pbc = pbc
        self.graph, self.hi = self._define_graph_and_hilbert()
        self.Ham = None

    def _index(self, row, col, spin=0):
        return spin * self.L * self.L2 + (row * self.L + col)
    
    @abstractmethod
    def _define_graph_and_hilbert(self):
        pass

    @abstractmethod
    def get_hamiltonian(self):
        pass

    @abstractmethod
    def get_exchange_graph(self):
        pass

    def get_ed_data(self, k = 2):
        # Convert the Hamiltonian to a sparse matrix
        if self.Ham is None:
            raise ValueError("Hamiltonian not defined")
        sp_h = self.Ham.to_sparse()

        eig_vals, eig_vecs = scipy.sparse.linalg.eigsh(sp_h, k=k, which="SA")

        sorted_indices = eig_vals.argsort()
        eig_vals = eig_vals[sorted_indices]
        eig_vecs = eig_vecs[:, sorted_indices]

        E_gs = eig_vals[0]

        print("Exact ground state energy:", E_gs)
        return eig_vals, eig_vecs


class NetketQWZSystem(NetketSystem):
    def __init__(self, L, args, L2=-1, N=-1, pbc=False):
        self.L = L
        self.args = args
        self.L2 = L2 if L2 != -1 else L
        self.N = N if N != -1 else L * self.L2
        self.pbc = pbc
        self.graph, self.hi = self._define_graph_and_hilbert()
        self.Ham = None

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
        #for spin in range(int(2 * self.hi.s + 1)):
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

    def get_hamiltonian(self):
        H = 0.0 + 0.0j
        s, p = 1, -1

        m = self.args['m']
        t = self.args['t']
        U = self.args['U']
        bias = self.args.get('bias', 0)
        complex  = self.args.get('complex', True)

        for i in self.graph.nodes():
            H += m * (nc(self.hi, i, s) - nc(self.hi, i, p))
            H += U * nc(self.hi, i, s) * nc(self.hi, i, p)
        for (i, j), color in zip(self.graph.edges(), self.graph.edge_colors):
            H += t * (cdag(self.hi, i, s) * c(self.hi, j, s) + cdag(self.hi, j, s) * c(self.hi, i, s) 
                      - cdag(self.hi, i, p) * c(self.hi, j, p) - cdag(self.hi, j, p) * c(self.hi, i, p))
            if color == 1:
                if complex:
                    H += 1j * t * (cdag(self.hi, i, s) * c(self.hi, j, p) - cdag(self.hi, j, p) * c(self.hi, i, s)
                                + cdag(self.hi, i, p) * c(self.hi, j, s) - cdag(self.hi, j, s) * c(self.hi, i, p))
                else:
                    H += t * (cdag(self.hi, i, s) * c(self.hi, j, p) + cdag(self.hi, j, p) * c(self.hi, i, s)
                                - cdag(self.hi, i, p) * c(self.hi, j, s) - cdag(self.hi, j, s) * c(self.hi, i, p))
            elif color == 2:
                H += t * (cdag(self.hi, i, s) * c(self.hi, j, p) + cdag(self.hi, j, p) * c(self.hi, i, s)
                          -cdag(self.hi, i, p) * c(self.hi, j, s) - cdag(self.hi, j, s) * c(self.hi, i, p))

            elif color == 3:
                # x direction hopping for periodic boundary condition
                if complex: 
                    H += 1j * t * (cdag(self.hi, j, s) * c(self.hi, i, p) - cdag(self.hi, i, p) * c(self.hi, j, s)
                        + cdag(self.hi, j, p) * c(self.hi, i, s) - cdag(self.hi, i, s) * c(self.hi, j, p))
                else:
                    H += t * (cdag(self.hi, j, s) * c(self.hi, i, p) + cdag(self.hi, i, p) * c(self.hi, j, s)
                                - cdag(self.hi, j, p) * c(self.hi, i, s) - cdag(self.hi, i, s) * c(self.hi, j, p))
                
            elif color == 4:
                # y direction hopping for periodic boundary condition
                H += t * (cdag(self.hi, j, s) * c(self.hi, i, p) + cdag(self.hi, i, p) * c(self.hi, j, s)
                        - cdag(self.hi, j, p) * c(self.hi, i, s) - cdag(self.hi, i, s) * c(self.hi, j, p))
            else:
                raise ValueError(f"Invalid edge color {color}")
        H += bias * nc(self.hi, 0, p)
        self.Ham = H
        return H

    def corr_func(self, i, o0, o1):
        # Unfold the shift i into (delta_y, delta_x)
        delta_y = i // self.L
        delta_x = i % self.L

        sum_corr = 0.0
        for y in range(self.L2):
            for x in range(self.L):
                site0 = self._index(y, x)  # row=y, col=x
                if not self.pbc and (y + delta_y >= self.L2 or x + delta_x >= self.L):
                    continue

                site1 = self._index((y + delta_y) % self.L2,(x + delta_x) % self.L,)
                sum_corr += nc(self.hi, site0, o0) * nc(self.hi, site1, o1)

        return sum_corr



class NetketHubbardSystem(NetketSystem):
    # This is a spinless Hubbard system with nearest neighbor interaction
    def _define_graph_and_hilbert(self):
        #graph = nk.graph.Grid((self.L, self.L2), pbc = self.pbc)
        edge_colors = []
        for row in range(self.L2):
            for col in range(self.L - 1):
                edge_colors.append([self._index(row, col), self._index(row, col + 1)])
            if self.pbc:
                edge_colors.append([self._index(row, self.L - 1), self._index(row, 0)])

        for col in range(self.L):
            for row in range(self.L2 - 1):
                edge_colors.append([self._index(row, col), self._index(row + 1, col)])
            if self.pbc:
                edge_colors.append([self._index(self.L2 - 1, col), self._index(0, col)])

        graph = nk.graph.Graph(edges=edge_colors)
        hilbert = nkx.hilbert.SpinOrbitalFermions(graph.n_nodes, s=None, n_fermions=self.N)
        #print(f"L = {self.L}, L2 = {self.L2}, Graph info: {graph.edges()}")
        return graph, hilbert
    
    def get_exchange_graph(self):
        return self.graph

    def get_hamiltonian(self):
        H = 0.0 + 0.0j
        t = self.args['t']
        U = self.args['U']
        bias = self.args.get('bias', 0)
        for (i, j) in self.graph.edges():
            H += t * (cdag(self.hi, i) * c(self.hi, j) + cdag(self.hi, j) * c(self.hi, i))
            H += U * nc(self.hi, i) * nc(self.hi, j) 
        H += bias * nc(self.hi, 0)
        self.Ham = H
        return H

    def corr_func(self, i):
        # Unfold the shift i into (delta_y, delta_x)
        delta_y = i // self.L
        delta_x = i % self.L

        sum_corr = 0.0
        for y in range(self.L2):
            for x in range(self.L):
                site0 = self._index(y, x)  # row=y, col=x
                if not self.pbc and (y + delta_y >= self.L2 or x + delta_x >= self.L):
                    continue

                site1 = self._index((y + delta_y) % self.L2,(x + delta_x) % self.L,)
                sum_corr += nc(self.hi, site0) * nc(self.hi, site1)

        return sum_corr
