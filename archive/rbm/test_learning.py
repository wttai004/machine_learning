import numpy as np
from numpy import tanh
import matplotlib.pyplot as plt
import random
from math import sqrt
from model import Model
from rbm import RBM
from rbm_operator import Operator, Sx_, Sy_, Sz_, SzSz_, set_h_Hamiltonian, set_J1_Hamiltonian, set_J2_Hamiltonian, set_Sz_operator, set_SzSz_operator


def test_h_hamiltonian(h):
    print("Testing h Hamiltonian")
    N = 1000
    model = Model(2,3)
    rbm = RBM(model)
    Ham =  set_h_Hamiltonian(model, h = h)
    gamma = 1
    rbm.train(Ham, gamma, N = 1000, n_iter = 40)

    batch = rbm.create_batch(N)
    #print(f"the updated energy is {rbm.expectation_value_batch(Ham, batch)}")
    print(f"The spin distributions are {np.average(batch, axis=0)}")

def test_J1_hamiltonian(J1):
    print("Testing J1 Hamiltonian")
    N = 2000
    n_iter = 40
    model = Model(2,4)
    rbm = RBM(model)
    Ham =  set_J1_Hamiltonian(model, J = J1)
    gamma = 0.1
    rbm.train(Ham, gamma, N = N, n_iter = n_iter, method = "sr")

    batch = rbm.create_batch(N)
    #print(f"the updated energy is {rbm.expectation_value_batch(Ham, batch)}")
    print(f"The spin distributions are {np.average(batch, axis=0)}")


def get_expectation_value_for_rbm(rbm):
    #get the Sz and SzSz expectation values
    model = rbm.model
    Szs = set_Sz_operator(model)
    SzSzs = set_SzSz_operator(model)
    N = 1000
    Sz_expectation = rbm.expectation_value_with_new_batch(Szs, N)
    SzSz_expectation = rbm.expectation_value_with_new_batch(SzSzs, N)
    print(f"The expectation value for Sz is {Sz_expectation}")
    print(f"The expectation value for SzSz is {SzSz_expectation}")
    return Sz_expectation, SzSz_expectation


if __name__ == "__main__":
    #test_h_hamiltonian(4)
    test_J1_hamiltonian(1)