import numpy as np
from numpy import tanh
import matplotlib.pyplot as plt
import random
from math import sqrt
from model import Model
from rbm import RBM
from rbm_operator import Operator, Sx_, Sy_, Sz_, SzSz_, set_h_Hamiltonian, set_J1_Hamiltonian, set_J2_Hamiltonian


def test_h_hamiltonian(h):
    print("Testing h Hamiltonian")
    N = 1000
    model = Model(2,3)
    rbm = RBM(model)
    Ham =  set_h_Hamiltonian(model, h = 4)
    gamma = 1
    rbm.train(Ham, gamma, N = 1000, n_iter = 20)

    batch = rbm.create_batch(N)
    #print(f"the updated energy is {rbm.expectation_value_batch(Ham, batch)}")
    print(f"The spin distributions are {np.average(batch, axis=0)}")

def test_J1_hamiltonian(J1):
    print("Testing J1 Hamiltonian")
    N = 1000
    model = Model(2,4)
    rbm = RBM(model)
    Ham =  set_J1_Hamiltonian(model, J = 4)
    gamma = 1
    rbm.train(Ham, gamma, N = 1000, n_iter = 40)

    batch = rbm.create_batch(N)
    #print(f"the updated energy is {rbm.expectation_value_batch(Ham, batch)}")
    print(f"The spin distributions are {np.average(batch, axis=0)}")


if __name__ == "__main__":
    #test_h_hamiltonian(4)
    test_J1_hamiltonian(4)