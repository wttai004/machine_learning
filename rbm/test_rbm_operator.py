from model import Model
from rbm import RBM
import numpy as np
import random
from rbm_operator import *

def test_J1():
    print("Testing J1 Hamiltonian on 2x1 lattice...")
    # Test the J1 Hamiltonian on 2x1 lattice
    model = Model(2,1)
    J1_Op = set_J1_Hamiltonian(model)
    #print(f"J1 Hamiltonian: {J1_Op.onesiteoperators}, {J1_Op.twositeoperators}")
    spin1 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
    spin2 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
    #print(f"Hamiltonian expectation: <spin1|H|spin2>={J1_Op.vdot(spin1, spin1)}")
    assert abs(J1_Op.vdot(spin1, spin1) -0.25) < 1e-5, f"Triplet state (up, up) has wrong energy: {J1_Op.vdot(spin1, spin1)}"
    spin2[1, 0] = [0,1]
    #print(f"Flipping a spin in s Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
    assert J1_Op.vdot(spin1, spin2) == 0.0, f"Two orthogonal spins are not orthogonal: {J1_Op.vdot(spin1, spin2)}"
    spinud = spin1.copy()
    spinud[1, 0] = [0, 1]
    spindu = spin1.copy()
    spindu[0, 0] = [0, 1]
    singlet_energy = (J1_Op.vdot(spinud, spinud)-J1_Op.vdot(spindu, spinud)-J1_Op.vdot(spinud, spindu)+J1_Op.vdot(spindu, spindu))/2
    triplet_energy = (J1_Op.vdot(spinud, spinud)+J1_Op.vdot(spindu, spinud)+J1_Op.vdot(spinud, spindu)+J1_Op.vdot(spindu, spindu))/2
    #print(f"J1 Hamiltonian on singlet state gives: <spin1|H|spin2>={singlet_energy}")
    #print(f"J1 Hamiltonian on triplet state gives: <spin1|H|spin2>={triplet_energy}")
    assert abs(singlet_energy + 0.75) == 0.0, f"Singlet energy is not correct: {singlet_energy}"
    assert abs(triplet_energy - 0.25) == 0.0, f"Triplet energy is not correct: {triplet_energy}"

    # Test the J1 Hamiltonian on 4x4 lattice
    print("Testing J1 Hamiltonian on 4x4 lattice...")
    model = Model(4, 4)
    spin1 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
    spin2 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
    J1_Op = set_J1_Hamiltonian(model)
    #print(f"J1 Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
    assert abs(J1_Op.vdot(spin1, spin1) - 0.25*model.L1*model.L2 - 2) < 1e-5, f"All up spins has wrong energy: {J1_Op.vdot(spin1, spin1)}"
    spin2[0,0] = [0, 1]
    #print(f"Flipping a corner spin in s Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
    #print(f"Energy is now: <spin2|H|spin2>={J1_Op.vdot(spin2, spin2)}")
    assert abs(J1_Op.vdot(spin2, spin2) - 5) < 1e-5, f"Flipping a corner spin has wrong energy: {J1_Op.vdot(spin2, spin2)}"
    spin2[2,2] = [0, 1]
    #print(f"Flipping a middle spin in s Hamiltonian gives: <spin1|H|spin2>={J1_Op.vdot(spin1, spin2)}")
    #print(f"Energy is now: <spin2|H|spin2>={J1_Op.vdot(spin2, spin2)}")
    assert abs(J1_Op.vdot(spin2, spin2) - 3) < 1e-5, f"Flipping a middle spin has wrong energy: {J1_Op.vdot(spin2, spin2)}"

def test_h():
    print("Testing h Hamiltonian on 2x1 lattice...")
    model = Model(4,4)
    Ham = set_h_Hamiltonian(model, h = 1)
    #Ham += set_J1_Hamiltonian(model, J = 1)
    spin1 = np.array([[[1,0] for _ in range(model.L2)] for _ in range(model.L1)])
    #print(f"Hamiltonian expectation: <spin1|H|spin2>={Ham.vdot(spin1, spin1)}")
    assert abs(Ham.vdot(spin1, spin1) - 0.5*model.L1*model.L2) < 1e-5, f"All up spins has wrong energy: {Ham.vdot(spin1, spin1)}"
    spin1[1, 0] = [0,1]
    #print(f"Flipping a spin in s Hamiltonian gives: <spin1|H|spin2>={Ham.vdot(spin1, spin1)}")
    assert abs(Ham.vdot(spin1, spin1) - 0.5*model.L1*model.L2 + 1) < 1e-5, f"Flipping a spin has wrong energy: {Ham.vdot(spin1, spin1)}"
    spin1[3, 0] = [0,1]
    #print(f"Flipping a spin in s Hamiltonian gives: <spin1|H|spin2>={Ham.vdot(spin1, spin1)}")
    assert abs(Ham.vdot(spin1, spin1) - 0.5*model.L1*model.L2 + 2) < 1e-5, f"Flipping two spin has wrong energy: {Ham.vdot(spin1, spin1)}"

def test_SzSz():
    print("Testing SzSz expectation value...")
    model = Model(2,2)
    rbm = RBM(model)
    average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])) for _ in range(40)]
    #print(f"The average expectation <SzSz|ud__|SzSz> is {np.mean(average_expectations)} with standard deviation {np.std(average_expectations)}")
    assert abs(np.mean(average_expectations) + 0.25) < 1e-5, f"Triplet state (up, down) has wrong expectation: {np.mean(average_expectations)}"
    average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), np.array([[[0, 1], [0, 1]], [[0, 1], [1, 0]]])) for _ in range(40)]
    #print(f"The average expectation  <SzSz|dd__|SzSz> is {np.mean(average_expectations)} with standard deviation {np.std(average_expectations)}")
    assert abs(np.mean(average_expectations) - 0.25) < 1e-5, f"Triplet state (down, down) has wrong expectation: {np.mean(average_expectations)}"
    
    model = Model(2,2)
    rbm = RBM(model, seed = 42)
    weight = 2
    rbm.set_evaluate_function(rbm.evaluate_function_dummy_two_site(2))
    batch = rbm.create_batch(400) #This uses evaluate_dummy, which gives 2/3 for spin up and 1/3 for spin down 
    average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), batch[i]) for i in range(len(batch))]
    #print(f"The average expectation for a 2:1 mixed state is {np.mean(average_expectations)} with standard deviation {np.std(average_expectations)}")
    assert abs(np.mean(average_expectations) - (0.25 * weight**2 - 0.25)/(weight**2 + 1)) < 4*np.std(average_expectations)/np.sqrt(400), f"Mixed state has wrong expectation: {np.mean(average_expectations)}"
    # rbm = RBM(model)
    # rbm.set_evaluate_function(rbm.evaluate_function_dummy_two_site(20))
    # batch = rbm.create_batch(200) #This uses evaluate_dummy, which gives 2/3 for spin up and 1/3 for spin down 
    # average_expectations = [rbm.expectation_value(SzSz_(0,0,0,1,model), batch[i]) for i in range(len(batch))]
    # print(f"The average expectation for a 2:1 mixed state is {np.mean(average_expectations)} with standard deviation {np.std(average_expectations)}")

def test_expectation():
    print("Testing implementation of expectation value...")
    model = Model(2,3)
    rbm = RBM(model, seed = 42)
    N = 10000
    #create a batch
    batch = rbm.create_batch(N)
    # Implement a Hamiltonian
    Ham =  set_J1_Hamiltonian(model, J = 1)#set_h_Hamiltonian(model, h = 4)
    Szs = set_h_Hamiltonian(model, h = 1)
    #print(f"the spin expectation value is {rbm.expectation_value_batch(Szs, batch)}")
    def calculate_Sz_expectation_brute_force(spins):
        return np.mean(np.sum(batch[:, :, :, 0]/2 - batch[:, :, :, 1]/2, axis=(1, 2)))
    #print(f"the naive spin expectation value is {calculate_Sz_expectation_brute_force(batch)}")
    assert abs(rbm.expectation_value_batch(Szs, batch) - calculate_Sz_expectation_brute_force(batch)) < 1e-5, f"Spin expectation value is wrong: {rbm.expectation_value_batch(Szs, batch)}"

if __name__ == "__main__":
    test_J1()
    test_h()
    # Test the SzSz expectation value
    test_SzSz()
    # Test the model expectation
    test_expectation()
    print("All tests passed!")