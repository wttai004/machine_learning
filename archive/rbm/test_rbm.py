import numpy as np
import random
from model import Model  # Assume the refactored code is saved in model.py
from rbm import RBM
from rbm_operator import set_h_Hamiltonian

def test_batch(N = 10000, weight = 2):
    print("Testing batch")
    model = Model(4,4)
    rbm = RBM(model)
    spin = model.get_random_spins()
    rbm.theta(spin)
    rbm.evaluate(spin)
    rbm.set_evaluate_function(rbm.evaluate_function_dummy(weight))
    rbm.evaluate(spin)
    rbm.metropolis_step(spin)
    batch = rbm.create_batch(10000)
    test = 0
    for spins in batch:
        test += spins[0,0,0]
    mean = test / N
    expected_mean = (weight**2)/(weight ** 2 + 1)
    print(test/N, np.sqrt(expected_mean * (1-expected_mean)/N) * 4)
    assert np.abs(mean -  (weight**2)/(weight ** 2 + 1)) < np.sqrt(expected_mean * (1-expected_mean)/N) * 4, f"mean = {mean}, expected = {1 * (weight**2)/(weight ** 2 + 1)}"


def test_expectation():
    print("Testing expectation")
    model = Model(2,3)
    rbm = RBM(model)
    Ham =  set_h_Hamiltonian(model, h = 4)
    weight = 5
    rbm.set_evaluate_function(rbm.evaluate_function_dummy(weight))
    batch = rbm.create_batch_complete()
    exact_value = rbm.expectation_value_exact(Ham)
    print(f"The exact energy is {exact_value}")

    N = 500
    test = [rbm.expectation_value_batch(Ham, rbm.create_batch(N)) for _ in range(5)]
    print(f"The average energy for batch size N = {N} is {np.mean(test)} with standard deviation {np.std(test)}")
    assert np.abs(np.mean(test) - exact_value) < 4 * np.std(test), "The energy is not within 4 standard deviations of the exact value"

    # N = 5000
    # test = [rbm.expectation_value_batch(Ham,  rbm.create_batch(N)) for _ in range(5)]
    # print(f"The average energy for batch size N = {N} is {np.mean(test)} with standard deviation {np.std(test)}")
    # assert np.abs(np.mean(test) - exact_value) < 4 * np.std(test), "The energy is not within 4 standard deviations of the exact value"


if __name__ == "__main__":
    np.random.seed(0)  # Set random seed for reproducibility
    test_batch()
    test_expectation()