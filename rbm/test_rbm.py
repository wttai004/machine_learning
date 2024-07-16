import numpy as np
import random
from model import Model  # Assume the refactored code is saved in model.py
from rbm import RBM

def test_batch():
    model = Model(4,4)
    rbm = RBM(model)
    spin = model.get_random_spins()
    rbm.theta(spin)
    rbm.evaluate(spin)
    rbm.set_evaluate_function(rbm.evaluate_function_dummy(0.01))
    rbm.evaluate(spin)
    rbm.metropolis_step(spin)
    batch = rbm.create_batch(10000)
    test = 0
    for spins in batch:
        test += spins[0,0,0]
    #print(test/100)
    #print(rbm.expectation_value_Sz_batch(batch))

if __name__ == "__main__":
    random.seed(0)  # Set random seed for reproducibility
    test_batch()
