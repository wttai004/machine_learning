from model import Model
import numpy as np
import random

class RBM:
    """
    A reduced Boltzmann machine that implements the spin wavefunction based on the model class
    """
    def __init__(self, model) -> None:
        self.model = model
        self.L1 = model.L1
        self.L2 = model.L2
        self.M = 3*int(self.L1*self.L2/2)

        self.a = np.random.random(self.L1*self.L2)-0.5#np.ones(self.L1 * self.L2)
        self.b = np.random.random(self.M)-0.5#np.zeros(self.M)
        self.M = np.random.random((self.M, self.L1 * self.L2))-0.5#np.ones((self.M, self.L1 * self.L2))

    def theta(self, spin)->float:
        """
        Helper function: evaluates the product bj + Wij sigma i
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        projected_spin = self.model.project_spin(spin)
        return self.b + np.dot(self.M, projected_spin)

    def evaluate(self, spin) -> float:
        """
        Evaluate the wavefunction for a given spin configuration
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        projected_spin = self.model.project_spin(spin)
        #print(self.M, projected_spin, np.prod(2 * np.cosh(self.b + np.dot(self.M, projected_spin))))
        return np.exp(np.dot(self.a, projected_spin)) * np.prod(2 * np.cosh(self.b + np.dot(self.M, projected_spin)))

    def evaluate_dummy(self, spin) -> float:
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        return 20 if (spin[0, 0, 0] == 1) else 1

    def metropolis_step(self, spin):
        """
        Perform a single Metropolis step
        """
        spin2 = self.model.flip_random_spin(spin)
        #print(self.evaluate(spin2),self.evaluate(spin))
        p = min(1, (self.evaluate(spin2) / self.evaluate(spin))**2)
        #p = min(1, (self.evaluate_dummy(spin2) / self.evaluate_dummy(spin)))
        if random.random() < p:
            return spin2
        return spin
    
    def create_batch(self, N, burn_in = 100, skip = 10):
        """
        Create a batch of N spins
        """
        assert type(N) == int and N >= 1
        result_array = np.zeros((N, self.L1, self.L2, 2), dtype = int)
        current_spins = self.model.get_random_spins()
        for _ in range(burn_in):
            current_spins = self.metropolis_step(current_spins)
        #print(result_array.shape, current_spins.shape, self.L1, self.L2)
        result_array[0] = current_spins
        for i in range(1, N*skip):
            current_spins = self.metropolis_step(current_spins)
            if i % skip == 0:
                result_array[i//skip] = current_spins
        return result_array

    def expectation_value(self, operator, spin):
        """
        Evaluate the expectation value of an operator for a given spin configuration
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        spin1 = spin
        spin2s = self.model.generate_local_spins(spin1, change = 2)
        result = 0
        for spin2 in spin2s:
            #print(spin1, spin2)
            #print(spin1, spin2, operator.vdot(spin1, spin2), self.evaluate_dummy(spin2), self.evaluate_dummy(spin1))
            result += operator.vdot(spin1, spin2) * self.evaluate(spin2) / self.evaluate(spin1)
        return result
    
    def expectation_value_batch(self, operator, spins):
        """
        Evaluate the expectation value of an operator for the batch spins
        """
        result = 0
        for spin in spins:
            result += self.expectation_value(operator, spin)
        return result / len(spins)
        
    def expectation_value_with_new_batch(self, operator, N = 10, burn_in = 100, skip = 10):
        """
        Evaluate the expectation value of an operator for a batch of spin configurations
        """
        spins = self.create_batch(N, burn_in = burn_in, skip = skip)
        result = 0
        for spin in spins:
            result += self.expectation_value(operator, spin)
        return result / len(spins)