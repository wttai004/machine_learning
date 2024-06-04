from model import Model
import numpy as np
import random
from numpy import tanh

class RBM:
    """
    A reduced Boltzmann machine that implements the spin wavefunction based on the model class
    """
    def __init__(self, model) -> None:
        self.model = model
        self.L1 = model.L1
        self.L2 = model.L2
        self.M = 3*int(self.L1*self.L2/2)

        self.a = (np.random.random(self.L1*self.L2)-0.5) + 1j * (np.random.random(self.L1*self.L2) -0.5) #np.ones(self.L1 * self.L2)
        self.b = (np.random.random(self.M)-0.5) + 1j * (np.random.random(self.M)-0.5)#np.zeros(self.M)
        self.M = (np.random.random((self.M, self.L1 * self.L2))-0.5) + 1j * (np.random.random((self.M, self.L1 * self.L2))-0.5)#np.ones((self.M, self.L1 * self.L2))

    def get_weights(self):
        #Helper function, return the weights
        return self.a, self.b, self.M
    
    def set_weights(self, a, b, M):
        #Helper function, set the weights
        self.a = a
        self.b = b
        self.M = M


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
        #This is just for debugging use
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        return 20 if (spin[0, 0, 0] == 1) else 1

    def metropolis_step(self, spin):
        """
        Perform a single Metropolis step
        """
        spin2 = self.model.flip_random_spin(spin)
        #print(self.evaluate(spin2),self.evaluate(spin))
        p = min(1, (self.evaluate(spin2) / self.evaluate(spin))**2)
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
    
    def expectation_value_Sz(self, spin):
        return np.sum(spin[:, :, 0]/2 - spin[:, :, 1]/2, axis=(0, 1))
    
    def decay(self, b):
        return max(100*0.9**b, 0.01)
    
    def get_deltas(self, Ham, p, N = 100):
        #create a batch
        batch = self.create_batch(N)
        # Implement a Hamiltonian
        Es = np.array([self.expectation_value(Ham, spin) for spin in batch]) 
        tanhs =  np.array([tanh(self.theta(spin)) for spin in batch])
        sigmas = np.array([self.model.project_spin(spin) for spin in batch])

        Oais =  sigmas
        Objs =  tanhs
        Owijs = tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :]
        # Flatten W and concatenate a, b, W
        O = np.concatenate([Oais, Objs, Owijs.reshape(N, -1)], axis=1)
        mean_O = np.mean(O, axis=0)
        # Compute <Oj Ok>
        Oj_Ok = np.einsum('ij,ik->jk', O, O) / N
        # Compute <Oj><Ok>
        mean_Oj_Ok = np.outer(mean_O, mean_O)
        # Correlation matrix
        Skk = Oj_Ok - mean_Oj_Ok

        Skk_reg = Skk + self.decay(1) * np.eye(Skk.shape[0])
        Skk_inv = np.linalg.inv(Skk_reg)

        delta_as = np.mean(Es[:, np.newaxis] * sigmas, axis = 0) - np.mean(Es) * np.mean(sigmas, axis = 0)
        delta_bs = np.mean(Es[:, np.newaxis] * tanhs, axis = 0) - np.mean(Es) * np.mean(tanhs, axis = 0)
        delta_Ws = np.mean(Es[:, np.newaxis, np.newaxis] * tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :], axis = 0) - np.mean(Es) * np.mean(tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :], axis = 0)

        deltas = np.concatenate([delta_as, delta_bs, delta_Ws.reshape(-1)], axis=0)

        deltas_reg = Skk_inv @ deltas
        delta_as_reg = deltas_reg[:sigmas.shape[1]]
        delta_bs_reg = deltas_reg[sigmas.shape[1]:sigmas.shape[1] + tanhs.shape[1]]
        delta_Ws_reg = deltas_reg[sigmas.shape[1] + tanhs.shape[1]:].reshape(tanhs.shape[1], sigmas.shape[1])
        return delta_as_reg, delta_bs_reg, delta_Ws_reg
    
    def update_weights(self, Ham, gamma, p, N = 100, verbose = True):
        if verbose:
            print(f"Updating weights at iteration {p}")
        delta_as, delta_bs, delta_Ws = self.get_deltas(Ham, p)
        self.a -= gamma * delta_as
        self.b -= gamma * delta_bs
        self.M -= gamma * delta_Ws

    def calculate_Sz_expectation_brute_force(self, batch):
        #Helper function
        return  np.mean(np.sum(batch[:, :, :, 0]/2 - batch[:, :, :, 1]/2, axis=(1, 2)))

    def train(self, Ham, gamma, operator = 0, N = 100, n_iter = 10, verbose = True):
        for p in range(n_iter):
            self.update_weights(Ham, gamma, p, N = N, verbose = verbose)
            if verbose:
                batch = self.create_batch(N)
                print("Current energy:", self.expectation_value_batch(Ham, batch))
                print("Current Sz:", self.calculate_Sz_expectation_brute_force(batch))
        return self.a, self.b, self.M