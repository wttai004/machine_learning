from model import Model
import numpy as np
import random
from numpy import tanh
from itertools import product
from tqdm import tqdm
class RBM:
    """
    A reduced Boltzmann machine that implements the spin wavefunction based on the model class
    """
    def __init__(self, model, seed = -1) -> None:
        self.model = model
        self.L1 = model.L1
        self.L2 = model.L2
        self.M = 3*int(self.L1*self.L2/2)

        self.a = (np.random.random(self.L1*self.L2)-0.5) + 1j * (np.random.random(self.L1*self.L2) -0.5) #np.ones(self.L1 * self.L2)
        self.b = (np.random.random(self.M)-0.5) + 1j * (np.random.random(self.M)-0.5)#np.zeros(self.M)
        self.M = (np.random.random((self.M, self.L1 * self.L2))-0.5) + 1j * (np.random.random((self.M, self.L1 * self.L2))-0.5)#np.ones((self.M, self.L1 * self.L2))

        self.evaluate_function = self.evaluate
        if seed != -1:
            np.random.seed(seed)

    def set_evaluate_function(self, evaluate_function):
        self.evaluate_function = evaluate_function

    def evaluate_function_dummy(self, weight):
        def evaluate_dummy(spin) -> float:
            return weight if (spin[0, 0, 0] == 1) else 1
        return evaluate_dummy
    

    def evaluate_function_dummy_two_site(self, weight):
        def evaluate_dummy(spin) -> float:
            return weight if (spin[0, 0, 0] == spin[0, 1, 0]) else 1
        return evaluate_dummy
    
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
        result = np.exp(np.dot(self.a, projected_spin)) * np.prod(2 * np.cosh(self.b + np.dot(self.M, projected_spin)))
        assert np.isreal(result), f"Invalid result {result}"
        return np.exp(np.dot(self.a, projected_spin)) * np.prod(2 * np.cosh(self.b + np.dot(self.M, projected_spin)))
    
    def metropolis_step(self, spin):
        """
        Perform a single Metropolis step
        """
        spin2 = self.model.flip_random_spin(spin)
        #print(self.evaluate(spin2),self.evaluate(spin))
        p = min(1, (self.evaluate_function(spin2) / self.evaluate_function(spin))**2)
        if np.random.random() < p:
            return spin2
        return spin
    
    def create_batch(self, N, burn_in = 1000, skip = 10):
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
    
    def create_batch_complete(self):
        """
        Create a batch of all possible spins (for debugging purpose)
        """
        L1 = self.L1
        L2 = self.L2
        total_spins = L1 * L2
        # Generate all possible binary configurations for total_spins
        binary_configs = product([0, 1], repeat=total_spins)
        
        # Map each binary configuration to the corresponding spin configuration
        spin_configurations = []
        for config in binary_configs:
            spin_array = np.zeros((L1, L2, 2), dtype=int)
            for i in range(L1):
                for j in range(L2):
                    index = i * L2 + j
                    if config[index] == 0:
                        spin_array[i, j] = (1, 0)
                    else:
                        spin_array[i, j] = (0, 1)
            spin_configurations.append(spin_array)
        
        return spin_configurations

    def expectation_value(self, operator, spin):
        """
        Evaluate the expectation value of an operator for a given spin configuration
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        spin1 = spin
        spin2s = self.model.generate_local_spins(spin1, change = 2)
        result = 0
        for spin2 in spin2s:
            result += operator.vdot(spin1, spin2) * self.evaluate_function(spin2) / self.evaluate_function(spin1)
        return result
    
    def expectation_value_batch(self, operator, spins):
        """
        Evaluate the expectation value of an operator for the batch spins
        """
        result = 0
        for spin in spins:
            result += self.expectation_value(operator, spin)
        return result / len(spins)
        
    def expectation_value_exact(self, operator):
        """
        Evaluate the expectation value of an operator for all possible spin configurations
        Warning: Exponential scaling to system size
        """
        spins = self.create_batch_complete()
        result = 0
        normalization = 0
        for spin in spins:
            result += self.expectation_value(operator, spin) * self.evaluate_function(spin)**2
            normalization += self.evaluate_function(spin)**2
        return result/ normalization

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

    def expectation_value_Sz_batch(self, spins):
        result = 0
        for spin in spins:
            result += self.expectation_value_Sz(spin)
        return result / len(spins)
    
    def decay(self, b):
        return max(100*0.9**b, 0.01)
    
    def get_deltas_sr(self, Ham, p, N = 100):
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

        Skk_reg = Skk + 0.001 * np.eye(Skk.shape[0]) # self.decay(p) * np.eye(Skk.shape[0])
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

    def get_deltas_sgd(self, Ham, p, N = 100):
        #create a batch
        batch = self.create_batch(N)
        # Implement a Hamiltonian
        Es = np.array([self.expectation_value(Ham, spin) for spin in batch]) 
        tanhs =  np.array([tanh(self.theta(spin)) for spin in batch])
        sigmas = np.array([self.model.project_spin(spin) for spin in batch])

        delta_as = np.mean(Es[:, np.newaxis] * sigmas, axis = 0) - np.mean(Es) * np.mean(sigmas, axis = 0)
        delta_bs = np.mean(Es[:, np.newaxis] * tanhs, axis = 0) - np.mean(Es) * np.mean(tanhs, axis = 0)
        delta_Ws = np.mean(Es[:, np.newaxis, np.newaxis] * tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :], axis = 0) - np.mean(Es) * np.mean(tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :], axis = 0)

        deltas = np.concatenate([delta_as, delta_bs, delta_Ws.reshape(-1)], axis=0)

        return delta_as, delta_bs, delta_Ws#delta_as_reg, delta_bs_reg, delta_Ws_reg
    

    def update_weights(self, Ham, gamma, p, N = 100, verbose = True, method = "sgd"):
        if verbose:
            print(f"Updating weights at iteration {p}")
            print(f"Current magnitude of the weights are {np.linalg.norm(self.a)}, {np.linalg.norm(self.b)}, {np.linalg.norm(self.M)}")
        if method == "sgd":
            delta_as, delta_bs, delta_Ws = self.get_deltas_sgd(Ham, p)
        elif method == "sr":
            delta_as, delta_bs, delta_Ws = self.get_deltas_sr(Ham, p)
        self.a -= gamma * delta_as
        self.b -= gamma * delta_bs
        self.M -= gamma * delta_Ws

    def calculate_Sz_expectation_brute_force(self, batch):
        #Helper function
        return  np.mean(np.sum(batch[:, :, :, 0]/2 - batch[:, :, :, 1]/2, axis=(1, 2)))

    def train(self, Ham, gamma, decay_gamma = 0.99, operator = 0, N = 100, n_iter = 10, verbose = True, method = "sr"):
        for p in tqdm(range(n_iter), desc = "Training progress"):
            self.update_weights(Ham, gamma * decay_gamma**p, p, N = N, verbose = verbose, method = "sr")
            if verbose:
                batch = self.create_batch(N)
                print("Current energy:", self.expectation_value_batch(Ham, batch))
                print(f"The spin distributions are {np.average(batch, axis=0)}")
                #print("Current Sz:", self.calculate_Sz_expectation_brute_force(batch))
        print("Training done")
        #return self.a, self.b, self.M