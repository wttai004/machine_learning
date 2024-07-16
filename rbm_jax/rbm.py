from model import Model, _project_spin, _flip_random_spin, _get_random_spins
import numpy as np
import random
from numpy import tanh
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

@jit
def _theta(spin, b, M)->float:
    """
    Helper function: evaluates the product bj + Wij sigma i
    """
    projected_spin = _project_spin(spin)
    return b + jnp.dot(M, projected_spin)

@jit
def _evaluate(spin, a, b, M) -> float:
    """
    Evaluate the wavefunction for a given spin configuration
    """
    projected_spin = _project_spin(spin)
    #print(self.M, projected_spin, np.prod(2 * np.cosh(self.b + np.dot(self.M, projected_spin))))
    return jnp.exp(jnp.dot(a, projected_spin)) * jnp.prod(2 * jnp.cosh(b + jnp.dot(M, projected_spin)))

@partial(jit, static_argnums=(1,))
def _metropolis_step(spin, evaluate_function, subkey1, subkey2):
    """
    Perform a single Metropolis step
    """
    spin2 = _flip_random_spin(spin, subkey1)
    #print(self.evaluate(spin2),self.evaluate(spin))
    p = jnp.minimum(1, (evaluate_function(spin2) / evaluate_function(spin))**2)
    #return jax.lax.cond(jax.random.uniform(subkey) < p, spin2, spin, operand=None)
    return jax.lax.cond(jax.random.uniform(subkey2) < p, lambda _: spin2, lambda _: spin, operand=None)

@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5))
def _create_batch(L1, L2, evaluate_function, N, burn_in, skip, subkey):
    result_array = jnp.zeros((N, L1, L2, 2), dtype=int)
    subkey, newkey = jax.random.split(subkey)
    current_spins = _get_random_spins(L1, L2, newkey)

    # Burn-in phase
    def burn_in_step(state, _):
        spins, key = state
        key, subkey1, subkey2 = jax.random.split(key, 3)
        new_spins = _metropolis_step(spins, evaluate_function, subkey1, subkey2)
        return (new_spins, key), None

    (current_spins, subkey), _ = jax.lax.scan(burn_in_step, (current_spins, subkey), None, length=burn_in)

    # Sampling phase
    def sample_step(state, i):
        spins, key, result_array = state
        key, subkey1, subkey2 = jax.random.split(key, 3)
        new_spins = _metropolis_step(spins, evaluate_function, subkey1, subkey2)
        result_array = jax.lax.cond(i % skip == 0,
                                    lambda arr: arr.at[i // skip].set(new_spins),
                                    lambda arr: arr,
                                    result_array)
        return (new_spins, key, result_array), None

    initial_state = (current_spins, subkey, result_array)
    final_state, _ = jax.lax.scan(sample_step, initial_state, jnp.arange(1, N * skip))

    _, _, result_array = final_state
    return result_array



class RBM:
    """
    A reduced Boltzmann machine that implements the spin wavefunction based on the model class
    """
    def __init__(self, model, seed = -1) -> None:
        self.model = model
        self.L1 = model.L1
        self.L2 = model.L2
        self.M = 3*int(self.L1*self.L2/2)
        self.key = jax.random.PRNGKey(seed)
        self.key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(self.key, 7)

        self.a = jax.random.uniform(subkey1, (self.L1 * self.L2,)) - 0.5 + 1j * (jax.random.uniform(subkey2, (self.L1 * self.L2,)) - 0.5)
        self.b = jax.random.uniform(subkey3, (self.M,)) - 0.5 + 1j * (jax.random.uniform(subkey4, (self.M,)) - 0.5)
        self.M = jax.random.uniform(subkey5, (self.M, self.L1 * self.L2)) - 0.5 + 1j * (jax.random.uniform(subkey6, (self.M, self.L1 * self.L2)) - 0.5)

        self.evaluate_function = self.evaluate
        # if seed != -1:
        #     np.random.seed(seed)

    def set_evaluate_function(self, evaluate_function):
        self.evaluate_function = evaluate_function

    def evaluate_function_dummy(self, weight):
        @jax.jit
        def evaluate_dummy(spin) -> float:
            return jax.lax.cond(spin[0, 0, 0] == 1, lambda _: float(weight), lambda _: 1.0, operand=None)
        return evaluate_dummy

    def evaluate_function_dummy_two_site(self, weight):
        @jax.jit
        def evaluate_dummy(spin) -> float:
            return jax.lax.cond(spin[0, 0, 0] == spin[0, 1, 0], lambda _: float(weight), lambda _: 1.0, operand=None)
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
        return _theta(spin, self.b, self.M)

    def evaluate(self, spin) -> float:
        """
        Evaluate the wavefunction for a given spin configuration
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        return _evaluate(spin, self.a, self.b, self.M)
    
    def metropolis_step(self, spin):
        """
        Perform a single Metropolis step
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        return _metropolis_step(spin, self.evaluate_function, subkey1, subkey2)
    
    def create_batch(self, N, burn_in = 1000, skip = 10):
        """
        Create a batch of N spins
        """
        assert type(N) == int and N >= 1
        self.key, subkey = jax.random.split(self.key, 2)
        return _create_batch(self.L1, self.L2, self.evaluate_function, N, burn_in, skip, subkey)

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
        return jnp.sum(spin[:, :, 0]/2 - spin[:, :, 1]/2, axis=(0, 1))
    
    def expectation_value_Sz_batch(self, spins):
        result = 0
        for spin in spins:
            result += self.expectation_value_Sz(spin)
        return result / len(spins)
    
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
    

if __name__ == "__main__":
    model = Model(2,3)
    rbm = RBM(model)
    spin = model.get_random_spins()
    rbm.theta(spin)
    rbm.evaluate(spin)
    rbm.set_evaluate_function(rbm.evaluate_function_dummy(2))
    rbm.evaluate(spin)
    rbm.metropolis_step(spin)
    batch = rbm.create_batch(100)
    print(batch)
