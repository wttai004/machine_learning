from model import Model
import numpy as np
import random
from numpy import tanh
import jax
import jax.numpy as jnp
from jax import jit, lax, tree_util
from functools import partial

class RBM:
    """
    A reduced Boltzmann machine that implements the spin wavefunction based on the model class
    """
    #model is static, key is dynamic, a, b, m, evaluate_function is dynamic
    def __init__(self, model, key = jax.random.PRNGKey(0), a = None, b = None, M = None, evaluate_function = None) -> None:
        self.model = model
        self.L1 = model.L1
        self.L2 = model.L2
        self.M = 3*int(self.L1*self.L2/2)
        #self.key = key
        self.key, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 7)

        self.a = jax.random.uniform(subkey1, (self.L1 * self.L2,)) - 0.5 + 1j * (jax.random.uniform(subkey2, (self.L1 * self.L2,)) - 0.5) if a is None else a
        self.b = jax.random.uniform(subkey3, (self.M,)) - 0.5 + 1j * (jax.random.uniform(subkey4, (self.M,)) - 0.5) if b is None else b
        self.M = jax.random.uniform(subkey5, (self.M, self.L1 * self.L2)) - 0.5 + 1j * (jax.random.uniform(subkey6, (self.M, self.L1 * self.L2)) - 0.5) if M is None else M

        
        #self.evaluate_function = self.evaluate_function_dummy(0.1)
        self.evaluate_function = self.evaluate if evaluate_function is None else evaluate_function
        # if seed != -1:
        #     np.random.seed(seed)

    def _tree_flatten(self):
        children = (self.key, self.a, self.b, self.M)
        aux_data = {"model": self.model, "evaluate_function": self.evaluate_function}
        return (children, aux_data)
    
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(aux_data['model'], *children, evaluate_function = aux_data['evaluate_function'])

    @jit
    def theta(self, spin)->float:
        """
        Helper function: evaluates the product bj + Wij sigma i
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        projected_spin = self.model.project_spin(spin)
        return self.b + jnp.dot(self.M, projected_spin)

    @jit
    def evaluate(self, spin) -> float:
        """
        Evaluate the wavefunction for a given spin configuration
        """
        assert spin.shape == (self.L1, self.L2, 2), f"Invalid spin shape {spin.shape}"
        projected_spin = self.model.project_spin(spin)
        return jnp.exp(jnp.dot(self.a, projected_spin)) * jnp.prod(2 * jnp.cosh(self.b + jnp.dot(self.M, projected_spin)))
        #return _evaluate(spin, self.a, self.b, self.M)
    

    def set_evaluate_function(self, evaluate_function):
        self.evaluate_function = evaluate_function

    def evaluate_function_dummy(self, weight):
        #@jax.jit
        def evaluate_dummy(spin) -> float:
            return jax.lax.cond(spin[0, 0, 0] == 1, lambda _: float(weight), lambda _: 1.0, operand=None)
        return evaluate_dummy

    def evaluate_function_dummy_two_site(self, weight):
        #@jax.jit
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

    @jit
    def _metropolis_step(self, spin, subkey):
        """
        Perform a single Metropolis step
        """
        subkey1, subkey2 = jax.random.split(subkey)
        spin2 = self.model._flip_random_spin(spin, subkey1)
        p = jnp.minimum(1, (self.evaluate_function(spin2) / self.evaluate_function(spin))**2)
        return jax.lax.cond(jax.random.uniform(subkey2) < p, lambda _: spin2, lambda _: spin, operand=None)
   
    def metropolis_step(self, spin):
        """
        Perform a single Metropolis step
        """
        key, subkey = jax.random.split(self.key, 2)
        return self._metropolis_step(spin, subkey)
    
    @partial(jit, static_argnums=(1, 3, 4))
    def _create_batch(self, N, key, burn_in = 1000, skip = 10):
        """
        Create a batch of N spins
        """
        result_array = jnp.zeros((N, self.L1,self.L2, 2), dtype=int)
        key, subkey = jax.random.split(key)
        current_spins = self.model._get_random_spins(subkey)

        def burn_in_step(state, _):
            spins, key = state
            new_spins = self._metropolis_step(spins, key)
            return (new_spins, key), None

        current_state = (current_spins, key)
        current_state, _ = jax.lax.scan(burn_in_step, current_state, None, length=burn_in)

        def sample_step(state, i):
            spins, result_array, key = state
            key, subkey = jax.random.split(key)
            new_spins = self._metropolis_step(spins, subkey)
            result_array = jax.lax.cond(i % skip == 0,
                                    lambda arr: arr.at[i // skip].set(new_spins),
                                    lambda arr: arr,
                                    result_array)
            return (new_spins, result_array, key), None

        initial_state = (current_state[0], result_array, current_state[1])
        final_state, _ = jax.lax.scan(sample_step, initial_state, jnp.arange(0, N * skip))

        _, result_array, _ = final_state
        return result_array

    def create_batch(self, N, burn_in = 1000, skip = 10):
        key, subkey = jax.random.split(self.key)
        return self._create_batch(N, subkey, burn_in = burn_in, skip = skip)

    #@jit
    @partial(jit, static_argnums=(1,))
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

    @partial(jit, static_argnums=(1,))
    def expectation_value_batch(self, operator, spins):
        """
        Evaluate the expectation value of an operator for the batch spins
        """
        result = 0
        for spin in spins:
            result += self.expectation_value(operator, spin)
        return result / len(spins)
        
    # def expectation_value_with_new_batch(self, operator, N = 10, burn_in = 100, skip = 10):
    #     """
    #     Evaluate the expectation value of an operator for a batch of spin configurations
    #     """
    #     spins = self.create_batch(N, burn_in = burn_in, skip = skip)
    #     result = 0
    #     for spin in spins:
    #         result += self.expectation_value(operator, spin)
    #     return result / len(spins)
    
    # def expectation_value_Sz(self, spin):
    #     return jnp.sum(spin[:, :, 0]/2 - spin[:, :, 1]/2, axis=(0, 1))
    
    # def expectation_value_Sz_batch(self, spins):
    #     result = 0
    #     for spin in spins:
    #         result += self.expectation_value_Sz(spin)
    #     return result / len(spins)
    
    # def decay(self, b):
    #     return max(100*0.9**b, 0.01)
    
    # def get_deltas(self, Ham, p, N = 100):
    #     #create a batch
    #     batch = self.create_batch(N)
    #     # Implement a Hamiltonian
    #     Es = np.array([self.expectation_value(Ham, spin) for spin in batch]) 
    #     tanhs =  np.array([tanh(self.theta(spin)) for spin in batch])
    #     sigmas = np.array([self.model.project_spin(spin) for spin in batch])

    #     Oais =  sigmas
    #     Objs =  tanhs
    #     Owijs = tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :]
    #     # Flatten W and concatenate a, b, W
    #     O = np.concatenate([Oais, Objs, Owijs.reshape(N, -1)], axis=1)
    #     mean_O = np.mean(O, axis=0)
    #     # Compute <Oj Ok>
    #     Oj_Ok = np.einsum('ij,ik->jk', O, O) / N
    #     # Compute <Oj><Ok>
    #     mean_Oj_Ok = np.outer(mean_O, mean_O)
    #     # Correlation matrix
    #     Skk = Oj_Ok - mean_Oj_Ok

    #     Skk_reg = Skk + self.decay(1) * np.eye(Skk.shape[0])
    #     Skk_inv = np.linalg.inv(Skk_reg)

    #     delta_as = np.mean(Es[:, np.newaxis] * sigmas, axis = 0) - np.mean(Es) * np.mean(sigmas, axis = 0)
    #     delta_bs = np.mean(Es[:, np.newaxis] * tanhs, axis = 0) - np.mean(Es) * np.mean(tanhs, axis = 0)
    #     delta_Ws = np.mean(Es[:, np.newaxis, np.newaxis] * tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :], axis = 0) - np.mean(Es) * np.mean(tanhs[:, :, np.newaxis] * sigmas[:, np.newaxis, :], axis = 0)

    #     deltas = np.concatenate([delta_as, delta_bs, delta_Ws.reshape(-1)], axis=0)

    #     deltas_reg = Skk_inv @ deltas
    #     delta_as_reg = deltas_reg[:sigmas.shape[1]]
    #     delta_bs_reg = deltas_reg[sigmas.shape[1]:sigmas.shape[1] + tanhs.shape[1]]
    #     delta_Ws_reg = deltas_reg[sigmas.shape[1] + tanhs.shape[1]:].reshape(tanhs.shape[1], sigmas.shape[1])
    #     return delta_as_reg, delta_bs_reg, delta_Ws_reg
    
    # def update_weights(self, Ham, gamma, p, N = 100, verbose = True):
    #     if verbose:
    #         print(f"Updating weights at iteration {p}")
    #     delta_as, delta_bs, delta_Ws = self.get_deltas(Ham, p)
    #     self.a -= gamma * delta_as
    #     self.b -= gamma * delta_bs
    #     self.M -= gamma * delta_Ws

    # def calculate_Sz_expectation_brute_force(self, batch):
    #     #Helper function
    #     return  np.mean(np.sum(batch[:, :, :, 0]/2 - batch[:, :, :, 1]/2, axis=(1, 2)))

    # def train(self, Ham, gamma, operator = 0, N = 100, n_iter = 10, verbose = True):
    #     for p in range(n_iter):
    #         self.update_weights(Ham, gamma, p, N = N, verbose = verbose)
    #         if verbose:
    #             batch = self.create_batch(N)
    #             print("Current energy:", self.expectation_value_batch(Ham, batch))
    #             print("Current Sz:", self.calculate_Sz_expectation_brute_force(batch))
    #     return self.a, self.b, self.M

tree_util.register_pytree_node(RBM,
                               RBM._tree_flatten,
                               RBM._tree_unflatten)    

if __name__ == "__main__":
    print("Testing compilation errors...")
    model = Model(2,3)
    rbm = RBM(model)
    spin = model.get_random_spins()
    rbm.theta(spin)
    rbm.evaluate(spin)
    rbm.set_evaluate_function(rbm.evaluate_function_dummy(2))
    rbm.evaluate(spin)
    rbm.metropolis_step(spin)
    batch = rbm.create_batch(100)
    # print(batch)
