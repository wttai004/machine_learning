import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random
from jax import jit, lax
from functools import partial

@jit
def _project_spin(spin):
    return jnp.array(spin[:,:,0].flatten(), dtype=jnp.float32) - 0.5

def project_element(value):
    return lax.cond(value == 0.5, 
                    lambda _: jnp.array([1, 0]), 
                    lambda _: jnp.array([0, 1]), 
                    operand=None)


@partial(jit, static_argnums=(1,2)) 
def _unproject_spin(spin, L1, L2):
    return jnp.array([[project_element(spin[i * L2 + j]) for j in range(L2)] for i in range(L1)])

    #return jnp.array([[[1, 0] if spin[i * L2 + j] == 0.5 else [0, 1] for j in range(L2)] for i in range(L1)])

@partial(jit, static_argnums=(0,1)) 
def _get_random_spins(L1, L2, subkey):
    return jax.random.choice(subkey, jnp.array([[1, 0], [0, 1]]), shape=(L1,L2))

@jit
def _flip_spin_at(spin, i, j):
    def flip_if_one(spin, i, j):
        spin = spin.at[i, j].set(jnp.array([1, 0]))
        return spin

    def flip_if_zero(spin, i, j):
        spin = spin.at[i, j].set(jnp.array([0, 1]))
        return spin
    
    condition = jnp.array_equal(spin[i, j], jnp.array([0, 1]))
    spin = jax.lax.cond(condition, flip_if_one, flip_if_zero, spin, i, j)
    
    return spin


@partial(jit, static_argnums=(1,2,3)) 
def _generate_local_spins(spin, L1, L2, change = 1):
    result = [spin]
    result.extend([_flip_spin_at(spin, i, j) for i in range(L1) for j in range(L2)])
    
    if change == 2:
        result.extend([_flip_spin_at(_flip_spin_at(spin, i, j), k, l)
                        for i in range(L1) for j in range(L2)
                        for k in range(L1) for l in range(L2)
                        if (k * L2 + l) > (i * L2 + j)])

    return result
    
@jit
def _vdot(spin1, spin2):
    dot_product = jnp.sum(spin1 * spin2, axis=-1)
    result = jnp.prod(dot_product)
    
    return result


class Model:
    """
    A model that represents a 2D lattice of spins with a Hamiltonian
    """

    def __init__(self, L1, L2, seed = 0) -> None:
        self.L1 = L1
        self.L2 = L2
        self.Hamiltonian = jnp.zeros((L1, L2, L1, L2, 2, 2, 2, 2), dtype=int)
        self.key = jax.random.PRNGKey(seed)

    def project_spin(self, spin):
        """
        Helper function. Given the spins (L1, L2, 2), project them down to a shape (L1*L2)

        In lieu of the one-hot spin representation, each spin entry [1,0] and [0,1] is mapped to 1/2 and -1/2 respectively
        """
        return _project_spin(spin)


    def unproject_spin(self, spin):
        """
        Helper function. Given the spins (L1*L2), project them back to a shape (L1, L2, 2)
        """
        assert spin.shape == (self.L1 * self.L2,), f"Invalid shape {spin.shape} for unproject_spin"
        return _unproject_spin(spin, self.L1, self.L2)

    def get_random_spins(self):
        """
        return: a 2D array (L1, L2, 2) of random spins
        Each site has either spin up [1,0] or spin down [0,1]
        """
        #raise NotImplementedError("get_random_spins is not implemented properly")
        self.key, subkey = jax.random.split(self.key)
        return _get_random_spins(self.L1, self.L2, subkey)
    
    def flip_spin_at(self, spin, i, j):
        """
        Flip the spin at position (i, j) in the input spin array
        """
        return _flip_spin_at(spin, i, j)
    
    def flip_random_spin(self, spin1):
        """
        return: a 2D array (L1, L2, 2) of spins with one random spin flipped
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        spin2 = spin1.copy()
        i = jax.random.randint(subkey1, (1,), 0, self.L1)[0]
        j = jax.random.randint(subkey2, (1,), 0, self.L2)[0]
        # Use JAX's conditional operations instead of assertions
        return _flip_spin_at(spin2, i, j)

        return _flip_random_spin(spin1, self.L1, self.L2, subkey1, subkey2)
        # spin2 = spin1.copy()
        # i = jax.random.randint(subkey1, (1,), 0, self.L1)[0]
        # j = jax.random.randint(subkey2, (1,), 0, self.L2)[0]
        # #i, j = random.randint(0, self.L1-1), random.randint(0, self.L2-1)
        # assert spin1[i,j,0] == 0 or spin1[i,j,0] == 1, f"Invalid spin value {spin1[i,j,0]}"
        # assert spin2[i,j,0] == 0 or spin2[i,j,0] == 1, f"Invalid spin value {spin2[i,j,0]}"
        # spin2 = spin2.at[i, j].set([1, 0] if spin1[i, j, 0] == 0 else [0, 1])
        # return spin2

    
    def generate_local_spins(self, spin, change=1):
        """
        Generate a set of spins with local perturbation around the input spin (flip one spins)

        Change: consider how much spins can be changed at maximum (currently only allows change = 1 or 2)
        """
        assert change == 1 or change == 2, f"Invalid change value {change}"
        return _generate_local_spins(spin, self.L1, self.L2, change)
        # result = [spin]
        # result.extend([self.flip_spin_at(spin, i, j) for i in range(self.L1) for j in range(self.L2)])
        
        # if change == 2:
        #     result.extend([self.flip_spin_at(self.flip_spin_at(spin, i, j), k, l)
        #                    for i in range(self.L1) for j in range(self.L2)
        #                    for k in range(self.L1) for l in range(self.L2)
        #                    if (k * self.L2 + l) > (i * self.L2 + j)])

        # return result



    def vdot(self, spin1, spin2):
        """
        #get the overlap expectation value <spin1|spin2>
        #where spin1 and spin2 are 2D arrays (L1, L2) generated by get_random_spins or equivalent
        """
        #assert spin1.dtype == int, f"Hdot cannot handle spins of type {spin1.dtype}. Please convert this to int"
        #assert np.all(np.sum(spin1, axis = -1) == 1), "spin1 is not properly normalized"
        #assert np.all(np.sum(spin2, axis = -1) == 1), "spin2 is not properly normalized"
        # Element-wise multiplication and summation
        return _vdot(spin1, spin2)
        #dot_product = jnp.sum(spin1 * spin2, axis=-1)
        #result = jnp.prod(dot_product)
        
        #return result


if __name__ == "__main__":
    print("Hello world")