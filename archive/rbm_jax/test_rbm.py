import jax.numpy as jnp
import random
from model import Model  # Assume the refactored code is saved in model.py
from rbm import RBM
from jax import jit, lax
import jax


def test_batch(N = 10000, seed = 0, weight = 2):
    print("Testing whether batch generates a reasonable spin configuration with reasonable mean...")
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    model = Model(4,4, key = key1)
    rbm = RBM(model, key = key2)
    # spin = model.get_random_spins()
    # rbm.theta(spin)
    # rbm.evaluate(spin)
    rbm.set_evaluate_function(rbm.evaluate_function_dummy(weight))
    # rbm.evaluate(spin)
    # rbm.metropolis_step(spin)
    batch = rbm.create_batch(N)
    test = 0
    print(batch.shape)
    for spins in batch:
        test += spins[0,0,0]
    mean = test / N
    expected_mean = (weight**2)/(weight ** 2 + 1)
    print(test/N, jnp.sqrt(expected_mean * (1-expected_mean)/N) * 4)
    assert jnp.abs(mean -  (weight**2)/(weight ** 2 + 1)) < jnp.sqrt(expected_mean * (1-expected_mean)/N) * 4, f"mean = {mean}, expected = {1 * (weight**2)/(weight ** 2 + 1)}"

if __name__ == "__main__":
    test_batch(N = 10000, seed = 1)
