import jax.numpy as jnp
import random
from model import Model  # Assume the refactored code is saved in model.py

def test_project_and_unproject_spin():
    print("Testing project_spin and unproject_spin...")
    model = Model(2, 2)
    spin = model.get_random_spins()
    assert spin.shape == (2, 2, 2), "Random spins have incorrect shape."
    projected_spin = model.project_spin(spin)
    assert projected_spin.shape == (4,), "Random spins have incorrect shape."
    unprojected_spin = model.unproject_spin(projected_spin)
    assert unprojected_spin.shape == (2, 2, 2), "Random spins have incorrect shape."
    assert jnp.array_equal(spin, unprojected_spin), "Unprojected spin does not match original spin."

def test_get_random_spins():
    print("Testing get_random_spins...")
    model = Model(4, 4)
    spin = model.get_random_spins()
    assert spin.shape == (4, 4, 2), "Random spins have incorrect shape."
    assert jnp.all(jnp.all(spin == jnp.array([1, 0]), axis=-1) | jnp.all(spin == jnp.array([0, 1]), axis=-1)), "Random spins are not properly initialized."
    spin2 = model.get_random_spins()
    assert not jnp.array_equal(spin, spin2), "Random spins are not random."

def test_flip_random_spin():
    print("Testing flip_random_spin...")
    model = Model(4, 4)
    spin = model.get_random_spins()
    flipped_spin = model.flip_random_spin(spin)
    #print(spin)
    #print(flipped_spin)
    assert spin.shape == flipped_spin.shape, "Flipped spin shape mismatch."
    assert jnp.any(spin != flipped_spin), "Flipped spin is identical to original spin."
    assert (spin != flipped_spin).sum() == 2, "More than one spin flipped."

def test_generate_local_spins():
    print("Testing generate_local_spins...")
    model = Model(3, 3)
    spin = model.get_random_spins()
    local_spins = model.generate_local_spins(spin, change=1)
    assert len(local_spins) == 10, "Incorrect number of local spins for change=1."
    for spin in local_spins:
        assert jnp.all(jnp.all(spin == jnp.array([1, 0]), axis=-1) | jnp.all(spin == jnp.array([0, 1]), axis=-1)), "Local spins are not properly initialized."
    local_spins_change_2 = model.generate_local_spins(spin, change=2)
    assert len(local_spins_change_2) == 46, "Incorrect number of local spins for change=2."
    for spin in local_spins_change_2:
        assert jnp.all(jnp.all(spin == jnp.array([1, 0]), axis=-1) | jnp.all(spin == jnp.array([0, 1]), axis=-1)), "Local spins are not properly initialized."

def test_vdot():
    print("Testing vdot...")
    model = Model(2, 2)
    spin1 = model.get_random_spins()
    spin2 = spin1.copy()
    v_value = model.vdot(spin1, spin2)
    assert v_value == 1, "vdot of identical spins should be 1."
    spin2 = model.flip_random_spin(spin1)
    v_value = model.vdot(spin1, spin2)
    assert v_value == 0, "vdot of orthogonal spins should be 0."

if __name__ == "__main__":
    random.seed(0)  # Set random seed for reproducibility
    test_project_and_unproject_spin()
    test_get_random_spins()
    test_flip_random_spin()
    test_generate_local_spins()
    test_vdot()
    print("All tests passed!")
