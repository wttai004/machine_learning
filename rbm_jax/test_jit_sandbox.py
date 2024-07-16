import jax
import jax.numpy as jnp

# Define a pytree with nested structures
pytree = {'a': jnp.array([1.0, 2.0, 3.0]), 'b': [jnp.array([4.0, 5.0]), (jnp.array([6.0]), jnp.array([7.0]))]}

# Apply a function to all leaves of the pytree
def add_one(x):
    return x + 1

new_pytree = jax.tree_map(add_one, pytree)
print(new_pytree)

# Flatten the pytree into a list of leaves and a treedef
leaves, treedef = jax.tree_flatten(pytree)
print(leaves)
print(treedef)

# Unflatten the list of leaves back into the original structure
restored_pytree = jax.tree_unflatten(treedef, leaves)
print(restored_pytree)