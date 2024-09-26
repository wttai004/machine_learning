import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MyPytreeClass:
    def __init__(self, a, b, func):
        self.a = a  # Leaf
        self.b = b  # Leaf
        self.func = func  # Function object

    def __repr__(self):
        return f"MyPytreeClass(a={self.a}, b={self.b}, func={self.func})"

    def _tree_flatten(self):
        # Leaves are the values that we want to be treated as JAX arrays
        leaves = (self.a, self.b)
        # Aux data includes the function object
        aux_data = (self.func,)
        return leaves, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        func, = aux_data  # Unpack the function object
        return cls(*leaves, func)

# Example function
def example_func(x):
    return x ** 2

# Example usage
obj = MyPytreeClass(a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6]), func=example_func)
flattened, aux_data = obj._tree_flatten()
print("Flattened:", flattened)
print("Aux data:", aux_data)

reconstructed = MyPytreeClass._tree_unflatten(aux_data, flattened)
print("Reconstructed:", reconstructed)

# Using the function
print("Function output:", reconstructed.func(3))
