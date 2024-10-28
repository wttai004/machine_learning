import flax.linen as nn
#import netket.nn as nn
import netket as nk
import netket.experimental as nkx
from netket.utils.types import NNInitFunc
from netket.nn.masked_linear import default_kernel_init
from typing import Any, Callable, Sequence
from functools import partial
import jax
import jax.numpy as jnp

DType = Any

class LogSlaterDeterminant(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    complex: bool = False  # Toggle for complex or real parameters

    def setup(self):
        if self.complex:
            self.param_dtype = jnp.complex64
            self.kernel_init = self.complex_kernel_init
        else:
            self.param_dtype = jnp.float32
            self.kernel_init = jax.nn.initializers.lecun_normal()

    def complex_kernel_init(self, key, shape, dtype=jnp.complex64):
        """Initializes complex parameters by combining real and imaginary parts."""
        real_init = jax.nn.initializers.lecun_normal()
        imag_init = jax.nn.initializers.lecun_normal()
        key_real, key_imag = jax.random.split(key)
        real = real_init(key_real, shape, dtype=jnp.float32)
        imag = imag_init(key_imag, shape, dtype=jnp.float32)
        return real + 1j * imag

    @nn.compact
    def __call__(self, n):
        # Initialize the parameter matrix M
        M = self.param(
            'M',
            self.kernel_init,
            (self.hilbert.size, self.hilbert.n_fermions),
            self.param_dtype
        )

        @partial(jnp.vectorize, signature='(n)->()')
        def log_sd(n):
            # Find the positions of the occupied orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            
            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = M[R]

            # Compute the logarithm of the determinant
            return nk.jax.logdet_cmplx(A)

        return log_sd(n)


class LogNeuralJastrowSlater(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    hidden_units: int
    num_hidden_layers: int = 1
    complex: bool = False  

    def setup(self):
        if self.complex:
            self.param_dtype = jnp.complex64
            self.kernel_init = self.complex_kernel_init
        else:
            self.param_dtype = jnp.float32
            self.kernel_init = jax.nn.initializers.lecun_normal()

    def complex_kernel_init(self, key, shape, dtype=jnp.complex64):
        """Initializes complex parameters by combining real and imaginary parts."""
        real_init = jax.nn.initializers.lecun_normal()
        imag_init = jax.nn.initializers.lecun_normal()
        key_real, key_imag = jax.random.split(key)
        real = real_init(key_real, shape, dtype=jnp.float32)
        imag = imag_init(key_imag, shape, dtype=jnp.float32)
        return real + 1j * imag

    @nn.compact
    def __call__(self, n):
       
        @partial(jnp.vectorize, signature='(n)->()')
        def log_wf(n):
            #Bare Slater Determinant (N x Nf matrix of the orbital amplitudes) 
            M = self.param('M', self.kernel_init, (self.hilbert.size, self.hilbert.n_fermions,), self.param_dtype)

            #Construct the Neural Jastrow
            for _ in range(self.num_hidden_layers):
                J = nn.Dense(self.hidden_units, param_dtype=self.param_dtype, kernel_init=self.kernel_init)(J)
                J = jax.nn.tanh(J)
            J = J.sum()
            
            # Find the positions of the occupied orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            # Select the N rows of M corresponding to the occupied orbitals, obtaining the Nf x Nf slater matrix
            A = M[R]
            # compute the (log) determinant and add the Jastrow
            # (when exponentiating this becomes a product of the slater and jastrow terms)
            return nk.jax.logdet_cmplx(A)+J

        return log_wf(n)
    
class LogNeuralBackflow(nn.Module):
    hilbert: nkx.hilbert.SpinOrbitalFermions
    hidden_units: int
    num_hidden_layers: int = 1
    complex: bool = True  # Toggle for complex or real parameters

    def setup(self):
        if self.complex:
            self.param_dtype = jnp.complex64
            self.kernel_init = self.complex_kernel_init
        else:
            self.param_dtype = jnp.float32
            self.kernel_init = jax.nn.initializers.lecun_normal()

    def complex_kernel_init(self, key, shape, dtype=jnp.complex64):
        """Initializes complex parameters by combining real and imaginary parts."""
        real_init = jax.nn.initializers.lecun_normal()
        imag_init = jax.nn.initializers.lecun_normal()
        key_real, key_imag = jax.random.split(key)
        real = real_init(key_real, shape, dtype=jnp.float32)
        imag = imag_init(key_imag, shape, dtype=jnp.float32)
        return real + 1j * imag

    @nn.compact
    def __call__(self, n):
       
        @partial(jnp.vectorize, signature='(n)->()')
        def log_sd(n):
            #Bare Slater Determinant (N x Nf matrix of the orbital amplitudes) 
            M = self.param('M', self.kernel_init, (self.hilbert.size, self.hilbert.n_fermions,), self.param_dtype)

            # Construct the Backflow. Takes as input strings of $N$ occupation numbers, outputs an $N x Nf$ matrix
            # that modifies the bare orbitals.
            for _ in range(self.num_hidden_layers):
                F = nn.Dense(self.hidden_units, param_dtype=self.param_dtype)(F)
                F = jax.nn.tanh(F)
            # last layer, outputs N x Nf values
            F = nn.Dense(self.hilbert.size * self.hilbert.n_fermions, param_dtype=self.param_dtype)(F)
            # reshape into M and add
            M += F.reshape(M.shape)
            
            #Find the positions of the occupied, backflow-modified orbitals
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            A = M[R]
            return nk.jax.logdet_cmplx(A)

        return log_sd(n)