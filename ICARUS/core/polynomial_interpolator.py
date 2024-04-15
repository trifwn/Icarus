from functools import partial

import jax
import jax.numpy as jnp


class Polynomial2DInterpolator:
    def __init__(self, kx, ky) -> None:
        self.kx = kx
        self.ky = ky
        self.coefficients = jnp.zeros((kx + 1, ky + 1))

        self.x_mean = jnp.empty(0)
        self.x_std = jnp.empty(0)

        self.y_mean = jnp.empty(0)
        self.y_std = jnp.empty(0)

        self.z_mean = jnp.empty(0)
        self.z_std = jnp.empty(0)

    def fit(self, x: jax.Array, y: jax.Array, z: jax.Array) -> None:
        """
        Works in the following way:

        A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
        B = zp

        coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=-1)
        print(coeff)

        def f(x, y):
        return coeff[0] + coeff[1]*x + coeff[2]*y + coeff[3]*x**2 + coeff[4]*x**2*y + coeff[5]*x**2*y**2 + coeff[6]*y**2 + coeff[7]*x*y**2 + coeff[8]* x*y

        Args:
            x (jax.Array): _description_
            y (jax.Array): _description_
        """
        # Normalize the input data
        x_mean, x_std = jnp.mean(x), jnp.std(x)
        y_mean, y_std = jnp.mean(y), jnp.std(y)
        z_mean, z_std = jnp.mean(z), jnp.std(z)

        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std
        z = (z - z_mean) / z_std

        self.x_mean = x_mean
        self.x_std = x_std
        self.z_mean = z_mean

        self.y_mean = y_mean
        self.y_std = y_std
        self.z_std = z_std

        N = x.size
        # # Construct the Vandermonde matrices
        X_powers = jnp.vander(x, (self.kx + 1), increasing=True)
        Y_powers = jnp.vander(y, (self.ky + 1), increasing=True)

        # Create arrays of powers of x and y up to the specified order
        X_powers = jnp.vander(x, (self.kx + 1), increasing=True)
        Y_powers = jnp.vander(y, (self.ky + 1), increasing=True)

        # Reshape powers of x and y to have the same shape
        X_powers_reshaped = X_powers.reshape(N, self.kx + 1, 1)
        Y_powers_reshaped = Y_powers.reshape(N, 1, self.ky + 1)

        # Compute the outer product of powers of x and y
        A = (X_powers_reshaped * Y_powers_reshaped).reshape(N, -1)

        # Solve the least squares problem
        coefficients, r, rank, s = jnp.linalg.lstsq(A, z, rcond=-1)
        # print(f"Rank: {rank}")
        # print(f"Residual: {r}")
        # print(f"Singular values: {s}")
        self.coefficients = coefficients

    partial(jax.jit, static_argnums=(0,))

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        # If the arrays are scalar convert to 1D arrays
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)

        # If the dimensions of x and y are different flatten them
        if x.ndim > 1:
            x = x.flatten()
        if y.ndim > 1:
            y = y.flatten()

        x_dim = x.shape
        y_dim = y.shape
        # Assert that the dimensions of x and y are the same
        if not x_dim == y_dim:
            raise ValueError(f"The dimensions of x and y must be equal got x.dim = {x_dim} and y.dim = {y_dim} ")

        # Normalize the input data
        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std

        X_powers = jnp.vander(x, self.kx + 1, increasing=True)
        Y_powers = jnp.vander(y, self.ky + 1, increasing=True)

        # Convolving the powers of x and y
        # Reshape the Vandermonde matrices to match the shape of the Vandermonde matrix A
        X_powers_reshaped = X_powers[:, :, None]  # Add a new axis
        Y_powers_reshaped = Y_powers[:, None, :]
        A = X_powers_reshaped * Y_powers_reshaped

        # Flatten the resulting matrix A to match the expected shape
        A = A.reshape(-1, (self.kx + 1) * (self.ky + 1))
        z = jnp.dot(A, self.coefficients).reshape(x_dim)

        # Denormalize the output data
        z = z * self.z_std + self.z_mean
        return z
