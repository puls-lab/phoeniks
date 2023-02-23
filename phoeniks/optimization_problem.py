import numpy as np
from scipy.constants import c as c_0
from numba import jit, complex128, float64, int64


@jit(complex128(float64, float64, float64, float64), nopython=True)
def p(n, k, omega, thickness):
    """Propagation in a material with complex refractive index and thickness."""
    # TODO: Possible bug? Ioachim wrote:
    #  (- omega * k * thickness - 1j * omega * n * thickness) / c_0
    #  but k should have the 1j, since its imaginary
    # complex_n = n + 1j * k
    return np.exp(-omega * (k + 1j * n) * thickness / c_0)


@jit(complex128(float64, float64, float64, float64), nopython=True)
def r(n1, n2, k1, k2):
    """Calculate the complex reflection coefficient."""
    complex_n1 = n1 - 1j * k1
    complex_n2 = n2 - 1j * k2
    return (complex_n2 - complex_n1) / (complex_n1 + complex_n2)


@jit(complex128(float64, float64, float64, float64), nopython=True)
def t(n1, n2, k1, k2):
    """Calculate the complex transmission coefficient."""
    complex_n1 = n1 - 1j * k1
    complex_n2 = n2 - 1j * k2
    return 2 * complex_n1 / (complex_n1 + complex_n2)


@jit(complex128(float64[:], int64, float64, float64, complex128[:], int64), nopython=True)
def get_H_approx(x, i, omega, thickness, H_approx, delta_max):
    """Calculates the approximate transfer function."""
    n = x[0]
    k = x[1]
    n_air = 1.00027
    k_air = 0
    H_approx[i] = p(n_air, k_air, omega, -thickness) * t(n_air, n, k_air, k) * t(n, n_air, k, k_air)
    # TODO: Possible bug / discrepancy of Ioachims code to the paper? fabry_perot should start at P_1, not zero?
    fabry_perot = 0
    # Careful with looping variable, m is specifically chosen because:
    # 1) i is taken as index for the arrays,
    # 2) k is taken for imaginary part of refractive index.
    # Loop starts at zero (and not like the sum in the paper at i=1), to multiply P_1 with the other terms above.
    for m in range(delta_max):
        fabry_perot += p(n, k, omega, thickness) * (r(n, n_air, k, k_air) ** 2 * p(n, k, omega, thickness) ** 2) ** m
    H_approx[i] *= fabry_perot
    return H_approx[i]


# @jit(float64(float64[:], int64, float64, float64, complex128[:], complex128[:], int64), nopython=True)
# np.unwrap is not supported by numba
def error_function(x, i, omega, thickness, H, H_approx, delta_max):
    """Compares the calculated transfer function (H_approx) with the numerical one (H) from the measurement data."""
    H_approx[i] = get_H_approx(x, i, omega, thickness, H_approx, delta_max)
    u = np.unwrap(np.angle(H[:i + 1]))
    u_approx = np.unwrap(np.angle(H_approx[:i + 1]))
    error = np.abs(np.abs(H_approx[i]) - np.abs(H[i])) + np.abs(u[i] - u_approx[i])
    return error


if __name__ == "__main__":
    pass
