import numpy as np
from scipy.constants import c as c_0
from numba import jit, complex128, float64, int64


@jit(complex128(float64, float64, float64, float64), nopython=True)
def p(n, k, omega, thickness):
    """Propagation in a material with complex refractive index and thickness."""
    # TODO:
    #  I wrote before
    #  np.exp(-omega * (k + 1j * n) * thickness / c_0)
    #  Possible bug? Ioachim wrote:
    #  (- omega * k * thickness - 1j * omega * n * thickness) / c_0
    #  but k should have the 1j, since its imaginary
    # I think its solved, the -1j before the bracket removes the 1j before the k
    # complex_n = n + 1j * k
    return np.exp((- omega * k * thickness - 1j * omega * n * thickness) / c_0)


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
    """Calculates the approximate transfer function based on a model."""
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
    """Compares the calculated transfer function (H_approx) with the numerical one (H) from the measurement data.

    Input:
    x (np.ndarray, size 1x2) : Possible values for n and k
    i (int) : Iteration variable for index in transfer function array
    thickness (float) : Thickness of material
    H (np.ndarray, complex) : Transfer function, calculated by experimental data
    H_approx (np.ndarray, complex) : Tranfer function, estimated by transfer function model
    delta_max (int) : Maximum number of fabry-perot echoes included in transfer function model

    Output:
    error (float) : Discrepancy for given x (n and k) between model and experimental transfer function."""
    H_approx[i] = get_H_approx(x, i, omega, thickness, H_approx, delta_max)
    u = np.unwrap(np.angle(H[:i + 1]))
    u_approx = np.unwrap(np.angle(H_approx[:i + 1]))
    error = np.abs(np.abs(H_approx[i]) - np.abs(H[i])) + np.abs(u[i] - u_approx[i])
    return error


def error_function2(x, i, omega, thickness, H, H_approx, delta_max):
    """Compares the calculated transfer function (H_approx) with the numerical one (H) from the measurement data.

    Input:
    x (np.ndarray, size 1x2) : Values used for n and k by the optimization algorithm
    i (int) : Iteration variable for index in transfer function array
    thickness (float) : Thickness of material
    H (np.ndarray, complex) : Transfer function, calculated by experimental data
    H_approx (np.ndarray, complex) : Tranfer function, estimated by transfer function model
    delta_max (int) : Maximum number of fabry-perot echoes included in transfer function model

    Output:
    error (float) : Discrepancy for given x (n and k) between model and experimental transfer function."""
    H_approx[i] = get_H_approx(x, i, omega, thickness, H_approx, delta_max)
    chi_1 = (np.log(np.abs(H_approx[i])) - np.log(np.abs(H[i]))) ** 2
    d = np.angle(H_approx[i]) - np.angle(H[i])
    chi_2 = (np.mod(d + np.pi, 2 * np.pi) - np.pi) ** 2
    error = chi_1 + chi_2
    return error


def error_function_thickness(thickness, obj):
    frequency, n, k, alpha = obj.run_optimization(thickness[0] * 1e-3)
    print(f"Thickness: {thickness[0] * 1e3:.1f} µm\tError: {np.sum(np.abs(np.diff(n))) * thickness[0] * 1e3:.4E}")
    return np.sum(np.abs(np.diff(n))) * thickness


if __name__ == "__main__":
    pass
