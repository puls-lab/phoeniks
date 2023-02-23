import numpy as np
from numba import jit, complex128, float64, int64
from scipy.constants import c as c_0
# Own libraries
from .optimization_problem import get_H_approx

"""
Based on the SVMAF algorithm developed by:

Ioachim Pupeza, Rafal Wilk, and Martin Koch, 
"Highly accurate optical material parameter determination with THz time-domain spectroscopy," 
Opt. Express 15, 4335-4350 (2007)

https://doi.org/10.1364/OE.15.004335
"""


def confidence_interval(dnx, dsx):
    return np.sqrt(dnx ** 2 + dsx ** 2)


def get_derivatives(a, b, c, d):
    derivatives = {"df": {}, "dg": {}}
    derivatives["df"]["da"] = c / (c ** 2 + d ** 2)
    derivatives["df"]["db"] = d / (c ** 2 + d ** 2)
    derivatives["df"]["dc"] = (a * d ** 2 - a * c ** 2 - 2 * b * c * d) / (c ** 2 + d ** 2) ** 2
    derivatives["df"]["dd"] = (b * c ** 2 - b * d ** 2 - 2 * a * c * d) / (c ** 2 + d ** 2) ** 2
    derivatives["dg"]["da"] = - d / (c ** 2 + d ** 2)
    derivatives["dg"]["db"] = c / (c ** 2 + d ** 2)
    derivatives["dg"]["dc"] = - (b * c ** 2 - b * d ** 2 - 2 * a * c * d) / (c ** 2 + d ** 2) ** 2
    derivatives["dg"]["dd"] = (a * d ** 2 - a * c ** 2 - 2 * b * c * d) / (c ** 2 + d ** 2) ** 2
    return derivatives


def error_propagation(data_std, derivatives):
    dH = 0
    for x_std, derivative_key in zip(data_std, ["da", "db", "dc", "dd"]):
        dH += (x_std * derivatives[derivative_key]) ** 2
    return np.sqrt(dH)


@jit(float64[:](float64[:], int64), nopython=True)
def smooth(x, p):
    ph = int((p - 1) / 2)
    a = np.zeros(len(x))
    a[0] = x[0]
    for i in range(1, ph):
        a[i] = x[i]
        for j in range(1, i + 1):
            a[i] += x[i - j] + x[i + j]
        a[i] /= (2 * (i - 1) + 1)

    for i in range(ph, len(x) - ph):
        a[i] = x[i]
        for j in range(1, ph + 1):
            a[i] += x[i - j] + x[i + j]
        a[i] /= p

    for i in range(len(x) - ph, len(x)):
        a[i] = x[i]
        for j in range(1, len(x) - i):
            a[i] += x[i - j] + x[i + j]
        a[i] = a[i] / (2 * (len(x) - i - 1) + 1)
    return a


class SVMAF:
    def __init__(self, extract_obj, svmaf_iterations=20):
        self.alpha_smooth = None
        self.k_smooth = None
        self.n_smooth = None
        self.data = extract_obj.data
        self.svmaf_iterations = svmaf_iterations

        self.dH_real = None
        self.dH_imag = None
        self.setup()

    def setup(self):
        a = self.data.fd_sample_std.real
        b = self.data.fd_sample_std.imag
        c = self.data.fd_reference_std.real
        d = self.data.fd_reference_std.imag

        data_std = [self.data.fd_sample_std.real,
                    self.data.fd_sample_std.imag,
                    self.data.fd_reference_std.real,
                    self.data.fd_reference_std.imag]
        dark_std = [self.data.fd_dark_std.real,
                    self.data.fd_dark_std.imag,
                    self.data.fd_dark_std.real,
                    self.data.fd_dark_std.imag]
        total_std = [confidence_interval(ds, dn) for ds, dn in zip(data_std, dark_std)]
        derivatives = get_derivatives(a, b, c, d)
        self.dH_real = error_propagation(total_std, derivatives["df"])
        self.dH_imag = error_propagation(total_std, derivatives["dg"])

    def run(self, thickness):
        for _ in range(self.svmaf_iterations):
            n_smooth = smooth(self.data.n, 3)
            k_smooth = smooth(self.data.k, 3)
            H_approx = np.zeros(len(self.data.n), dtype=np.complex128)
            fd_sample_corrected = np.zeros(len(self.data.n), dtype=np.complex128)
            for i in range(len(self.data.n) - 1):
                H_approx[i] = get_H_approx(np.array([n_smooth[i], k_smooth[i]]),
                                           i,
                                           self.data.omega[i],
                                           thickness,
                                           H_approx,
                                           self.data.delta_max)
                fd_sample_corrected[i] = self.data.fd_reference[i] * H_approx[i]
                if np.abs(self.data.H[i].real - H_approx[i].real) > self.dH_real[i] or \
                        np.abs(self.data.H[i].imag - H_approx[i].imag) > self.dH_imag[i]:
                    n_smooth[i] = self.data.n[i]
                    k_smooth[i] = self.data.k[i]
            self.data.n = n_smooth
            self.data.k = k_smooth
        self.n_smooth = self.data.n
        self.k_smooth = self.data.k
        self.alpha_smooth = 4 * np.pi * self.k_smooth * self.data.frequency / c_0
        return self.n_smooth, self.k_smooth, self.alpha_smooth
