import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import fmin
from scipy.constants import c as c_0
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import copy
# Internal libraries
from .thz_data import Data
from .optimization_problem import error_function
from .svmaf import SVMAF


def _D(n, k, m):
    """D-Parameter for thickness extraction with total variation method, see Eq. (19)

    Timothy D. Dorney, Richard G. Baraniuk, and Daniel M. Mittleman
    Material parameter estimation with terahertz time-domain spectroscopy
    J. Opt. Soc. Am. A 18, 1562-1571 (2001)
    https://doi.org/10.1364/JOSAA.18.001562

    Input:
    n (np.ndarray, float): Real part of refractive index
    k (np.ndarray, float): Imaginary part of refractive index
    m (np.ndarray, int): Index for both arrays.

    Output:
    D-parameter (np.ndarray, float): See get_thickness method of Extraction class."""
    return np.abs(n[m - 1] - n[m]) + np.abs(k[m - 1] - k[m])


class Extraction:
    def __init__(self, data: Data, progress_bar=True):
        # Data contains all time and frequency domain data as well as derived values
        self.data = data
        self.original = copy.deepcopy(data)
        # Should there be a progress bar (Useful for analyzing a single file, but disturbing for multiple files)
        self.progress_bar = progress_bar
        # Accuracy of minimization algorithm
        self.accuracy = 1e-4
        # Refractive index (real and imaginary) of air
        self.n_air = 1.00027
        self.k_air = 0
        # How many Fabry-Perot echoes fit in the rest of the time trace
        self.delta_max = None

    def unwrap_phase(self, frequency_start=None, frequency_stop=None):
        if frequency_start is not None and frequency_stop is None:
            if frequency_start < self.original.frequency[0] or frequency_start > self.original.frequency[-1]:
                raise ValueError("frequency_start outside frequency range of data.")
            idx_start = np.where(self.original.frequency >= frequency_start)[0][0]
            self.data.frequency = self.original.frequency[idx_start:]
            self.data.omega = 2 * np.pi * self.data.frequency
            if self.data.mode != "reference_sample":
                self.data.fd_dark = self.data.fd_dark[idx_start:]
            self.data.fd_reference = self.data.fd_reference[idx_start:]
            self.data.fd_sample = self.data.fd_sample[idx_start:]
        elif frequency_start is None and frequency_stop is not None:
            if frequency_stop < self.original.frequency[0] or frequency_stop > self.original.frequency[-1]:
                raise ValueError("frequency_stop outside frequency range of data.")
            idx_stop = np.where(self.original.frequency >= frequency_stop)[0][0]
            self.data.frequency = self.original.frequency[:idx_stop]
            self.data.omega = 2 * np.pi * self.data.frequency
            if self.data.mode != "reference_sample":
                self.data.fd_dark = self.data.fd_dark[:idx_stop]
            self.data.fd_reference = self.data.fd_reference[:idx_stop]
            self.data.fd_sample = self.data.fd_sample[:idx_stop]
        else:
            if frequency_start < self.original.frequency[0] or frequency_start > self.original.frequency[-1]:
                raise ValueError("frequency_start outside frequency range of data.")
            if frequency_stop < self.original.frequency[0] or frequency_stop > self.original.frequency[-1]:
                raise ValueError("frequency_stop outside frequency range of data.")
            idx_start = np.where(self.original.frequency >= frequency_start)[0][0]
            idx_stop = np.where(self.original.frequency >= frequency_stop)[0][0]
            self.data.frequency = self.original.frequency[idx_start:idx_stop]
            self.data.omega = 2 * np.pi * self.data.frequency
            if self.data.mode != "reference_sample":
                self.data.fd_dark = self.data.fd_dark[idx_start:]
            self.data.fd_reference = self.data.fd_reference[idx_start:]
            self.data.fd_sample = self.data.fd_sample[idx_start:]
        self.data.H = self.data.fd_sample / self.data.fd_reference
        self.data.phase_reference = np.unwrap(np.angle(self.data.fd_reference))
        self.data.phase_sample = np.unwrap(np.angle(self.data.fd_sample))
        self.data.phase_H = np.unwrap(np.angle(self.data.H))

    def advanced_unwrap_phase(self, amplitude_threshold=0.1, debug=False):
        # data_abs = np.abs(self.data.fd_sample)-np.abs(self.data.fd_dark)
        # data_abs /= np.max(data_abs)
        data_abs = np.abs(self.data.fd_sample)
        dark_abs = np.abs(self.data.fd_dark)
        data_abs_lower = data_abs[:np.argmax(data_abs) + 1]
        dark_abs_lower = dark_abs[:np.argmax(data_abs) + 1]
        data_abs_upper = data_abs[np.argmax(data_abs):]
        dark_abs_upper = dark_abs[np.argmax(data_abs):]
        if debug:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            ax[0].semilogy(self.data.frequency[:np.argmax(data_abs) + 1], data_abs_lower, alpha=0.6,
                           label="signal lower")
            ax[0].semilogy(self.data.frequency[:np.argmax(data_abs) + 1], dark_abs_lower, alpha=0.6, color="black",
                           label="dark lower")
            ax[0].semilogy(self.data.frequency[np.argmax(data_abs):], data_abs_upper, alpha=0.6, label="upper")
            ax[0].semilogy(self.data.frequency[np.argmax(data_abs):], dark_abs_upper, alpha=0.6, color="black",
                           label="dark upper")
            ax[0].legend(loc="upper right")
            ax[0].grid(True)
            ax[0].xaxis.set_major_formatter(EngFormatter("Hz"))
            ax[0].set_xlabel("Frequency")
            ax[1].plot(self.data.frequency, self.data.phase_reference, label="Reference")
            ax[1].plot(self.data.frequency, self.data.phase_sample, label="Sample")
            ax[1].plot(self.data.frequency, self.data.phase_H, label="H, before correction")
            # plt.show()

        idx_start = (data_abs_lower < 10 * dark_abs_lower).nonzero()[0][-1]
        idx_stop = (data_abs_upper < 10 * dark_abs_upper).nonzero()[0][0]
        self.data.frequency = self.data.frequency[idx_start:idx_stop]
        self.data.omega = 2 * np.pi * self.data.frequency
        self.data.fd_dark = self.data.fd_dark[idx_start:idx_stop]
        self.data.fd_reference = self.data.fd_reference[idx_start:idx_stop]
        self.data.fd_sample = self.data.fd_sample[idx_start:idx_stop]
        self.data.H = self.data.fd_sample / self.data.fd_reference
        self.data.phase_H = np.unwrap(np.angle(self.data.H))
        z = np.polyfit(self.data.frequency, self.data.phase_H, 1)
        p = np.poly1d(z)
        self.data.phase_H -= np.round(p[0] / np.pi) * np.pi
        self.data.H = np.abs(self.data.H) * np.exp(1j * self.data.phase_H)
        if debug:
            ax[1].plot(self.data.frequency, self.data.phase_H, color="black", label="H, after correction")
            ax[1].legend(loc="upper right")
            ax[1].grid(True)
            ax[1].xaxis.set_major_formatter(EngFormatter("Hz"))
            ax[1].set_xlabel("Frequency")

    def get_reliable_frequency_range(self) -> (float, float):
        """Sample amplitude in frequency domain needs to be at least two times stronger than the dark measurement."""
        idx = np.where(np.abs(self.data.fd_sample) > 2 * np.abs(self.data.fd_dark))[0]
        idx_start = idx[0]
        idx_stop = idx[-1]
        return self.data.frequency[idx_start], self.data.frequency[idx_stop]

    def get_time_shift(self) -> float:
        """Calculates the time delta between the peak of the THz pulse in reference trace and the peak in the sample
        trace. """
        idx_peak_reference = np.argmax(np.abs(self.data.td_reference))
        idx_peak_sample = np.argmax(np.abs(self.data.td_sample))
        delta_time = np.abs(self.data.time[idx_peak_reference] - self.data.time[idx_peak_sample])
        return delta_time

    def get_initial_nk(self, thickness: float) -> np.ndarray:
        """Get initial values for n and k.

        Input:
        thickness (float): Thickness of sample in [m]

        Output:
        n (np.ndarray, float): Real part of refractive index
        k (np.ndarray, float): Imaginary part of refractive index
        """
        delta_time = self.get_time_shift()
        n = np.zeros(len(self.data.frequency))
        k = np.zeros(len(self.data.frequency))
        n += (c_0 / thickness) * delta_time + self.n_air
        amplitude_ratio = np.max(np.abs(self.data.td_sample)) / np.max(np.abs(self.data.td_reference))
        if self.data.omega[0] == 0:
            # Circumvent division by 0 error
            k[0] = 0
            k[1:] = -c_0 * np.log(amplitude_ratio) / (thickness * self.data.omega[1:])
        else:
            k += -c_0 * np.log(amplitude_ratio) / (thickness * self.data.omega)
        self.data.n = n
        self.data.k = k
        return self.data.n, self.data.k

    def get_max_delta(self, thickness) -> int:
        """How many echoes for a given sample thickness with given refractive index "n"
        can fit in the time span between reference peak and end of the time trace?"""
        timespan = np.abs(self.data.time[-1] - self.data.time[np.argmax(np.abs(self.data.td_reference))])
        optical_thickness = thickness * np.mean(self.data.n) / c_0
        delta_max = int(np.round((timespan / optical_thickness) / 2))
        self.data.delta_max = delta_max
        return delta_max

    def get_thickness(self, thickness, thickness_range=50e-6, step_size=5e-6):
        """Tries to extract the thickness by total variation method, based on:

        Timothy D. Dorney, Richard G. Baraniuk, and Daniel M. Mittleman
        Material parameter estimation with terahertz time-domain spectroscopy
        J. Opt. Soc. Am. A 18, 1562-1571 (2001)
        https://doi.org/10.1364/JOSAA.18.001562

        Input:
        thickness (float): Guessed thickness of sample in [m]
        thickness_range (float): Range of thicknesses to test, an array is created with thickness +- thickness_range in [m].
        step_size (float): Step size of the thickness array in [m].

        Output:
        thickness_array (np.ndarray, float): Tested thickness array
        tv_1 (np.ndarray, float): Total variation method of degree 1, minimum gives the best thickness approximation
        tv_2 (np.ndarray, float): Total variation method of degree 2, minimum gives the best thickness approximation
        tv_s (np.ndarray, float): Total variation method compared to SVMAF data,
                                  only calculated when data is provided with standard deviation values.
                                  Minimum gives the best thickness approximation
        """
        thickness_array = np.arange(thickness - thickness_range, thickness + thickness_range, step_size)
        # Total variation dictionary
        tv_dict = {}
        # Total variation of order 1
        tv_dict["tv_1"] = np.zeros(len(thickness_array))
        # Total variation of order 2
        tv_dict["tv_2"] = np.zeros(len(thickness_array))
        # Total variation, Nick's method
        tv_dict["tv_n"] = np.zeros(len(thickness_array))
        # Total variation with SVMAF values (only possible when standard deviation values are provided)
        if self.data.mode == "reference_sample_dark_standard_deviations":
            tv_dict["tv_s"] = np.zeros(len(thickness_array))
        # Don't show progress bar for each single iteration, instead we initialize global progress bar
        self.progress_bar = False
        for idx, thickness in enumerate(tqdm(thickness_array)):
            # Get refractive index for given thickness
            frequency, n, k, alpha = self.run_optimization(thickness)
            # Index array needs to start at 1, since D parameter used m-1 as index
            m = np.arange(1, len(n) - 1)
            tv_dict["tv_1"][idx] = np.sum(_D(n, k, m))
            tv_dict["tv_2"][idx] = np.sum(np.abs(_D(n, k, m) - _D(n, k, m + 1)))
            tv_dict["tv_n"][idx] = np.sum(np.abs(np.diff(n)))
            if self.data.mode == "reference_sample_dark_standard_deviations":
                svmaf_obj = SVMAF(self)
                n_smooth, k_smooth, alpha_smooth = svmaf_obj.run(thickness=thickness)
                tv_dict["tv_s"][idx] = np.abs(n - n_smooth) + np.abs(k - k_smooth)
        for key, value in tv_dict.items():
            print(f"{key}, optimal thickness: {EngFormatter('m')(thickness_array[np.argmin(value)])}")
        return thickness_array, tv_dict

    def run_optimization(self, thickness, delta_max=None):
        """Runs the optimization with Nelder-Mead minimization algorithm for each frequency.
        Major improvement: Use solution from previous frequency as start point instead of the initial n and k.

        Input:
        thickness (float): Thickness of sample in [m]
        delta_max (int): Maximum number of possible echoes in rest of sample time trace.

        Output:
        frequency (np.ndarray, float): Positive frequencies of evaluated frequency range in [Hz]
        n (np.ndarray, float): Real part of refractive index
        k (np.ndarray, float): Imaginary part of refractive index
        alpha (np.ndarray, float): Absorption coefficient in [m^-1], needs to be multiplied by 0.01 to get [cm^-1]
        """
        if delta_max is not None:
            self.delta_max = delta_max
        else:
            self.delta_max = self.get_max_delta(thickness)
        self.data.H_approx = np.zeros(len(self.data.omega), dtype=np.complex128)
        if self.progress_bar:
            for i, w in enumerate(tqdm(self.data.omega)):
                res = fmin(error_function,
                           x0=np.array([self.data.n[i], self.data.k[i]]),
                           xtol=self.accuracy,
                           args=(
                               i, self.data.omega[i], thickness, self.data.H, self.data.H_approx, self.data.delta_max),
                           disp=False)
                self.data.n[i] = res[0]
                self.data.k[i] = res[1]
        else:
            for i, w in enumerate(self.data.omega):
                res = fmin(error_function,
                           x0=np.array([self.data.n[i], self.data.k[i]]),
                           xtol=self.accuracy,
                           args=(
                               i, self.data.omega[i], thickness, self.data.H, self.data.H_approx, self.data.delta_max),
                           disp=False)
                self.data.n[i] = res[0]
                self.data.k[i] = res[1]
        self.data.alpha = 4 * np.pi * self.data.frequency * self.data.k / c_0
        return self.data.frequency, self.data.n, self.data.k, self.data.alpha
