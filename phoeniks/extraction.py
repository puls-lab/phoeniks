import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import fmin, curve_fit
from scipy.constants import c as c_0
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import copy
# Internal libraries
from .thz_data import Data
from .optimization_problem import error_function, error_function2, error_function_thickness
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

    def _cut_frequency(self, idx_start, idx_stop):
        self.data.frequency = np.copy(self.original.frequency[idx_start:idx_stop])
        self.data.omega = 2 * np.pi * self.data.frequency
        if self.data.mode != "reference_sample":
            self.data.fd_dark = np.copy(self.original.fd_dark[idx_start:idx_stop])
        self.data.fd_reference = np.copy(self.original.fd_reference[idx_start:idx_stop])
        self.data.fd_sample = np.copy(self.original.fd_sample[idx_start:idx_stop])

    def unwrap_phase(self, frequency_start=None, frequency_stop=None):
        if frequency_start is not None and frequency_stop is None:
            if frequency_start < self.original.frequency[0] or frequency_start > self.original.frequency[-1]:
                raise ValueError("frequency_start outside frequency range of data.")
            idx_start = np.where(self.original.frequency >= frequency_start)[0][0]
            self._cut_frequency(idx_start=idx_start, idx_stop=-1)
        elif frequency_start is None and frequency_stop is not None:
            if frequency_stop < self.original.frequency[0] or frequency_stop > self.original.frequency[-1]:
                raise ValueError("frequency_stop outside frequency range of data.")
            idx_stop = np.where(self.original.frequency >= frequency_stop)[0][0]
            self._cut_frequency(idx_start=0, idx_stop=idx_stop)
        else:
            if frequency_start < self.original.frequency[0] or frequency_start > self.original.frequency[-1]:
                raise ValueError("frequency_start outside frequency range of data.")
            if frequency_stop < self.original.frequency[0] or frequency_stop > self.original.frequency[-1]:
                raise ValueError("frequency_stop outside frequency range of data.")
            idx_start = np.where(self.original.frequency >= frequency_start)[0][0]
            idx_stop = np.where(self.original.frequency >= frequency_stop)[0][0]
            self._cut_frequency(idx_start=idx_start, idx_stop=idx_stop)
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
        can fit in the time span between reference peak and end of the time trace?
        Subtract pure zero-padded data"""
        non_padded_time_data = self.data.time
        non_padded_time_data = non_padded_time_data[~(self.data.td_reference == 0)]
        timespan = np.abs(non_padded_time_data[-1] - non_padded_time_data[np.argmax(np.abs(self.data.td_reference))])
        optical_thickness = thickness * np.mean(self.data.n) / c_0
        delta_max = int(np.round((timespan / optical_thickness) / 2))
        self.data.delta_max = delta_max
        return delta_max

    def get_thickness_array(self, thickness, thickness_range=50e-6, step_size=5e-6):
        """Tries to extract the thickness by total variation method, based on:

        Timothy D. Dorney, Richard G. Baraniuk, and Daniel M. Mittleman
        Material parameter estimation with terahertz time-domain spectroscopy
        J. Opt. Soc. Am. A 18, 1562-1571 (2001)
        https://doi.org/10.1364/JOSAA.18.001562

        Nick's method, slightly different implementation of TV method:
        Greenall, Nicholas Robert
        Parameter Extraction and Uncertainty in Terahertz Time-Domain Spectroscopic Measurements
        PhD thesis, University of Leeds (2017)
        https://etheses.whiterose.ac.uk/19045/

        Chen's and Pickwell-MacPherson's method based on the offset exponential method:
        Chen X, Pickwell-MacPherson E.
        A Sensitive and Versatile Thickness Determination Method Based on Non-Inflection Terahertz Property Fitting.
        Sensors. 2019; 19(19):4118.
        https://doi.org/10.3390/s19194118

        Ioachim's method, based on the standard deviation at each frequency
        Ioachim Pupeza, Rafal Wilk, and Martin Koch,
        "Highly accurate optical material parameter determination with THz time-domain spectroscopy,"
        Opt. Express 15, 4335-4350 (2007)
        https://doi.org/10.1364/OE.15.004335

        Input:
        thickness (float): Guessed thickness of sample in [m]
        thickness_range (float): Range of thicknesses to test, an array is created with thickness +- thickness_range in [m].
        step_size (float): Step size of the thickness array in [m].

        Output:
        thickness_array (np.ndarray, float): Tested thickness array
        tv_1 (np.ndarray, float): Total variation method of degree 1.
        tv_2 (np.ndarray, float): Total variation method of degree 2.
        tv_2 (np.ndarray, float): Total variation based on Nick's method.
        offset_exponential (np.ndarray, float): Offset exponential method based on
                                                Chen's and Pickwell-MacPherson's method.
        tv_s (np.ndarray, float): Total variation method compared to SVMAF data,
                                  only calculated when data is provided with standard deviation values.
        """
        thickness_array = np.arange(thickness - thickness_range, thickness + thickness_range, step_size)
        # Thickness error dictionary
        thickness_error_dict = {}
        # Total variation of order 1
        thickness_error_dict["tv_1"] = np.zeros(len(thickness_array))
        # Total variation of order 2
        thickness_error_dict["tv_2"] = np.zeros(len(thickness_array))
        # Total variation, Nick's method
        thickness_error_dict["tv_n"] = np.zeros(len(thickness_array))
        # Offset exponential method
        thickness_error_dict["offset_exponential"] = np.zeros(len(thickness_array))
        # Total variation with SVMAF values (only possible when standard deviation values are provided)
        if self.data.mode == "reference_sample_dark_standard_deviations":
            thickness_error_dict["tv_s"] = np.zeros(len(thickness_array))
        # Don't show progress bar for each single iteration, instead we initialize global progress bar
        self.progress_bar = False
        for idx, thickness in enumerate(tqdm(thickness_array)):
            # Get refractive index for given thickness
            frequency, n, k, alpha = self.run_optimization(thickness)
            # Index array needs to start at 1, since D parameter used m-1 as index
            m = np.arange(1, len(n) - 1)
            thickness_error_dict["tv_1"][idx] = np.sum(_D(n, k, m))
            thickness_error_dict["tv_2"][idx] = np.sum(np.abs(_D(n, k, m) - _D(n, k, m + 1)))
            thickness_error_dict["tv_n"][idx] = np.sum(np.abs(np.diff(n))) * thickness
            thickness_error_dict["offset_exponential"][idx] = self.get_RMSE_oe(frequency, n, k)
            if self.data.mode == "reference_sample_dark_standard_deviations":
                svmaf_obj = SVMAF(self)
                n_smooth, k_smooth, alpha_smooth = svmaf_obj.run(thickness=thickness)
                thickness_error_dict["tv_s"][idx] = np.sum(np.abs(n - n_smooth)) + np.sum(np.abs(k - k_smooth))
        thickness_error_dict["tv_1"] /= np.max(thickness_error_dict["tv_1"])
        thickness_error_dict["tv_2"] /= np.max(thickness_error_dict["tv_2"])
        thickness_error_dict["tv_n"] /= np.max(thickness_error_dict["tv_n"])
        thickness_error_dict["offset_exponential"] /= np.max(thickness_error_dict["offset_exponential"])
        for key, value in thickness_error_dict.items():
            print(f"{key}, optimal thickness: {EngFormatter('m')(thickness_array[np.argmin(value)])}")
        return thickness_array, thickness_error_dict

    def get_RMSE_oe(self, frequency, n, k):
        """
        Calculates the root-mean-square-error (RMSE) from the offset exponential function to extracted refractive index
        and attenuation coefficient. Since the wrong thickness introduces oscillations to both parameters which the
        offset exponential function cannot follow, the RMSE will increase. The minimum RMSE for multiple tested
        thicknesses is a good indicator for the thickness of the sample.

        Based on the paper
        > Chen X, Pickwell-MacPherson E.
        > A Sensitive and Versatile Thickness Determination Method Based on Non-Inflection Terahertz Property Fitting.
        > Sensors. 2019; 19(19):4118.
        > https://doi.org/10.3390/s19194118

        with the additional modification to evaluate RMSE for n and k, based on
        > Mukherjee, S., Kumar, N.M.A., Upadhya, P.C. et al.
        > A review on numerical methods for thickness determination in  terahertz time-domain spectroscopy.
        > Eur. Phys. J. Spec. Top. 230, 4099–4111 (2021).
        > https://doi.org/10.1140/epjs/s11734-021-00215-9
        """
        rmse_real = np.nan
        rmse_imag = np.nan
        a, b, c = self.get_inital_guess_oe(frequency / 1e12, n)
        try:
            popt, pcov = curve_fit(self.offset_exponential, p0=np.array([a, b, c]), xdata=frequency / 1e12, ydata=n)
            rmse_real = np.sqrt(np.sum((n - self.offset_exponential(frequency / 1e12, *popt)) ** 2) / len(n))
        except RuntimeError:
            if self.debug:
                print(
                    "INFO: Could not find good fitting parameter for freq. vs. n for the offset exponential function.")
        a, b, c = self.get_inital_guess_oe(frequency / 1e12, k)
        try:
            popt, pcov = curve_fit(self.offset_exponential, p0=np.array([a, b, c]), xdata=frequency / 1e12, ydata=k)
            rmse_imag = np.sqrt(np.sum((k - self.offset_exponential(frequency / 1e12, *popt)) ** 2) / len(k))
        except RuntimeError:
            if self.debug:
                print(
                    "INFO: Could not find good fitting parameter for freq. vs. k for the offset exponential function.")
        total_rmse = np.nansum(np.array([rmse_real, rmse_imag]))
        return total_rmse

    @staticmethod
    def offset_exponential(x, a, b, c):
        # Offset exponential function, defined as
        # y = a + b * np.exp(c * x)
        # Careful, this is a different (but equivalent) definition as in the original publication:
        #
        # Chen X, Pickwell-MacPherson E.
        # A Sensitive and Versatile Thickness Determination Method Based on Non-Inflection Terahertz Property Fitting.
        # Sensors. 2019; 19(19):4118.
        # https://doi.org/10.3390/s19194118
        return a + b * np.exp(c * x)

    def get_inital_guess_oe(self, x, y):
        # Finding good initial guesses for a, b, c for the function
        # y = a + b * np.exp(c * x)
        # "Regressions et equations integrales" from Jean Jacquelin
        # https://en.scribd.com/doc/14674814/Regressions-et-equations-integrales
        # See also: https://github.com/scipy/scipy/pull/9158
        # https://scikit-guess.readthedocs.io/en/latest/appendices/reei/
        # translation.html#non-linear-regression-of-the-types-power-exponential-logarithmic-weibull
        x, y = self.check_input(x, y)

        S = np.zeros(len(x))
        S[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))

        dx = (x - x[0])
        dy = (y - y[0])

        e_1 = np.sum(dx ** 2)
        e_2 = np.sum(dx * S)
        e_3 = e_2
        e_4 = np.sum(S ** 2)

        M_1 = np.array([[e_1, e_2], [e_3, e_4]])
        # res_1 = np.linalg.inv(M_1) @ np.array([[np.sum(dx * dy)], [np.sum(dy * S)]])
        res_1 = np.linalg.solve(M_1, np.array([[np.sum(dx * dy)], [np.sum(dy * S)]]))  # More robust method
        c = res_1[1, 0]

        e_1 = len(x)
        e_2 = np.sum(np.exp(c * x))
        e_3 = e_2
        e_4 = np.sum(np.exp(2 * c * x))

        M_2 = np.array([[e_1, e_2], [e_3, e_4]])

        # res_2 = np.linalg.inv(M_2) @ np.array([[np.sum(y)], [np.sum(y * np.exp(c * x))]])
        res_2 = np.linalg.solve(M_2, np.array([[np.sum(y)], [np.sum(y * np.exp(c * x))]]))  # More robust method
        a = res_2[0, 0]
        b = res_2[1, 0]
        return a, b, c

    @staticmethod
    def check_input(x, y):
        if len(x) != len(y):
            raise ValueError("x and y must be the same length!")
        if len(x) < 3:
            raise ValueError(
                "For an offset exponential fit, at least 3 x,y values need be provided, better >30 x,y values. "
                "Increase the bandwidth/number of frequency points you evaluate over the thickness.")
        sorted_idx = np.argsort(x)
        y = y[sorted_idx]
        x = x[sorted_idx]
        # Keep only indices when x is increasing
        keed_idx = np.diff(x) > 0
        x = np.append(x[0], x[1:][keed_idx])
        y = np.append(y[0], y[1:][keed_idx])
        return x, y

    def run_optimization(self, thickness, delta_max=None):
        """Runs the optimization with Nelder-Mead minimization algorithm for each frequency.

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
        n_previous = self.data.n[0]
        k_previous = self.data.k[0]
        if self.progress_bar:
            for i, w in enumerate(tqdm(self.data.omega)):
                res = fmin(error_function2,
                           x0=np.array([n_previous, k_previous]),
                           xtol=self.accuracy,
                           args=(
                               i, self.data.omega[i], thickness, self.data.H, self.data.H_approx, self.data.delta_max),
                           disp=False)
                self.data.n[i] = res[0]
                self.data.k[i] = res[1]
                n_previous = self.data.n[i]
                k_previous = self.data.k[i]
        else:
            for i, w in enumerate(self.data.omega):
                res = fmin(error_function2,
                           x0=np.array([n_previous, k_previous]),
                           xtol=self.accuracy,
                           args=(
                               i, self.data.omega[i], thickness, self.data.H, self.data.H_approx, self.data.delta_max),
                           disp=False)
                self.data.n[i] = res[0]
                self.data.k[i] = res[1]
                n_previous = self.data.n[i]
                k_previous = self.data.k[i]
        self.data.alpha = 4 * np.pi * self.data.frequency * self.data.k / c_0
        return self.data.frequency, self.data.n, self.data.k, self.data.alpha
