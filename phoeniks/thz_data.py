import numpy as np
from scipy import interpolate


def interpolate_fd(frequency, new_frequency, trace_std):
    # Interpol1d only works with real numbers, needs to split complex number in real and imag
    f_interpol_real = interpolate.interp1d(frequency, trace_std.real)
    f_interpol_imag = interpolate.interp1d(frequency, trace_std.imag)
    new_std_real = f_interpol_real(new_frequency)
    new_std_iamg = f_interpol_imag(new_frequency)
    return new_std_real + 1j * new_std_iamg


class Data:
    """This class contains all THz data in time and frequency domain for the cases of
    1. Dark measurement (THz blocked)
    2. Reference measurement (without sample)
    3. Sample measurement (with sample).

    The analysis can be done with different amounts of input data:

    1. The user provide time-domain data of reference,
    sample and dark measurements (averaged over multiple measurements) as well as standard deviations (based on the
    single measurements) in frequency-domain of the three measurements. This allows to use the SVMA-filter later and
    is the best case to use this program.

    2. The user provides reference, sample and dark measurement (averaged) in time domain. The dark trace allows to
    quantify the dynamic range for all frequencies. THe SVMA-filter cannot be used.

    3. The user provides only reference and sample trace (averaged) in time domain. The user can supply an upper
    frequency, from which the noise floor is determined. Can lead to wrong results (especially if data is taken from a
    Lock-In amplifier with an integrated low-pass filter, which leads to a non-flat noise floor)

    available_modes = ["reference_sample_dark_standard_deviations",
                       "reference_sample_dark",
                       "reference_sample"]"""
    def __init__(self,
                 time,
                 td_reference,
                 td_sample,
                 td_dark=None,
                 fd_reference_std=None,
                 fd_sample_std=None,
                 fd_dark_std=None) -> None:
        # Define instance variables in time domain
        self.time = time
        self.td_reference = td_reference
        self.td_sample = td_sample
        self.td_dark = td_dark
        # Define instance variables in frequency domain
        self.fd_reference_std = fd_reference_std
        self.fd_sample_std = fd_sample_std
        self.fd_dark_std = fd_dark_std
        # Variables for later extraction
        self.frequency = np.fft.rfftfreq(len(self.time), (self.time[-1] - self.time[0]) / (len(self.time) - 1))
        self.omega = 2 * np.pi * self.frequency
        self.phase_reference = None
        self.phase_sample = None
        self.n = None
        self.k = None
        self.alpha = None
        self.delta_max = None
        # Depending, how much data is provided, the right mode will be selected
        self.mode = "reference_sample_dark_standard_deviations"
        if fd_reference_std is None or fd_sample_std is None or fd_dark_std is None:
            self.mode = "reference_sample_dark"
        else:
            self.fd_reference_std = fd_reference_std
            self.fd_sample_std = fd_sample_std
            self.fd_dark_std = fd_dark_std
        if td_dark is None:
            self.mode = "reference_sample"
        else:
            self.fd_dark = np.fft.rfft(self.td_dark)
            if len(self.fd_dark) != len(self.frequency):
                raise ValueError("Supplied frequency data does not match calculated frequency array. " +
                                 "Frequency data should be only defined for positive frequencies.")
        self.fd_reference = np.fft.rfft(self.td_reference)
        self.fd_sample = np.fft.rfft(self.td_sample)
        self.H = self.fd_sample / self.fd_reference
        self.phase_H = None
        self.H_approx = None
        if (len(self.fd_reference) != len(self.frequency)) or \
                (len(self.fd_sample) != len(self.frequency)):
            raise ValueError("Supplied frequency data does not match calculated frequency array. " +
                             "Frequency data should be only defined for positive frequencies.")

    def offset_time_to_reference_peak(self) -> None:
        """Shift peak of reference trace to 0."""
        idx = np.argmax(self.td_reference)
        self.time -= self.time[idx]

    def pad_zeros(self, new_frequency_resolution=1e9) -> None:
        """Pad the time domain data with zeros to increase resolution in frequency domain."""
        dt = (self.time[-1] - self.time[0]) / (len(self.time) - 1)
        # Check if time traces got windowed
        if not self._data_is_windowed():
            raise ValueError("Data is not windowed. The data must be windowed before zeros can be padded.")
        current_td_length = np.abs(self.time[-1] - self.time[0])
        new_td_length = 1 / new_frequency_resolution
        delta_td_idx = int(np.floor((new_td_length - current_td_length) / dt)) + 1
        zero_array = np.zeros(len(self.time) + delta_td_idx)
        # TODO: Extending array properly, current implementation is not elegant :/
        tmp = np.copy(zero_array)
        self.time = np.linspace(self.time[0], self.time[-1] + delta_td_idx * dt, len(self.time) + delta_td_idx)
        new_frequency = np.fft.rfftfreq(len(self.time), (self.time[-1] - self.time[0]) / (len(self.time) - 1))
        tmp[:len(self.td_reference)] = self.td_reference
        self.td_reference = np.copy(tmp)
        tmp = np.copy(zero_array)
        tmp[:len(self.td_sample)] = self.td_sample
        self.td_sample = np.copy(tmp)
        if self.mode != "reference_sample":
            self.fd_dark = interpolate_fd(self.frequency, new_frequency, self.fd_dark)
            tmp = np.copy(zero_array)
            tmp[:] = np.nan
            tmp[:len(self.td_dark)] = self.td_dark
            self.td_dark = np.copy(tmp)
        del zero_array, tmp
        # Update frequency domain data
        self.fd_reference = np.fft.rfft(self.td_reference)
        self.fd_sample = np.fft.rfft(self.td_sample)
        if self.mode == "reference_sample_dark_standard_deviations":
            self.fd_reference_std = interpolate_fd(self.frequency, new_frequency, self.fd_reference_std)
            self.fd_sample_std = interpolate_fd(self.frequency, new_frequency, self.fd_sample_std)
            self.fd_dark_std = interpolate_fd(self.frequency, new_frequency, self.fd_dark_std)
        self.frequency = np.fft.rfftfreq(len(self.time), (self.time[-1] - self.time[0]) / (len(self.time) - 1))

    def get_window(self, trace: np.ndarray, time_start: float, time_end: float, alpha=0.16) -> np.ndarray:
        r = len(trace)
        rel_start = np.where(self.time >= time_start)[0][0] / r
        rel_end = 1 - np.where(self.time >= time_end)[0][0] / r
        width_start = round(r * rel_start) - 1
        width_end = round(r * rel_end) - 1
        beginning_end = round(r * (1 - rel_end)) - 1
        window = np.zeros(r)
        for i in range(r):
            if i <= width_start:
                i_argument = i / (2 * width_start)
                window[i] = 0.5 * (
                        1 - np.cos(2 * np.pi * i_argument) + alpha * np.cos(4 * np.pi * i_argument) - alpha)
            if i >= beginning_end:
                i_argument = (i - beginning_end + width_end - 1) / (2 * width_end)
                window[i] = 0.5 * (
                        1 - np.cos(2 * np.pi * i_argument) + alpha * np.cos(4 * np.pi * i_argument) - alpha)
            if width_start < i < beginning_end:
                window[i] = 1
        return window

    def window_traces(self, time_start: float, time_end: float, alpha=0.16) -> None:
        """Windows time domain traces with smoothness-factor alpha according to an (asymmetric) Blackman window.

        alpha (float): smoothness-factor.
        time_start (float): Time, where the window starts to let through signal in [s].
        time_stop (float): Time, where the window stops to let through signal in [s]."""
        # TODO: Implement Tukey-window, too.
        # We only need to create a window once, since all three time domain traces have the same length
        window = self.get_window(self.td_reference, time_start, time_end, alpha)
        self.td_reference *= window
        self.td_sample *= window
        if self.mode != "reference_sample":
            self.td_dark *= window
            self.fd_dark = np.fft.rfft(self.td_dark)
        # Update frequency domain data
        self.fd_reference = np.fft.rfft(self.td_reference)
        self.fd_sample = np.fft.rfft(self.td_sample)

    def _data_is_windowed(self) -> bool:
        """Check if the data is close to zero at beginning and end == is windowed."""
        windowed = True
        if self.mode != "reference_sample":
            traces = [self.td_reference, self.td_sample, self.td_dark]
        else:
            traces = [self.td_reference, self.td_sample]
        for trace in traces:
            if not np.isclose(trace[0], 0):
                windowed = False
            if not np.isclose(trace[-1], 0):
                windowed = False
        return windowed

    def linear_offset(self, time_trace, idx_beginning=10, idx_end=-31):
        """Subtract offset from the beginning of the time trace and interpolates a line with the last data points.
         This line is then subtracted from the time trace.

         Input:
         time_trace (np.array, 1D, float) : Array containing the THz signal in time domain.
         idx_beginning (int) :              How many data samples from the beginning are taken to create an average
                                            and subtract it?
         idx_end (int, negative) :          How many data samples from the end (thus negative) should be taken,
                                            to create a linear fit and substract it from the data?

        Output:
        time_trace (np.array, 1D, float) : THz time data with linear offset correction.
         """
        # Original code:
        # ma = np.mean(rv[:10])
        # rv -= ma
        # Calculating average of the last 30 sampling points in reference trace
        # me = np.mean(rv[-30:])
        # Creating a linear function with the slope between beginning (which is already subtracted, so 0) and end.
        # It will automatically create a linear function,
        # which is defined from the average of the first 10 datapoints and the last 30 data points.
        # o1 = np.arange(n - 1) * me / (n - 1)
        # rv -= o1
        n = len(time_trace)
        time_trace -= np.mean(time_trace[:idx_beginning])
        linear_function = np.arange(n) * np.mean(time_trace[idx_end:]) / (n - 1)
        time_trace -= linear_function
        return time_trace
