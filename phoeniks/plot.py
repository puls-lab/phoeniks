import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
from scipy.constants import c as c_0


class Plot:
    def __init__(self, lower_freq=None, upper_freq=None):
        if lower_freq is None:
            self.lower_freq = 0.1e12
        else:
            self.lower_freq = lower_freq
        if upper_freq is None:
            self.upper_freq = 10e12
        else:
            self.upper_freq = upper_freq

    def plot_data(self, data, lower_freq=None, upper_freq=None):
        """Plots the input data."""
        if lower_freq is not None:
            self.lower_freq = lower_freq
        if upper_freq is not None:
            self.upper_freq = upper_freq
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(data.time, data.td_reference, label="Reference")
        ax[0].plot(data.time, data.td_sample, label="Sample")
        if data.mode != "reference_sample":
            ax[0].plot(data.time, data.td_dark, label="Dark")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        ax[0].xaxis.set_major_formatter(EngFormatter("s"))
        ax[0].set_xlabel("Time")
        ax[0].yaxis.set_major_formatter(EngFormatter("V"))
        ax[0].set_ylabel("Amplitude")

        filter_frequency = (data.frequency > self.lower_freq) & (data.frequency < self.upper_freq)
        # Normalize the spectrum
        ref_power_spectrum_max = np.max(20 * np.log10(np.abs(data.fd_reference[filter_frequency])))
        ax[1].plot(data.frequency[filter_frequency],
                   20 * np.log10(np.abs(data.fd_reference[filter_frequency])) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Reference")
        ax[1].plot(data.frequency[filter_frequency],
                   20 * np.log10(np.abs(data.fd_sample[filter_frequency])) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Sample")
        if data.mode != "reference_sample":
            ax[1].plot(data.frequency[filter_frequency],
                       20 * np.log10(np.abs(data.fd_dark[filter_frequency])) - ref_power_spectrum_max,
                       alpha=0.9,
                       label="Dark")
        ax[1].legend(loc="upper right")
        ax[1].grid(True)
        ax[1].xaxis.set_major_formatter(EngFormatter("Hz"))
        ax[1].set_xlabel("Frequency")
        ax[1].set_ylabel("Power spectrum")
        plt.tight_layout()

    def plot_phase(self, extract_data):
        # Normalize the spectrum
        ref_power_spectrum_max = np.max(20 * np.log10(np.abs(extract_data.data.fd_reference)))

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(extract_data.data.frequency,
                   20 * np.log10(np.abs(extract_data.data.fd_reference)) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Reference")
        ax[0].plot(extract_data.data.frequency,
                   20 * np.log10(np.abs(extract_data.data.fd_sample)) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Sample")
        if extract_data.data.mode != "reference_sample":
            ax[0].plot(extract_data.data.frequency,
                       20 * np.log10(np.abs(extract_data.data.fd_dark)) - ref_power_spectrum_max,
                       color="black",
                       alpha=0.9,
                       label="Dark")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        ax[0].xaxis.set_major_formatter(EngFormatter("Hz"))
        ax[0].set_xlabel("Frequency")
        ax[0].set_ylabel("Power spectrum")

        ax[1].plot(extract_data.data.frequency, extract_data.data.phase_reference, ".-", alpha=0.9, label="Reference")
        ax[1].plot(extract_data.data.frequency, extract_data.data.phase_sample, ".-", alpha=0.9, label="Sample")
        ax[1].plot(extract_data.data.frequency, extract_data.data.phase_H, ".-", alpha=0.9, label="Sample/Reference")
        ax[1].legend(loc="upper right")
        ax[1].grid(True)
        ax[1].xaxis.set_major_formatter(EngFormatter("Hz"))
        ax[1].set_xlabel("Frequency")
        ax[1].set_ylabel("Phase")
        plt.tight_layout()

    def plot_refractive_index(self, f, n, k):
        alpha = 0.01 * 2 * f * 2 * np.pi * k / c_0  # In [cm^-1]
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(f, n)
        ax[0].xaxis.set_major_formatter(EngFormatter("Hz"))
        ax[0].set_xlabel("Frequency")
        ax[0].set_ylabel("Refractive Index")
        ax[0].grid(True)
        # ax[0].set_ylim([1.6, 2.4])
        ax[1].plot(f, alpha, color="tab:orange")
        ax[1].xaxis.set_major_formatter(EngFormatter("Hz"))
        ax[1].set_xlabel("Frequency")
        ax[1].set_ylabel(r"Absorption [$\mathrm{cm}^{-1}$]")
        ax[1].grid(True)
        plt.tight_layout()

    def total_variation(self, thickness_array, tv_dict):
        fig, ax = plt.subplots()
        ax.plot(thickness_array, tv_dict["tv_1"] / np.max(tv_dict["tv_1"]), label="TotalVariation, deg=1")
        ax.plot(thickness_array, tv_dict["tv_2"] / np.max(tv_dict["tv_2"]), label="TotalVariation, deg=2")
        if len(tv_dict) > 2:
            ax.plot(thickness_array, tv_dict["tv_s"] / np.max(tv_dict["tv_s"]), label="TotalVariation vs. SVMAF")
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(EngFormatter("m"))
        plt.tight_layout()

    def plot_output_data(self, data, time_pulse_td, time_pulse_fd, lower_freq=None, upper_freq=None):
        if lower_freq is not None:
            self.lower_freq = lower_freq
        if upper_freq is not None:
            self.upper_freq = upper_freq
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(data.time, data.td_reference, label="Reference")
        ax[0].plot(data.time, data.td_sample, label="Sample")
        ax[0].plot(data.time, time_pulse_td, label="Fitted sample")
        ax[0].legend(loc="upper right")
        ax[0].grid(True)
        ax[0].xaxis.set_major_formatter(EngFormatter("s"))
        ax[0].set_xlabel("Time")
        ax[0].yaxis.set_major_formatter(EngFormatter("V"))
        ax[0].set_ylabel("Amplitude")

        filter_frequency = (data.frequency > self.lower_freq) & (data.frequency < self.upper_freq)
        # Normalize the spectrum
        ref_power_spectrum_max = np.max(20 * np.log10(np.abs(data.fd_reference[filter_frequency])))
        ax[1].plot(data.frequency[filter_frequency],
                   20 * np.log10(np.abs(data.fd_reference[filter_frequency])) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Reference")
        ax[1].plot(data.frequency[filter_frequency],
                   20 * np.log10(np.abs(data.fd_sample[filter_frequency])) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Sample")
        ax[1].plot(data.frequency[filter_frequency],
                   20 * np.log10(np.abs(time_pulse_fd[filter_frequency])) - ref_power_spectrum_max,
                   alpha=0.9,
                   label="Fitted Sample")
        ax[1].legend(loc="upper right")
        ax[1].grid(True)
        ax[1].xaxis.set_major_formatter(EngFormatter("Hz"))
        ax[1].set_xlabel("Frequency")
        ax[1].set_ylabel("Power spectrum")
        plt.tight_layout()
