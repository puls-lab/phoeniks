import numpy as np
from scipy.signal import find_peaks, peak_widths


def zero_runs(a):
    # From Warren Weckesser, https://stackoverflow.com/a/24892274/8599759
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def unwrap_refractive_index(refractive_index):
    w = len(refractive_index) ** 2 / (np.arange(1, len(refractive_index)) ** 2)
    filtered_diff = np.diff(refractive_index) / w
    peaks_pos, _ = find_peaks(filtered_diff, height=5e-3, distance=3)
    peaks_neg, _ = find_peaks(-filtered_diff, height=5e-3, distance=3)
    peaks = np.sort(np.append(peaks_pos, peaks_neg))
    results_full = peak_widths(np.abs(np.diff(refractive_index)), peaks, rel_height=0.9)
    compensation = np.zeros(len(refractive_index))
    for lower, upper in zip(results_full[2], results_full[3]):
        # Select samples where peak starts and stops
        lower = int(np.round(lower))
        upper = int(np.round(upper))
        compensation[lower + 1:upper + 1] = np.diff(refractive_index)[lower:upper]
    compensation = np.cumsum(compensation)
    ranges = zero_runs(np.diff(refractive_index - compensation))
    d_ref = np.diff(refractive_index - compensation)
    idx = []
    for peak in peaks:
        idx.append(
            [i for i, single_range in enumerate(ranges) if (peak >= single_range[0]) and (peak <= single_range[1])])
    idx = np.unique(idx)
    slope_compensation = np.zeros(len(refractive_index))
    if len(idx) > 0:
        for confirmed_range in ranges[idx, :]:
            # print("###")
            # print(confirmed_range)
            slopes = d_ref[confirmed_range[0] - 1:confirmed_range[1] + 1]
            # print(slopes)
            slopes_interpol = np.linspace(slopes[0], slopes[-1], len(slopes))[1:-1]
            # print(slopes_interpol)
            slope_compensation[confirmed_range[0] + 1:confirmed_range[1] + 1] = slopes_interpol
    slope_compensation = np.cumsum(slope_compensation)
    return refractive_index - compensation + slope_compensation


w = 1 / (np.arange(1, len(refractive_index)) ** 2)
filtered_diff = np.diff(refractive_index) / w
peaks_pos, _ = find_peaks(filtered_diff / np.max(np.abs(filtered_diff)), height=0.1, distance=3)
peaks_neg, _ = find_peaks(-filtered_diff / np.max(np.abs(filtered_diff)), height=0.1, distance=3)
debug = False
# if debug:
#    fig,ax = plt.subplots()
#    ax.plot(filtered_diff / np.max(np.abs(filtered_diff)))
#    ax.plot(peaks_pos, filtered_diff[peaks_pos] / np.max(np.abs(filtered_diff)), "x")
#    ax.plot(peaks_neg, filtered_diff[peaks_neg] / np.max(np.abs(filtered_diff)), "x")
#    ax.grid(True)
peaks = np.sort(np.append(peaks_pos, peaks_neg))
results_full = peak_widths(np.abs(np.diff(refractive_index)), peaks, rel_height=0.9)
compensation = np.zeros(len(refractive_index))
for lower, upper in zip(results_full[2], results_full[3]):
    # Select samples where peak starts and stops
    lower = int(np.round(lower))
    upper = int(np.round(upper))
    compensation[lower + 1:upper + 1] = np.diff(refractive_index)[lower:upper]
    # compensation[lower] = np.diff(refractive_index)[lower - 1]
compensation = np.cumsum(compensation)
if debug:
    plt.figure()
    plt.plot(compensation)
    plt.grid(True)
    plt.figure()
    plt.plot(refractive_index)
    plt.plot(refractive_index - compensation)
    tmp = np.zeros(len(refractive_index))
    tmp[peaks + 1] = np.diff(refractive_index)[peaks]
    tmp = np.cumsum(tmp)
    plt.plot(refractive_index - 2 * tmp)
    plt.grid(True)

ranges = zero_runs(np.diff(refractive_index - compensation))
d_ref = np.diff(refractive_index - compensation)
idx = []
for peak in peaks:
    idx.append([i for i, single_range in enumerate(ranges) if (peak >= single_range[0]) and (peak <= single_range[1])])
idx = np.unique(idx)
slope_compensation = np.zeros(len(refractive_index))
for confirmed_range in ranges[idx, :]:
    # print("###")
    # print(confirmed_range)
    slopes = d_ref[confirmed_range[0] - 1:confirmed_range[1] + 1]
    # print(slopes)
    slopes_interpol = np.linspace(slopes[0], slopes[-1], len(slopes))[1:-1]
    # print(slopes_interpol)
    slope_compensation[confirmed_range[0] + 1:confirmed_range[1] + 1] = slopes_interpol
slope_compensation = np.cumsum(slope_compensation)
