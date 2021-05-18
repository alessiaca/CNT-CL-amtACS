from itertools import permutations
import numpy as np
import random

perm = permutations([1, 2, 3,1,2,3])
perm = list(perm)
orders = []
for i in list(perm):
      diffs = np.diff(i)
      if not np.any(diffs==0):
            orders.append(i)
orders_final = random.sample(orders,20)








import numpy as np
import mne
from scipy.io import savemat
from mne.time_frequency import psd_welch, psd_array_welch
from mne.preprocessing import compute_current_source_density
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt
from make_fir_coeffs import fir_coeffs
from scipy.stats import zscore


# After the run without stimulation use this script to determine the individual frequency and to generate the covariance
# matrix of the data without stimulation

# PARAMETERS TO CHANGE
par_n = '3'
experiment = 'validation/pilot'
#experiment = 'working_memory/pilot'
#experiment = 'memory_consolidation/pilot'
#experiment = 'working_memory'
#experiment = 'memory_consolidation'
bads = ['CPz']

# Load the data without stimulation
i_par = 5
data_path = "C:/Users/alessia/Documents/Praktika/Berlin/CL-amtACS/CL-Validation/data/pilot/"
file_path = data_path + f'{i_par}/'+'no_stim.vhdr'
bads = ["env","sass_output","Flicker","CPz","Cz"]
no_stim = mne.io.read_raw_brainvision(file_path,preload=True)
sfreq = no_stim.info['sfreq']
no_stim.drop_channels(['env','sass_output',"Flicker"])
ch_names = no_stim.ch_names
montage = make_standard_montage("easycap-M1")
no_stim.set_montage(montage,match_case=False)
#no_stim = compute_current_source_density(no_stim)

# Determine if there are any channels that should be excluded bases on their band power
ch_names_arr = np.array(ch_names)
var = np.var(no_stim._data,1)
mask_bad = np.abs(zscore(var)) > 1.645
bads.append(ch_names_arr[mask_bad][0])
bads = np.unique(bads)

events = mne.events_from_annotations(no_stim)[0]
no_stim_copy = no_stim.copy()
# Choose different epochs and frequency ranges for the working memory and memory consolidation experiment
# if experiment[:14] == "working_memory":
#     fmin = 7
#     fmax = 13
#     no_stim_copy.filter(fmin,fmax)
#     epochs = mne.Epochs(raw=no_stim_copy, events=events, event_id=7, detrend=0, baseline=None, tmin=0.7, tmax=3.1,preload=True)
#     ch_names_pick = [ch for ch in ch_names if ch not in bads and (ch[:1] == 'O' or ch[:2] == 'PO')]
#     chidx = [ch_names.index(ch) + 1 for ch in ch_names if ch not in bads and (ch[:1] == 'O' or ch[:2] == 'PO')]
#
# else:
#     # CHANGE here the parameters for the memory consolidation !!!!!!!!!!!!!!!!!!
#     fmin = 3
#     fmax = 8
#     no_stim_copy.filter(fmin, fmax)
#     epochs = mne.Epochs(raw=no_stim_copy, events=events, event_id=[2,3,4,5,6,7,8,9], detrend=0, baseline=None, tmin=0, tmax=2.5,preload=True)
#     ch_names_pick = [ch for ch in ch_names if ch not in bads and ch in ['Fz','F1','F2']]
#     chidx = [ch_names.index(ch) + 1 for ch in ch_names if ch not in bads and ch in ['Fz','F1','F2']]

# Compute the power of those channels in the set frequency range
ch_names_pick = [ch for ch in ch_names if ch not in bads and (ch[:1] == 'O' or ch[:2] == 'PO')]
chidx = [ch_names.index(ch) + 1 for ch in ch_names if ch not in bads and (ch[:1] == 'O' or ch[:2] == 'PO')]
fmin = 1
fmax = 30
test = no_stim.get_data("O2")-no_stim.get_data("P8")
psds, freqs = psd_array_welch(test, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5),sfreq=sfreq)
#psds, freqs = psd_welch(no_stim, fmin=fmin, fmax=fmax, picks = ["O2"],n_fft=int(sfreq*5))

# # Transform output into decibel
# psds = 10. * np.log10(psds)

# Plot the PSD
plt.semilogy(freqs, psds.T)
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.show()

# Print the maximum frequency
print(f"Largest power spectral density"
      f" at (median): {np.round(freqs[np.argmax(np.median(a=psds, axis=(0, 1)))], 2)}"  # This code first collapses across epochs and then across channels
      f", at (mix): {np.round(freqs[np.argmax(np.mean(np.median(psds, 0),0))], 2)}"     # Collapsing via median across epochs is good, to avoid extreme noise
      f", at (mean): {np.round(freqs[np.argmax(np.mean(psds, (0, 1)))], 2)}")           # Collapsing across channels with median is questionable, because then one channel represents the whole

# Choose and save the individual frequency
print("Which frequency to choose)")
frequency = input("> ")
frequency = float(frequency)
data_dict = {"frequency": frequency}
savemat(folder_path + f"{par_n}_frequency.mat", data_dict)

# Given the frequency, compute the covariance matrix

# Generate the filter coefficients
t = int((500/frequency)*3)
coeffs = fir_coeffs(freq =frequency)

# Save the filter coefficients
data_dict = {"filter_coeffs": coeffs}
savemat(folder_path + f"{par_n}_filter_coeffs.mat", data_dict)

# Filter the data
filt_sig = filtfilt(coeffs,1,no_stim._data)

# Compute and save the covariance matrix
data_dict = {"C_B_64": np.cov(filt_sig)}
savemat(folder_path + f"{par_n}_C_B.mat", data_dict)

# Save the indexes to exclude as well as the channels to use for the experiment
exclude_idx = [ch_names.index(ch) + 1 for ch in bads]
data_dict = {"exclude_idx": exclude_idx}
savemat(folder_path + f"{par_n}_exclude_idx.mat", data_dict)
data_dict = {"chidx": chidx}
savemat(folder_path + f"{par_n}_chidx.mat", data_dict)

# Print which channels are excluded
print(f"Channels {bads} are excluded")