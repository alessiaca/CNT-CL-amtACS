import numpy as np
import mne
from scipy.io import savemat
from mne.time_frequency import psd_welch, psd_array_welch
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.stats import zscore
from utils import  get_target_chan, fir_coeffs, compute_phase_diff
import tkinter as tk
from tkinter import filedialog
from mne.filter import filter_data

# CHANGE HERE BAD ELECTRODES !!!!
bads = ["F9","FT9","T7","TP7","TP9","FT7","C5","CP5"]

# After the run without stimulation use this script to determine the individual frequency and to generate the covariance
# matrix of the data without stimulation

# Load the data without stimulation
root = tk.Tk()
root.withdraw()
folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\memory_consolidation")
raw_no_stim = mne.io.read_raw_brainvision(folder_path+'\\no_stim.vhdr',preload=True)
sfreq = raw_no_stim.info['sfreq']
env = raw_no_stim.get_data("env").flatten()
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker']])
ch_names = raw_no_stim.ch_names

# Determine if there are any channels that should be excluded bases on their band power
ch_names_arr = np.array(ch_names)
var = np.var(raw_no_stim._data,1)
mask_bad = np.abs(zscore(var)) > 1.645
bads.append(ch_names_arr[mask_bad][0])
bads = np.unique(bads)

# Plot the power spectrum of all channels and the virtual channel
ch_picks = ["F3","Fz","F4"]
target_chan = get_target_chan(raw_no_stim,ch_picks)
fmin = 1
fmax = 30
n_sec_ffts = [3,4,5]
plt.subplot(1,2,1)
for n_sec_fft in n_sec_ffts:
      psds, freqs = psd_array_welch(target_chan, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq * n_sec_fft))
      plt.semilogy(freqs, psds.T,label=n_sec_fft)
      plt.xlabel("Frequency in Hz")
      plt.ylabel("Power Spectral Density in dB")

      # Print the maximum frequency
      mask = np.where((freqs >= 4) & (freqs <= 8))
      freqs = freqs[mask]
      psds = psds[mask]
      print(f"Largest power spectral density"
            f" at (median): {np.round(freqs[np.argmax(psds)], 2)}")  # This code first collapses across epochs and then across channels
      # Collapsing across channels with median is questionable, because then one channel represents the whole

plt.legend()

psds, freqs = psd_welch(inst=raw_no_stim, fmin=fmin, fmax=fmax, picks=ch_names, n_fft=int(sfreq*4))
plt.subplot(1,2,2)
plt.semilogy(freqs, psds.T)
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Target channel")
plt.show()

# Choose and save the individual frequency
print("Which frequency to choose?")
frequency = input("> ")
frequency = float(frequency)
data_dict = {"frequency": frequency}
savemat(folder_path + "/frequency.mat", data_dict)

# Filter the data given the frequency
t = int((500/frequency)*3)
coeffs = fir_coeffs(freq=frequency,n_taps=825)
filt_sig = filtfilt(coeffs,1,raw_no_stim._data)

# Compute and save the covariance matrix
data_dict = {"C_B_64": np.cov(filt_sig)}
savemat(folder_path + "/C_B.mat", data_dict)

# Save the indexes to exclude as well as the channels to use for the experiment
exclude_idx = [ch_names.index(ch) + 1 for ch in bads]
data_dict = {"exclude_idx": exclude_idx}
savemat(folder_path + "/exclude_idx.mat", data_dict)
chidx = [ch_names.index(ch)+1 for ch in ch_picks]
data_dict = {"chidx": chidx}
savemat(folder_path + "/chidx.mat", data_dict)

# Print which channels were excluded
print(f"Channels {bads} are excluded")

# PLot the phase locking of the target channel and the envelope
target_chan_filt = filter_data(target_chan,sfreq,frequency-1,frequency+1)
phase_diff = compute_phase_diff(target_chan_filt, env)
plt.subplot(111, projection='polar')
plt.hist(phase_diff)
plt.title("No stim")
plt.show()