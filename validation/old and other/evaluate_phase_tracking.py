import numpy as np
import pandas
import mne
import matplotlib.pyplot as plt
from utils import get_target_chan, compute_phase_diff, circular_hist, window_SASS, fir_coeffs,SASS
import tkinter as tk
from tkinter import filedialog
from mne.time_frequency import psd_array_welch, psd_welch
from scipy.signal import filtfilt

# Define parameters
bads = ["CPz"]
frequency = 6.5
lfreq = frequency-1
hfreq = frequency+1
winsize_SASS = 25
ch_picks = ["F7","T7","P7"] # Order important

# Load data
folder_path = "C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\CL-Validation\\data\\06-03\\temporal_theta_no_stim"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
#raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)
env = raw_no_stim.get_data("env").flatten()

target_chan = get_target_chan(raw_no_stim,ch_picks)
plt.figure()
psds, freqs = psd_array_welch(target_chan, fmin=1, fmax=30, sfreq = 500, n_fft=int(500*5))
plt.subplot(1,2,1)
plt.semilogy(freqs,psds.flatten(),label="no stim")
psds, freqs = psd_welch(raw_no_stim, fmin=1, fmax=30, picks = "env", n_fft=int(500*5))
plt.subplot(1,2,2)
plt.semilogy(freqs,psds.flatten(),label="no stim")
plt.show()

# Filter the data and extract the target channel
coeffs = fir_coeffs(frequency)
target_chan = filtfilt(coeffs,1,target_chan)
env = filtfilt(coeffs,1,env)


# Compute the phase difference and plot it
phase_diff = compute_phase_diff(target_chan[:-1], env[:-1])
plt.subplot(111, projection='polar')
plt.hist(phase_diff)
plt.title("No stim")
plt.show()

