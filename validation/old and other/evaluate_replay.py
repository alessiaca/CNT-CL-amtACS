import numpy as np
import pandas
import mne
import matplotlib.pyplot as plt
from utils import get_target_chan, compute_phase_diff, circular_hist, window_SASS, fir_coeffs,SASS, combined_SASS, SASS
import tkinter as tk
from tkinter import filedialog
from mne.time_frequency import psd_array_welch, psd_welch
from scipy.signal import filtfilt

# Define parameters
frequency = 6.5
lfreq = frequency-1
hfreq = frequency+1
winsize_SASS = 25
ch_picks = ["F3","Fz","F4"] # Order important
bads = ["CPz", "CP2", "Fpz","sass_output","env","Flicker","FC2","Fp2","F8","F9","FT10","AF8"]

# Load data
folder_path = "C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\CL-Validation\\data\\06-03\\frontal_theta_no_stim"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
folder_path = "C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\CL-Validation\\data\\06-03\\frontal_theta_cl_stim_40_carrier"
raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)
raw_replay = mne.io.read_raw_brainvision(folder_path + '/replay.vhdr', preload=True)
winsize_SASS = raw_no_stim.time_as_index(winsize_SASS)[0]

# Get the envelope
env_no_stim = raw_no_stim.get_data("env").flatten()
env_stim = raw_stim.get_data("env").flatten()
env_replay = raw_replay.get_data("env").flatten()

# Drop bad channels
raw_no_stim.drop_channels(bads)
raw_stim.drop_channels(bads)
raw_replay.drop_channels(bads)

# Apply SASS to stim and replay (same filter matrix)
coeffs = fir_coeffs(frequency)
#raw_stim_SASS, raw_replay_SASS = combined_SASS(raw_stim.copy(),raw_replay.copy(),raw_no_stim.copy(),coeffs)
raw_stim_SASS = SASS(raw_stim.copy(),raw_no_stim.copy(),coeffs)
raw_replay_SASS = SASS(raw_replay.copy(),raw_no_stim.copy(),coeffs)
#raw_stim_SASS = window_SASS(raw_stim.copy(),raw_no_stim.copy(),coeffs,winsize_SASS)
#raw_replay_SASS = window_SASS(raw_replay.copy(),raw_no_stim.copy(),coeffs,winsize_SASS)
#raw_stim_SASS = raw_stim
#raw_replay_SASS = raw_replay

# Plot the power spectra
plt.figure()
plt.subplot(1,2,1)
target_chan_no_stim = get_target_chan(raw_no_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_no_stim, fmin=1, fmax=30, sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="no stim")

target_chan_stim = get_target_chan(raw_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_stim, fmin=1, fmax=30, sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="stim")
target_chan_stim_SASS = get_target_chan(raw_stim_SASS,ch_picks)
psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=1, fmax=30, sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="stim_SASS")

target_chan_replay = get_target_chan(raw_replay,ch_picks)
psds, freqs = psd_array_welch(target_chan_replay, fmin=1, fmax=30, sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="replay")
target_chan_replay_SASS = get_target_chan(raw_replay_SASS,ch_picks)
psds, freqs = psd_array_welch(target_chan_replay_SASS, fmin=1, fmax=30, sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="replay_SASS")

plt.legend()
plt.subplot(1,2,2)
psds, freqs = psd_array_welch(env_no_stim, fmin=1, fmax=30,  sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="no_stim")
psds, freqs = psd_array_welch(env_stim, fmin=1, fmax=30,  sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="stim")
psds, freqs = psd_array_welch(env_replay, fmin=1, fmax=30,  sfreq = 500, n_fft=int(500*5))
plt.semilogy(freqs,psds.flatten(),label="replay")
plt.legend()
plt.show()


# Filter the data and extract the target channel
target_chan_no_stim = filtfilt(coeffs,1,target_chan_no_stim)
target_chan_stim_SASS = filtfilt(coeffs,1,target_chan_stim_SASS)
target_chan_replay_SASS = filtfilt(coeffs,1,target_chan_replay_SASS)
target_chan_stim = filtfilt(coeffs,1,target_chan_stim)
target_chan_replay = filtfilt(coeffs,1,target_chan_replay)
# Filter the envelopes

# Compute the phase difference and plot it
phase_diff = compute_phase_diff(target_chan_no_stim[:-1], env_no_stim[:-1])
plt.figure()
plt.subplot(231, projection='polar')
plt.hist(phase_diff)
plt.title("No stim")

phase_diff = compute_phase_diff(target_chan_stim_SASS[:-1], env_stim[:-1])
plt.subplot(232, projection='polar')
plt.hist(phase_diff)
plt.title("stim")

phase_diff = compute_phase_diff(target_chan_replay_SASS[:-1], env_replay[:-1])
plt.subplot(233, projection='polar')
plt.hist(phase_diff)
plt.title("Replay")

phase_diff = compute_phase_diff(target_chan_stim[:-1], env_stim[:-1])
plt.subplot(234, projection='polar')
plt.hist(phase_diff)
plt.title("stim_no_SASS")

phase_diff = compute_phase_diff(target_chan_replay[:-1], env_replay[:-1])
plt.subplot(235, projection='polar')
plt.hist(phase_diff)
plt.title("Replay_no_SASS")
plt.show()

