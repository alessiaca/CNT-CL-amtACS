import numpy as np
import pandas
import mne
import matplotlib.pyplot as plt
from utils import get_target_chan, compute_phase_diff, fir_coeffs
import tkinter as tk
from tkinter import filedialog
from mne.time_frequency import psd_array_welch, psd_welch
from scipy.signal import filtfilt
from mne.filter import  filter_data

# Define parameters
bads = ["F9","FT9","T7","TP9","TP10"]
frequency = 6
lfreq = frequency-1
hfreq = frequency+1
ch_picks = ["F3","Fz","F4"] # Order important
filter_type = 'long' # short or long
n_sec_fft = 3

# Load data
n_par = 2
#root = tk.Tk()
#root.withdraw()
#folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\memory_consolidation")
folder_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\eeg\\data\\{n_par}"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
sfreq = raw_no_stim.info['sfreq']

# Get the envelope and drop the non EEG channels
env = raw_no_stim.get_data("env").flatten()
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in bads])
sfreq = raw_no_stim.info['sfreq']
ch_names = raw_no_stim.ch_names

# Plot the power spectrum of all channels, the target channel and the envelope
fmin = 1
fmax = 30
target_chan = get_target_chan(raw_no_stim,ch_picks)
psds, freqs = psd_welch(inst=raw_no_stim, fmin=fmin, fmax=fmax, picks=ch_names, n_fft=int(sfreq*n_sec_fft))
plt.subplot(1,3,1)
plt.semilogy(freqs, psds.T)
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
psds, freqs = psd_array_welch(target_chan, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*n_sec_fft))
plt.subplot(1,3,2)
plt.semilogy(freqs, psds.T)
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Target channel")
psds, freqs = psd_array_welch(env, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*n_sec_fft))
plt.subplot(1,3,3)
plt.semilogy(freqs, psds.T)
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Envelope")
plt.show()

if filter_type == 'short':
    # Filter the target channels around the target frequency (same filter length as applied in real-time)
    t = int((500/frequency)*3)
    coeffs = fir_coeffs(freq=frequency,n_cycles=1)
    target_chan_filt = filtfilt(coeffs, 1, target_chan)
elif filter_type == 'long':
    target_chan_filt = filter_data(target_chan,sfreq,frequency-1,frequency+1)

# Compute the phase difference and plot it
phase_diff = compute_phase_diff(target_chan_filt, env)
plt.subplot(111, projection='polar')
plt.hist(phase_diff)
plt.title("No stim")
plt.show()