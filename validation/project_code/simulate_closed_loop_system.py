# Simulate the behaviour of the closed loop system

import numpy as np
import mne
from PyEMD import CEEMDAN
from mne.time_frequency import psd_welch, psd_array_welch,tfr_array_morlet
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert, filtfilt
import tkinter as tk
from tkinter import filedialog
from scipy.io import savemat, loadmat
from scipy.optimize import curve_fit
from scipy.stats import zscore
from main_code.utils import get_target_chan, compute_phase_diff, combined_window_SASS, DFT, compute_P, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func, plv, make_P_laplace
ceemdan = CEEMDAN()

# Load the data used for simulation
folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/working memory/data/pilot/1"
#folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/validation/data/pilot/1"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)
bad_channels = ["Cz", "CPz", "CP1", "CP2", "C1", "C2", "Fz"]

# Drop non EEG and bad channels
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env', 'stim', 'Flicker'] or ch in bad_channels])
raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in ['env', 'stim', 'Flicker'] or ch in bad_channels])

# Get the data used for simulation
data_stim = raw_stim._data

# Set some parameters
frequency = 12
lfreq = frequency - 1; hfreq = frequency + 1
n_samps = data_stim.shape[1]
sfreq = raw_no_stim.info['sfreq']
ch_names = raw_stim.ch_names
buffer_size = sfreq
buffer_SASS_size = sfreq * 25
coeffs = fir_coeffs(frequency, n_taps=50)
coeffs_SASS = fir_coeffs(frequency, n_taps=96)
fmin_plot = 1
fmax_plot = 40

# Get the stimulation onsets
events = mne.events_from_annotations(raw_stim)[0]
stim_onset_idx = np.arange(len(events))[np.isin(events[:, 2], [2, 3, 4])]
sample_stim_onset = events[stim_onset_idx, 0]
sample_stim_offset = events[stim_onset_idx, 0] + 1 * sfreq
# Get an array with all the samples where simulation was present (for 3 sec after onset) - every 100 ms
sample_stim = np.array([sample_stim_onset[5:] + i for i in np.arange(0, 3 * sfreq, 50)]).flatten()

# Get an array of the stimulation data with the
# Initialize the SASS Matrix, buffer and Laplace matrix
P_SASS = np.zeros((len(ch_names), len(ch_names)))
inner_pick = "Pz"
outer_picks = ["O1", "O2", "P3", "P4"]
P_Laplace = make_P_laplace(raw_stim.ch_names, inner_pick, outer_picks)

# Create the covariance matrix of the data without stimulation (filtered around the target frequency)
no_stim_data = filtfilt(coeffs_SASS, 1, raw_no_stim._data.copy())
B = np.cov(no_stim_data)
# Compute the power spectrum of the no stimulation data
psds_no_stim, freqs = psd_array_welch((P_Laplace @ raw_no_stim._data).flatten(), fmin=fmin_plot, fmax=fmax_plot, sfreq=sfreq, n_fft=int(sfreq * 1))

# Compute a SASS matrix based on the whole dataset - to test
stim_data = filtfilt(coeffs_SASS, 1, raw_stim._data.copy())
A = np.cov(stim_data)
P_SASS, n_nulls = compute_P(A, B)

for i_samp in np.arange(buffer_SASS_size+sfreq, n_samps):

    # Update the SASS Matrix every 2 seconds (after 25 seconds)
    if i_samp >= buffer_SASS_size and i_samp % sfreq * 2 == 0:
        # Get the buffer
        buffer_long = data_stim[:, int(i_samp - buffer_SASS_size):int(i_samp)].copy()
        # Filter the buffer (with a longer filter)
        buffer_long_filt = filtfilt(coeffs_SASS, 1, buffer_long.copy())
        # Compute the covariance matrix
        A = np.cov(buffer_long_filt)
        # Compute the SASS Matrix
        P_SASS, n_nulls = compute_P(A, B)

    # Only plot if stimulation is present - after events 2,3,4
    if i_samp in sample_stim:
        # Generate the envelope
        # Get the buffer (1 sec)
        buffer = data_stim[:, int(i_samp - buffer_size):int(i_samp)].copy()
        # Detrend the buffer
        buffer -= buffer.mean(1)[:,np.newaxis]
        # Apply SASS
        buffer_SASS = P_SASS @ buffer
        # Compute the target channel
        target_chan = P_Laplace @ buffer_SASS
        # Filter the target channel (with a short filter)
        target_chan_filt = filtfilt(coeffs, 1, target_chan.copy())
        # Compute the phase using the DFT at the target frequency using the last 3 cycles
        amplitude, phase = DFT(target_chan_filt[-int((sfreq / frequency) * 3):], frequency, sfreq)
        # Transform the phase into an envelope
        envelope = np.cos(2 * np.pi* frequency * np.linspace(0, 1, 500) + phase)

        plt.figure()
        # Plot the data in the time domain at the different processing stages and the envelope
        plt.subplot(2, 2, 1); plt.plot((P_Laplace @ buffer).flatten()); plt.title("Target Channel raw")
        plt.subplot(2, 2, 2); plt.plot(target_chan.flatten()); plt.title(f"SASS, # nulls {n_nulls.mean()}")
        plt.subplot(2, 2, 3); plt.plot(zscore(target_chan_filt.flatten())); plt.title("SASS+Filt")
        plt.subplot(2, 2, 3); plt.plot(np.arange(500-int((sfreq / frequency) * 3), 500), zscore(envelope[:int((sfreq / frequency) * 3)]), label="Envelope"); plt.title("Envelope")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Plot the power spectrum of the target channel at the different processing stages
        plt.figure()
        plt.subplot(2, 2, 1)
        psds, freqs = psd_array_welch((P_Laplace @ buffer).flatten(), fmin=fmin_plot, fmax=fmax_plot, sfreq=sfreq, n_fft=int(sfreq * 1))
        plt.semilogy(freqs, psds, label=f"Raw")
        psds, freqs = psd_array_welch(target_chan.flatten(), fmin=fmin_plot, fmax=fmax_plot, sfreq=sfreq, n_fft=int(sfreq * 1))
        plt.semilogy(freqs, psds, label=f"SASS, # nulls {n_nulls.mean()}")
        psds, freqs = psd_array_welch(target_chan_filt.flatten(), fmin=fmin_plot, fmax=fmax_plot, sfreq=sfreq, n_fft=int(sfreq * 1))
        plt.semilogy(freqs, psds, label=f"SASS+Filt")
        # Add the power spectrum of the no stimulation data
        plt.semilogy(freqs, psds_no_stim, label=f"No Stim")
        # Plot the power spectrum of the buffer used for SASS
        psds, freqs = psd_array_welch((P_Laplace @ (P_SASS @ buffer_long)).flatten(), fmin=fmin_plot, fmax=fmax_plot, sfreq=sfreq, n_fft=int(sfreq * 1))
        plt.semilogy(freqs, psds, label=f"SASS buffer SASS")
        psds, freqs = psd_array_welch((P_Laplace @ buffer_long).flatten(), fmin=fmin_plot, fmax=fmax_plot, sfreq=sfreq, n_fft=int(sfreq * 1))
        plt.semilogy(freqs, psds, label=f"SASS buffer no SASS")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.hist(n_nulls)
        plt.subplot(2, 2, 3)
        IMFs = ceemdan(target_chan.squeeze())
        plt.plot(IMFs.T)
        plt.show()