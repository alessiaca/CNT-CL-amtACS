import numpy as np
import pandas
import mne
import matplotlib.pyplot as plt
from utils import get_target_chan, compute_phase_diff, window_SASS, fir_coeffs,SASS,find_bad_channels
import tkinter as tk
from tkinter import filedialog
from mne.time_frequency import psd_array_welch, psd_welch
from scipy.signal import filtfilt
from mne.filter import filter_data
mne.set_log_level('CRITICAL')
from scipy.fftpack import next_fast_len

# Define parameters
initial_bads = ["CPz"]
frequency = 6.3
lfreq = frequency-1
hfreq = frequency+1
winsize_SASS = 25
ch_picks = ["O1","O2","P8"] # Order important
sass_type = 'regular' # regular or window
filter_type = 'short' # short or long
t = int((500 / frequency) * 3)
coeffs = fir_coeffs(freq=frequency)

# Load data
root = tk.Tk()
root.withdraw()
# folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\validation\\pilot")
folder_path = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\validation\\pilot\\7"
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
sfreq = raw_no_stim.info['sfreq']
raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)
sfreq = raw_no_stim.info['sfreq']

# Get the envelope and drop non EEG channels or those that we know are bad (above stimulation electrode)
env_stim = raw_stim.get_data("env").flatten()
env_no_stim = raw_no_stim.get_data("env").flatten()
raw_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])

# Identify bad channels in the no_stim and stim dataset and drop them too
stim_bads = find_bad_channels(raw_stim)
no_stim_bads = find_bad_channels(raw_no_stim,use_zscore=True)
bads = np.unique(np.concatenate((no_stim_bads, stim_bads)))
print(f"{bads} are dropped")
#raw_stim.drop_channels(bads)
#raw_no_stim.drop_channels(bads)

# Plot the power spectrum of the target_channel in no_stim, stim and stim with SASS and the envelope
fmin = 1
fmax = 30

plt.figure()
plt.subplot(1,2,1)
psds, freqs = psd_array_welch(env_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="Stim")
psds, freqs = psd_array_welch(env_no_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="No Stim")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Envelope")
plt.legend()

plt.subplot(1,2,2)
target_chan_no_stim = get_target_chan(raw_no_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_no_stim,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="No Stim")
target_chan_stim = get_target_chan(raw_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim")

winsize_SASS = raw_stim.time_as_index(winsize_SASS)[0]
if sass_type == 'window':
    if filter_type == 'long':
        raw_stim = window_SASS(raw_stim, raw_no_stim, lfreq,hfreq,winsize_SASS)
    elif filter_type == 'short':
        raw_stim = window_SASS(raw_stim, raw_no_stim, lfreq, hfreq,winsize_SASS,filter_type='short',filter_coeffs=coeffs)
elif sass_type == 'regular':
    if filter_type == 'long':
        raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq)
    elif filter_type == 'short':
        raw_stim = SASS(raw_stim, raw_no_stim, lfreq, hfreq,filter_type='short',filter_coeffs=coeffs)

target_chan_stim_SASS = get_target_chan(raw_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="After SASS")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Target channel")
plt.legend()
plt.show()

if filter_type == 'short':
    # Filter the target channels around the target frequency (same filter length as applied in real-time)
    target_chan_stim_SASS = filtfilt(coeffs, 1, target_chan_stim_SASS)
    target_chan_no_stim = filtfilt(coeffs, 1, target_chan_no_stim)
elif filter_type == 'long':
    target_chan_stim_SASS = filter_data(target_chan_stim_SASS,sfreq,lfreq,hfreq)
    target_chan_no_stim = filter_data(target_chan_no_stim,sfreq,lfreq,hfreq)

# Compute the phase differences and plot them
signal_len = target_chan_stim_SASS.size
fft_len = next_fast_len(signal_len)
phase_diff_stim_SASS = compute_phase_diff(target_chan_stim_SASS, env_stim,fft_len=fft_len)
plt.subplot(121, projection='polar')
plt.hist(phase_diff_stim_SASS)
plt.title("Stim with SASS")

signal_len = target_chan_no_stim.size
fft_len = next_fast_len(signal_len)
phase_diff_no_stim = compute_phase_diff(target_chan_no_stim, env_no_stim,fft_len=fft_len)
plt.subplot(122, projection='polar')
plt.hist(phase_diff_no_stim)
plt.title("No Stim")
plt.show()