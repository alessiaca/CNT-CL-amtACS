import numpy as np
import mne
from mne.time_frequency import psd_welch, psd_array_welch,tfr_array_morlet
import matplotlib.pyplot as plt
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
import tkinter as tk
from tkinter import filedialog
from scipy.io import savemat, loadmat
from scipy.optimize import curve_fit
from utils import get_target_chan, compute_phase_diff, combined_window_SASS, \
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func
mne.set_log_level('CRITICAL')

# Set parameters

# Choose the dataset
n_par = 1
#root = tk.Tk()
#root.withdraw()
#folder_path = filedialog.askdirectory(initialdir = "C:\\Users\\David\\Documents\\Closed-Loop amtacs\\Data\\memory_consolidation")
folder_path = f"C:\\Users\\alessia\\Documents\\Jobs\\CNT\\CL-amtACS\\memory_consolidation\\eeg\\data\\{n_par}"

# Load data
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/encoding.vhdr', preload=True)
ch_names_array = np.array(raw_stim.ch_names)
initial_bads = loadmat(folder_path+ "/exclude_idx.mat")["exclude_idx"][0]
initial_bads = ch_names_array[initial_bads - 1]
chidx = loadmat(folder_path+ "/chidx.mat")["chidx"][0]
ch_picks = ch_names_array[chidx - 1]
frequency = loadmat(folder_path+ "/frequency.mat")["frequency"][0][0]
lfreq = frequency-1
hfreq = frequency+1
sfreq = raw_no_stim.info['sfreq']

# Print the parameters for visual check
print(f"Bads: {initial_bads}")
print(f"Channel picks: {ch_picks}")
print(f"Frequency: {frequency}")

# Get the envelope
raw_no_stim_env = raw_no_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])
raw_stim_env = raw_stim.copy().filter(lfreq,hfreq).pick_channels(["env"])

# Drop non EEG channels and those based on the non_stim dataset
raw_no_stim.drop_channels([ch for ch in raw_no_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])
raw_stim.drop_channels([ch for ch in raw_stim.ch_names if ch in ['env','sass_output','Flicker'] or ch in initial_bads])

# Identify bad channels from the stim dataset
bads = find_bad_channels(raw_stim,use_zscore=False)
print(f"{bads} would be dropped")
# Comment this out if you don't want to drop them
raw_stim.drop_channels(bads)
raw_no_stim.drop_channels(bads)
ch_names = raw_stim.ch_names

# Plot the power of the envelopes
fmin = 1
fmax = 30
plt.figure()
plt.subplot(1,2,1)
psds, freqs = psd_welch(raw_no_stim_env, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="Stim")
psds, freqs = psd_welch(raw_stim_env, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="No Stim")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Envelope")
plt.legend()

# Plot the power spectrum of the target channel
plt.subplot(1,2,2)
target_chan_no_stim = get_target_chan(raw_no_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_no_stim,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="No Stim")
target_chan_stim = get_target_chan(raw_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim")

# Apply SASS and plot the power spectrum of the target channel
ch_names = raw_stim.ch_names
chidx = [ch_names.index(ch) for ch in ch_picks]
P_laplace = np.zeros((1,len(ch_names)))
P_laplace[0,chidx[0]] = 1
P_laplace[0,chidx[1]] = -2
P_laplace[0,chidx[2]] = 1
raw_stim = SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace)
target_chan_stim_SASS = get_target_chan(raw_stim,ch_picks)
psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim after SASS")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Target channel")
plt.legend()
#plt.show()

# Filter the data around the target frequency
raw_stim.filter(lfreq,hfreq)

# Cut the data into epochs, get the mean amplitude for each condition and plot the phase locking in that condition
events = mne.events_from_annotations(raw_stim)[0]
conditions = ['no stim','in phase', 'open loop']
event_ids = [1,2,3]
amplitudes = np.zeros((len(event_ids),1))
stds = np.zeros((len(event_ids),1))
plt.figure()
for i,event_id in enumerate(event_ids):
      epochs = mne.Epochs(raw_stim, events, event_id=event_id, tmin=40, tmax=60*5.5, detrend=0, baseline=None,preload=True)
      epochs_env = mne.Epochs(raw_stim_env, events, event_id=event_id, tmin=40, tmax=60*5.5, detrend=0, baseline=None,preload=True)
      env = epochs_env._data.squeeze()

      # Get the target channel and its hilbert transform
      target_chan = get_target_chan(epochs, ch_picks)
      n_samps = target_chan.shape[1]
      fft_len = next_fast_len(n_samps)
      hil = hilbert(target_chan,fft_len)[:,:n_samps]

      psds = tfr_array_morlet(target_chan[:, np.newaxis, :], sfreq, [frequency], n_cycles=4, output='power')

      # Get the amplitude averaged over epochs
      amplitudes[i] = np.median(np.abs(hil))
      #amplitudes[i] = np.median(psds.flatten())
      stds[i] = np.std(np.median(np.abs(hil),-1))
      #stds[i] = np.std(psds.flatten())

      # Plot the phase locking of the target_channel with the envelope
      phase_diff = compute_phase_diff(target_chan, env)
      ax = plt.subplot(1, 3, i + 1, projection='polar')
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      plt.hist(phase_diff.flatten(), color="blue", alpha=0.5)
      plt.title(f"{conditions[i]} °")

plt.show()

# Plot the amplitudes as bars
amplitudes = amplitudes.flatten()
stds = stds.flatten()
x = np.arange(len(conditions))  # the label locations
width = 0.35  # the width of the bars
plt.errorbar(x,amplitudes,stds,color="red")
plt.bar(x,amplitudes,tick_label=conditions)
plt.show()



