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
      fir_coeffs,combined_SASS,find_bad_channels, SASS, window_SASS, sine_func, plv

# Set parameters

# Choose the dataset
#root = tk.Tk()
#root.withdraw()
#folder_path = filedialog.askdirectory(initialdir = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/WM-Task/data")
folder_path = "C:/Users/alessia/Documents/Jobs/CNT/CL-amtACS/WM-Task/data/pilot/1"

# Load data
raw_no_stim = mne.io.read_raw_brainvision(folder_path + '/no_stim.vhdr', preload=True)
raw_stim = mne.io.read_raw_brainvision(folder_path + '/stim.vhdr', preload=True)
ch_names_array = np.array(raw_stim.ch_names)
initial_bads = ["Cz", "CPz", "CP1", "CP2", "C1", "C2","Fz"]
frequency = 12
lfreq = frequency-1
hfreq = frequency+1
sfreq = raw_no_stim.info['sfreq']
inner_pick = "Pz"
outer_picks = ["O1","O2","P3","P4"]

# Get the envelope
raw_no_stim_env = raw_no_stim.copy().pick_channels(["env"])
raw_stim_env = raw_stim.copy().pick_channels(["env"])

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
fmax = 50
plt.figure()
plt.subplot(1,2,1)
psds, freqs = psd_welch(raw_no_stim_env, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="No Stim")
psds, freqs = psd_welch(raw_stim_env, fmin=fmin, fmax=fmax, n_fft=int(sfreq*5))
plt.semilogy(freqs, psds.T, label="Stim")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Envelope")
plt.legend()

# Plot the power spectrum of the target channel
plt.subplot(1,2,2)
target_chan_no_stim = get_target_chan(raw_no_stim,inner_pick,outer_picks)
psds, freqs = psd_array_welch(target_chan_no_stim,fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="No Stim")
target_chan_stim = get_target_chan(raw_stim,inner_pick,outer_picks)
psds, freqs = psd_array_welch(target_chan_stim, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim")

# Apply SASS and plot the power spectrum of the target channel
ch_names = raw_stim.ch_names
chidx = [ch_names.index(ch) for ch in outer_picks]
P_laplace = np.zeros((1,len(ch_names)))
P_laplace[0,chidx[0]] = 1
P_laplace[0,chidx[1]] = 1
P_laplace[0,chidx[2]] = 1
P_laplace[0,chidx[3]] = 1
P_laplace[0,raw_stim.ch_names.index(inner_pick)] = -4
coeffs = fir_coeffs(frequency, n_taps=96)
win = raw_stim.time_as_index(25)[0]; step = raw_stim.time_as_index(2)[0]
raw_stim_SASS = SASS(raw_stim.copy(), raw_no_stim, lfreq,hfreq,P_laplace=P_laplace,filter_type='long',filter_coeffs=coeffs)
#raw_stim = window_SASS(raw_stim, raw_no_stim, lfreq,hfreq,P_laplace=P_laplace,filter_type='long',filter_coeffs=coeffs, window_size=win)

target_chan_stim_SASS = get_target_chan(raw_stim_SASS,inner_pick,outer_picks)
psds, freqs = psd_array_welch(target_chan_stim_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq*5))
plt.semilogy(freqs,psds,label="Stim after SASS")
plt.xlabel("Frequency in Hz")
plt.ylabel("Power Spectral Density in dB")
plt.title("Target channel")
plt.legend()

# Filter the data around the target frequency
#raw_stim.filter(lfreq,hfreq)

# Cut the data into epochs, get the mean amplitude for each condition and plot the phase locking in that condition
events = mne.events_from_annotations(raw_stim)[0]
# Delete the events of 1 that are not showing the delay (but after the delay)
cond_one = np.where(events[:,2] == 1)
smaller_200 = np.where(np.diff(events[:,0]) < 200)[0]
delete = np.intersect1d(smaller_200,cond_one)
events = np.delete(events, delete,axis=0)
conditions = ['no stim','in phase', 'anti phase', 'open loop']
event_ids = [1,2,3,4]
amplitudes = np.zeros((len(event_ids),1))
stds = np.zeros((len(event_ids),1))
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()

for i,event_id in enumerate(event_ids):
      epochs = mne.Epochs(raw_stim, events, event_id=event_id, tmin=0.2, tmax=3, detrend=0, baseline=None,preload=True)
      epochs_SASS = mne.Epochs(raw_stim_SASS, events, event_id=event_id, tmin=0.2, tmax=3, detrend=0, baseline=None,preload=True)

      # Compute the frequency spectrum
      target_chan = get_target_chan(epochs, inner_pick, outer_picks)
      target_chan_SASS = get_target_chan(epochs_SASS, inner_pick, outer_picks)
      psds, freqs = psd_array_welch(target_chan, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq * 2.5))
      ax = fig3.add_subplot(1, 1, 1)
      ax.semilogy(freqs, psds.mean(0), label=f"Stim {conditions[i]}")
      psds, freqs = psd_array_welch(target_chan_SASS, fmin=fmin, fmax=fmax, sfreq=sfreq, n_fft=int(sfreq * 2.5))
      ax.semilogy(freqs, psds.mean(0), label=f"Stim SASS {conditions[i]}")

      epochs_env = mne.Epochs(raw_stim_env, events, event_id=event_id, tmin=0.2, tmax=3, detrend=0, baseline=None,preload=True)
      env = epochs_env._data.squeeze()

      # Compute the frequency spectrum
      #psds, freqs = psd_array_welch(env.flatten(), fmin=fmin, fmax=fmax, sfreq=sfreq,n_fft=int(sfreq * 5))
      #ax = fig3.add_subplot(1,1,1)
      #ax.semilogy(freqs, psds, label=conditions[i])

      # Get the target channel and its hilbert transform
      target_chan = get_target_chan(epochs, inner_pick, outer_picks)

      # Plot the phase locking of the target_channel with the envelope
      phase_diff = compute_phase_diff(target_chan, env)
      ax = fig1.add_subplot(1, 4, i + 1,  projection='polar')
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      ax.hist(phase_diff.flatten(), color="blue", alpha=0.5)
      ax.set_title(f"{conditions[i]} °")

      # Plot the PLVs
      ax = fig2.add_subplot(1, 4, i + 1)
      plvs = np.array([plv(epoch_diff) for epoch_diff in phase_diff])
      ax.hist(plvs, color="blue", alpha=0.5)
      ax.set_title(f"{conditions[i]} °")
      ax.set_xlabel("PLV")

plt.subplots_adjust(wspace=0.5)
plt.legend()
plt.show()


